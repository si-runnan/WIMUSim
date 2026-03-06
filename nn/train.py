"""
PhyNeSim training script.

Trains the PhyNeSim neural residual corrector: the network learns to predict
the residual between WIMUSim physics output and real IMU signals,
using MoVi paired data (SMPL poses + real IMU).

Usage (with real Xsens IMU as target — recommended):
    python -m nn.train \
        --amass_root     /data/MoVi/amass \
        --xsens_root     /data/MoVi/xsens \
        --smpl_model     path/to/smpl/models \
        --imu_names      HED STER PELV RUA LUA RLA LLA RHD LHD RTH LTH RSH LSH RFT LFT \
        --output_dir     output/checkpoints \
        [--window        128] \
        [--stride        32] \
        [--epochs        100] \
        [--lr            3e-4] \
        [--wandb_project wimusim_corrector]

Usage (with v3d-derived kinematics as target):
    python -m nn.train \
        --amass_root     /data/MoVi/amass \
        --v3d_root       /data/MoVi/v3d \
        --smpl_model     path/to/smpl/models \
        --output_dir     output/checkpoints

Training objective:
    Minimise Smooth L1 loss between corrected IMU and real IMU.
    Loss is computed separately for accelerometer and gyroscope channels
    (different physical scales) then summed.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from nn.model import NeuralSimulator, simulator_loss
from nn.dataset import SimulatorDataset


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, n_imus):
    model.train()
    total = 0.0
    for pose_6d, phys_imu, real_imu in loader:
        pose_6d  = pose_6d.to(device)
        phys_imu = phys_imu.to(device)
        real_imu = real_imu.to(device)

        pred = model(pose_6d, phys_imu)
        loss = simulator_loss(pred, real_imu, n_imus=n_imus)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, n_imus):
    model.eval()
    total = 0.0
    for pose_6d, phys_imu, real_imu in loader:
        pose_6d  = pose_6d.to(device)
        phys_imu = phys_imu.to(device)
        real_imu = real_imu.to(device)

        pred = model(pose_6d, phys_imu)
        loss = simulator_loss(pred, real_imu, n_imus=n_imus)
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def physics_baseline(loader, device, n_imus):
    """Compute loss of physics-only output (no neural correction), as baseline."""
    total = 0.0
    for _, phys_imu, real_imu in loader:
        phys_imu = phys_imu.to(device)
        real_imu = real_imu.to(device)
        total += simulator_loss(phys_imu, real_imu, n_imus=n_imus).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    amass_root: str,
    smpl_model: str,
    imu_names: list,
    output_dir: str,
    v3d_root: str = None,
    xsens_root: str = None,
    subjects: list = None,
    activity_indices: list = None,
    window: int = 128,
    stride: int = 32,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    residual: bool = True,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    val_split: float = 0.15,
    device: str = "cuda",
    wandb_project: str = None,
    seed: int = 42,
):
    if v3d_root is None and xsens_root is None:
        raise ValueError("--v3d_root is required when --xsens_root is not provided.")

    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print("=== Loading MoVi data ===")
    dataset = SimulatorDataset.from_movi(
        amass_root=amass_root,
        v3d_root=v3d_root,
        smpl_model_path=smpl_model,
        imu_names=imu_names,
        xsens_root=xsens_root,
        subjects=subjects,
        activity_indices=activity_indices,
        window=window,
        stride=stride,
        device=device,
    )
    dataset.generate_data()

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    n_imus = dataset.n_imus
    print(f"  Train: {n_train} windows  |  Val: {n_val} windows  |  IMUs: {n_imus}")

    # ------------------------------------------------------------------
    # 2. Physics baseline
    # ------------------------------------------------------------------
    baseline = physics_baseline(val_loader, device, n_imus)
    print(f"\n  Physics-only baseline loss: {baseline:.4f}")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = NeuralSimulator(
        n_imus=n_imus,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        residual=residual,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== NeuralSimulator ({n_params:,} params) ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------
    # 4. W&B
    # ------------------------------------------------------------------
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            config=dict(
                imu_names=imu_names, window=window, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers, residual=residual,
                batch_size=batch_size, epochs=epochs, lr=lr,
                physics_baseline=baseline,
            ),
        )
        wandb.watch(model, log_freq=50)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print("\n=== Training ===")
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr  = train_epoch(model, train_loader, optimizer, device, n_imus)
        val = eval_epoch(model,   val_loader,   device,   n_imus)
        scheduler.step()

        improvement = (baseline - val) / baseline * 100
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:4d}/{epochs}"
            f"  train {tr:.4f}  val {val:.4f}"
            f"  ({improvement:+.1f}% vs physics)  {elapsed:.1f}s"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "train/loss": tr, "val/loss": val,
                "val/improvement_pct": improvement,
                "val/physics_baseline": baseline,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

        if val < best_val:
            best_val = val
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val,
                "physics_baseline": baseline,
                "config": dict(
                    n_imus=n_imus, imu_names=imu_names,
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    d_ff=d_ff, residual=residual, window=window,
                ),
            }, out_dir / "best.pt")

    torch.save({
        "epoch": epochs,
        "model": model.state_dict(),
        "val_loss": val,
        "physics_baseline": baseline,
        "config": dict(
            n_imus=n_imus, imu_names=imu_names,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, residual=residual, window=window,
        ),
    }, out_dir / "last.pt")

    improvement = (baseline - best_val) / baseline * 100
    print(f"\nBest val: {best_val:.4f}  ({improvement:+.1f}% vs physics baseline {baseline:.4f})")

    if use_wandb:
        import wandb
        wandb.finish()

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PhyNeSim neural residual corrector"
    )
    parser.add_argument("--amass_root", required=True,
                        help="Root directory of F_amass downloads (F_amass_Subject_N.mat)")
    parser.add_argument("--v3d_root",   default=None,
                        help="Root directory of MoVi v3d data (F_v3d_Subject_N.mat). "
                             "Required when xsens_root is not set. When xsens_root is "
                             "set, used for activity segmentation (optional — omit to "
                             "derive boundaries from AMASS file lengths instead).")
    parser.add_argument("--xsens_root", default=None,
                        help="Root directory of MoVi Xsens data (imu_Subject_N.mat). "
                             "When provided, uses real physical IMU as training target "
                             "instead of v3d-derived kinematics.")
    parser.add_argument("--smpl_model", required=True,
                        help="Path to SMPL model directory (contains SMPL_NEUTRAL.pkl)")
    parser.add_argument("--imu_names",  nargs="+",
                        default=["HED","STER","PELV","RUA","LUA","RLA","LLA",
                                 "RHD","LHD","RTH","LTH","RSH","LSH","RFT","LFT"],
                        help="IMU sensor names to use")
    parser.add_argument("--output_dir", default="output/checkpoints")
    parser.add_argument("--subjects",          nargs="+", type=int, default=None,
                        help="Subject numbers to use (1-90). Default: subjects 1-60.")
    parser.add_argument("--activity_indices",  nargs="+", type=int, default=None,
                        help="Activity indices to use (0-20). Default: all 21.")
    parser.add_argument("--window",     type=int,   default=128)
    parser.add_argument("--stride",     type=int,   default=32)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--d_ff",       type=int,   default=512)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--no_residual", action="store_true",
                        help="Predict IMU directly (no residual mode)")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        amass_root=args.amass_root,
        v3d_root=args.v3d_root,
        smpl_model=args.smpl_model,
        imu_names=args.imu_names,
        output_dir=args.output_dir,
        xsens_root=args.xsens_root,
        subjects=args.subjects,
        activity_indices=args.activity_indices,
        window=args.window,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        residual=not args.no_residual,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
