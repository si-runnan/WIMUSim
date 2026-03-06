"""
Training script for the IMU → SMPL body_pose estimator.

Usage:
    python -m nn.train \
        --cpm_pkl    output/cpm_params_movi.pkl \
        --output_dir output/checkpoints \
        [--n_combinations 500] \
        [--window 128] \
        [--stride 32] \
        [--d_model 256] \
        [--n_heads 4] \
        [--n_layers 4] \
        [--batch_size 64] \
        [--epochs 100] \
        [--lr 3e-4] \
        [--val_split 0.1] \
        [--device cuda] \
        [--wandb_project wimusim_pose]

Training data:
    Loaded from cpm_params_movi.pkl (produced by examples/parameter_identification.ipynb).
    For each sampled (B, D, P, H) combination, WIMUSim generates a virtual IMU sequence
    and the corresponding SMPL body_pose is extracted from D.orientation.

Loss:
    Geodesic (angular) loss — average rotation angle error in radians.
    At inference, multiply by 180/π to convert to degrees.
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from nn.model import PoseEstimator, geodesic_loss, rot6d_to_rotmat
from nn.dataset import PoseEstimDataset


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for imu, pose_6d in loader:
        imu     = imu.to(device)      # (B, T, N_imus * 6)
        pose_6d = pose_6d.to(device)  # (B, T, 23 * 6)

        pred_rotmat = model(imu)       # (B, T, 23, 3, 3)
        gt_rotmat   = rot6d_to_rotmat(
            pose_6d.view(*pose_6d.shape[:2], 23, 6)
        )                              # (B, T, 23, 3, 3)

        loss = geodesic_loss(pred_rotmat, gt_rotmat)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for imu, pose_6d in loader:
        imu     = imu.to(device)
        pose_6d = pose_6d.to(device)

        pred_rotmat = model(imu)
        gt_rotmat   = rot6d_to_rotmat(
            pose_6d.view(*pose_6d.shape[:2], 23, 6)
        )
        loss = geodesic_loss(pred_rotmat, gt_rotmat)
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cpm_pkl: str,
    output_dir: str,
    n_combinations: int = 500,
    window: int = 128,
    stride: int = 32,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    val_split: float = 0.1,
    device: str = "cuda",
    wandb_project: str = None,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print("=== Building dataset ===")
    dataset = PoseEstimDataset.from_cpm_pkl(
        cpm_pkl,
        window=window,
        stride=stride,
        n_combinations=n_combinations,
        device=device,
    )
    dataset.generate_data()

    n_val  = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"  Train windows : {n_train}")
    print(f"  Val   windows : {n_val}")
    print(f"  IMUs          : {dataset.n_imus}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = PoseEstimator(
        n_imus=dataset.n_imus,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model  ({n_params:,} parameters) ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------
    # 3. Optional W&B
    # ------------------------------------------------------------------
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            config=dict(
                n_combinations=n_combinations, window=window, stride=stride,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                batch_size=batch_size, epochs=epochs, lr=lr,
            ),
        )
        wandb.watch(model, log_freq=50)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print("\n=== Training ===")
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model,   val_loader,   device)
        scheduler.step()

        # Convert radians → degrees for readability
        train_deg = train_loss * 180 / 3.14159265
        val_deg   = val_loss   * 180 / 3.14159265
        elapsed   = time.time() - t0

        print(
            f"  Epoch {epoch:4d}/{epochs}"
            f"  train {train_deg:6.2f}°  val {val_deg:6.2f}°"
            f"  ({elapsed:.1f}s)"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "train/loss_rad": train_loss, "train/loss_deg": train_deg,
                "val/loss_rad":   val_loss,   "val/loss_deg":   val_deg,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = out_dir / "best.pt"
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": dict(
                    n_imus=dataset.n_imus,
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    d_ff=d_ff, dropout=dropout, window=window,
                ),
            }, ckpt_path)

    # Save final checkpoint
    torch.save({
        "epoch":    epochs,
        "model":    model.state_dict(),
        "val_loss": val_loss,
        "config": dict(
            n_imus=dataset.n_imus,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout, window=window,
        ),
    }, out_dir / "last.pt")

    print(f"\nBest val: {best_val * 180 / 3.14159265:.2f}°  →  {out_dir / 'best.pt'}")

    if use_wandb:
        import wandb
        wandb.finish()

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train IMU → SMPL body pose estimator"
    )
    parser.add_argument("--cpm_pkl",       required=True,
                        help="Path to cpm_params_movi.pkl")
    parser.add_argument("--output_dir",    default="output/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--n_combinations", type=int, default=500,
                        help="Number of CPM (B,D,P,H) combinations to simulate (-1 = all)")
    parser.add_argument("--window",        type=int, default=128)
    parser.add_argument("--stride",        type=int, default=32)
    parser.add_argument("--d_model",       type=int, default=256)
    parser.add_argument("--n_heads",       type=int, default=4)
    parser.add_argument("--n_layers",      type=int, default=4)
    parser.add_argument("--d_ff",          type=int, default=512)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--val_split",     type=float, default=0.1)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name (omit to disable)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    train(
        cpm_pkl=args.cpm_pkl,
        output_dir=args.output_dir,
        n_combinations=args.n_combinations,
        window=args.window,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
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
