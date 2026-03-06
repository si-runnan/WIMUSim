"""
PhyNeSim inference — augment WIMUSim physics simulation with neural correction.

Given SMPL poses, PhyNeSim runs WIMUSim physics first, then applies the
neural residual corrector to produce more accurate virtual IMU signals.

Usage:
    python -m nn.infer \
        --checkpoint  output/checkpoints/best.pt \
        --smpl_npz    /data/movi/F_Subject01/F_Subject01_walk01_poses.npz \
        --smpl_model  path/to/smpl/models \
        --output      output/corrected_imu.npz

Or use as a drop-in replacement for WIMUSim.simulate() in Python:

    from nn.infer import corrected_simulate

    imu_dict = corrected_simulate(
        checkpoint="output/checkpoints/best.pt",
        B=B, D=D, P=P, H=H,
    )
    # imu_dict: {imu_name: (acc, gyro)}  — same format as WIMUSim.simulate()
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nn.model import NeuralSimulator, quat_wxyz_to_rot6d
from nn.dataset import JOINT_ORDER, _stack_imu_dict
from wimusim.wimusim import WIMUSim


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[NeuralSimulator, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = NeuralSimulator(
        n_imus=cfg["n_imus"],
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        d_ff=cfg.get("d_ff", 512),
        dropout=0.0,
        residual=cfg.get("residual", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def corrected_simulate(
    checkpoint: str,
    B: "WIMUSim.Body",
    D: "WIMUSim.Dynamics",
    P: "WIMUSim.Placement",
    H: "WIMUSim.Hardware",
    window: Optional[int] = None,
    stride: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Drop-in replacement for WIMUSim.simulate() with neural correction.

    Runs WIMUSim physics, then applies the neural residual corrector,
    returning corrected virtual IMU signals in the same format.

    Args:
        checkpoint: Path to trained checkpoint (.pt file).
        B, D, P, H: WIMUSim parameters (same as WIMUSim.simulate()).
        window:     Sliding window size. Uses value from checkpoint if None.
        stride:     Inference stride between windows.
        device:     Torch device.

    Returns:
        {imu_name: (acc, gyro)}  — corrected virtual IMU, same as WIMUSim output.
    """
    dev = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model, cfg = load_checkpoint(checkpoint, dev)
    win = window or cfg["window"]
    imu_names = cfg["imu_names"]

    # --- Physics simulation ---
    env  = WIMUSim(B=B, D=D, P=P, H=H)
    virt = env.simulate(mode="generate")   # {name: (acc, gyro)}

    avail = [n for n in imu_names if n in virt]
    phys_tensor = _stack_imu_dict(
        {n: virt[n] for n in avail}, avail
    ).unsqueeze(0)                          # (1, T, N*6)

    # --- Pose features ---
    T = phys_tensor.shape[1]
    pose_feats = []
    for joint in JOINT_ORDER:
        q = D.orientation.get(joint)
        if q is None:
            q = torch.zeros(T, 4, device=dev); q[:, 0] = 1.0
        pose_feats.append(quat_wxyz_to_rot6d(q.to(dev)))
    pose_tensor = torch.cat(pose_feats, dim=-1).unsqueeze(0)  # (1, T, 24*6)

    # --- Sliding-window inference with overlap averaging ---
    n_ch  = phys_tensor.shape[-1]
    acc_c = torch.zeros(1, T, n_ch, device=dev)
    cnt   = torch.zeros(1, T, 1,    device=dev)

    phys_d = phys_tensor.to(dev)
    pose_d = pose_tensor.to(dev)

    for start in range(0, T - win + 1, stride):
        end  = start + win
        pred = model(pose_d[:, start:end], phys_d[:, start:end])
        acc_c[:, start:end] += pred
        cnt[:, start:end]   += 1.0

    # Handle tail
    if T > win and cnt[:, -1, 0] == 0:
        start = T - win
        pred  = model(pose_d[:, start:], phys_d[:, start:])
        acc_c[:, start:] += pred
        cnt[:, start:]   += 1.0

    cnt = cnt.clamp(min=1.0)
    corrected = (acc_c / cnt).squeeze(0)   # (T, N*6)

    # --- Unpack back into {name: (acc, gyro)} ---
    result = {}
    for i, name in enumerate(avail):
        base = i * 6
        result[name] = (
            corrected[:, base:base+3].cpu(),
            corrected[:, base+3:base+6].cpu(),
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PhyNeSim inference"
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--smpl_npz",   required=True,
                        help="Path to SMPL params .npz "
                             "(keys: betas, global_orient, body_pose)")
    parser.add_argument("--smpl_model", required=True,
                        help="Path to SMPL model directory")
    parser.add_argument("--output",     default="output/corrected_imu.npz")
    parser.add_argument("--stride",       type=int,   default=32)
    parser.add_argument("--sample_rate", type=float, default=60.0,
                        help="SMPL pose sample rate in Hz (default: 60)")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--gender",     default="neutral",
                        choices=["neutral", "male", "female"])
    args = parser.parse_args()

    import numpy as np
    from dataset_configs.smpl.utils import compute_B_from_beta, smpl_pose_to_D_orientation
    from dataset_configs.movi.utils import generate_default_placement_params
    from wimusim import utils as wu

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load SMPL params
    data = np.load(args.smpl_npz)
    betas        = data["betas"]
    global_orient = data["global_orient"]
    body_pose    = data["body_pose"]

    # Build WIMUSim params
    B_rp = compute_B_from_beta(betas, smpl_model_path=args.smpl_model,
                                gender=args.gender)
    orientation = smpl_pose_to_D_orientation(global_orient, body_pose)

    def _to_tensor(d):
        return {k: torch.tensor(v, dtype=torch.float32, device=device)
                for k, v in d.items()}

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    imu_names = ckpt["config"]["imu_names"]

    B = WIMUSim.Body(rp=_to_tensor(B_rp), device=device)
    D = WIMUSim.Dynamics(
        orientation=_to_tensor(orientation),
        sample_rate=args.sample_rate,
        device=device,
    )
    P_all = generate_default_placement_params(B_rp)
    P = WIMUSim.Placement(
        rp=_to_tensor({k: v for k, v in P_all["rp"].items() if k[1] in imu_names}),
        ro=_to_tensor({k: v for k, v in P_all["ro"].items() if k[1] in imu_names}),
        device=device,
    )
    _H = wu.generate_default_H_configs(imu_names)
    H = WIMUSim.Hardware(
        ba=_H["ba"], bg=_H["bg"], sa=_H["sa"], sg=_H["sg"],
        sa_range_dict=_H["sa_range_dict"], sg_range_dict=_H["sg_range_dict"],
    )

    print("Running corrected simulation...")
    imu_dict = corrected_simulate(
        checkpoint=args.checkpoint,
        B=B, D=D, P=P, H=H,
        stride=args.stride,
        device=device,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, (acc, gyro) in imu_dict.items():
        save_data[f"{name}_acc"]  = acc.numpy()
        save_data[f"{name}_gyro"] = gyro.numpy()
    np.savez(str(out_path), **save_data)
    print(f"Saved corrected IMU to {out_path}")


if __name__ == "__main__":
    main()
