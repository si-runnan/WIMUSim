"""
Inference: real IMU data → SMPL body_pose.

Usage:
    python -m nn.infer \
        --checkpoint  output/checkpoints/best.pt \
        --imu_pkl     /data/tc_processed/s5/walking1/imu.pkl \
        --imu_names   HED STER PELV RUA LUA RLA LLA RHD LHD RTH LTH RSH LSH \
        --output      output/pred_pose.npz

Input IMU format (imu.pkl):
    {imu_name: {'acc': np.ndarray (T,3), 'gyro': np.ndarray (T,3)}}

Output (pred_pose.npz):
    body_pose: np.ndarray (T, 23, 3, 3)  — predicted rotation matrices
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from nn.model import PoseEstimator
from dataset_configs.smpl.consts import SMPL_BODY_POSE_JOINT_NAMES


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple:
    """
    Load a PoseEstimator from a checkpoint file.

    Returns:
        (model, config)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = PoseEstimator(
        n_imus=cfg["n_imus"],
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        d_ff=cfg.get("d_ff", 512),
        dropout=0.0,   # inference — no dropout
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_imu_pkl(pkl_path: str, imu_names: List[str]) -> np.ndarray:
    """
    Load IMU data from a .pkl file and stack into (T, N_imus * 6).

    Args:
        pkl_path:  Path to imu.pkl
        imu_names: Ordered list of IMU names to use (must match training order).

    Returns:
        np.ndarray (T, N_imus * 6)  float32
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    arrays = []
    for name in imu_names:
        if name not in raw:
            raise KeyError(f"IMU '{name}' not found in {pkl_path}. "
                           f"Available: {list(raw.keys())}")
        d = raw[name]
        acc  = np.asarray(d["acc"],  dtype=np.float32)   # (T, 3)
        gyro = np.asarray(d["gyro"], dtype=np.float32)   # (T, 3)
        arrays.append(np.concatenate([acc, gyro], axis=-1))  # (T, 6)

    return np.concatenate(arrays, axis=-1)   # (T, N_imus * 6)


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(
    model: PoseEstimator,
    imu: np.ndarray,
    window: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run sliding-window inference and average overlapping predictions.

    Args:
        imu:    (T, N_imus * 6)  float32
        window: same as used during training
        stride: step between windows (can differ from training stride)

    Returns:
        body_pose: (T, 23, 3, 3)  float32
    """
    T = imu.shape[0]
    imu_t = torch.from_numpy(imu).to(device)

    # Accumulator for averaging overlapping windows
    acc   = torch.zeros(T, 23, 3, 3, device=device)
    count = torch.zeros(T, 1, 1, 1,  device=device)

    for start in range(0, T - window + 1, stride):
        end = start + window
        x = imu_t[start:end].unsqueeze(0)        # (1, T_win, N_imus*6)
        pred = model(x).squeeze(0)               # (T_win, 23, 3, 3)
        acc[start:end]   += pred
        count[start:end] += 1.0

    # Handle the tail (last incomplete stride that wasn't covered)
    if T > window:
        start = T - window
        x = imu_t[start:].unsqueeze(0)
        pred = model(x).squeeze(0)
        mask = count[start:] == 0
        acc[start:][mask.expand_as(acc[start:])] = \
            pred[mask.squeeze(-1).squeeze(-1)]
        count[start:][count[start:] == 0] = 1.0

    count = count.clamp(min=1.0)
    body_pose = (acc / count).cpu().numpy().astype(np.float32)
    return body_pose


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IMU → SMPL body_pose inference"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--imu_pkl", required=True,
                        help="Path to imu.pkl with real IMU data")
    parser.add_argument("--imu_names", nargs="+", required=True,
                        help="Ordered IMU names (must match training order)")
    parser.add_argument("--output", default="output/pred_pose.npz",
                        help="Output .npz path")
    parser.add_argument("--stride", type=int, default=32,
                        help="Inference stride (default: 32 frames)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_checkpoint(args.checkpoint, device)
    window = cfg["window"]
    print(f"  Window: {window} frames  |  IMUs: {cfg['n_imus']}")

    print(f"Loading IMU data: {args.imu_pkl}")
    imu = load_imu_pkl(args.imu_pkl, args.imu_names)
    T = imu.shape[0]
    print(f"  {T} frames × {imu.shape[1]} channels")

    if T < window:
        # Pad with reflection
        pad = window - T
        imu = np.concatenate([imu, imu[-pad:][::-1]], axis=0)
        print(f"  (padded to {imu.shape[0]} frames)")

    print("Running inference...")
    body_pose = infer(model, imu, window=window, stride=args.stride, device=device)
    body_pose = body_pose[:T]   # trim any padding

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        body_pose=body_pose,
        imu_names=args.imu_names,
        joint_names=SMPL_BODY_POSE_JOINT_NAMES,
    )
    print(f"Saved: {out_path}  (body_pose: {body_pose.shape})")


if __name__ == "__main__":
    main()
