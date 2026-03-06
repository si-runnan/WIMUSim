"""
End-to-end pipeline: Video → Virtual IMU data.

Usage:
    python pipeline/run.py \
        --video input.mp4 \
        --smpl_model path/to/smpl/models \
        --imu LLA RLA LSH RSH PELV \
        --output output/imu_data.npz

Steps:
    1. video_to_smpl()            Video → SMPL (β, global_orient, body_pose)
    2. compute_B_from_beta()      β → bone vectors (B)
    3. smpl_pose_to_D_orientation() pose → joint quaternions (D)
    4. generate_default_placement_params() → IMU placement (P)
    5. WIMUSim.simulate()         B + D + P + H → virtual IMU (acc, gyro)
    6. Save results to .npz / .csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.video_to_smpl import video_to_smpl
from dataset_configs.smpl.utils import compute_B_from_beta, smpl_pose_to_D_orientation
from dataset_configs.movi.utils import generate_default_placement_params, generate_B_range
from wimusim import WIMUSim, utils


def run(
    video_path: str,
    smpl_model_path: str,
    imu_names: list,
    output_path: str,
    gender: str = "neutral",
    sample_rate: int = 30,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    smooth: bool = True,
    save_csv: bool = False,
):
    """
    Full pipeline from video to virtual IMU data.

    Args:
        video_path:      Path to input video.
        smpl_model_path: Directory containing SMPL_NEUTRAL.pkl (or gender variants).
        imu_names:       List of IMU names to simulate, e.g. ["LLA", "RLA", "LSH"].
                         Must be a subset of MoVi IMU names (see dataset_configs/movi/consts.py).
        output_path:     Path to save output .npz file.
        gender:          "neutral", "male", or "female".
        sample_rate:     Frame rate of the SMPL pose output (default: 30 Hz from HMR2.0).
        device:          "cuda" or "cpu".
        smooth:          Apply smoothing to HMR2.0 output.
        save_csv:        Also save individual IMU CSVs alongside the .npz.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_npz = str(output_path.parent / "smpl_params.npz")

    print("=" * 60)
    print(f"Step 1/5  Video → SMPL  ({video_path})")
    print("=" * 60)
    betas, global_orient, body_pose = video_to_smpl(
        video_path=video_path,
        output_npz=smpl_npz,
        device=device,
        smooth=smooth,
    )

    print("\n" + "=" * 60)
    print("Step 2/5  β → Body (B) parameters")
    print("=" * 60)
    B_rp = compute_B_from_beta(betas, smpl_model_path=smpl_model_path, gender=gender)
    print(f"  Computed {len(B_rp)} bone vectors.")

    print("\n" + "=" * 60)
    print("Step 3/5  Pose → Dynamics (D) parameters")
    print("=" * 60)
    orientation = smpl_pose_to_D_orientation(global_orient, body_pose)
    T = list(orientation.values())[0].shape[0]
    print(f"  {T} frames × 24 joints.")

    print("\n" + "=" * 60)
    print("Step 4/5  Placement (P) + Hardware (H) defaults")
    print("=" * 60)
    P_all = generate_default_placement_params(B_rp)

    # Filter to only the requested IMU names
    P_dict = {
        "rp": {k: v for k, v in P_all["rp"].items() if k[1] in imu_names},
        "ro": {k: v for k, v in P_all["ro"].items() if k[1] in imu_names},
    }
    if not P_dict["rp"]:
        raise ValueError(
            f"None of the requested IMUs {imu_names} are defined in the default placement. "
            f"Available: {[k[1] for k in P_all['rp']]}"
        )
    H_dict = utils.generate_default_H_configs(imu_names)
    print(f"  IMUs: {imu_names}")

    print("\n" + "=" * 60)
    print("Step 5/5  WIMUSim → Virtual IMU data")
    print("=" * 60)
    D_dict = {"orientation": orientation, "sample_rate": sample_rate}
    wimusim_env = WIMUSim(
        B={"rp": B_rp},
        D=D_dict,
        P=P_dict,
        H=H_dict,
        dataset_name="SMPL",
        device=device,
    )
    virtual_IMU_dict = wimusim_env.simulate(mode="generate")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_data = {}
    for imu_name in imu_names:
        if imu_name not in virtual_IMU_dict:
            print(f"  Warning: {imu_name} not in simulation output, skipping.")
            continue
        acc, gyro = virtual_IMU_dict[imu_name]
        save_data[f"{imu_name}_acc"]  = acc.detach().cpu().numpy()
        save_data[f"{imu_name}_gyro"] = gyro.detach().cpu().numpy()

    np.savez(str(output_path), **save_data, sample_rate=sample_rate)
    print(f"\n  Saved virtual IMU data to {output_path}")

    if save_csv:
        import pandas as pd
        for imu_name in imu_names:
            if f"{imu_name}_acc" not in save_data:
                continue
            acc  = save_data[f"{imu_name}_acc"]
            gyro = save_data[f"{imu_name}_gyro"]
            df = pd.DataFrame({
                "acc_x":  acc[:, 0],  "acc_y":  acc[:, 1],  "acc_z":  acc[:, 2],
                "gyro_x": gyro[:, 0], "gyro_y": gyro[:, 1], "gyro_z": gyro[:, 2],
            })
            csv_path = output_path.parent / f"{imu_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path}")

    print("\nDone.")
    return virtual_IMU_dict


def main():
    parser = argparse.ArgumentParser(
        description="PhyNeSim: Video → Virtual IMU data"
    )
    parser.add_argument("--video",       required=True,  help="Path to input video")
    parser.add_argument("--smpl_model",  required=True,  help="Path to SMPL model directory (contains SMPL_NEUTRAL.pkl)")
    parser.add_argument("--imu",         nargs="+",      default=["LLA", "RLA", "LSH", "RSH", "PELV"],
                        help="IMU names to simulate (default: LLA RLA LSH RSH PELV)")
    parser.add_argument("--output",      default="output/imu_data.npz", help="Output .npz path")
    parser.add_argument("--gender",      default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--sample_rate", type=int, default=30, help="HMR2.0 output frame rate")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_smooth",   action="store_true", help="Disable pose smoothing")
    parser.add_argument("--csv",         action="store_true", help="Also save individual CSV files")
    args = parser.parse_args()

    run(
        video_path=args.video,
        smpl_model_path=args.smpl_model,
        imu_names=args.imu,
        output_path=args.output,
        gender=args.gender,
        sample_rate=args.sample_rate,
        device=args.device,
        smooth=not args.no_smooth,
        save_csv=args.csv,
    )


if __name__ == "__main__":
    main()
