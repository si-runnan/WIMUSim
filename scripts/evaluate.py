"""
Evaluate PhyNeSim on TotalCapture or EMDB test sets.

Physics-only baseline:
    python scripts/evaluate.py \
        --dataset     totalcapture \
        --data_root   /data/tc_processed \
        --smpl_model  path/to/smpl/models \
        --output_dir  results/

With neural residual correction:
    python scripts/evaluate.py \
        --dataset     totalcapture \
        --data_root   /data/tc_processed \
        --smpl_model  path/to/smpl/models \
        --checkpoint  output/checkpoints/best.pt \
        --output_dir  results/

Outputs:
    results/
        {dataset}_metrics.csv          per-sequence per-IMU metrics
        {dataset}_summary.csv          mean across all sequences
        {dataset}_per_imu_rmse.png     per-IMU RMSE bar chart (acc + gyro)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset_configs.smpl.utils import compute_B_from_beta, smpl_pose_to_D_orientation
from pipeline.evaluate import evaluate, save_metrics
from wimusim import WIMUSim, utils


# ---------------------------------------------------------------------------
# Single-sequence runner
# ---------------------------------------------------------------------------

def _run_sequence(betas, global_orient, body_pose, imu_names,
                  placement_fn, sample_rate, smpl_model, checkpoint, device):
    """Simulate one sequence. Returns {imu_name: (acc, gyro)}."""
    B_rp        = compute_B_from_beta(betas, smpl_model_path=smpl_model)
    orientation = smpl_pose_to_D_orientation(global_orient, body_pose)
    P_params    = placement_fn(B_rp)
    H           = utils.generate_default_H_configs(imu_names)

    if checkpoint is None:
        env = WIMUSim(
            B={"rp": B_rp},
            D={"orientation": orientation, "sample_rate": sample_rate},
            P=P_params,
            H=H,
            dataset_name="SMPL",
            device=device,
        )
        return env.simulate(mode="generate")

    # Neural-corrected path
    from nn.infer import corrected_simulate
    dev = torch.device(device)

    def _t(v):
        if isinstance(v, np.ndarray):
            return torch.tensor(v, dtype=torch.float32, device=dev)
        return v.to(dev) if isinstance(v, torch.Tensor) else v

    B_obj = WIMUSim.Body(rp={k: _t(v) for k, v in B_rp.items()}, device=dev)
    D_obj = WIMUSim.Dynamics(
        orientation={k: _t(v) for k, v in orientation.items()},
        sample_rate=sample_rate,
        device=dev,
    )
    P_obj = WIMUSim.Placement(
        rp={k: _t(v) for k, v in P_params["rp"].items()},
        ro={k: _t(v) for k, v in P_params["ro"].items()},
        device=dev,
    )
    _H = utils.generate_default_H_configs(imu_names)
    H_obj = WIMUSim.Hardware(
        ba=_H["ba"], bg=_H["bg"], sa=_H["sa"], sg=_H["sg"],
        sa_range_dict=_H["sa_range_dict"], sg_range_dict=_H["sg_range_dict"],
    )

    return corrected_simulate(
        checkpoint=checkpoint, B=B_obj, D=D_obj, P=P_obj, H=H_obj,
        device=dev,
    )


# ---------------------------------------------------------------------------
# Per-dataset evaluation loops
# ---------------------------------------------------------------------------

def eval_totalcapture(data_root, smpl_model, checkpoint, device, subjects, activities):
    from dataset_configs.totalcapture import consts as tc
    from dataset_configs.totalcapture.utils import (
        load_smpl_params, load_imu_data,
        generate_default_placement_params as tc_placement,
    )

    subj_list = subjects or tc.SUBJECT_LIST
    act_list  = activities or tc.ACTIVITY_LIST
    rows = []

    for subj in subj_list:
        for act in act_list:
            print(f"  {subj}/{act} ...", end=" ", flush=True)
            try:
                betas, go, bp = load_smpl_params(data_root, subj, act)
                real_imu      = load_imu_data(data_root, subj, act)
            except FileNotFoundError:
                print("skipped (no file)")
                continue

            try:
                virt_imu = _run_sequence(
                    betas, go, bp, tc.IMU_NAMES, tc_placement,
                    tc.SMPL_SAMPLE_RATE, smpl_model, checkpoint, device,
                )
            except Exception as e:
                print(f"skipped ({e})")
                continue

            df = evaluate(virt_imu, real_imu)
            df["subject"]  = subj
            df["activity"] = act
            rows.append(df)
            print("done")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def eval_emdb(data_root, smpl_model, checkpoint, device, split):
    from dataset_configs.emdb import consts as emdb
    from dataset_configs.emdb.utils import (
        load_smpl_params, load_imu_data, iter_sequences,
        generate_default_placement_params as emdb_placement,
    )

    rows = []
    for subj, seq_name, pkl_path in iter_sequences(data_root, split=split):
        print(f"  {subj}/{seq_name} ...", end=" ", flush=True)
        try:
            betas, go, bp = load_smpl_params(pkl_path)
            real_imu      = load_imu_data(pkl_path)
        except Exception as e:
            print(f"skipped ({e})")
            continue

        try:
            virt_imu = _run_sequence(
                betas, go, bp, emdb.IMU_NAMES, emdb_placement,
                emdb.SMPL_SAMPLE_RATE, smpl_model, checkpoint, device,
            )
        except Exception as e:
            print(f"skipped ({e})")
            continue

        df = evaluate(virt_imu, real_imu)
        df["subject"]  = subj
        df["sequence"] = seq_name
        rows.append(df)
        print("done")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_per_imu_rmse(df: pd.DataFrame, output_dir: Path, prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{prefix} — Per-IMU RMSE", fontsize=13)

    for ax, modality, unit in [
        (axes[0], "acc",  "m/s²"),
        (axes[1], "gyro", "rad/s"),
    ]:
        sub = df[(df["modality"] == modality) & (df["imu"] != "MEAN")]
        per_imu = sub.groupby("imu")["rmse"].mean().sort_values()
        per_imu.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{modality.upper()} RMSE ({unit})")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out = output_dir / f"{prefix}_per_imu_rmse.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PhyNeSim evaluation")
    parser.add_argument("--dataset",    required=True,
                        choices=["totalcapture", "emdb"],
                        help="Dataset to evaluate on")
    parser.add_argument("--data_root",  required=True,
                        help="Root of preprocessed dataset (output of preprocess.py)")
    parser.add_argument("--smpl_model", required=True,
                        help="Path to SMPL model directory (contains SMPL_NEUTRAL.pkl)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained PhyNeSim checkpoint (.pt). "
                             "Omit to run physics-only baseline.")
    parser.add_argument("--output_dir", default="results",
                        help="Where to save CSV and plots (default: results/)")
    parser.add_argument("--device",     default="cpu",
                        help="Torch device (cpu or cuda)")

    # TotalCapture filters
    parser.add_argument("--subjects",   nargs="+", default=None,
                        help="TotalCapture subjects to evaluate (e.g. --subjects s1 s5)")
    parser.add_argument("--activities", nargs="+", default=None,
                        help="TotalCapture activities to evaluate")

    # EMDB split
    parser.add_argument("--split",      default="EMDB_2",
                        choices=["EMDB_1", "EMDB_2"],
                        help="EMDB split (EMDB_1=indoor, EMDB_2=outdoor, default: EMDB_2)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode   = "PhyNeSim" if args.checkpoint else "Physics-only"
    prefix = args.dataset if args.dataset == "totalcapture" else f"emdb_{args.split.lower()}"

    print(f"\n=== Evaluating {args.dataset}  [{mode}] ===\n")

    if args.dataset == "totalcapture":
        df = eval_totalcapture(
            args.data_root, args.smpl_model, args.checkpoint,
            args.device, args.subjects, args.activities,
        )
    else:
        df = eval_emdb(
            args.data_root, args.smpl_model, args.checkpoint,
            args.device, args.split,
        )

    if df.empty:
        print("\nNo sequences evaluated. Check --data_root path.")
        sys.exit(1)

    # Summary: mean over all sequences (rows where imu == "MEAN")
    summary = (
        df[df["imu"] == "MEAN"]
        .groupby("modality")[["rmse", "mae", "pearson"]]
        .mean()
    )

    print(f"\n{'=' * 55}")
    print(f"  {args.dataset.upper()}  [{mode}]  — Overall Mean")
    print(f"{'=' * 55}")
    print(summary.round(4).to_string())

    # Save outputs
    metrics_path = out_dir / f"{prefix}_metrics.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    save_metrics(df, str(metrics_path))
    summary.to_csv(summary_path)
    print(f"\n  Summary saved to {summary_path}")

    plot_per_imu_rmse(df, out_dir, prefix)


if __name__ == "__main__":
    main()
