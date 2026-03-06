"""
Evaluation module: compare virtual IMU data against real IMU data.

Metrics computed per IMU, per modality (acc / gyro):
    - RMSE   Root Mean Square Error      (lower is better)
    - MAE    Mean Absolute Error         (lower is better)
    - PCC    Pearson Correlation Coeff.  (higher is better, max 1.0)

Usage:
    from pipeline.evaluate import evaluate, print_metrics

    results = evaluate(virtual_IMU_dict, real_IMU_dict)
    print_metrics(results)
"""

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Pearson correlation across axes."""
    if pred.ndim == 1:
        pred, target = pred[:, None], target[:, None]
    pccs = []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], target[:, i]
        denom = np.std(p) * np.std(t)
        if denom < 1e-8:
            pccs.append(0.0)
        else:
            pccs.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(pccs))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    virtual_IMU_dict: dict,
    real_IMU_dict: dict,
    metrics: list = ("rmse", "mae", "pearson"),
    align_length: bool = True,
) -> pd.DataFrame:
    """
    Compare virtual IMU data against real IMU data.

    Args:
        virtual_IMU_dict: {imu_name: (acc_tensor, gyro_tensor)}
                          output of WIMUSim.simulate()
        real_IMU_dict:    {imu_name: (acc_array, gyro_array)}
                          ground-truth IMU recordings
        metrics:          which metrics to compute
        align_length:     if True, trim both signals to the shorter length

    Returns:
        pd.DataFrame with columns:
            imu | modality | rmse | mae | pearson
        Plus a summary row (mean across all IMUs).
    """
    metric_fns = {"rmse": rmse, "mae": mae, "pearson": pearson}
    rows = []

    common_imus = [k for k in virtual_IMU_dict if k in real_IMU_dict]
    if not common_imus:
        raise ValueError(
            f"No common IMU names between virtual {list(virtual_IMU_dict.keys())} "
            f"and real {list(real_IMU_dict.keys())}"
        )

    for imu_name in sorted(common_imus):
        virt_acc,  virt_gyro  = virtual_IMU_dict[imu_name]
        real_acc,  real_gyro  = real_IMU_dict[imu_name]

        # Convert tensors to numpy
        if isinstance(virt_acc, torch.Tensor):
            virt_acc  = virt_acc.detach().cpu().numpy()
        if isinstance(virt_gyro, torch.Tensor):
            virt_gyro = virt_gyro.detach().cpu().numpy()
        real_acc  = np.asarray(real_acc,  dtype=np.float32)
        real_gyro = np.asarray(real_gyro, dtype=np.float32)

        # Align length
        if align_length:
            n = min(virt_acc.shape[0], real_acc.shape[0])
            virt_acc,  real_acc  = virt_acc[:n],  real_acc[:n]
            virt_gyro, real_gyro = virt_gyro[:n], real_gyro[:n]

        for modality, virt, real in [("acc", virt_acc, real_acc),
                                      ("gyro", virt_gyro, real_gyro)]:
            row = {"imu": imu_name, "modality": modality}
            for m in metrics:
                row[m] = metric_fns[m](virt, real)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Summary row: mean across all IMUs for each modality
    for modality in ("acc", "gyro"):
        sub = df[df["modality"] == modality]
        summary = {"imu": "MEAN", "modality": modality}
        for m in metrics:
            summary[m] = sub[m].mean()
        rows.append(summary)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_metrics(df: pd.DataFrame) -> None:
    """Pretty-print the evaluation results."""
    for modality in ("acc", "gyro"):
        unit = "m/s²" if modality == "acc" else "rad/s"
        print(f"\n{'─' * 55}")
        print(f"  {modality.upper()}  ({unit})")
        print(f"{'─' * 55}")
        sub = df[df["modality"] == modality].copy()
        sub = sub.set_index("imu").drop(columns=["modality"])

        # Highlight MEAN row
        mean_row = sub.loc["MEAN"]
        other = sub.drop(index="MEAN")

        col_fmt = {c: "{:.4f}" for c in sub.columns}
        print(other.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"{'─' * 55}")
        print(f"{'MEAN':<12}", end="")
        for c in sub.columns:
            print(f"  {mean_row[c]:.4f}", end="")
        print()


def save_metrics(df: pd.DataFrame, path: str) -> None:
    """Save metrics DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Metrics saved to {path}")
