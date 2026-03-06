"""
TotalCapture preprocessing script for WIMUSim.

Converts raw TotalCapture files into the format expected by
dataset_configs/totalcapture/utils.py:

    {output_root}/
        s1/
            walking1/
                smpl.npz   — betas (10,), global_orient (T,3,3), body_pose (T,23,3,3)
                imu.pkl    — {wimusim_name: {'acc': (T,3), 'gyro': (T,3)}}
            ...
        s2/
            ...

Usage:
    python -m dataset_configs.totalcapture.preprocess \
        --tc_root    /data/TotalCapture \
        --amass_root /data/AMASS/TotalCapture \
        --output     /data/tc_processed \
        [--subjects  s1 s2 s3 s4 s5] \
        [--activities walking1 walking2 ...]

Data sources:
    - SMPL fits:  AMASS/TotalCapture subset (https://amass.is.tue.mpg.de/)
      Expected structure:
        {amass_root}/{Subject}/
            {subject}_{activity}_poses.npz   (AMASS format)
    - IMU data:   TotalCapture raw release (https://cvssp.org/data/totalcapture/)
      Expected structure:
        {tc_root}/{Subject}/
            {activity}/
                Sensors_{activity}.txt       (Xsens text export)

AMASS file format (standard across all AMASS subsets):
    poses:  (T, 156)   axis-angle, first 3 = global_orient, next 69 = body_pose (23 joints)
    betas:  (16,)      shape params (first 10 used)
    mocap_framerate: float

TotalCapture Xsens sensor text format:
    Tab-separated columns per sensor block:
        AccX AccY AccZ  GyrX GyrY GyrZ
    Sensor order (hardcoded in TC release): see SENSOR_ORDER below.
"""

import argparse
import pickle
import pathlib
import re
import sys
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_configs.totalcapture.consts import (
    ACTIVITY_LIST,
    IMU_NAME_PAIRS,
    SUBJECT_LIST,
)

# ---------------------------------------------------------------------------
# TotalCapture sensor column order in the raw Xsens text file
# ---------------------------------------------------------------------------
# Each sensor contributes 6 columns: AccX AccY AccZ GyrX GyrY GyrZ
# The order below matches the TotalCapture official release.
SENSOR_ORDER = [
    "Head",
    "Sternum",
    "Pelvis",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftLowerArm",
    "RightLowerArm",
    "LeftHand",
    "RightHand",
    "LeftUpperLeg",
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
]

# Map TotalCapture sensor label → WIMUSim IMU name
TC_LABEL_TO_WS = {tc: ws for ws, tc in IMU_NAME_PAIRS}

# AMASS subject folder names differ from TotalCapture "s1" notation
# TotalCapture AMASS subset uses e.g. "S1", "S2", …
AMASS_SUBJECT_MAP = {s: s.upper() for s in SUBJECT_LIST}

# Activity name map: TC activity name → AMASS file suffix
# AMASS TotalCapture files are named like: s1_acting1_poses.npz
# (subject lowercase, activity exactly as in ACTIVITY_LIST)
# No remapping needed — they match.


# ---------------------------------------------------------------------------
# SMPL loading from AMASS
# ---------------------------------------------------------------------------

def load_amass_smpl(amass_root: str, subject: str, activity: str):
    """
    Load SMPL parameters from an AMASS TotalCapture .npz file.

    AMASS convention:
        poses:  (T, 156)  — axis-angle, first 3 global_orient, rest body joints
        betas:  (16,)     — first 10 used as shape params
        mocap_framerate: float

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)
        body_pose:     np.ndarray (T, 23, 3, 3)
        framerate:     float
    """
    amass_subject = AMASS_SUBJECT_MAP[subject]  # e.g. "S1"
    # Filename convention in AMASS TotalCapture: {subject}_{activity}_poses.npz
    fname = f"{subject}_{activity}_poses.npz"
    npz_path = pathlib.Path(amass_root) / amass_subject / fname

    if not npz_path.exists():
        raise FileNotFoundError(
            f"AMASS file not found: {npz_path}\n"
            f"Expected structure: {{amass_root}}/S1/s1_walking1_poses.npz"
        )

    data = np.load(str(npz_path))
    poses = data["poses"].astype(np.float32)       # (T, 156)
    betas = data["betas"][:10].astype(np.float32)  # (10,)
    framerate = float(data.get("mocap_framerate", 60.0))

    T = poses.shape[0]

    # Global orient: first 3 columns → axis-angle → rotation matrix
    global_orient_aa = poses[:, :3]               # (T, 3)
    global_orient = (
        Rotation.from_rotvec(global_orient_aa.reshape(-1, 3))
        .as_matrix()
        .reshape(T, 3, 3)
        .astype(np.float32)
    )

    # Body pose: columns 3..72 → 23 joints × 3 axis-angle
    body_pose_aa = poses[:, 3:72].reshape(T, 23, 3)  # (T, 23, 3)
    body_pose = (
        Rotation.from_rotvec(body_pose_aa.reshape(-1, 3))
        .as_matrix()
        .reshape(T, 23, 3, 3)
        .astype(np.float32)
    )

    return betas, global_orient, body_pose, framerate


# ---------------------------------------------------------------------------
# IMU loading from raw TotalCapture text files
# ---------------------------------------------------------------------------

def _find_imu_file(tc_root: str, subject: str, activity: str) -> pathlib.Path:
    """
    Locate the Xsens sensor text file for a given subject/activity.

    TotalCapture releases the IMU data as:
        {tc_root}/{Subject}/{activity}/Sensors_{activity}.txt
    where {Subject} is capitalised (S1, S2, …).
    """
    subject_folder = pathlib.Path(tc_root) / subject.upper() / activity
    candidates = list(subject_folder.glob("Sensors*.txt"))
    if not candidates:
        # Also try lowercase subject folder
        subject_folder = pathlib.Path(tc_root) / subject / activity
        candidates = list(subject_folder.glob("Sensors*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Sensors*.txt found in {subject_folder}\n"
            f"Expected TotalCapture IMU file at: "
            f"{{tc_root}}/S1/{activity}/Sensors_{activity}.txt"
        )
    return candidates[0]


def load_tc_imu(tc_root: str, subject: str, activity: str) -> dict:
    """
    Parse the TotalCapture Xsens text export and return a WIMUSim-compatible
    IMU dict.

    Returns:
        imu_dict: {wimusim_name: (acc, gyro)}
                  acc, gyro: np.ndarray (T, 3)  [m/s², rad/s]
    """
    imu_file = _find_imu_file(tc_root, subject, activity)

    # Read all numeric lines (skip header lines starting with non-numeric chars)
    rows = []
    with open(imu_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header lines
            if re.match(r"^[A-Za-z#]", line):
                continue
            vals = line.split()
            if len(vals) == len(SENSOR_ORDER) * 6:
                rows.append([float(v) for v in vals])

    if not rows:
        raise ValueError(f"No valid data rows found in {imu_file}")

    data = np.array(rows, dtype=np.float32)  # (T, N_sensors * 6)

    imu_dict = {}
    for col_idx, tc_label in enumerate(SENSOR_ORDER):
        ws_name = TC_LABEL_TO_WS.get(tc_label)
        if ws_name is None:
            continue
        base = col_idx * 6
        acc  = data[:, base + 0 : base + 3]   # (T, 3) m/s²
        gyro = data[:, base + 3 : base + 6]   # (T, 3) rad/s
        imu_dict[ws_name] = (acc, gyro)

    return imu_dict


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------

def _trim_to_min_length(*arrays) -> list:
    """Trim all arrays to the shortest length along axis 0."""
    min_len = min(a.shape[0] for a in arrays)
    return [a[:min_len] for a in arrays]


def align_sequences(global_orient, body_pose, imu_dict):
    """
    Trim all sequences to the same length.

    For TotalCapture, SMPL and IMU share the same 60 Hz rate,
    so no resampling is needed — only length alignment.

    Returns aligned (global_orient, body_pose, imu_dict).
    """
    arrays = [global_orient, body_pose] + [v[0] for v in imu_dict.values()]
    min_len = min(a.shape[0] for a in arrays)

    global_orient = global_orient[:min_len]
    body_pose     = body_pose[:min_len]
    imu_dict_out  = {
        k: (acc[:min_len], gyro[:min_len])
        for k, (acc, gyro) in imu_dict.items()
    }
    return global_orient, body_pose, imu_dict_out


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_smpl(out_dir: pathlib.Path, betas, global_orient, body_pose):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "smpl.npz"),
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
    )


def save_imu(out_dir: pathlib.Path, imu_dict: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Convert to {tc_label: {'acc': ..., 'gyro': ...}} format
    raw_dict = {}
    for ws_name, (acc, gyro) in imu_dict.items():
        raw_dict[ws_name] = {"acc": acc, "gyro": gyro}
    with open(str(out_dir / "imu.pkl"), "wb") as f:
        pickle.dump(raw_dict, f)


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def preprocess(
    tc_root: str,
    amass_root: str,
    output_root: str,
    subjects: Optional[List[str]] = None,
    activities: Optional[List[str]] = None,
    skip_existing: bool = True,
):
    """
    Preprocess TotalCapture: convert AMASS SMPL fits + raw IMU data into
    WIMUSim-compatible format.

    Args:
        tc_root:      Root directory of TotalCapture raw release.
        amass_root:   Root directory of AMASS TotalCapture subset.
        output_root:  Where to write processed data.
        subjects:     List of subjects to process (default: all).
        activities:   List of activities to process (default: all).
        skip_existing: Skip if output already exists.
    """
    subjects   = subjects   or SUBJECT_LIST
    activities = activities or ACTIVITY_LIST
    out_root   = pathlib.Path(output_root)

    total = len(subjects) * len(activities)
    done = 0

    for subject in subjects:
        for activity in activities:
            out_dir = out_root / subject / activity
            done += 1
            prefix = f"[{done:3d}/{total}] {subject}/{activity}"

            if skip_existing and (out_dir / "smpl.npz").exists() and (out_dir / "imu.pkl").exists():
                print(f"{prefix}  — skipped (already exists)")
                continue

            # --- SMPL from AMASS ---
            try:
                betas, global_orient, body_pose, smpl_fps = load_amass_smpl(
                    amass_root, subject, activity
                )
            except FileNotFoundError as e:
                print(f"{prefix}  — SMPL not found, skipping: {e}")
                continue

            # Warn if AMASS framerate ≠ 60 Hz
            if abs(smpl_fps - 60.0) > 0.5:
                print(f"  WARNING: AMASS framerate {smpl_fps:.1f} Hz ≠ 60 Hz for {subject}/{activity}")

            # --- IMU from raw TotalCapture ---
            try:
                imu_dict = load_tc_imu(tc_root, subject, activity)
            except (FileNotFoundError, ValueError) as e:
                print(f"{prefix}  — IMU not found, skipping: {e}")
                continue

            # --- Align lengths ---
            global_orient, body_pose, imu_dict = align_sequences(
                global_orient, body_pose, imu_dict
            )

            # --- Save ---
            save_smpl(out_dir, betas, global_orient, body_pose)
            save_imu(out_dir, imu_dict)

            T = global_orient.shape[0]
            print(f"{prefix}  — OK  ({T} frames, {len(imu_dict)} IMUs)")

    print("\nPreprocessing complete.")
    print(f"Output: {out_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TotalCapture for WIMUSim "
                    "(AMASS SMPL fits + raw Xsens IMU → smpl.npz + imu.pkl)"
    )
    parser.add_argument(
        "--tc_root", required=True,
        help="Root directory of TotalCapture raw release "
             "(contains S1/, S2/, … with Sensors_*.txt files)"
    )
    parser.add_argument(
        "--amass_root", required=True,
        help="Root directory of AMASS TotalCapture subset "
             "(contains S1/s1_walking1_poses.npz, …)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        choices=SUBJECT_LIST,
        help="Subjects to process (default: all)"
    )
    parser.add_argument(
        "--activities", nargs="+", default=None,
        choices=ACTIVITY_LIST,
        help="Activities to process (default: all)"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-process even if output already exists"
    )
    args = parser.parse_args()

    preprocess(
        tc_root      = args.tc_root,
        amass_root   = args.amass_root,
        output_root  = args.output,
        subjects     = args.subjects,
        activities   = args.activities,
        skip_existing= not args.no_skip,
    )


if __name__ == "__main__":
    main()
