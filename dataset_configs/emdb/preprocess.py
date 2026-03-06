"""
EMDB preprocessing script for WIMUSim.

EMDB provides SMPL-X parameters.  WIMUSim only needs the first 21 body joints
(shared with SMPL), so preprocessing simply:
  1. Reads raw EMDB .pkl files
  2. Extracts SMPL-compatible parameters (betas, global_orient, body_pose 1-21)
  3. Pads joints 22-23 (L_HAND / R_HAND) with identity rotations
  4. Re-saves as standardised .pkl files compatible with emdb/utils.py

The output format for each sequence matches what emdb/utils.py expects:
    {
        'smpl': {
            'betas':         np.ndarray (10,)
            'global_orient': np.ndarray (T, 3, 3)
            'body_pose':     np.ndarray (T, 21, 3, 3)   joints 1–21 only
        },
        'imu': {
            'acc':  np.ndarray (T, 6, 3)   sensor order: LLA RLA LSH RSH PELV BACK
            'gyro': np.ndarray (T, 6, 3)
        },
        'fps': int
    }

Usage:
    python -m dataset_configs.emdb.preprocess \
        --emdb_root  /data/EMDB \
        --output     /data/emdb_processed \
        [--split     EMDB_2]        # EMDB_1=indoor, EMDB_2=outdoor

Raw EMDB structure (as downloaded):
    {emdb_root}/
        EMDB_1/
            P1/
                seq_01.pkl
                ...
        EMDB_2/
            P1/
                seq_01.pkl

Each raw .pkl contains SMPL-X parameters and EM-tracker IMU readings.
The script handles both SMPL-X and SMPL raw formats transparently.
"""

import argparse
import pathlib
import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm

from dataset_configs.emdb.consts import SUBJECT_LIST, TRAIN_SPLIT, TEST_SPLIT

# Sensor name order expected by emdb/utils.py
IMU_ORDER = ["LLA", "RLA", "LSH", "RSH", "PELV", "BACK"]

# Map WIMUSim sensor name → typical EMDB raw sensor key variants
# (Raw EMDB uses different key names depending on release version)
SENSOR_KEY_VARIANTS = {
    "LLA":  ["left_forearm",   "left_lower_arm"],
    "RLA":  ["right_forearm",  "right_lower_arm"],
    "LSH":  ["left_lower_leg", "left_shin"],
    "RSH":  ["right_lower_leg","right_shin"],
    "PELV": ["pelvis",         "hips"],
    "BACK": ["upper_back",     "sternum", "spine"],
}


# ---------------------------------------------------------------------------
# Raw data parsing
# ---------------------------------------------------------------------------

def _load_raw(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _parse_smpl(raw: dict) -> tuple:
    """
    Extract SMPL-compatible parameters from a raw EMDB .pkl dict.

    Handles both SMPL-X and SMPL raw key layouts.

    Returns:
        betas:         (10,)    float32
        global_orient: (T,3,3)  float32
        body_pose:     (T,21,3,3) float32   joints 1-21 (rotation matrices)
    """
    # Prefer 'smpl' key, fall back to 'smplx'
    smpl_data = raw.get("smpl") or raw.get("smplx") or raw

    betas_raw = smpl_data.get("betas", np.zeros(10, dtype=np.float32))
    betas = np.asarray(betas_raw, dtype=np.float32).ravel()[:10]   # (10,)

    # global_orient: (T, 3, 3) or (T, 1, 3, 3) or (T, 3) axis-angle
    go = smpl_data.get("global_orient") or smpl_data.get("root_orient")
    go = np.asarray(go, dtype=np.float32)
    if go.ndim == 4:
        go = go[:, 0]           # (T, 1, 3, 3) → (T, 3, 3)
    elif go.shape[-1] == 3 and go.ndim == 2:
        # axis-angle (T, 3) → rotation matrix (T, 3, 3)
        from scipy.spatial.transform import Rotation
        go = Rotation.from_rotvec(go).as_matrix().astype(np.float32)
    global_orient = go          # (T, 3, 3)

    # body_pose: (T, 21/23, 3, 3) or (T, 21/23, 3) axis-angle
    bp = smpl_data.get("body_pose") or smpl_data.get("pose_body")
    bp = np.asarray(bp, dtype=np.float32)
    if bp.ndim == 3 and bp.shape[-1] == 3:
        # axis-angle (T, N, 3) → rotation matrix (T, N, 3, 3)
        from scipy.spatial.transform import Rotation
        T, N = bp.shape[:2]
        bp = Rotation.from_rotvec(bp.reshape(-1, 3)).as_matrix().reshape(T, N, 3, 3).astype(np.float32)
    body_pose = bp[:, :21]      # (T, 21, 3, 3) — first 21 body joints

    return betas, global_orient, body_pose


def _parse_imu(raw: dict) -> tuple:
    """
    Extract IMU acc and gyro in the order defined by IMU_ORDER.

    Returns:
        acc:  (T, 6, 3)  float32
        gyro: (T, 6, 3)  float32
    """
    imu_data = raw.get("imu") or raw

    # Case 1: array format  {'acc': (T, 6, 3), 'gyro': (T, 6, 3)}
    if "acc" in imu_data and isinstance(imu_data["acc"], np.ndarray):
        acc  = np.asarray(imu_data["acc"],  dtype=np.float32)   # (T, N, 3)
        gyro = np.asarray(imu_data["gyro"], dtype=np.float32)
        if acc.ndim == 2:   # (T*N, 3) flat → reshape
            N = len(IMU_ORDER)
            T = acc.shape[0] // N
            acc  = acc.reshape(T, N, 3)
            gyro = gyro.reshape(T, N, 3)
        return acc[:, :6], gyro[:, :6]

    # Case 2: dict format  {'left_forearm': {'acc': (T,3), 'gyro': (T,3)}, ...}
    T = None
    acc_list, gyro_list = [], []
    for ws_name in IMU_ORDER:
        found = False
        for key_variant in SENSOR_KEY_VARIANTS[ws_name]:
            if key_variant in imu_data:
                sensor = imu_data[key_variant]
                a = np.asarray(sensor.get("acc",  sensor.get("acceleration", np.zeros((1,3)))), dtype=np.float32)
                g = np.asarray(sensor.get("gyro", sensor.get("gyroscope",    np.zeros((1,3)))), dtype=np.float32)
                if T is None:
                    T = a.shape[0]
                acc_list.append(a[:T])
                gyro_list.append(g[:T])
                found = True
                break
        if not found:
            # Sensor not available — fill with zeros
            if T is None:
                T = 1
            acc_list.append(np.zeros((T, 3), dtype=np.float32))
            gyro_list.append(np.zeros((T, 3), dtype=np.float32))

    acc  = np.stack(acc_list,  axis=1)   # (T, 6, 3)
    gyro = np.stack(gyro_list, axis=1)
    return acc, gyro


def _get_fps(raw: dict) -> int:
    return int(raw.get("fps", raw.get("mocap_framerate", 60)))


# ---------------------------------------------------------------------------
# Main preprocessing loop
# ---------------------------------------------------------------------------

def preprocess(
    emdb_root: str,
    output_root: str,
    splits: Optional[list] = None,
    skip_existing: bool = True,
):
    """
    Preprocess EMDB sequences into WIMUSim-compatible format.

    Args:
        emdb_root:   Root directory of raw EMDB download.
        output_root: Where to write processed .pkl files (same structure).
        splits:      List of split names to process (default: both EMDB_1 and EMDB_2).
        skip_existing: Skip if output already exists.
    """
    splits = splits or [TRAIN_SPLIT, TEST_SPLIT]
    root_in  = pathlib.Path(emdb_root)
    root_out = pathlib.Path(output_root)

    sequences = []
    for split in splits:
        for subject_dir in sorted((root_in / split).iterdir()):
            if not subject_dir.is_dir():
                continue
            for pkl_path in sorted(subject_dir.glob("*.pkl")):
                sequences.append((split, subject_dir.name, pkl_path))

    print(f"Found {len(sequences)} sequences across splits: {splits}")

    ok = 0
    for split, subject, pkl_path in tqdm(sequences):
        out_path = root_out / split / subject / pkl_path.name
        if skip_existing and out_path.exists():
            continue

        try:
            raw = _load_raw(str(pkl_path))
            betas, global_orient, body_pose = _parse_smpl(raw)
            acc, gyro = _parse_imu(raw)
            fps = _get_fps(raw)
        except Exception as e:
            print(f"  SKIP {split}/{subject}/{pkl_path.name}: {e}")
            continue

        # Trim to same length
        T = min(global_orient.shape[0], body_pose.shape[0], acc.shape[0])
        global_orient = global_orient[:T]
        body_pose     = body_pose[:T]
        acc           = acc[:T]
        gyro          = gyro[:T]

        out_dict = {
            "smpl": {
                "betas":         betas,
                "global_orient": global_orient,
                "body_pose":     body_pose,
            },
            "imu": {
                "acc":  acc,
                "gyro": gyro,
            },
            "fps": fps,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "wb") as f:
            pickle.dump(out_dict, f)
        ok += 1

    print(f"\nDone. Processed {ok} / {len(sequences)} sequences → {root_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EMDB for WIMUSim (SMPL-X → SMPL-compatible format)"
    )
    parser.add_argument("--emdb_root", required=True,
                        help="Root directory of raw EMDB download")
    parser.add_argument("--output",    required=True,
                        help="Output directory for processed data")
    parser.add_argument("--split",     nargs="+", default=None,
                        choices=["EMDB_1", "EMDB_2"],
                        help="Splits to process (default: both)")
    parser.add_argument("--no_skip",   action="store_true",
                        help="Re-process even if output already exists")
    args = parser.parse_args()

    preprocess(
        emdb_root=args.emdb_root,
        output_root=args.output,
        splits=args.split,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
