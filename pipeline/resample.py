"""
Temporal resampling utilities for WIMUSim.

Problem: different data sources run at different sample rates.
    MoVi  — SMPL fits: 60 Hz  |  IMU: 100 Hz
    EMDB  — SMPL fits: 60 Hz  |  IMU:  60 Hz
    TotalCapture — SMPL: 60 Hz | IMU:  60 Hz
    HMR2.0 output — 30 Hz (video FPS)

All signals fed into WIMUSim must share the same sample rate.
The recommended approach:
    - Downsample IMU to match SMPL rate  (avoids SMPL up-sampling artefacts)
    - OR up-sample SMPL to match IMU rate using SLERP on rotation matrices

Usage:
    from pipeline.resample import resample_imu_dict, resample_smpl

    # Downsample MoVi IMU from 100 Hz → 60 Hz
    imu_60 = resample_imu_dict(imu_dict, from_hz=100, to_hz=60)

    # Up-sample HMR2.0 SMPL from 30 Hz → 60 Hz
    go_60, bp_60 = resample_smpl(global_orient, body_pose, from_hz=30, to_hz=60)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# IMU resampling  (simple linear interpolation per axis)
# ---------------------------------------------------------------------------

def resample_signal(signal: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """
    Resample a 1-D or 2-D time-series signal.

    Args:
        signal:   (T,) or (T, N)  float32/float64
        from_hz:  original sample rate
        to_hz:    target sample rate

    Returns:
        Resampled signal of shape (T', ) or (T', N).
    """
    if from_hz == to_hz:
        return signal

    T = signal.shape[0]
    t_orig   = np.linspace(0, T / from_hz, T, endpoint=False)
    T_new    = int(round(T * to_hz / from_hz))
    t_new    = np.linspace(0, T / from_hz, T_new, endpoint=False)

    if signal.ndim == 1:
        f = interp1d(t_orig, signal, kind="linear", fill_value="extrapolate")
        return f(t_new).astype(signal.dtype)
    else:
        out = np.zeros((T_new, signal.shape[1]), dtype=signal.dtype)
        for i in range(signal.shape[1]):
            f = interp1d(t_orig, signal[:, i], kind="linear", fill_value="extrapolate")
            out[:, i] = f(t_new)
        return out


def resample_imu_dict(
    imu_dict: dict,
    from_hz: float,
    to_hz: float,
) -> dict:
    """
    Resample all IMU signals (acc + gyro) in a dict to a new sample rate.

    Args:
        imu_dict: {imu_name: (acc, gyro)}  acc/gyro shape (T, 3)
        from_hz:  original sample rate (e.g. 100 for MoVi IMU)
        to_hz:    target sample rate   (e.g.  60 for MoVi SMPL)

    Returns:
        Resampled dict with same structure.
    """
    if from_hz == to_hz:
        return imu_dict

    out = {}
    for name, (acc, gyro) in imu_dict.items():
        out[name] = (
            resample_signal(np.asarray(acc,  dtype=np.float32), from_hz, to_hz),
            resample_signal(np.asarray(gyro, dtype=np.float32), from_hz, to_hz),
        )
    return out


# ---------------------------------------------------------------------------
# SMPL pose resampling  (SLERP on rotation matrices)
# ---------------------------------------------------------------------------

def _rotmat_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    """(T, 3, 3) → (T, 3) axis-angle via scipy."""
    T = rotmat.shape[0]
    return Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec().reshape(T, 3)


def _slerp_rotmat(rotmat: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """
    SLERP a sequence of rotation matrices from t_orig to t_new.

    Args:
        rotmat: (T, 3, 3)
        t_orig: (T,) original time stamps
        t_new:  (T',) target time stamps

    Returns:
        (T', 3, 3)
    """
    rots  = Rotation.from_matrix(rotmat)
    slerp = Slerp(t_orig, rots)
    return slerp(t_new).as_matrix().astype(np.float32)


def resample_smpl(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    from_hz: float,
    to_hz: float,
) -> tuple:
    """
    Resample SMPL rotation matrices to a new frame rate using SLERP.

    Args:
        global_orient: (T, 3, 3)
        body_pose:     (T, 23, 3, 3)
        from_hz:       original frame rate  (e.g. 30 for HMR2.0)
        to_hz:         target frame rate    (e.g. 60 for MoVi IMU)

    Returns:
        global_orient_resampled: (T', 3, 3)
        body_pose_resampled:     (T', 23, 3, 3)
    """
    if from_hz == to_hz:
        return global_orient, body_pose

    T     = global_orient.shape[0]
    T_new = int(round(T * to_hz / from_hz))

    t_orig = np.linspace(0, T / from_hz, T,     endpoint=False)
    t_new  = np.linspace(0, T / from_hz, T_new, endpoint=False)

    # Clamp t_new to valid range
    t_new = np.clip(t_new, t_orig[0], t_orig[-1])

    go_r = _slerp_rotmat(global_orient, t_orig, t_new)          # (T', 3, 3)

    N = body_pose.shape[1]
    bp_r = np.stack(
        [_slerp_rotmat(body_pose[:, j], t_orig, t_new) for j in range(N)],
        axis=1,
    )                                                             # (T', 23, 3, 3)

    return go_r, bp_r


# ---------------------------------------------------------------------------
# Convenience: align everything to a common rate
# ---------------------------------------------------------------------------

def align_to_smpl_rate(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    imu_dict: dict,
    smpl_hz: float,
    imu_hz: float,
    video_hz: float = None,
) -> tuple:
    """
    Align all signals to the SMPL frame rate (recommended approach).

    1. If video_hz is given (HMR2.0 output), resample SMPL to smpl_hz first.
    2. Downsample IMU from imu_hz → smpl_hz.

    Returns:
        global_orient: (T, 3, 3)
        body_pose:     (T, 23, 3, 3)
        imu_dict:      {name: (acc, gyro)} at smpl_hz
        target_hz:     smpl_hz
    """
    # Step 1: resample SMPL if it came from video (different fps)
    if video_hz is not None and video_hz != smpl_hz:
        global_orient, body_pose = resample_smpl(
            global_orient, body_pose, from_hz=video_hz, to_hz=smpl_hz
        )

    # Step 2: resample IMU to match SMPL rate
    if imu_hz != smpl_hz:
        imu_dict = resample_imu_dict(imu_dict, from_hz=imu_hz, to_hz=smpl_hz)

    return global_orient, body_pose, imu_dict, smpl_hz
