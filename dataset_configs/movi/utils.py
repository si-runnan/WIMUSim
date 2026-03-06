"""
Utility functions for loading MoVi dataset and converting to WIMUSim parameters.

Data layout:

    amass_root/
        F_amass_Subject_{N}.mat         all 21 activities per subject, 120 Hz
        Structure: Subject_N_F_amass['move'][i]:
            description          : activity name string
            jointsExpMaps_amass  : (T, 52, 3) axis-angle (joints 0=root, 1-21=body)
            jointsBetas_amass    : (16,) shape params
            RootTranslation_amass: (T, 3) root translation in metres

    xsens_root/
        imu_Subject_{N}.mat             real Xsens IMU, 21 joints, 100 Hz

v3d .mat structure (optional, for activity segmentation):
    Subject_{N}_F['move']:
        jointsTranslation_v3d : (T_all, 15, 3)  mm, 120 Hz, all activities concat'd
        jointsAffine_v3d      : (T_all, 15, 4, 4) rotation+translation, 120 Hz
        flags120              : (21, 2)  1-indexed start/end frames per activity
        motions_list          : (21,)   activity names

IMU signals derived from v3d kinematics use the same conventions as WIMUSim:
    gravity vector  = [0, 0, -9.8] m/s²  (Z-up world frame)
    acc = R^T @ (d²p/dt² - g_world)      (specific force in body frame)
    gyro = R^T @ ω_world                  (angular velocity in body frame)
"""

from pathlib import Path

import numpy as np
import torch
import pytorch3d.transforms.rotation_conversions as rc

from dataset_configs.movi.consts import (
    V3D_SAMPLE_RATE, SMPL_SAMPLE_RATE, XSENS_SAMPLE_RATE,
    V3D_SEGMENT_NAMES, V3D_SEG_TO_IMU, IMU_TO_SEG_IDX,
    IMU_PARENT_JOINT, XSENS_JOINT_TO_IMU,
)
from dataset_configs.smpl.utils import _to_rotmat, _rotmat_to_quat_wxyz


# ---------------------------------------------------------------------------
# SMPL parameter loading  (F_amass_Subject_N.mat format)
# ---------------------------------------------------------------------------

# Activity name normalisation: F_amass .mat names → V3D_MOTION_LIST names
_MAT_TO_V3D_NAME = {
    "cross_arms":    "crossarms",
    "jumping_jacks": "jumping_jack",
    "throw/catch":   "throw_catch",
    "squatting_rm":  "dancing_rm",   # subject-specific variant of dancing_rm
}

def _norm_activity_name(name: str) -> str:
    """Normalise an activity name for matching across data sources."""
    return _MAT_TO_V3D_NAME.get(name.lower().strip(), name.lower().strip())


def load_smpl_params(amass_root: str, subject_num: int, seq_id: int):
    """
    Load SMPL parameters for one activity from F_amass_Subject_N.mat.

    Args:
        amass_root:  Root directory containing F_amass_Subject_N.mat files.
        subject_num: Subject number (1-90).
        seq_id:      Activity index 1-21 (1-indexed, V3D_MOTION_LIST order).

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)  rotation matrices
        body_pose:     np.ndarray (T, 23, 3, 3)  (joints 22-23 = identity)
        trans:         np.ndarray (T, 3)  root translation in metres
    """
    import scipy.io
    from dataset_configs.movi.consts import V3D_MOTION_LIST

    mat_path = Path(amass_root) / f"F_amass_Subject_{subject_num}.mat"
    mat  = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    subj = mat[f"Subject_{subject_num}_F_amass"]

    target = _norm_activity_name(V3D_MOTION_LIST[seq_id - 1])
    move_data = None
    for move in subj["move"]:
        if _norm_activity_name(move["description"]) == target:
            move_data = move
            break
    if move_data is None:
        raise KeyError(
            f"Activity '{V3D_MOTION_LIST[seq_id - 1]}' not found in {mat_path}"
        )

    exp_maps = move_data["jointsExpMaps_amass"].astype(np.float32)  # (T, 52, 3)
    betas    = move_data["jointsBetas_amass"][:10].astype(np.float32)
    trans    = move_data["RootTranslation_amass"].astype(np.float32)
    T        = exp_maps.shape[0]

    global_orient = rc.axis_angle_to_matrix(
        torch.tensor(exp_maps[:, 0, :])
    ).numpy()                                                            # (T, 3, 3)

    body_rot_21 = rc.axis_angle_to_matrix(
        torch.tensor(exp_maps[:, 1:22, :].reshape(-1, 3))
    ).numpy().reshape(T, 21, 3, 3)                                       # (T, 21, 3, 3)

    eye = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 2, -1, -1)
    body_pose = torch.cat(
        [torch.tensor(body_rot_21), eye], dim=1
    ).numpy()                                                            # (T, 23, 3, 3)

    return betas, global_orient, body_pose, trans


# ---------------------------------------------------------------------------
# IMU data derived from v3d kinematics
# ---------------------------------------------------------------------------

_GRAVITY = np.array([0.0, 0.0, -9.8], dtype=np.float64)   # WIMUSim default, m/s²

# Kinematic chain: segment_index → parent_segment_index (-1 = root)
# RPV (5) is the root; all local affine matrices must be multiplied up to get
# global pose.  Order matters: parents must appear before children.
_SEG_PARENT = {
    5: -1,   # RPV  → root
    0:  5,   # RTA  → RPV
    1:  0,   # RHE  → RTA
    2:  0,   # LAR  → RTA
    3:  2,   # LFA  → LAR
    4:  3,   # LHA  → LFA
    9:  0,   # RAR  → RTA
   10:  9,   # RFA  → RAR
   11: 10,   # RHA  → RFA
    6:  5,   # LTH  → RPV
    7:  6,   # LSK  → LTH
    8:  7,   # LFT  → LSK
   12:  5,   # RTH  → RPV
   13: 12,   # RSK  → RTH
   14: 13,   # RFT  → RSK
}
# Topological order (parents before children)
_SEG_TOPO = [5, 0, 1, 2, 3, 4, 9, 10, 11, 6, 7, 8, 12, 13, 14]


def _global_affines(local_affine_act: np.ndarray) -> np.ndarray:
    """
    Convert local affine matrices to global (world-frame) affine matrices.

    In v3d data only the root segment (RPV, index 5) is stored in global
    coordinates.  All other segments are stored as local transforms relative
    to their parent.  Global pose is obtained by multiplying up the chain:
        global[child] = global[parent] @ local[child]

    Args:
        local_affine_act: (T, 15, 4, 4) local affine matrices for one activity.

    Returns:
        global_affine: (T, 15, 4, 4) world-frame affine matrices.
    """
    T = local_affine_act.shape[0]
    global_aff = np.zeros_like(local_affine_act)

    for seg_idx in _SEG_TOPO:
        parent = _SEG_PARENT[seg_idx]
        if parent == -1:
            # Root: already in global frame
            global_aff[:, seg_idx] = local_affine_act[:, seg_idx]
        else:
            # Chain: global[child] = global[parent] @ local[child]
            global_aff[:, seg_idx] = (
                global_aff[:, parent] @ local_affine_act[:, seg_idx]
            )

    return global_aff


def _compute_segment_imu(global_affine_seg: np.ndarray, dt: float) -> tuple:
    """
    Compute accelerometer and gyroscope signals from a segment's global affine.

    Matches WIMUSim's simulate_imu convention:
        acc  = R^T @ (d²p/dt² - g_world)   [specific force, body frame, m/s²]
        gyro = R^T @ ω_world                [angular velocity, body frame, rad/s]

    Args:
        global_affine_seg: (T, 4, 4) world-frame affine for one segment.
        dt:                Time step in seconds (1 / V3D_SAMPLE_RATE).

    Returns:
        acc:  (T, 3) float32 m/s²
        gyro: (T, 3) float32 rad/s
    """
    # Global position in metres (from affine translation column)
    p = global_affine_seg[:, :3, 3] / 1000.0     # (T, 3)

    # Global rotation matrix R: body-to-world
    R = global_affine_seg[:, :3, :3]              # (T, 3, 3)

    # ---- Accelerometer --------------------------------------------------
    a_world       = np.zeros_like(p)
    a_world[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / (dt ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    diff = a_world - _GRAVITY[None, :]
    acc  = np.einsum("tij,tj->ti", R.transpose(0, 2, 1), diff)   # (T, 3)

    # ---- Gyroscope -------------------------------------------------------
    dR         = np.zeros_like(R)
    dR[1:-1]   = (R[2:] - R[:-2]) / (2.0 * dt)
    dR[0]      = dR[1]
    dR[-1]     = dR[-2]

    Omega       = np.einsum("tij,tkj->tik", dR, R)
    omega_world = np.stack([Omega[:, 2, 1], Omega[:, 0, 2], Omega[:, 1, 0]], axis=-1)
    gyro        = np.einsum("tij,tj->ti", R.transpose(0, 2, 1), omega_world)  # (T, 3)

    return acc.astype(np.float32), gyro.astype(np.float32)


def load_imu_data(v3d_root: str, subject_num: int, activity_idx: int,
                  imu_names=None) -> dict:
    """
    Load IMU signals for one activity by computing them from Vicon kinematics.

    Signals are computed at 120 Hz (V3D_SAMPLE_RATE = SMPL_SAMPLE_RATE),
    matching WIMUSim's simulation output frame-for-frame.

    Args:
        v3d_root:     Root directory containing F_v3d_Subject_N.mat files.
        subject_num:  Subject number (1-90).
        activity_idx: Activity index 0-20 (into V3D_MOTION_LIST).
        imu_names:    Subset of IMU names to return (default: all 15).

    Returns:
        imu_dict: {wimusim_imu_name: (acc_np, gyro_np)}
                  acc/gyro shapes (T, 3), units m/s² and rad/s at 120 Hz.
    """
    import scipy.io

    mat_path = Path(v3d_root) / f"F_v3d_Subject_{subject_num}.mat"
    subject_key = f"Subject_{subject_num}_F"
    mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    move = mat[subject_key]["move"]

    flags  = move["flags120"]                        # (21, 2) 1-indexed
    start  = int(flags[activity_idx, 0]) - 1         # convert to 0-indexed
    end    = int(flags[activity_idx, 1])             # exclusive

    affine_all = move["jointsAffine_v3d"]            # (T_all, 15, 4, 4)
    affine_act = affine_all[start:end]               # (T_act, 15, 4, 4)

    # Convert local affines → global world-frame affines
    global_aff = _global_affines(affine_act)         # (T_act, 15, 4, 4)

    dt_120    = 1.0 / V3D_SAMPLE_RATE
    requested = imu_names if imu_names is not None else list(V3D_SEG_TO_IMU.values())
    imu_dict  = {}

    for imu_name in requested:
        seg_idx = IMU_TO_SEG_IDX.get(imu_name)
        if seg_idx is None:
            continue

        acc, gyro = _compute_segment_imu(global_aff[:, seg_idx], dt_120)
        imu_dict[imu_name] = (acc, gyro)

    return imu_dict


# ---------------------------------------------------------------------------
# Real Xsens IMU loading  (imu_Subject_N.mat)
# ---------------------------------------------------------------------------

def load_xsens_imu(xsens_root: str, subject_num: int, activity_idx: int,
                   v3d_root: str = None, amass_root: str = None,
                   imu_names=None) -> dict:
    """
    Load real Xsens IMU measurements for one activity from imu_Subject_N.mat.

    The .mat file stores the full session as a continuous array (S1_Synched).
    Activity boundaries are determined by one of two methods (in priority order):
        1. v3d flags120 (if v3d_root is provided) — uses 120 Hz frame boundaries
           from the Vicon file and converts to 100 Hz Xsens indices.
        2. AMASS file lengths (if amass_root is provided, v3d_root not given) —
           each activity's frame count in F_amass_Subject_N.mat × (100/120) gives
           the Xsens frame count for that activity.

    At least one of v3d_root or amass_root must be provided.

    Data layout inside the .mat (per joint block of 16 columns):
        [X(3), V(3), Q(4), A(3), W(3)]
        A = acceleration in g  →  multiply by 9.81 for m/s²
        W = angular velocity in rad/s  (same unit as WIMUSim)

    Args:
        xsens_root:   Root directory containing imu_Subject_N.mat files.
        subject_num:  Subject number (1-90).
        activity_idx: Activity index 0-20 (into V3D_MOTION_LIST).
        v3d_root:     Root directory containing F_v3d_Subject_N.mat files
                      (used only to read flags120 for activity segmentation).
                      If None, amass_root must be provided instead.
        amass_root:   Root of AMASS BMLmovi download. Used for segmentation
                      when v3d_root is not available.
        imu_names:    Subset of WIMUSim IMU names to return (default: all 17).

    Returns:
        imu_dict: {wimusim_imu_name: (acc_np, gyro_np)}
                  acc/gyro shapes (T, 3), units m/s² and rad/s at 120 Hz.
    """
    import scipy.io

    # --- Activity boundaries ----------------------------------------------
    if v3d_root is not None:
        # Method 1: v3d flags120 (preferred — exact boundaries)
        v3d_path = Path(v3d_root) / f"F_v3d_Subject_{subject_num}.mat"
        v3d_mat  = scipy.io.loadmat(str(v3d_path), simplify_cells=True)
        move     = v3d_mat[f"Subject_{subject_num}_F"]["move"]
        flags    = move["flags120"]                      # (21, 2) 1-indexed
        start120 = int(flags[activity_idx, 0]) - 1      # 0-indexed
        end120   = int(flags[activity_idx, 1])           # exclusive
        start100 = round(start120 * XSENS_SAMPLE_RATE / V3D_SAMPLE_RATE)
        end100   = round(end120   * XSENS_SAMPLE_RATE / V3D_SAMPLE_RATE)
    elif amass_root is not None:
        # Method 2: derive from AMASS file lengths (no v3d needed)
        import scipy.io as _sio
        from dataset_configs.movi.consts import V3D_MOTION_LIST
        mat_path_a  = Path(amass_root) / f"F_amass_Subject_{subject_num}.mat"
        lengths_100 = []

        # Read T for each activity in V3D_MOTION_LIST order from F_amass .mat
        subj = _sio.loadmat(str(mat_path_a), simplify_cells=True)[
            f"Subject_{subject_num}_F_amass"
        ]
        name_to_T = {
            _norm_activity_name(m["description"]): m["jointsExpMaps_amass"].shape[0]
            for m in subj["move"]
        }
        for v3d_name in V3D_MOTION_LIST:
            T_120 = name_to_T.get(_norm_activity_name(v3d_name), 0)
            lengths_100.append(round(T_120 * XSENS_SAMPLE_RATE / V3D_SAMPLE_RATE))

        start100 = sum(lengths_100[:activity_idx])
        end100   = start100 + lengths_100[activity_idx]
    else:
        raise ValueError("Either v3d_root or amass_root must be provided for segmentation.")

    # --- Load Xsens data --------------------------------------------------
    xsens_path = Path(xsens_root) / f"imu_Subject_{subject_num}.mat"
    xsens_mat  = scipy.io.loadmat(str(xsens_path))
    imu_top    = xsens_mat["IMU"][0, 0]
    s1         = imu_top["S1_Synched"][0, 0]

    data = s1["data"]                                    # (T_session, 336)
    # Clamp end index to actual data length
    end100 = min(end100, data.shape[0])

    # Extract activity slice at 100 Hz
    act_data = data[start100:end100]                     # (T_act_100, 336)

    # Build joint name → column-base-index map
    joint_names = [s1["jointNames"][0, i][0] for i in range(s1["jointNames"].shape[1])]
    joint_base  = {name: idx * 16 for idx, name in enumerate(joint_names)}

    # --- Extract acc / gyro per requested IMU ----------------------------
    requested = imu_names if imu_names is not None else list(XSENS_JOINT_TO_IMU.values())
    # Build reverse map: wimusim_name → xsens_joint_name
    imu_to_xsens = {v: k for k, v in XSENS_JOINT_TO_IMU.items()}

    imu_dict_100hz = {}
    for imu_name in requested:
        joint_name = imu_to_xsens.get(imu_name)
        if joint_name is None or joint_name not in joint_base:
            continue
        base = joint_base[joint_name]
        # A channels (g) → m/s²
        acc  = act_data[:, base + 10 : base + 13].astype(np.float32) * 9.81
        gyro = act_data[:, base + 13 : base + 16].astype(np.float32)
        imu_dict_100hz[imu_name] = (acc, gyro)

    # --- Upsample 100 Hz → 120 Hz (linear interp) ----------------------
    T_100 = act_data.shape[0]
    T_120 = round(T_100 * V3D_SAMPLE_RATE / XSENS_SAMPLE_RATE)
    t_100 = np.linspace(0.0, 1.0, T_100)
    t_120 = np.linspace(0.0, 1.0, T_120)

    imu_dict = {}
    for imu_name, (acc, gyro) in imu_dict_100hz.items():
        acc_120  = np.stack([np.interp(t_120, t_100, acc[:, i])  for i in range(3)], axis=1)
        gyro_120 = np.stack([np.interp(t_120, t_100, gyro[:, i]) for i in range(3)], axis=1)
        imu_dict[imu_name] = (acc_120.astype(np.float32), gyro_120.astype(np.float32))

    return imu_dict


# ---------------------------------------------------------------------------
# Default Placement (P) parameters
# ---------------------------------------------------------------------------

def generate_default_placement_params(B_rp: dict) -> dict:
    """
    Generate default IMU placement parameters for MoVi's 15 v3d-derived IMU positions.

    Args:
        B_rp: bone vector dict {(parent, child): np.ndarray(3,)} from compute_B_from_beta.

    Returns:
        dict with "rp" and "ro" sub-dicts keyed by (parent_joint, imu_name).
    """
    rp, ro = {}, {}

    def _bone(p, c):
        return B_rp.get((p, c), np.zeros(3))

    # Head
    rp[("HEAD",       "HED")]  = np.array([0.0,  0.0,  0.05])
    ro[("HEAD",       "HED")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Sternum
    rp[("SPINE3",     "STER")] = np.array([0.0,  0.1,  0.0])
    ro[("SPINE3",     "STER")] = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Pelvis
    rp[("BASE",       "PELV")] = np.array([0.0,  0.1,  0.0])
    ro[("BASE",       "PELV")] = np.deg2rad(np.array([-90.0, 0.0, -90.0]))

    # Upper arms
    rp[("L_SHOULDER", "LUA")]  = _bone("L_SHOULDER", "L_ELBOW") * 0.5
    ro[("L_SHOULDER", "LUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    rp[("R_SHOULDER", "RUA")]  = _bone("R_SHOULDER", "R_ELBOW") * 0.5
    ro[("R_SHOULDER", "RUA")]  = np.deg2rad(np.array([0.0, 0.0, -90.0]))

    # Forearms
    rp[("L_ELBOW",    "LLA")]  = _bone("L_ELBOW", "L_WRIST") * 0.5
    ro[("L_ELBOW",    "LLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    rp[("R_ELBOW",    "RLA")]  = _bone("R_ELBOW", "R_WRIST") * 0.5
    ro[("R_ELBOW",    "RLA")]  = np.deg2rad(np.array([0.0, 0.0, 180.0]))

    # Hands
    rp[("L_WRIST",    "LHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("L_WRIST",    "LHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("R_WRIST",    "RHD")]  = np.array([0.05, 0.0, 0.0])
    ro[("R_WRIST",    "RHD")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    # Thighs
    rp[("L_HIP",      "LTH")]  = _bone("L_HIP", "L_KNEE") * 0.5
    ro[("L_HIP",      "LTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    rp[("R_HIP",      "RTH")]  = _bone("R_HIP", "R_KNEE") * 0.5
    ro[("R_HIP",      "RTH")]  = np.deg2rad(np.array([90.0, 180.0, 0.0]))

    # Shins
    rp[("L_KNEE",     "LSH")]  = _bone("L_KNEE", "L_ANKLE") * 0.5
    ro[("L_KNEE",     "LSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    rp[("R_KNEE",     "RSH")]  = _bone("R_KNEE", "R_ANKLE") * 0.5
    ro[("R_KNEE",     "RSH")]  = np.deg2rad(np.array([90.0, 90.0, 0.0]))

    # Feet
    rp[("L_ANKLE",    "LFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("L_ANKLE",    "LFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    rp[("R_ANKLE",    "RFT")]  = np.array([0.1, 0.0, -0.03])
    ro[("R_ANKLE",    "RFT")]  = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    return {"rp": rp, "ro": ro}


# ---------------------------------------------------------------------------
# B range helper
# ---------------------------------------------------------------------------

def generate_B_range(B_rp: dict, min_scale=0.85, max_scale=1.15,
                     near_zero=(-0.01, 0.01)) -> dict:
    """Generate per-axis search ranges for each bone vector entry."""
    return {
        k: np.array([
            sorted([e * min_scale, e * max_scale]) if abs(e) > 1e-4 else list(near_zero)
            for e in v
        ])
        for k, v in B_rp.items()
    }
