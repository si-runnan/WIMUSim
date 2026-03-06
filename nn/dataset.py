"""
Dataset for IMU-based SMPL body pose estimation.

Data flow:
    cpm_params_movi.pkl
        → B_list, D_list, P_list, H_list
            → WIMUSim.simulate()  →  virtual IMU (T, N_imus * 6)
            → D.orientation       →  body_pose   (T, 23, 3, 3)
        → sliding windows (T_win, …)

Each dataset item:
    imu:      (T_win, N_imus * 6)   float32   acc+gyro concatenated per sensor
    pose_6d:  (T_win, 23 * 6)       float32   6D rotation (first 2 cols of rotmat)
"""

import pickle
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from wimusim.wimusim import WIMUSim
from nn.model import rotmat_to_rot6d

# SMPL body_pose joint order (joints 1-23, matching SMPL_BODY_POSE_JOINT_NAMES)
from dataset_configs.smpl.consts import SMPL_BODY_POSE_JOINT_NAMES


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix  (WXYZ PyTorch3D convention)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (WXYZ) to rotation matrix.

    Args:
        q: (..., 4)  WXYZ

    Returns:
        (..., 3, 3)
    """
    # Normalise
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    rotmat = torch.stack(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1
    ).reshape(*q.shape[:-1], 3, 3)
    return rotmat


def _extract_pose_6d(D: "WIMUSim.Dynamics") -> torch.Tensor:
    """
    Extract SMPL body_pose from a WIMUSim Dynamics object as 6D rotation.

    Returns:
        (T, 23 * 6)  float32
    """
    T = D.n_samples
    pose_6d = []
    for joint_name in SMPL_BODY_POSE_JOINT_NAMES:
        q = D.orientation[joint_name]       # (T, 4)  WXYZ
        rotmat = _quat_wxyz_to_rotmat(q)   # (T, 3, 3)
        rot6d = rotmat_to_rot6d(rotmat)    # (T, 6)
        pose_6d.append(rot6d)
    return torch.cat(pose_6d, dim=-1)      # (T, 23 * 6)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseEstimDataset(Dataset):
    """
    PyTorch Dataset for IMU → SMPL body_pose estimation.

    Call generate_data() once before training to simulate all IMU sequences.
    Each __getitem__ returns a sliding window of (imu, pose_6d).

    Args:
        B_list:         List of WIMUSim.Body parameters
        D_list:         List of WIMUSim.Dynamics parameters (one per sequence)
        P_list:         List of WIMUSim.Placement parameters
        H_list:         List of WIMUSim.Hardware parameters
        window:         Sliding window length in frames
        stride:         Step between windows
        n_combinations: How many (B, D, P, H) combos to sample.
                        -1 = all combinations (Cartesian product).
        device:         Torch device
    """

    def __init__(
        self,
        B_list: list,
        D_list: list,
        P_list: list,
        H_list: list,
        window: int = 128,
        stride: int = 64,
        n_combinations: int = -1,
        device: Optional[torch.device] = None,
    ):
        self.B_list = B_list
        self.D_list = D_list
        self.P_list = P_list
        self.H_list = H_list
        self.window = window
        self.stride = stride
        self.n_combinations = n_combinations
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Populated by generate_data()
        self._imu_seqs: List[torch.Tensor] = []    # each (T_i, N_imus*6)
        self._pose_seqs: List[torch.Tensor] = []   # each (T_i, 23*6)
        self._index_map: List[tuple] = []          # (seq_idx, frame_start)
        self._n_imus: Optional[int] = None

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_data(self):
        """
        Simulate virtual IMU sequences and extract pose targets.

        Call this once before using the dataset.
        """
        import itertools
        from tqdm import tqdm

        if self.n_combinations == -1:
            combos = list(
                itertools.product(
                    range(len(self.B_list)),
                    range(len(self.D_list)),
                    range(len(self.P_list)),
                    range(len(self.H_list)),
                )
            )
        else:
            combos = [
                (
                    random.randint(0, len(self.B_list) - 1),
                    random.randint(0, len(self.D_list) - 1),
                    random.randint(0, len(self.P_list) - 1),
                    random.randint(0, len(self.H_list) - 1),
                )
                for _ in range(self.n_combinations)
            ]

        print(f"Generating {len(combos)} virtual IMU sequence(s)...")
        for B_i, D_i, P_i, H_i in tqdm(combos):
            B = self.B_list[B_i]
            D = self.D_list[D_i]
            P = self.P_list[P_i]
            H = self.H_list[H_i]

            # Simulate virtual IMU
            env = WIMUSim(B=B, D=D, P=P, H=H)
            virt = env.simulate(mode="generate")   # {name: (acc, gyro)}
            imu_tensor = torch.cat(
                [torch.cat(virt[name], dim=-1) for name in env.P.imu_names],
                dim=-1,
            ).detach().cpu()                       # (T, N_imus * 6)

            if self._n_imus is None:
                self._n_imus = len(env.P.imu_names)

            # Extract pose target
            pose_tensor = _extract_pose_6d(D).detach().cpu()  # (T, 23 * 6)

            # Trim to same length (should already be equal)
            T = min(imu_tensor.shape[0], pose_tensor.shape[0])
            imu_tensor  = imu_tensor[:T]
            pose_tensor = pose_tensor[:T]

            self._imu_seqs.append(imu_tensor)
            self._pose_seqs.append(pose_tensor)

        # Build index map
        self._index_map = []
        for seq_idx, seq in enumerate(self._imu_seqs):
            T = seq.shape[0]
            for start in range(0, T - self.window + 1, self.stride):
                self._index_map.append((seq_idx, start))

        print(f"Dataset ready: {len(self._imu_seqs)} sequences, "
              f"{len(self._index_map)} windows, "
              f"{self._n_imus} IMUs.")

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int):
        seq_idx, start = self._index_map[idx]
        end = start + self.window
        imu   = self._imu_seqs[seq_idx][start:end]    # (T_win, N_imus * 6)
        pose  = self._pose_seqs[seq_idx][start:end]   # (T_win, 23 * 6)
        return imu, pose

    @property
    def n_imus(self) -> Optional[int]:
        return self._n_imus

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cpm_pkl(
        cls,
        pkl_path: str,
        window: int = 128,
        stride: int = 64,
        n_combinations: int = -1,
        device: Optional[torch.device] = None,
    ) -> "PoseEstimDataset":
        """
        Convenience constructor: load cpm_params from a pickle file and build
        the dataset.

        Args:
            pkl_path: Path to cpm_params_movi.pkl (or any compatible file).
                      Expected keys: B_list, D_list, P_list, H_list.

        Returns:
            PoseEstimDataset (call generate_data() before use).
        """
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)

        return cls(
            B_list=params["B_list"],
            D_list=params["D_list"],
            P_list=params["P_list"],
            H_list=params["H_list"],
            window=window,
            stride=stride,
            n_combinations=n_combinations,
            device=device,
        )
