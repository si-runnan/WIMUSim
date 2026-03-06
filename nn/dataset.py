"""
PhyNeSim dataset — training data for the neural residual corrector.

Each sample pairs SMPL pose features with:
    - WIMUSim physics output (from simulate())
    - Real IMU signals (from MoVi or other datasets with ground truth)

PhyNeSim learns:
    real_imu ≈ phys_imu + NN(pose_6d, phys_imu)

Data flow:
    MoVi sequence
        SMPL poses  →  smpl_pose_to_D_orientation()  →  D (Dynamics)
                    →  WIMUSim.simulate()             →  phys_imu
                    →  D.orientation (joint quats)    →  pose_6d features
        real IMU    →  load_imu_data()                →  real_imu
        → sliding windows → (pose_6d, phys_imu, real_imu)
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from wimusim.wimusim import WIMUSim
from nn.model import quat_wxyz_to_rot6d
from dataset_configs.smpl.consts import SMPL_JOINT_ID_DICT


# Joint order: all 24 SMPL joints sorted by joint index
JOINT_ORDER = sorted(SMPL_JOINT_ID_DICT.keys(), key=lambda k: SMPL_JOINT_ID_DICT[k])


def _extract_pose_6d(D: "WIMUSim.Dynamics") -> torch.Tensor:
    """
    Extract all 24 SMPL joint orientations from a Dynamics object as 6D rotation.

    Returns:
        (T, 24 * 6)  float32
    """
    feats = []
    for joint in JOINT_ORDER:
        q = D.orientation.get(joint)
        if q is None:
            # Joints not tracked (e.g. L_HAND/R_HAND padded with identity)
            T = next(iter(D.orientation.values())).shape[0]
            q = torch.zeros(T, 4, device=next(iter(D.orientation.values())).device)
            q[:, 0] = 1.0   # identity quaternion  WXYZ
        feats.append(quat_wxyz_to_rot6d(q))   # (T, 6)
    return torch.cat(feats, dim=-1)            # (T, 24 * 6)


def _stack_imu_dict(
    imu_dict: dict,
    imu_names: List[str],
) -> torch.Tensor:
    """
    Stack IMU signals from a dict into a tensor (T, N_imus * 6).

    Channel order per sensor: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].

    Args:
        imu_dict:  {name: (acc, gyro)}  acc/gyro: np.ndarray (T, 3) or Tensor
        imu_names: Ordered list of sensor names.

    Returns:
        (T, N_imus * 6)  float32
    """
    parts = []
    for name in imu_names:
        acc, gyro = imu_dict[name]
        if isinstance(acc, np.ndarray):
            acc  = torch.from_numpy(acc.astype(np.float32))
            gyro = torch.from_numpy(gyro.astype(np.float32))
        parts.append(torch.cat([acc, gyro], dim=-1))   # (T, 6)
    return torch.cat(parts, dim=-1)                    # (T, N_imus * 6)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimulatorDataset(Dataset):
    """
    Dataset of (pose_6d, phys_imu, real_imu) triples for training the
    WIMUSim neural residual corrector.

    Call generate_data() once to run WIMUSim on all sequences before training.

    Args:
        sequences:   List of (D, real_imu_dict, B, P, H) tuples where
                       D            — WIMUSim.Dynamics for this sequence
                       real_imu_dict— {imu_name: (acc_np, gyro_np)}
                       B, P, H      — WIMUSim body / placement / hardware params
        imu_names:   Ordered sensor names (determines output channel order)
        window:      Sliding window size in frames
        stride:      Step between windows
        device:      Torch device
    """

    def __init__(
        self,
        sequences: list,
        imu_names: List[str],
        window: int = 128,
        stride: int = 32,
        device: Optional[torch.device] = None,
    ):
        self.sequences = sequences
        self.imu_names = imu_names
        self.window = window
        self.stride = stride
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._pose_seqs: List[torch.Tensor] = []      # (T, 24*6)
        self._phys_seqs: List[torch.Tensor] = []      # (T, N_imus*6)
        self._real_seqs: List[torch.Tensor] = []      # (T, N_imus*6)
        self._index_map: List[Tuple[int, int]] = []   # (seq_idx, frame_start)

    def generate_data(self):
        """Run WIMUSim on all sequences to produce (pose_6d, phys_imu) pairs."""
        print(f"Generating physics IMU for {len(self.sequences)} sequences...")
        for D, real_imu_dict, B, P, H in tqdm(self.sequences):
            # Physics simulation
            env = WIMUSim(B=B, D=D, P=P, H=H)
            virt = env.simulate(mode="generate")   # {name: (acc, gyro)}

            # Filter to requested IMU names
            avail = set(virt.keys()) & set(self.imu_names)
            if not avail:
                continue

            names_in_order = [n for n in self.imu_names if n in avail]

            phys_tensor = _stack_imu_dict(
                {n: virt[n] for n in names_in_order}, names_in_order
            ).detach().cpu()                         # (T, N_imus*6)

            real_tensor = _stack_imu_dict(
                {n: real_imu_dict[n] for n in names_in_order if n in real_imu_dict},
                [n for n in names_in_order if n in real_imu_dict],
            )                                        # (T, N_imus*6)

            pose_tensor = _extract_pose_6d(D).detach().cpu()  # (T, 24*6)

            # Trim all to same length
            T = min(phys_tensor.shape[0], real_tensor.shape[0], pose_tensor.shape[0])
            if T < self.window:
                continue
            phys_tensor = phys_tensor[:T]
            real_tensor = real_tensor[:T]
            pose_tensor = pose_tensor[:T]

            self._pose_seqs.append(pose_tensor)
            self._phys_seqs.append(phys_tensor)
            self._real_seqs.append(real_tensor)

        # Build sliding-window index
        self._index_map = []
        for seq_idx, seq in enumerate(self._pose_seqs):
            T = seq.shape[0]
            for start in range(0, T - self.window + 1, self.stride):
                self._index_map.append((seq_idx, start))

        n = len(self.imu_names)
        print(f"Dataset ready: {len(self._pose_seqs)} seqs, "
              f"{len(self._index_map)} windows, {n} IMUs.")

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pose_6d:  (T_win, 24 * 6)
            phys_imu: (T_win, N_imus * 6)
            real_imu: (T_win, N_imus * 6)
        """
        seq_idx, start = self._index_map[idx]
        end = start + self.window
        return (
            self._pose_seqs[seq_idx][start:end],
            self._phys_seqs[seq_idx][start:end],
            self._real_seqs[seq_idx][start:end],
        )

    @property
    def n_imus(self) -> int:
        return len(self.imu_names)

    # ------------------------------------------------------------------
    # Factory: build from MoVi data directory
    # ------------------------------------------------------------------

    @classmethod
    def from_movi(
        cls,
        movi_root: str,
        smpl_model_path: str,
        imu_names: List[str],
        subjects: Optional[List[str]] = None,
        activities: Optional[List[str]] = None,
        trials: Optional[List[int]] = None,
        window: int = 128,
        stride: int = 32,
        device: Optional[torch.device] = None,
    ) -> "SimulatorDataset":
        """
        Build a SimulatorDataset from MoVi raw data.

        Requires MoVi data downloaded and MoVi/SMPL utils available.
        See dataset_configs/movi/utils.py for expected directory structure.

        Args:
            movi_root:       Root directory of MoVi dataset.
            smpl_model_path: Path to SMPL model files (for compute_B_from_beta).
            imu_names:       IMU sensor names to include.
            subjects:        Subjects to include (default: TRAIN_SUBJECTS).
            activities:      Activity names (default: all).
            trials:          Trial numbers (default: [1, 2, 3, 4, 5]).
        """
        from dataset_configs.movi.utils import (
            load_smpl_params,
            load_imu_data,
            generate_default_placement_params,
        )
        from dataset_configs.movi.consts import TRAIN_SUBJECTS, ACTIVITY_LIST
        from dataset_configs.smpl.utils import (
            compute_B_from_beta,
            smpl_pose_to_D_orientation,
        )
        from wimusim import utils as wu

        subjects   = subjects   or TRAIN_SUBJECTS
        activities = activities or ACTIVITY_LIST
        trials     = trials     or [1, 2, 3, 4, 5]
        device_    = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        sequences = []
        for subject in subjects:
            for activity in activities:
                for trial in trials:
                    try:
                        betas, global_orient, body_pose = load_smpl_params(
                            movi_root, subject, activity, trial
                        )
                        real_imu_dict = load_imu_data(
                            movi_root, subject, activity, trial
                        )
                    except (FileNotFoundError, KeyError):
                        continue

                    # Build WIMUSim parameters
                    B_rp = compute_B_from_beta(
                        betas, smpl_model_path=smpl_model_path
                    )
                    B = WIMUSim.Body(
                        rp={k: torch.tensor(v, dtype=torch.float32, device=device_)
                            for k, v in B_rp.items()},
                        device=device_,
                    )

                    from dataset_configs.movi.consts import SMPL_SAMPLE_RATE
                    orientation = smpl_pose_to_D_orientation(global_orient, body_pose)
                    D = WIMUSim.Dynamics(
                        orientation={
                            k: torch.tensor(v, dtype=torch.float32, device=device_)
                            for k, v in orientation.items()
                        },
                        sample_rate=SMPL_SAMPLE_RATE,
                        device=device_,
                    )

                    P_all = generate_default_placement_params(B_rp)
                    P = WIMUSim.Placement(
                        rp={k: torch.tensor(v, dtype=torch.float32, device=device_)
                            for k, v in P_all["rp"].items()
                            if k[1] in imu_names},
                        ro={k: torch.tensor(v, dtype=torch.float32, device=device_)
                            for k, v in P_all["ro"].items()
                            if k[1] in imu_names},
                        device=device_,
                    )
                    H = wu.generate_default_H_configs(imu_names)

                    sequences.append((D, real_imu_dict, B, P, H))

        print(f"Loaded {len(sequences)} MoVi sequences "
              f"({len(subjects)} subjects × {len(activities)} activities × {len(trials)} trials)")

        return cls(
            sequences=sequences,
            imu_names=imu_names,
            window=window,
            stride=stride,
            device=device_,
        )
