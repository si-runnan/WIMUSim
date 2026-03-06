"""
Neural residual corrector for WIMUSim IMU simulation.

Motivation:
    WIMUSim uses physics-based simulation (rigid body dynamics, gravity model)
    to generate virtual IMU signals from SMPL poses.  Physics alone cannot
    capture all real-world effects:
        - Soft-tissue artefacts (sensor moves with skin, not only the bone)
        - Non-linear sensor characteristics and cross-axis sensitivity
        - Complex noise patterns beyond simple additive Gaussian models
        - Inaccuracies in the placement / orientation model (P parameter)

    A neural corrector trained on real paired data (SMPL pose + real IMU)
    learns to predict the residual between physics output and reality.
    Physics provides a strong, interpretable prior; the network only needs
    to model the remaining gap — a much smaller learning task.

Architecture:
    Input:
        pose_feat  (T, 24 * 6)    SMPL joint orientations as 6D representation
        phys_imu   (T, N_imus*6)  WIMUSim physics output (acc+gyro per sensor)
        → concat → (T, 24*6 + N_imus*6)

    Model:
        Linear projection → Sinusoidal PE → Transformer Encoder → MLP head

    Output (residual mode):
        residual   (T, N_imus*6)    additive correction to physics output
        → final = phys_imu + residual

    Output (direct mode, residual=False):
        imu_pred   (T, N_imus*6)    direct IMU prediction from pose only

Loss:
    Smooth L1 (Huber) on accelerometer and gyroscope channels separately,
    with an optional physics consistency regulariser.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation utilities  (pose → 6D features)
# ---------------------------------------------------------------------------

def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Extract the first two columns of a rotation matrix as a 6D feature.

    Args:
        rotmat: (..., 3, 3)

    Returns:
        (..., 6)  — continuous rotation representation (Zhou et al., CVPR 2019)
    """
    return rotmat[..., :2].reshape(*rotmat.shape[:-2], 6)


def quat_wxyz_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (WXYZ) to 6D rotation representation.

    Args:
        q: (..., 4)

    Returns:
        (..., 6)
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2*(y*y + z*z);  r01 = 2*(x*y - w*z);  r02 = 2*(x*z + w*y)
    r10 = 2*(x*y + w*z);      r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z - w*x)

    # First two columns of the rotation matrix → 6D
    return torch.stack([r00, r10, r01, r11, r02, r12], dim=-1)   # (..., 6)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Neural Residual Corrector
# ---------------------------------------------------------------------------

class NeuralSimulator(nn.Module):
    """
    Transformer-based neural corrector for WIMUSim.

    Given SMPL joint orientations and WIMUSim physics output, predicts
    a residual correction so that:

        corrected_imu = phys_imu + model(pose_feat, phys_imu)

    This residual formulation keeps physics as the dominant prior and
    only requires the network to learn the sim-to-real gap.

    Args:
        n_joints:   Number of SMPL joints (24 for full body incl. root)
        n_imus:     Number of IMU sensors
        d_model:    Transformer hidden dimension
        n_heads:    Multi-head attention heads
        n_layers:   Number of Transformer encoder layers
        d_ff:       Feed-forward dimension inside each layer
        dropout:    Dropout probability
        residual:   If True (default), predicts residual on top of physics.
                    If False, predicts IMU directly from pose only.
    """

    def __init__(
        self,
        n_imus: int,
        n_joints: int = 24,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints
        self.residual = residual

        # Input: pose 6D + physics IMU (if residual) or pose 6D only
        in_dim = n_joints * 6 + n_imus * 6 if residual else n_joints * 6
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pe = SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Output: residual (or direct IMU) — separate heads for acc and gyro
        # Acc and gyro have very different scales, so separate heads help
        self.acc_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_imus * 3),
        )
        self.gyro_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_imus * 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pose_6d: torch.Tensor,
        phys_imu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pose_6d:   (B, T, n_joints * 6)  joint orientations as 6D rotation
            phys_imu:  (B, T, n_imus * 6)    WIMUSim physics output (acc+gyro)

        Returns:
            corrected_imu: (B, T, n_imus * 6)
                If residual=True:  phys_imu + predicted_residual
                If residual=False: direct prediction from pose_6d only
        """
        if self.residual:
            x = torch.cat([pose_6d, phys_imu], dim=-1)   # (B, T, joints*6 + imus*6)
        else:
            x = pose_6d

        h = self.pe(self.input_proj(x))   # (B, T, d_model)
        h = self.transformer(h)            # (B, T, d_model)

        acc_out  = self.acc_head(h)        # (B, T, n_imus * 3)
        gyro_out = self.gyro_head(h)       # (B, T, n_imus * 3)

        # Interleave acc and gyro per sensor: [acc_s0, gyro_s0, acc_s1, gyro_s1, ...]
        B, T = h.shape[:2]
        acc_out  = acc_out.view(B, T, self.n_imus, 3)
        gyro_out = gyro_out.view(B, T, self.n_imus, 3)
        out = torch.cat([acc_out, gyro_out], dim=-1).view(B, T, self.n_imus * 6)

        if self.residual:
            return phys_imu + out
        return out


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def simulator_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_imus: int,
    acc_weight: float = 1.0,
    gyro_weight: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute separate Smooth L1 (Huber) loss on accelerometer and gyroscope.

    Splitting the two modalities prevents the gyro (small magnitude, rad/s)
    from being dominated by the acc (large magnitude, m/s²).

    Args:
        pred:        (B, T, n_imus * 6)  predicted IMU  (acc+gyro interleaved)
        target:      (B, T, n_imus * 6)  real IMU
        n_imus:      number of IMU sensors
        acc_weight:  loss weight for accelerometer channels
        gyro_weight: loss weight for gyroscope channels
        beta:        Smooth L1 beta parameter

    Returns:
        Scalar loss.
    """
    B, T = pred.shape[:2]
    pred   = pred.view(B, T, n_imus, 6)
    target = target.view(B, T, n_imus, 6)

    acc_loss  = F.smooth_l1_loss(pred[..., :3], target[..., :3], beta=beta)
    gyro_loss = F.smooth_l1_loss(pred[..., 3:], target[..., 3:], beta=beta)

    return acc_weight * acc_loss + gyro_weight * gyro_loss
