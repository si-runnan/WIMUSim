"""
Transformer-based IMU → SMPL body pose estimator.

Architecture overview:
    raw IMU (acc+gyro)
        → Linear projection
        → Sinusoidal positional encoding
        → Transformer Encoder (self-attention over time)
        → MLP pose head
        → 6D rotation representation
        → Gram-Schmidt → rotation matrices (3×3)

Rotation representation:
    Uses the continuous 6D representation from
    "On the Continuity of Rotation Representations in Neural Networks"
    (Zhou et al., CVPR 2019).  The first two columns of a rotation matrix
    are predicted; the third is recovered via cross product.

Loss:
    Geodesic (angular) loss on rotation matrices — measures the angle between
    the predicted and ground-truth rotation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to a rotation matrix via Gram-Schmidt.

    Args:
        x: (..., 6)

    Returns:
        (..., 3, 3)
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # (..., 3, 3)


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Extract the first two columns of a rotation matrix as 6D representation.

    Args:
        rotmat: (..., 3, 3)

    Returns:
        (..., 6)
    """
    return rotmat[..., :2].reshape(*rotmat.shape[:-2], 6)


def geodesic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Geodesic (angular) loss between predicted and ground-truth rotation matrices.

    L = mean( arccos( (trace(R_err) - 1) / 2 ) )
    where R_err = pred^T @ target.

    Args:
        pred:   (..., 3, 3)
        target: (..., 3, 3)
        reduction: "mean" | "sum" | "none"

    Returns:
        Scalar loss (or tensor if reduction="none").
    """
    R_err = pred.transpose(-2, -1) @ target
    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_angle)
    if reduction == "mean":
        return angle.mean()
    elif reduction == "sum":
        return angle.sum()
    return angle


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

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
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PoseEstimator(nn.Module):
    """
    Transformer encoder that maps windowed IMU signals to SMPL body_pose.

    Input:  (B, T, n_imus * 6)   raw IMU — acc_xyz + gyro_xyz per sensor
    Output: (B, T, n_joints, 3, 3)  per-joint rotation matrices (body_pose)

    Args:
        n_imus:   number of IMU sensors
        d_model:  transformer hidden dimension
        n_heads:  number of attention heads
        n_layers: number of transformer encoder layers
        d_ff:     feedforward dimension inside each layer
        dropout:  dropout probability
        n_joints: number of body joints to estimate (23 for SMPL body_pose)
    """

    def __init__(
        self,
        n_imus: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_joints: int = 23,
    ):
        super().__init__()
        self.n_imus = n_imus
        self.n_joints = n_joints

        self.input_proj = nn.Linear(n_imus * 6, d_model)
        self.pe = SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_joints * 6),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_imus * 6)

        Returns:
            (B, T, n_joints, 3, 3)
        """
        B, T, _ = x.shape
        h = self.pe(self.input_proj(x))         # (B, T, d_model)
        h = self.transformer(h)                  # (B, T, d_model)
        rot6d = self.pose_head(h)                # (B, T, n_joints * 6)
        rot6d = rot6d.view(B, T, self.n_joints, 6)
        return rot6d_to_rotmat(rot6d)            # (B, T, n_joints, 3, 3)

    def predict_6d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same as forward but returns 6D representation instead of rotation matrices.
        Useful for computing loss directly in 6D space.

        Returns:
            (B, T, n_joints, 6)
        """
        B, T, _ = x.shape
        h = self.pe(self.input_proj(x))
        h = self.transformer(h)
        rot6d = self.pose_head(h)
        return rot6d.view(B, T, self.n_joints, 6)
