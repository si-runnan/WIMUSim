"""
Video → SMPL parameter extraction using HMR2.0 / 4D-Humans.

Installation:
    pip install git+https://github.com/shubham-goel/4D-Humans.git
    pip install git+https://github.com/facebookresearch/detectron2.git

Usage:
    from pipeline.video_to_smpl import video_to_smpl

    betas, global_orient, body_pose = video_to_smpl(
        video_path="input.mp4",
        output_npz="smpl_output.npz",  # optional, saves to disk
    )

Output format (compatible with dataset_configs.smpl.utils):
    betas:         np.ndarray (10,)         — averaged over all frames
    global_orient: np.ndarray (T, 3, 3)    — per-frame rotation matrix
    body_pose:     np.ndarray (T, 23, 3, 3) — per-frame rotation matrices
"""

import numpy as np
from pathlib import Path


def video_to_smpl(
    video_path: str,
    output_npz: str = None,
    checkpoint: str = None,
    device: str = "cuda",
    batch_size: int = 16,
    person_idx: int = 0,
    smooth: bool = True,
):
    """
    Run HMR2.0 on a video and extract SMPL parameters.

    Args:
        video_path:  Path to the input video file.
        output_npz:  If provided, save SMPL params to this .npz path.
        checkpoint:  Path to HMR2.0 checkpoint. Uses default if None.
        device:      "cuda" or "cpu".
        batch_size:  Number of frames processed per forward pass.
        person_idx:  Which detected person to track (0 = largest/most central).
        smooth:      Apply Savitzky-Golay smoothing to reduce per-frame jitter.

    Returns:
        betas:         np.ndarray (10,)
        global_orient: np.ndarray (T, 3, 3)
        body_pose:     np.ndarray (T, 23, 3, 3)
    """
    try:
        from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hmr2
    except ImportError:
        raise ImportError(
            "4D-Humans (HMR2.0) is required.\n"
            "Install with:\n"
            "  pip install git+https://github.com/shubham-goel/4D-Humans.git\n"
            "  pip install git+https://github.com/facebookresearch/detectron2.git"
        )

    import torch
    import cv2

    # ------------------------------------------------------------------
    # 1. Load HMR2.0 model
    # ------------------------------------------------------------------
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    model, model_cfg = load_hmr2(checkpoint)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Load person detector (ViTDet via detectron2)
    # ------------------------------------------------------------------
    cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # ------------------------------------------------------------------
    # 3. Extract frames from video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Video: {len(frames)} frames @ {fps:.1f} fps")

    # ------------------------------------------------------------------
    # 4. Detect people and run HMR2.0 frame by frame
    # ------------------------------------------------------------------
    all_global_orient = []
    all_body_pose     = []
    all_betas         = []

    dataset = ViTDetDataset(model_cfg, frames, detector, person_idx=person_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            batch = recursive_to(batch, device)
            out   = model(batch)

            # HMR2.0 output: rotation matrices
            # global_orient: (B, 1, 3, 3), body_pose: (B, 23, 3, 3), betas: (B, 10)
            all_global_orient.append(out["pred_smpl_params"]["global_orient"].cpu().numpy())
            all_body_pose.append(    out["pred_smpl_params"]["body_pose"].cpu().numpy())
            all_betas.append(        out["pred_smpl_params"]["betas"].cpu().numpy())

    global_orient = np.concatenate(all_global_orient, axis=0)[:, 0]  # (T, 3, 3)
    body_pose     = np.concatenate(all_body_pose,     axis=0)         # (T, 23, 3, 3)
    betas_all     = np.concatenate(all_betas,         axis=0)         # (T, 10)

    # Average betas across frames — shape params are person-specific, not frame-specific
    betas = betas_all.mean(axis=0)                                    # (10,)

    # ------------------------------------------------------------------
    # 5. Optional smoothing (reduces per-frame jitter)
    # ------------------------------------------------------------------
    if smooth:
        global_orient, body_pose = _smooth_rotations(global_orient, body_pose)

    # ------------------------------------------------------------------
    # 6. Save and return
    # ------------------------------------------------------------------
    if output_npz is not None:
        np.savez(
            output_npz,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            fps=fps,
        )
        print(f"Saved SMPL params to {output_npz}")

    return betas, global_orient, body_pose


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def _smooth_rotations(global_orient: np.ndarray, body_pose: np.ndarray,
                      window: int = 15, poly: int = 3):
    """
    Apply Savitzky-Golay filter to rotation matrices via their rotation vectors.
    Reduces per-frame jitter from HMR2.0 without distorting the motion.

    Args:
        global_orient: (T, 3, 3)
        body_pose:     (T, 23, 3, 3)
        window:        Filter window length (frames). Must be odd.
        poly:          Polynomial order.

    Returns:
        Smoothed global_orient (T, 3, 3) and body_pose (T, 23, 3, 3).
    """
    from scipy.signal import savgol_filter
    import torch
    import pytorch3d.transforms.rotation_conversions as rc

    def _smooth(rotmats):
        # rotmats: (T, [N,] 3, 3) → rotvec → smooth → rotmat
        shape = rotmats.shape
        flat  = rotmats.reshape(-1, 3, 3)
        rvec  = rc.matrix_to_axis_angle(torch.tensor(flat)).numpy()  # (T*N, 3)
        rvec_s = savgol_filter(rvec, window_length=window, polyorder=poly, axis=0)
        rotmat_s = rc.axis_angle_to_matrix(torch.tensor(rvec_s)).numpy()
        return rotmat_s.reshape(shape)

    global_orient_s = _smooth(global_orient)

    T, N = body_pose.shape[:2]
    body_pose_s = np.stack(
        [_smooth(body_pose[:, j]) for j in range(N)], axis=1
    )

    return global_orient_s, body_pose_s
