# WIMUSim — SMPL + Neural Network Branch

WIMUSim is a physics-based IMU simulation framework.
This branch (`smpl-nn`) extends the `smpl` branch with a **neural residual corrector**
that reduces the sim-to-real gap in WIMUSim's physics simulation.
The network is trained on MoVi paired data (SMPL poses + real IMU) and learns
to correct the residual between physics output and real sensor measurements.

**Branch strategy**

| Branch | Skeleton | Pose source | Extra |
|--------|----------|-------------|-------|
| `master` | H3.6M (17 joints) | MotionBERT | — |
| `smpl` | SMPL (24 joints) | HMR2.0 / 4D-Humans | simulation only |
| `smpl-nn` ← you are here | SMPL (24 joints) | HMR2.0 / 4D-Humans | + neural residual corrector |

---

## How It Works

```
Video
  └─ HMR2.0 / 4D-Humans          pipeline/video_to_smpl.py
       └─ β, global_orient, body_pose
            ├─ compute_B_from_beta(β)           → B (bone vectors)
            └─ smpl_pose_to_D_orientation(θ)    → D (joint orientations)
                  └─ WIMUSim(B, D, P, H)
                        └─ simulate()           → virtual IMU (acc, gyro)
```

**Four parameters:**

| Param | Meaning | Source |
|-------|---------|--------|
| **B** (Body) | Bone lengths / vectors | SMPL β via `compute_B_from_beta` |
| **D** (Dynamics) | Per-joint orientation over time | SMPL θ via `smpl_pose_to_D_orientation` |
| **P** (Placement) | Where each IMU sits on the body | Default or manual |
| **H** (Hardware) | Sensor noise & bias | Default or device spec |

---

## Installation

### 1. Clone the repo (smpl branch)

```bash
git clone -b smpl https://github.com/si-runnan/WIMUSim.git
cd WIMUSim
pip install -e .
```

### 2. Core dependencies

```bash
pip install torch torchvision
pip install pytorch3d          # follow https://github.com/facebookresearch/pytorch3d
pip install smplx scipy pandas
```

### 3. HMR2.0 / 4D-Humans (for video input)

```bash
pip install git+https://github.com/shubham-goel/4D-Humans.git
pip install git+https://github.com/facebookresearch/detectron2.git
```

### 4. Download SMPL model files

1. Register at https://smpl.is.tue.mpg.de/ and download **SMPL_python_v.1.1.0.zip**
2. Extract and place the model files:

```
path/to/smpl/models/
    SMPL_NEUTRAL.pkl
    SMPL_MALE.pkl
    SMPL_FEMALE.pkl
```

---

## Quickstart: Video → Virtual IMU

```bash
python pipeline/run.py \
    --video        input.mp4 \
    --smpl_model   path/to/smpl/models \
    --imu          LLA RLA LSH RSH PELV \
    --output       output/imu_data.npz \
    --csv
```

Output files:

```
output/
    imu_data.npz     # all IMUs in one file (LLA_acc, LLA_gyro, ...)
    LLA.csv          # per-IMU CSV (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    RLA.csv
    ...
    smpl_params.npz  # intermediate SMPL parameters
```

### Available IMU positions

The default placement supports these IMU names (from MoVi's 17-sensor setup):

| Name | Body location |
|------|--------------|
| `HED` | Head |
| `STER` | Sternum |
| `PELV` | Pelvis |
| `RSHO` / `LSHO` | Right / Left shoulder |
| `RUA` / `LUA` | Right / Left upper arm |
| `RLA` / `LLA` | Right / Left forearm |
| `RHD` / `LHD` | Right / Left hand |
| `RTH` / `LTH` | Right / Left thigh |
| `RSH` / `LSH` | Right / Left shin |
| `RFT` / `LFT` | Right / Left foot |

### Use in Python

```python
from pipeline.run import run

virtual_IMU_dict = run(
    video_path="input.mp4",
    smpl_model_path="path/to/smpl/models",
    imu_names=["LLA", "RLA", "LSH", "RSH"],
    output_path="output/imu_data.npz",
)

acc_LLA, gyro_LLA = virtual_IMU_dict["LLA"]  # torch.Tensor (T, 3)
```

---

## Datasets

### Training — MoVi

90 subjects × 20 activities × 5 trials. SMPL fits + 17 synchronized IMUs.

Download: https://www.biomotionlab.ca/movi/

Expected directory structure:

```
movi/
    F_Subject01/
        F_Subject01_walk01_poses.npz   ← SMPL fits (AMASS format)
        F_Subject01_walk01_v3d.pkl     ← IMU data
    F_Subject02/
    ...
```

### Test — TotalCapture

5 subjects, 13 IMUs, indoor lab setting.

- Raw IMU data: https://cvssp.org/data/totalcapture/
- SMPL fits (AMASS TotalCapture subset): https://amass.is.tue.mpg.de/

After downloading both, run the preprocessing script:

```bash
python -m dataset_configs.totalcapture.preprocess \
    --tc_root    /data/TotalCapture \
    --amass_root /data/AMASS/TotalCapture \
    --output     /data/tc_processed
```

This converts AMASS SMPL fits + raw Xsens text files and writes:

```
tc_processed/
    s1/
        walking1/
            smpl.npz   # betas (10,), global_orient (T,3,3), body_pose (T,23,3,3)
            imu.pkl    # {imu_name: {'acc': (T,3), 'gyro': (T,3)}}
        acting1/
        ...
    s2/
    ...
```

Optional flags: `--subjects s1 s5`, `--activities walking1 acting1`, `--no_skip`.

### Test — EMDB

9 subjects, 6 IMUs, outdoor in-the-wild sequences.

Download: https://ait.ethz.ch/emdb

EMDB provides SMPL-X parameters. Run the preprocessing script to convert them
to SMPL-compatible format:

```bash
python -m dataset_configs.emdb.preprocess \
    --emdb_root /data/EMDB \
    --output    /data/emdb_processed \
    --split     EMDB_2          # outdoor split (test set)
```

Output structure:

```
emdb_processed/
    EMDB_1/          ← indoor (optional, can use as additional train)
        P1/
            sequence_01.pkl
    EMDB_2/          ← outdoor (test set)
        P1/
            sequence_01.pkl
```

Each processed `.pkl` contains `smpl` (betas, global_orient, body_pose) and
`imu` (acc, gyro) arrays, compatible with `dataset_configs/emdb/utils.py`.

---

## Parameter Identification (fitting to real IMU data)

When you have a dataset with synchronized video + real IMU (e.g. MoVi),
you can optimize WIMUSim parameters to match the real sensor output.

Open and run:

```
examples/parameter_identification.ipynb
```

This minimizes the error between WIMUSim's simulated IMU and the real IMU
by gradient descent over B, D, P, and H.

---

## Data Augmentation with CPM

After parameter identification, use **Comprehensive Parameter Mixing (CPM)**
to generate large-scale diverse virtual IMU training data.

Open and run:

```
examples/parameter_transformation.ipynb
```

CPM mixes B/D/P/H across subjects to simulate different body shapes,
motions, sensor placements, and hardware characteristics.

---

## Sample Rate Alignment

Different sources run at different rates:

| Source | SMPL rate | IMU rate |
|--------|-----------|----------|
| MoVi | 60 Hz | 100 Hz |
| TotalCapture | 60 Hz | 60 Hz |
| EMDB | 60 Hz | 60 Hz |
| HMR2.0 (video) | 30 Hz | — |

Use `pipeline/resample.py` to align them before feeding into WIMUSim:

```python
from pipeline.resample import align_to_smpl_rate

# MoVi: downsample IMU 100 Hz → 60 Hz
global_orient, body_pose, imu_dict, hz = align_to_smpl_rate(
    global_orient, body_pose, imu_dict, smpl_hz=60, imu_hz=100
)

# Video (HMR2.0): upsample SMPL 30 Hz → 60 Hz, then downsample IMU
global_orient, body_pose, imu_dict, hz = align_to_smpl_rate(
    global_orient, body_pose, imu_dict, smpl_hz=60, imu_hz=100, video_hz=30
)
```

SMPL rotation matrices are resampled with SLERP; IMU signals use linear interpolation.

---

## Neural Residual Corrector for WIMUSim

Physics-based simulation has a **sim-to-real gap**: WIMUSim's rigid-body model
cannot perfectly replicate soft-tissue artefacts, non-linear sensor characteristics,
and complex noise patterns found in real IMUs.

The `nn/` package provides a Transformer-based **neural residual corrector**
trained on MoVi paired data (SMPL poses + real IMU) to bridge this gap.

### How it works

```
SMPL pose  →  WIMUSim (physics)  →  phys_imu
     │                                  │
     └──────────── concat ───────────────┘
                       │
              Transformer Encoder
                       │
                   residual                    (learned sim-to-real correction)
                       │
        phys_imu + residual  =  corrected_imu  ≈  real IMU
```

Physics provides a strong, interpretable prior.
The network only needs to model the remaining gap — a much smaller learning task.

Loss: **Smooth L1** on acc and gyro channels separately (different physical scales).
The training log reports improvement over the physics-only baseline, e.g.:

```
Physics-only baseline loss: 0.3841
Epoch  50  train 0.1203  val 0.1389  (+63.8% vs physics)
```

### Step 1 — Train on MoVi

```bash
python -m nn.train \
    --movi_root  /data/MoVi \
    --smpl_model path/to/smpl/models \
    --imu_names  HED STER PELV RUA LUA RLA LLA RHD LHD RTH LTH RSH LSH RFT LFT \
    --output_dir output/checkpoints \
    --epochs     100 \
    --wandb_project wimusim_corrector
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--window` | 128 | Sliding window length (frames) |
| `--d_model` | 256 | Transformer hidden dim |
| `--n_layers` | 4 | Number of Transformer layers |
| `--epochs` | 100 | Training epochs |
| `--wandb_project` | None | W&B project name (omit to disable) |
| `--no_residual` | — | Direct mode: predict IMU from pose only |

Checkpoints saved to `output/checkpoints/best.pt` and `last.pt`.

### Step 2 — Use as drop-in for WIMUSim.simulate()

```python
from nn.infer import corrected_simulate

# Same B, D, P, H as WIMUSim.simulate()
imu_dict = corrected_simulate(
    checkpoint="output/checkpoints/best.pt",
    B=B, D=D, P=P, H=H,
)
# Returns {imu_name: (acc, gyro)} — identical format to WIMUSim output
acc_RLA, gyro_RLA = imu_dict["RLA"]
```

### Step 3 — CLI inference

```bash
python -m nn.infer \
    --checkpoint output/checkpoints/best.pt \
    --smpl_npz   smpl_params.npz \
    --smpl_model path/to/smpl/models \
    --output     output/corrected_imu.npz
```

---

## Example Notebooks

| Notebook | What it does |
|----------|-------------|
| `examples/generate_D_from_3d_pose.ipynb` | Load MoVi SMPL data → simulate virtual IMU |
| `examples/parameter_identification.ipynb` | Fit WIMUSim params to real MoVi IMU data |
| `examples/parameter_transformation.ipynb` | CPM data augmentation for model training |
| `examples/evaluation.ipynb` | Evaluate on TotalCapture / EMDB, output metrics |

---

## Project Structure

```
WIMUSim/
├── wimusim/                   Core simulation engine
│   ├── wimusim.py             WIMUSim class (B, D, P, H → IMU)
│   ├── optimizer.py           Gradient-based parameter identification
│   └── datasets.py            CPM dataset class
├── dataset_configs/
│   ├── smpl/                  SMPL format converters
│   │   ├── consts.py          24-joint skeleton definitions
│   │   └── utils.py           compute_B_from_beta, smpl_pose_to_D_orientation
│   ├── movi/                  MoVi dataset (train)
│   ├── totalcapture/          TotalCapture dataset (test)
│   │   └── preprocess.py      Convert AMASS + raw Xsens → smpl.npz + imu.pkl
│   └── emdb/                  EMDB dataset (test)
│       └── preprocess.py      Convert SMPL-X → SMPL-compatible format
├── pipeline/
│   ├── video_to_smpl.py       HMR2.0 wrapper: video → SMPL params
│   ├── run.py                 End-to-end: video → virtual IMU (CLI)
│   ├── resample.py            Sample rate alignment (SLERP + linear interp)
│   └── evaluate.py            Evaluation metrics (RMSE, MAE, Pearson)
├── nn/                        Neural residual corrector (smpl-nn branch)
│   ├── model.py               NeuralSimulator (Transformer) + rotation utils
│   ├── dataset.py             SimulatorDataset — MoVi pose+IMU pairs
│   ├── train.py               Training script — MoVi paired data (CLI)
│   └── infer.py               corrected_simulate() drop-in for WIMUSim (CLI)
└── examples/                  Jupyter notebooks
```

---

## Citation

```bibtex
@article{xxxx,
  title   = {WIMUSim: Wearable IMU Simulation Framework},
  author  = {xxxx},
  journal = {xxxx},
  year    = {xxxx}
}
```

## License

Apache License 2.0
