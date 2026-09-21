# HPC Training and Replication Guide

This guide details the hardware specifications, software dependencies, hyperparameters, and execution commands required to replicate the training of the PAVEN spatio-temporal video saliency model, matching the experiments conducted on the **CeSViMa Magerit Supercomputer** (Universidad Politécnica de Madrid).

---

## 1. Environment and Hardware Specifications

- **Compute Node:** CeSViMa Magerit HPC Cluster (`standard-gpu` partition)
- **Accelerator:** 1 $\times$ NVIDIA A100-SXM4-40GB / A100-PCIE-80GB GPU (allocated with at least 24 GB RAM per task)
- **Host CPU:** AMD EPYC / Intel Xeon with at least 4 worker threads
- **Software Stack:**
  - Linux (CentOS / Rocky Linux / Ubuntu)
  - Python 3.10.8
  - PyTorch 2.2.1
  - CUDA 12.3.0
  - cuDNN 8.9.2.26

---

## 2. Training Hyperparameters (Table 4.4 from Master Thesis)

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Optimizer** | Adam | Standard adaptive gradient optimization |
| **Learning Rate ($lr$)** | $1 \times 10^{-4}$ | Stable convergence without exploding gradients |
| **Batch Size** | 4 (or 8) | Fits temporal 3D tensor clips into 24 GB GPU memory |
| **Temporal Window (Clip Size)** | 32 frames | Captures spatio-temporal motion dynamics across 1 second |
| **Spatial Resolution** | $384 \times 224$ | Balanced resolution for 3D Inception blocks |
| **Input Mode** | `IS_RGB = 2` (`YYY`) | Replicates luma channel to prevent chrominance distortion |
| **Loss Function** | Combined | $\mathcal{L} = \mathcal{L}_{KLD} + \alpha \mathcal{L}_{CC} + \beta \mathcal{L}_{SIM}$ |
| **Number of Epochs** | 40 | Typically converges within 15–20 epochs ($\sim 16$ hours runtime) |

---

## 3. Launching Training

### A. Local Workstation / Single GPU
```bash
cd training/
python train.py \
    --batch_size 4 \
    --no_workers 4 \
    --clip_size 32 \
    --lr 1e-4 \
    --no_epochs 40 \
    --model_val_path "paven_vinet_checkpoint.pt"
```

### B. High-Performance Cluster (SLURM)
Submit the provided job script directly:
```bash
sbatch training/train_slurm.sh
```

Inspect job status:
```bash
squeue -u $USER
```

---

## 4. Checkpoints and Verification

During training, checkpoints are evaluated against the validation split using Kullback-Leibler Divergence (KLD), Pearson Correlation Coefficient (CC), and Similarity (SIM). The optimal checkpoint is saved and corresponds directly to the official weights hosted on Hugging Face: `paven_vinet_yyy.pt`.
