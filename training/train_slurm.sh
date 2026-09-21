#!/bin/bash
##----------------------- Start SLURM job description -----------------------
#SBATCH --partition=standard-gpu
#SBATCH --job-name=paven_vinet_train
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=END,FAIL
## Uncomment and set your email to receive job notifications:
## #SBATCH --mail-user=your_email@domain.com
##------------------------ End SLURM job description ------------------------

# Environment modules configuration (e.g. CeSViMa Magerit cluster)
# module purge && ml cuDNN/8.9.2.26-CUDA-12.2.0 && ml CUDA/12.3.0 && ml Python/3.10.8-GCCcore-12.2.0

echo "[PAVEN HPC] Starting training job on node: $(hostname)"
echo "[PAVEN HPC] CUDA Device Information:"
nvidia-smi

# Execute training with PyTorch
srun python train.py --batch_size 4 --no_workers 4 --clip_size 32 --lr 1e-4 --no_epochs 40