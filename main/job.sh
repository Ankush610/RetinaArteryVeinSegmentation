#!/bin/bash
#SBATCH --job-name=av_train           # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (one per GPU per node)
#SBATCH --gres=gpu:1                   # Number of GPUs on each node
#SBATCH --cpus-per-task=20            # Number of CPU cores per task
#SBATCH --partition=gpu                # GPU partition
#SBATCH --output=logs/logs_%j.out           # Output log file
#SBATCH --error=logs/logs_%j.err            # Error log file
#SBATCH --time=24:00:00                # Time limit

# -----------------------------
# Environment setup
# -----------------------------

source /home/omjadhav/ankush/medical_project/Multiclass-Segmentation-in-PyTorch/.venv/bin/activate

# -----------------------------
# Run the training script
# -----------------------------

python train.py
