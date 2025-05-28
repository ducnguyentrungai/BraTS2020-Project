#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=ddt_acc23
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=test_gpu.%j.out
#SBATCH --error=test_gpu.%j.err

# Chạy test.py
python test.py

# #GPU
# srun --pty --job-name=SLRT --partition=gpu --gres=gpu:1 --account=ddt_acc23 --mem=64gb --time=4:00:00 --cpus-per-task=8 --nodes=1 /bin/bash