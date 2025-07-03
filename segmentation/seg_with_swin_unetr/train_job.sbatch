#!/bin/bash
#SBATCH --job-name=swinunetr
#SBATCH --partition=dgx-small
#SBATCH --account=ddt_acc23
#SBATCH --time=12:00:00
#SBATCH --output=swin_unetr_batch1_32/job_%j.out
#SBATCH --error=swin_unetr_batch1_32/job_%j.err

# Kích hoạt môi trường Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seg-env

# Chạy script Python
python run.py
