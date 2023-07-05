#!/bin/bash -eux
#SBATCH --job-name=sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@mattermost
#SBATCH --partition=gpupro # -p
#SBATCH --gpus=1
#SBATCH --time=20:00:00

 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate mednlp
set +eu

wandb agent mednlp/deterministic/qawt064s