#!/bin/bash -eux
#SBATCH --job-name=train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@mattermost
#SBATCH --partition=gpu # -p
#SBATCH --gpus=1

 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate mednlp
set +eu

python -m src task=train