#!/bin/bash -eux
#SBATCH --job-name=sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@mattermost
#SBATCH --partition=gpu # -p
#SBATCH --gpus=1

 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate mednlp
set +eu

wandb agent mednlp/MedNLP/77j94i1e
