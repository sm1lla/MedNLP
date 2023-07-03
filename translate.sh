#!/bin/bash -eux
#SBATCH --job-name=translate
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@mattermost
#SBATCH --partition=gpu # -p
#SBATCH --gpus=1
#SBATCH --time=23:00:00
#SBATCH --mem=30G

 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate mednlp
set +eu

python -m src task=translate