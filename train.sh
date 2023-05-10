#!/bin/bash -eux
#SBATCH --job-name=trocrPartitions
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.preiss@student.hpi.uni-potsdam.de
#SBATCH --partition=gpupro # -p
#SBATCH --gpus=1
#SBATCH --output=trocrPartitions2kto10k.out

 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate mlens
set +eu


python -m src train
