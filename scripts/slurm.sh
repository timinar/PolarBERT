#!/bin/bash
#SBATCH --job-name=flash
#SBATCH --output=../logs/flash_%j.log
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28000
#SBATCH --time=8:00:00


source /groups/pheno/inar/mambaforge/etc/profile.d/conda.sh  # activate conda
conda activate pytorch2  # activate your environment

# cd scripts

# srun python base_training.py --config ../configs/mss_transformer_short.yaml --model_type flash
# srun python base_training.py --config ../configs/mss_transformer_10ep.yaml --model_type flash

srun python prometheus_training.py --config ../configs/mss_transformer_prometheus.yaml --model_type flash 