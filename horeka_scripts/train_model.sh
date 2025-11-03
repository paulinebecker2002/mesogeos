#!/bin/bash
#SBATCH --job-name=train_transformer
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=23:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mesogeos_py38

cd /hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger

python trainer/timelag_seed_training.py