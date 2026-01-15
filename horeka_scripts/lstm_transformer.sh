#!/bin/bash
#SBATCH --job-name=cval_gtn
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


cd /hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger

for tlag in 5 10 15 20 25 30; do
    ~/miniconda3/envs/mesogeos_py38/bin/python train.py --config configs/config_mlp/config_train.json --tlag $tlag
done