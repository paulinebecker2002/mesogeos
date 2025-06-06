#!/bin/bash
#SBATCH --job-name=train_rf
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

~/miniconda3/envs/mesogeos_py38/bin/python /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/train.py \
  --config /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/configs/config_rf/config_train.json
