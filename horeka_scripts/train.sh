#!/bin/bash
#SBATCH --job-name=cnn_c
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu


cd /hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger


~/miniconda3/envs/mesogeos_py38/bin/python train.py --config configs/config_cnn/config_train.json --pos_source positives_inland.csv --neg_source negatives_inland.csv
