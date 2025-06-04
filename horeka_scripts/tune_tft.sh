#!/bin/bash
#SBATCH --job-name=tft_train_fire
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu


cd /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger

~/miniconda3/envs/mesogeos_py38/bin/python /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/train.py --config /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/configs/config_tft/config_train.json
