#!/bin/bash
#SBATCH --job-name=a_transformer_tune
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu

~/miniconda3/envs/mesogeos_bw/bin/python /pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/optuna_tune.py --config /pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/configs/config_transformer/config_train.json
#~/miniconda3/envs/mesogeos_bw/bin/python /pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a_fire_danger/train.py --config /pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a_fire_danger/configs/config_transformer/config_train.json
