#!/bin/bash
#SBATCH --job-name=ale_all_models
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

MODEL_NAMES=("cnn" "gru" "lstm" "transformer" "gtn" "rf" "tft")

for MODEL_NAME in "${MODEL_NAMES[@]}";
do
    echo "Running ALE for model: $MODEL_NAME"
    CONFIG_TRAIN_PATH="configs/config_${MODEL_NAME}/config_train.json"
    ~/miniconda3/envs/mesogeos_py38/bin/python xai/ale.py --config "$CONFIG_TRAIN_PATH"
done