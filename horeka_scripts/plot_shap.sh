#!/bin/bash
#SBATCH --job-name=shap_all_models
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu


cd /hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger

MODEL_NAMES=("cnn" "mlp" "gru" "lstm" "transformer" "gtn" "rf" "tft")

for MODEL_NAME in "${MODEL_NAMES[@]}";
do
    echo "Running SHAP for model: $MODEL_NAME"
    CONFIG_TRAIN_PATH="configs/config_${MODEL_NAME}/config_train.json"
    ~/miniconda3/envs/mesogeos_py38/bin/python shap_local/plot_shap.py --config "$CONFIG_TRAIN_PATH"
done