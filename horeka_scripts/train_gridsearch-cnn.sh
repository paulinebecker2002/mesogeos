#!/bin/bash
#SBATCH --job-name=cnn-gridsearch
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0024498
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu

PYTHON="/home/hk-project-pai00005/uyxib/miniconda3/envs/mesogeos_py38/bin/python"
MODEL_NAME="cnn"

CONFIG_TRAIN_PATH="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/configs/config_cnn/config_train.json"
TRAIN_SCRIPT="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/train.py"
SAVE_DIR="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/models"

CONFIG_TEST_PATH="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/configs/config_cnn/config_test.json"
TEST_SCRIPT="/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/test.py"

cd /hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger
# GridSearch Parameters
for lr in 0.0005 0.001 0.002
do
  for dr in 0.0 0.025 0.05 0.075
  do
    for bs in 256
    do
      echo "Running with lr=$lr, dropout=$dr, batch_size=$bs"
      $PYTHON $TRAIN_SCRIPT \
        --config $CONFIG_TRAIN_PATH \
        --lr $lr \
        --dr $dr \
        --bs $bs

      # Get latest run_id (last modified dir in models/$MODEL_NAME/)
      RUN_ID=$(ls -t $SAVE_DIR/$MODEL_NAME/ | head -n 1)
      MODEL_PATH="$SAVE_DIR/$MODEL_NAME/$RUN_ID/model_best.pth"

      echo "Testing model: $MODEL_PATH"
      $PYTHON $TEST_SCRIPT \
        --config $CONFIG_TEST_PATH \
        --mp $MODEL_PATH

    done
  done
done
