#!/bin/bash
#SBATCH --job-name=gtn_train
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pauline.becker@student.kit.edu

PYTHON="/pfs/data6/home/ka/ka_iti/ka_hr7238/miniconda3/envs/mesogeos_bw/bin/python3.8"
MODEL_NAME="gtn"

CONFIG_TRAIN_PATH="/pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/configs/config_gtn/config_train.json"
TRAIN_SCRIPT="/pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/train.py"
SAVE_DIR="/pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/train.py/saved/models"

CONFIG_TEST_PATH="/pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/configs/config_gtn/config_test.json"
TEST_SCRIPT="/pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger/test.py"

cd /pfs/work9/workspace/scratch/ka_hr7238-mesogeos/code/ml_tracks/a.fire_danger
# GridSearch Parameters
for lr in 0.001
do
  for dr in 0.1
  do
    for bs in 128
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