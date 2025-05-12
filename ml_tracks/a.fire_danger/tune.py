import subprocess
from itertools import product


config_path = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/configs/config_gru/config_train.json"
train_script = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/train.py"
#Porblem here that python Path might change with hk-project id
python_exec = "/home/hk-project-pai00005/uyxib/miniconda3/envs/mesogeos_py38/bin/python"

# Grid: Parameterkombinationen
#lr_values = [0.001, 0.005, 0.01]
#dropout_values = [0.1, 0.3, 0.5]
#batch_sizes = [128, 256]

# Grid: Parameterkombinationen
lr_values = [0.001]
dropout_values = [0.1, 0.3]
batch_sizes = [128]

combinations = list(product(lr_values, dropout_values, batch_sizes))
for i, (lr, dr, bs) in enumerate(combinations, 1):
    #only print not stored in logger
    print(f"\nStarting run {i}/{len(combinations)}: lr={lr}, dropout={dr}, batch_size={bs}")

    cmd = [
        python_exec, train_script,
        "--config", config_path,
        "--lr", str(lr),
        "--dr", str(dr),
        "--bs", str(bs)
    ]

    subprocess.run(cmd)
