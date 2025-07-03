import json
import re
from pathlib import Path

# Basisverzeichnis, in dem alle logs liegen
BASE_DIR = Path("/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/log/transformer")

best_model = None
best_f1 = -1

DESIRED_TIMESTEPS = 5
timesteps_pattern = re.compile(r"Last n timesteps:\s+(\d+)")
f1_pattern = re.compile(r"'f1_score': ([0-9.]+)")
model_path_pattern = re.compile(r"Loading checkpoint:\s*(.*model_best\.pth)")

results = []

for log_file in BASE_DIR.glob("*/info.log"):
    with log_file.open('r') as f:
        lines = f.readlines()
        if not lines:
            continue

        # get Model path from first line
        first_line = lines[0]
        model_path_match = model_path_pattern.search(first_line)
        if model_path_match:
            model_path_str = model_path_match.group(1)
            model_path = Path(model_path_str)

            # get f1_score from last line (only necessary if model_path is found)
            last_line = lines[-1]
            match = f1_pattern.search(last_line)
            if match:
                f1_score = float(match.group(1))
                results.append((log_file, model_path, f1_score))

                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model = (log_file, model_path, f1_score)

if best_model:
    log_file, model_path, best_f1 = best_model
    print(f"Best model log: {log_file}")
    print(f"Best f1_score: {best_f1}")
    print(f"Best model path: {model_path}")

    config_path = model_path.parent / "config_train.json"
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config = json.load(f)

        lr = config.get('optimizer', {}).get('args', {}).get('lr', 'N/A')
        batch_size = config.get('dataloader', {}).get('args', {}).get('batch_size', 'N/A')
        dropout = config.get('model_args', {}).get('dropout', 'N/A')

        print(f"Model Parameters:")
        print(f"   - Learning rate : {lr}")
        print(f"   - Batch size    : {batch_size}")
        print(f"   - Dropout       : {dropout}")
    else:
        print(f"config_train.json not found at {config_path}")