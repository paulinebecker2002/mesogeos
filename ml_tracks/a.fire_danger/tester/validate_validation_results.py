import json
import re
from pathlib import Path

# Basisverzeichnis, in dem alle logs liegen
BASE_DIR = Path("/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/log/transformer")

best_model = None
best_f1 = -1

# Muster für die f1_score-Zeile
f1_pattern = re.compile(r"val_f1_score\s+:\s+([0-9.]+)")
# Muster für die model_best.pth Zeile
model_best_pattern = re.compile(r"Saving current best: model_best\.pth")

results = []

for log_file in BASE_DIR.glob("*/info.log"):
    with log_file.open('r') as f:
        lines = f.readlines()
        if not lines:
            continue

        # Wir suchen von unten nach oben das letzte model_best.pth
        for i in range(len(lines)-1, -1, -1):
            if model_best_pattern.search(lines[i]):
                # die f1_score steht ein paar Zeilen darüber, suche von (i) rückwärts
                for j in range(i-1, -1, -1):
                    if 'f1_score' in lines[j]:
                        match = f1_pattern.search(lines[j])
                        if match:
                            f1_score = float(match.group(1))
                            results.append((log_file, f1_score))

                            if f1_score > best_f1:
                                best_f1 = f1_score
                                best_model = log_file
                        break  # wir brauchen nur die nächste f1_score darüber
                break  # wir brauchen nur das letzte model_best.pth

if best_model:
    print(f"Best model log: {best_model}")
    print(f"Best f1_score: {best_f1}")

    # model_path: log → models tauschen
    model_path = Path(str(best_model).replace("/log/", "/models/")).parent / "model_best.pth"
    print(f"Best model path: {model_path}")

    # config_train.json im selben model Ordner
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
