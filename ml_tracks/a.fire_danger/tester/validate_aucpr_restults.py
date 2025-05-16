import json
import re
import subprocess
from pathlib import Path

MODEL_NAME = "cnn"  # Change this to the model name you want to analyze
BASE_DIR = Path(f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/log/{MODEL_NAME}")

best_model = None
best_aucpr = -1


f1_pattern = re.compile(r"val_f1_score\s+:\s+([0-9.]+)")
aucpr_pattern = re.compile(r"val_aucpr\s+:\s+([0-9.]+)")
precision_pattern = re.compile(r"val_precision\s+:\s+([0-9.]+)")
recall_pattern = re.compile(r"val_recall\s+:\s+([0-9.]+)")
model_best_pattern = re.compile(r"Saving current best: model_best\.pth")

results = []

for log_file in BASE_DIR.glob("*/info.log"):
    with log_file.open('r') as f:
        lines = f.readlines()
        if not lines:
            continue
        for i in range(len(lines)-1, -1, -1):
            if model_best_pattern.search(lines[i]):
                f1_score = None
                auprc = None
                precision = None
                recall = None

                for j in range(i-1, -1, -1):
                    if f1_score is None and (match := f1_pattern.search(lines[j])):
                        f1_score = float(match.group(1))
                    if auprc is None and (match := aucpr_pattern.search(lines[j])):
                        auprc = float(match.group(1))
                    if precision is None and (match := precision_pattern.search(lines[j])):
                        precision = float(match.group(1))
                    if recall is None and (match := recall_pattern.search(lines[j])):
                        recall = float(match.group(1))

                    if f1_score is not None and auprc is not None and precision is not None and recall is not None:
                        break

                if auprc is not None:
                    if auprc > best_aucpr:
                        best_aucpr = auprc
                        best_model = log_file
                        best_metrics = {
                            "f1_score": f1_score,
                            "auprc": auprc,
                            "precision": precision,
                            "recall": recall
                        }
                break

if best_model:
    print(f"Best model log: {best_model}")
    print(f"Best val_f1_score : {best_metrics['f1_score']}")
    print(f"Best val_auprc    : {best_metrics['auprc']}")
    print(f"Best val_precision: {best_metrics['precision']}")
    print(f"Best val_recall   : {best_metrics['recall']}")

    model_path = Path(str(best_model).replace("/log/", "/models/")).parent / "model_best.pth"
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

    print(f"\nRunning test.py with model_path: {model_path}")
    cmd = [
        "python",
        "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/test.py",
        "--config", f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/configs/config_{MODEL_NAME}/config_test.json",
        "--mp", str(model_path)
    ]
    subprocess.run(cmd, check=True)
