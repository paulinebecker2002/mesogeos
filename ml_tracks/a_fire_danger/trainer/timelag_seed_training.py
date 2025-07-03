import os
import subprocess
from pathlib import Path

MODEL_NAME = "transformer"

BASE_DIR = Path("/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger")
LOG_DIR = BASE_DIR / "saved/log" / MODEL_NAME
MODEL_DIR = BASE_DIR / "saved/models" / MODEL_NAME
TEST_SCRIPT = BASE_DIR / "test.py"
TRAIN_SCRIPT = BASE_DIR / "train.py"
CONFIG_TEST = BASE_DIR / f"configs/config_{MODEL_NAME}/config_test.json"
CONFIG_TRAIN = BASE_DIR / f"configs/config_{MODEL_NAME}/config_train.json"
TEST_LOG_BASE = BASE_DIR / "tester/saved/log" / MODEL_NAME
SAVE_DIR = BASE_DIR / "saved/crossValidation"

# Die Time-Lags und Seeds
time_lags = [5, 10, 15, 20, 25, 30]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12345]


for lag in time_lags:
    for seed in seeds:
        print(f"⏳ Training mit Lag: {lag}, Seed: {seed}")

        train_cmd = [
            "python", str(TRAIN_SCRIPT),
            "--config", str(CONFIG_TRAIN),
            "--tlag", str(lag),       # Fix hier
            "--seed", str(seed),
        ]

        subprocess.run(train_cmd)

        subdirs = sorted(MODEL_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
        if not subdirs:
            raise RuntimeError(f"❌ Kein Modellordner in {MODEL_DIR} gefunden.")
        last_model_dir = subdirs[-1]
        run_id = last_model_dir.name




        model_path = os.path.join(MODEL_DIR, run_id, "model_best.pth")
        print(f"✅ Verwende Modell: {model_path}")

        test_cmd = [
            "python", str(TEST_SCRIPT),
            "--config", str(CONFIG_TEST),
            "--tlag", str(lag),       # Fix hier
            "--seed", str(seed),
            "--model_path", str(model_path)
        ]

        subprocess.run(test_cmd)
