import csv
import re
from pathlib import Path

"""
Parses training log files for a specific model and extracts key evaluation metrics for selected input timelag configurations.

This script collects F1 scores and AUCPR values from training logs of different model runs (e.g., Transformer, LSTM),
but only includes those runs that were trained with a specified set of timelag values (e.g., 5, 10, 15, 20, 25, 30).
The results are written to a CSV file for further analysis and comparison.

@Author: [Your Name]
@Date: [Insert Date]
@Model: Set via MODEL_NAME variable (e.g., "transformer", "lstm")
@LogDirectory: Defined by BASE_DIR; expected to contain subfolders named like MMDD_HHMMSS/info.log

Inputs:
    - MODEL_NAME: name of the model to process
    - BASE_DIR: base directory containing subdirectories with log files
    - TIMESTEP_VALUES: list of desired timestep values (e.g., [5, 10, 15, 20, 25, 30])
    - Each log file must contain:
        * "Last n timesteps: <int>"
        * "val_f1_score : <float>"
        * "test_f1_score : <float>"
        * "val_aucpr : <float>"
        * "test_aucpr : <float>"

Outputs:
    - A CSV file named {MODEL_NAME}_summary.csv in the current working directory
    - Each row in the CSV contains:
        * Timelag: timestep value used in the run
        * Model: model name
        * F1 Training, Validation, and Testing scores
        * AUCPR Training, Validation, and Testing scores
        * Seed: training seed, if found
        * Date: parsed from subfolder name (e.g., "0709_152301" -> "07-09")

Limitations:
    - Assumes consistent log formatting (INFO lines and metric labels)
    - Assumes log directories are named as MMDD_HHMMSS
"""

MODEL_NAME = "transformer"  # Change this to the model name you want to analyze
BASE_DIR = Path(f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/log/{MODEL_NAME}")
OUTPUT_CSV = f"{MODEL_NAME}_summary.csv"
TIMESTEP_VALUES = [5, 10, 15, 20, 25, 30]


timesteps_pattern = re.compile(r"Last n timesteps:\s*(\d+)")
f1_train_pattern = re.compile(r"^\s*.*INFO\s+-\s+f1_score\s+:\s+([0-9.]+)")
f1_val_pattern = re.compile(r"val_f1_score\s+:\s+([0-9.]+)")
f1_test_pattern = re.compile(r"test_f1_score\s+:\s+([0-9.]+)")
aucpr_train_pattern = re.compile(r"^\s*.*INFO\s+-\s+aucpr\s+:\s+([0-9.]+)")
aucpr_val_pattern = re.compile(r"val_aucpr\s+:\s+([0-9.]+)")
aucpr_test_pattern = re.compile(r"test_aucpr\s+:\s+([0-9.]+)")
seed_pattern = re.compile(r"Seed:\s*(\d+)")

rows = []

for log_file in BASE_DIR.glob("*/info.log"):
    with log_file.open('r') as f:
        lines = f.readlines()
        if not lines:
            continue

        timestep = None
        for line in lines:
            if (m := timesteps_pattern.search(line)):
                timestep = int(m.group(1))
                break

        if timestep not in TIMESTEP_VALUES:
            continue

        f1_train = f1_val = f1_test = None
        aucpr_train = aucpr_val = aucpr_test = None
        seed = None

        for line in reversed(lines):
            if f1_test is None and (m := f1_test_pattern.search(line)):
                f1_test = float(m.group(1))
            if f1_val is None and (m := f1_val_pattern.search(line)):
                f1_val = float(m.group(1))
            if f1_train is None and (m := f1_train_pattern.search(line)):
                f1_train = float(m.group(1))

            if aucpr_test is None and (m := aucpr_test_pattern.search(line)):
                aucpr_test = float(m.group(1))
            if aucpr_val is None and (m := aucpr_val_pattern.search(line)):
                aucpr_val = float(m.group(1))
            if aucpr_train is None and (m := aucpr_train_pattern.search(line)):
                aucpr_train = float(m.group(1))

            if seed is None and (m := seed_pattern.search(line)):
                seed = m.group(1)

            if all(x is not None for x in [f1_train, f1_val, f1_test, aucpr_train, aucpr_val, aucpr_test]):
                break

        try:
            date_obj = log_file.parent.name
        except ValueError:
            date_obj = "unknown"

        row = {
            "Timelag": timestep,
            "Model": MODEL_NAME,
            "F1 Training": f1_train,
            "F1 Validation": f1_val,
            "F1 Testing": f1_test,
            "AUCPR Training": aucpr_train,
            "AUCPR Validation": aucpr_val,
            "AUCPR Testing": aucpr_test,
            "Seed": seed,
            "Date": date_obj
        }
        rows.append(row)

# Save to CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ["Timelag", "Model", "F1 Training", "F1 Validation", "F1 Testing",
                  "AUCPR Training", "AUCPR Validation", "AUCPR Testing", "Seed", "Date"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved results to {OUTPUT_CSV}")
