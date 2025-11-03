import re
import subprocess
from pathlib import Path
import pandas as pd


MODEL_NAME = "gtn"

BASE_DIR = Path("/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger")
LOG_DIR = BASE_DIR / "saved/log" / MODEL_NAME
MODEL_DIR = BASE_DIR / "saved/models" / MODEL_NAME
TEST_SCRIPT = BASE_DIR / "test.py"
CONFIG_TEST = BASE_DIR / f"configs/config_{MODEL_NAME}/config_test.json"
TEST_LOG_BASE = BASE_DIR / "tester/saved/log" / MODEL_NAME
SAVE_DIR = BASE_DIR / "saved/crossValidation"

patterns = {
    "train_f1_score": re.compile(r"^\s*.*INFO\s+-\s+f1_score\s+:\s+([0-9.]+)"),
    "train_aucpr": re.compile(r"^\s*.*INFO\s+-\s+aucpr\s+:\s+([0-9.]+)"),
    "val_f1_score": re.compile(r"val_f1_score\s+:\s+([0-9.]+)"),
    "val_aucpr": re.compile(r"val_aucpr\s+:\s+([0-9.]+)"),
    "test_dict": re.compile(r"INFO\s+-\s+\{.*?'f1_score':\s*([0-9.]+).*?'aucpr':\s*([0-9.]+).*?\}")
}
model_best_pattern = re.compile(r"Saving current best: model_best\.pth")

results = []
log_paths = sorted(LOG_DIR.glob("*/info.log"), key=lambda p: p.parent.stat().st_mtime, reverse=True)[:15]

VAL_SETS = [
    "2020", "2019", "2018", "2017", "2016",
    "2015", "2014", "2013", "2012", "2011",
    "2010", "2009", "2008", "2007", "2006"
]

TEST_SETS = [
    "2021 2022", "2020 2021", "2019 2020", "2018 2019", "2017 2018",
    "2016 2017", "2015 2016", "2014 2015", "2013 2014", "2012 2013",
    "2011 2012", "2010 2011", "2009 2010", "2008 2009", "2007 2008"
]

TRAIN_SETS = [
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019",
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2022",
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2021 2022",
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2020 2021 2022",
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2019 2020 2021 2022",
    "2006 2007 2008 2009 2010 2011 2012 2013 2014 2018 2019 2020 2021 2022",
    "2006 2007 2008 2009 2010 2011 2012 2013 2017 2018 2019 2020 2021 2022",
    "2006 2007 2008 2009 2010 2011 2012 2016 2017 2018 2019 2020 2021 2022",
    "2006 2007 2008 2009 2010 2011 2015 2016 2017 2018 2019 2020 2021 2022",
    "2006 2007 2008 2009 2010 2014 2015 2016 2017 2018 2019 2020 2021 2022",
    "2006 2007 2008 2009 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022",
    "2006 2007 2008 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022",
    "2006 2007 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022",
    "2006 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022",
    "2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022",
]



for idx, log_file in enumerate(log_paths):
    run_id = log_file.parent.name
    result = {"run_id": run_id}

    with log_file.open("r") as f:
        lines = f.readlines()


    for i in range(len(lines) - 1, -1, -1):
        if model_best_pattern.search(lines[i]):
            for j in range(i-1, -1, -1):
                for key in ["train_f1_score", "train_aucpr", "val_f1_score", "val_aucpr"]:
                    if result.get(key) is None and (m := patterns[key].search(lines[j])):
                        result[key] = float(m.group(1))
                if all(result.get(k) is not None for k in ["train_f1_score", "train_aucpr", "val_f1_score", "val_aucpr"]):
                    break
            break

    # test.py
    model_path = MODEL_DIR / run_id / "model_best.pth"
    if model_path.exists():
        try:
            train_year = TRAIN_SETS[idx]
            val_year = VAL_SETS[idx]
            test_year = TEST_SETS[idx]
            subprocess.run([
                "python", str(TEST_SCRIPT),
                "--config", str(CONFIG_TEST),
                "--mp", str(model_path),
                "--train_year", *train_year.split(),
                "--val_year", *val_year.split(),
                "--test_year", *test_year.split()
            ], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error for test.py at {run_id}: {e}")
            continue


        test_subdirs  = sorted(
            TEST_LOG_BASE.glob("*/info.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if test_subdirs:
            test_log = test_subdirs[0]
            print(f" info.log found at: {test_log}")
            with test_log.open() as f:
                for line in reversed(f.readlines()):
                    if (m := patterns["test_dict"].search(line)):
                        test_f1_score = float(m.group(1))
                        test_aucpr = float(m.group(2))
                        result["test_f1_score"] = test_f1_score
                        result["test_aucpr"] = test_aucpr
                        print(f"Test-Scores: f1_score={test_f1_score}, aucpr={test_aucpr}")
                        break
        else:
            print("no info.log found.")

    results.append(result)

df = pd.DataFrame(results)
csv_path = SAVE_DIR / f"{MODEL_NAME}_cv_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results stored at: {csv_path}")
