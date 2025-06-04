import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import re

# Fixed base directory where CSVs are stored
BASE_PATH = Path("/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/csv/validation_softmax_csv")

def load_predictions(filename, model_name):
    path = BASE_PATH / filename
    df = pd.read_csv(path)
    df = df.rename(columns={
        "true_label": "true",
        "pred_label": f"pred_{model_name}",
        "prob": f"prob_{model_name}"
    })
    df["model"] = model_name
    return df

def extract_model_name(filename):
    filename = filename.replace(".csv", "")
    filename = re.sub(r'^(val_|test_)', '', filename)
    filename = re.sub(r'_outputs_epoch\d+', '', filename)
    return filename

def assign_classification_result(df, pred_col, true_col="true"):
    return df.apply(lambda row:
                    "TP" if row[true_col] == 1 and row[pred_col] == 1 else
                    "TN" if row[true_col] == 0 and row[pred_col] == 0 else
                    "FP" if row[true_col] == 0 and row[pred_col] == 1 else
                    "FN", axis=1), df[true_col]

def main(args):
    model1_name = extract_model_name(args.model1_name)
    model2_name = extract_model_name(args.model2_name)

    df1 = load_predictions(args.model1_name, "model1")
    df2 = load_predictions(args.model2_name, "model2")

    # Merge predictions using coordinates and ground truth as keys
    df = pd.merge(df1, df2, on=["lat", "lon", "true"])

    df["result_model1"], _ = assign_classification_result(df, "pred_model1")
    df["result_model2"], _ = assign_classification_result(df, "pred_model2")

    # Find samples where one model is better than the other
    model1_better = df[(df["result_model1"].isin(["TP", "TN"])) & (df["result_model2"].isin(["FP", "FN"]))]
    model2_better = df[(df["result_model2"].isin(["TP", "TN"])) & (df["result_model1"].isin(["FP", "FN"]))]

    # Print statistics
    print("Number of False Positives:")
    print(f"  {model1_name}: {(df['result_model1'] == 'FP').sum()}")
    print(f"  {model2_name}: {(df['result_model2'] == 'FP').sum()}")
    print("\nNumber of False Negatives:")
    print(f"  {model1_name}: {(df['result_model1'] == 'FN').sum()}")
    print(f"  {model2_name}: {(df['result_model2'] == 'FN').sum()}")
    print("\nPer-sample model performance comparison:")
    print(f"  {model1_name} performs better on: {len(model1_better)} samples")
    print(f"  {model2_name} performs better on: {len(model2_better)} samples")

    # Plot: False Positives
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[model1_name, model2_name],
                y=[(df["result_model1"] == "FP").sum(), (df["result_model2"] == "FP").sum()])
    plt.title("False Positives Comparison")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"false_positives_{model1_name}_vs_{model2_name}.png")

    # Plot: False Negatives
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[model1_name, model2_name],
                y=[(df["result_model1"] == "FN").sum(), (df["result_model2"] == "FN").sum()])
    plt.title("False Negatives Comparison")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"false_negatives_{model1_name}_vs_{model2_name}.png")

    # Define all four pairwise comparison types
    cases = [
        ("TP", "FN", "correct true positive", "false negative"),
        ("TN", "FP", "correct true negative", "false positive"),
        ("FN", "TP", "false negative", "correct true positive"),
        ("FP", "TN", "false positive", "correct true negative"),
    ]

    print("Detailed cross-model comparison:")

    for r1, r2, desc1, desc2 in cases:
        mask = (df["result_model1"] == r1) & (df["result_model2"] == r2)
        count = mask.sum()
        if count > 0:
            print(f"{model1_name} predicts {desc1} while {model2_name} predicts {desc2}: {count} samples")

    print("\nPlots saved as:")
    print(f"  - false_positives_{model1_name}_vs_{model2_name}.png")
    print(f"  - false_negatives_{model1_name}_vs_{model2_name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predictions of two models on the same validation data.")
    parser.add_argument('--model1_name', type=str, required=True,
                        help="Filename of model 1 CSV (e.g. lstm_predictions_epoch10.csv)")
    parser.add_argument('--model2_name', type=str, required=True,
                        help="Filename of model 2 CSV (e.g. transformer_predictions_epoch10.csv)")
    args = parser.parse_args()
    main(args)
