import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_predictions(path, model_name):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "true_label": "true",
        "pred_label": f"pred_{model_name}",
        "prob": f"prob_{model_name}"
    })
    df["model"] = model_name
    return df

def assign_classification_result(df, pred_col, true_col="true"):
    return df.apply(lambda row:
                    "TP" if row[true_col] == 1 and row[pred_col] == 1 else
                    "TN" if row[true_col] == 0 and row[pred_col] == 0 else
                    "FP" if row[true_col] == 0 and row[pred_col] == 1 else
                    "FN", axis=1), df[true_col]

def main(args):
    df1 = load_predictions(args.model1_path, "model1")
    df2 = load_predictions(args.model2_path, "model2")

    # Merge predictions by lat/lon
    df = pd.merge(df1, df2, on=["lat", "lon", "true"])

    # Assign FP/FN/TP/TN categories
    df["result_model1"], _ = assign_classification_result(df, "pred_model1")
    df["result_model2"], _ = assign_classification_result(df, "pred_model2")

    # Vergleich: wo ist welches Modell besser?
    model1_besser = df[(df["result_model1"].isin(["TP", "TN"])) & (df["result_model2"].isin(["FP", "FN"]))]
    model2_besser = df[(df["result_model2"].isin(["TP", "TN"])) & (df["result_model1"].isin(["FP", "FN"]))]

    # Statistiken
    print("Anzahl falsch positiver Vorhersagen:")
    print("  Model 1:", (df["result_model1"] == "FP").sum())
    print("  Model 2:", (df["result_model2"] == "FP").sum())
    print("\nAnzahl falsch negativer Vorhersagen:")
    print("  Model 1:", (df["result_model1"] == "FN").sum())
    print("  Model 2:", (df["result_model2"] == "FN").sum())
    print("\nVergleich:")
    print("  Model 1 besser:", len(model1_besser))
    print("  Model 2 besser:", len(model2_besser))

    # Plot: Balkendiagramm falsch positive
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Model 1", "Model 2"],
                y=[(df["result_model1"] == "FP").sum(), (df["result_model2"] == "FP").sum()])
    plt.title("Falsch Positive")
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig("falsch_positive_vergleich.png")

    # Plot: Balkendiagramm falsch negative
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Model 1", "Model 2"],
                y=[(df["result_model1"] == "FN").sum(), (df["result_model2"] == "FN").sum()])
    plt.title("Falsch Negative")
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.savefig("falsch_negative_vergleich.png")

    # Plot: Wer ist besser
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Model 1 besser", "Model 2 besser"],
                y=[len(model1_besser), len(model2_besser)])
    plt.title("Modellvergleich – Sampleweise")
    plt.ylabel("Anzahl Samples")
    plt.tight_layout()
    plt.savefig("modellvergleich_samples.png")

    print("\nPlots gespeichert als:")
    print("  - falsch_positive_vergleich.png")
    print("  - falsch_negative_vergleich.png")
    print("  - modellvergleich_samples.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_path', type=str, required=True, help="Pfad zur CSV-Datei von Modell 1")
    parser.add_argument('--model2_path', type=str, required=True, help="Pfad zur CSV-Datei von Modell 2")
    args = parser.parse_args()
    main(args)
