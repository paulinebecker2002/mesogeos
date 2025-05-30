import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch


def get_feature_names(config):
    dynamic = config['features']['dynamic']
    static = config['features']['static']
    lag = config['dataset']['args']['lag']

    feature_names = []
    for t in range(lag):
        for name in dynamic:
            feature_names.append(f"{name}_t-{lag - t}")
        for name in static:
            feature_names.append(f"{name}_t-{lag - t}")

    return feature_names



def plot_shap_summary(shap_values, shap_class, input_tensor, feature_names, checkpoint_path, base_path, model_type, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_file = os.path.join(base_path, f"shap_summary_plot_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")

    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        # For shap.summary_plot we need 2D input: [B, F_total]
        if input_tensor.dim() == 3:
            input_for_plot = input_tensor.view(input_tensor.shape[0], -1)
        else:
            input_for_plot = input_tensor
    else:
        input_for_plot = input_tensor
    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")

    shap.summary_plot(
        shap_values,
        input_for_plot.cpu().numpy(),
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    class_label = "Fire Danger (Class 1)" if shap_class == 1 else "No Fire Danger (Class 0)"
    plt.title(f"SHAP Summary Plot – {class_label}", fontsize=14)
    plt.savefig(save_file, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Summary Plot stored at: {save_file}")


def plot_grouped_feature_importance(shap_values, shap_class, feature_names, checkpoint_path, base_path, model_type, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_file = os.path.join(base_path, f"grouped_shap_plot_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["lstm", "gru", "tft", "transformer"]:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    group_names = [name.split("_t-")[0] for name in feature_names]
    shap_df.columns = group_names

    grouped = shap_df.abs().groupby(axis=1, level=0).mean()
    mean_effect = grouped.mean(axis=0).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mean_effect.plot(kind="bar")
    plt.ylabel("Mean |SHAP value|")
    plt.title("Total influence per feature (aggregated over time)")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Grouped Plot stored at: {save_file}")


def plot_shap_temporal_heatmap(shap_values, shap_class, feature_names, checkpoint_path, base_path, model_type, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_file = os.path.join(base_path, f"shap_temporal_heatmap_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["mlp", "lstm", "gru", "cnn", "transformer", "gtn", "tft"]:
        shap_df = pd.DataFrame(shap_values).T
    elif model_type == "None":
        n_samples, seq_len, n_features = shap_values.shape
        shap_values = shap_values.transpose(2, 1, 0).reshape(n_features * seq_len, n_samples)
        shap_df = pd.DataFrame(shap_values)
    else:
        shap_df = pd.DataFrame(shap_values).T

    shap_df.columns = [f"instance_{i}" for i in range(shap_df.shape[1])]
    shap_df["feature"] = [name.split("_t-")[0] for name in feature_names]
    shap_df["time"] = [int(name.split("_t-")[1]) for name in feature_names]

    shap_df_long = pd.melt(shap_df, id_vars=["feature", "time"], var_name="instance", value_name="shap")
    heatmap_data = shap_df_long.groupby(["feature", "time"])["shap"].apply(lambda x: np.mean(np.abs(x))).unstack()

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cbar_kws={"label": "Mean SHAP value"}, cmap="coolwarm", center=0)
    plt.title("Mean SHAP-Value per Feature und Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP-Heatmap saved at: {save_file}")

def plot_shap_difference_bar(shap_class0, shap_class1, feature_names, checkpoint_path, base_path, model_type, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_file = os.path.join(base_path, f"shap_difference_plot_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Compute mean SHAP difference per feature
    shap_diff = np.mean(shap_class1, axis=0) - np.mean(shap_class0, axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Difference": shap_diff
    })

    # Sort by absolute difference for most impactful features
    df["abs_diff"] = df["SHAP_Difference"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns="abs_diff")

    # Plot
    df_top = df.head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(df_top["Feature"], df_top["SHAP_Difference"], color="steelblue")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("SHAP-Difference (class 1 - class 0)")
    plt.title("SHAP-Difference per Feature: Fire Danger (class 1) vs. No Fire Danger (class 0)")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP Difference Plot saved at: {save_file}")


def plot_shap_difference_aggregated(shap_class0, shap_class1, feature_names, checkpoint_path, base_path, model_type, logger=None):

    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_file = os.path.join(base_path, f"shap_difference_aggregated_plot_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)


    shap_diff = np.mean(shap_class1, axis=0) - np.mean(shap_class0, axis=0)
    base_feature_names = [name.split("_t-")[0] for name in feature_names]

    # DataFrame erstellen
    df = pd.DataFrame({
        "FullFeature": feature_names,
        "BaseFeature": base_feature_names,
        "SHAP_Diff": shap_diff
    })

    # Über Zeitfenster hinweg aggregieren (Mittelwert der Differenzen)
    grouped = df.groupby("BaseFeature")["SHAP_Diff"].mean().reset_index()

    grouped["abs_diff"] = grouped["SHAP_Diff"].abs()
    grouped = grouped.sort_values("abs_diff", ascending=False).drop(columns="abs_diff")

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.barh(grouped["BaseFeature"], grouped["SHAP_Diff"], color="darkorange")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Aggregated SHAP-Difference over time (class 1 - class 0)")
    plt.title("Mean Feature Contribution Difference Between Classes: ")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Aggregated SHAP Difference Plot saved at: {save_file}")