import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from collections import defaultdict


def plot_bar(ig_values, feature_names, model_id, model_type, base_path, logger=None):
    save_file = os.path.join(base_path, f"ig_bar_plot_{model_id}_{model_type}.png")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)

    # Compute mean absolute IG for each feature
    mean_ig = np.mean(np.abs(ig_values), axis=0)
    #mean_ig = np.mean(ig_values, axis=0)


    amount_of_feature = 25

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |IG value|": mean_ig
    })

    # Sort by absolute value for most impactful features
    df = df.sort_values("Mean |IG value|", ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    df.tail(amount_of_feature).plot(kind='barh', x='Feature', y='Mean |IG value|', color='skyblue', legend=False)
    plt.xlabel("Mean |IG value|")
    plt.title(f"Top {amount_of_feature} Features with Highest Integrated Gradients - Model: {model_type}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"IG Bar Plot saved at: {save_file}")


def plot_temporal_heatmap(ig_values, feature_names, model_id, model_type, base_path, logger=None, scaled=False):
    if scaled:
        save_file = os.path.join(base_path, f"ig_squared_temporal_heatmap_{model_id}_{model_type}.png")
    else:
       save_file = os.path.join(base_path, f"ig_temporal_heatmap_{model_id}_{model_type}.png")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Feature und Zeit extrahieren
    base_names = [name.split("_t-")[0] for name in feature_names]
    time_steps = [int(name.split("_t-")[1]) for name in feature_names]

    df = pd.DataFrame(ig_values.reshape(ig_values.shape[0], -1), columns=feature_names)
    df.columns = pd.MultiIndex.from_arrays([base_names, time_steps], names=("Feature", "Timestep"))

    df_long = df.stack(level=[0, 1]).reset_index()
    df_long.columns = ["Sample", "Feature", "Timestep", "IG"]
    df_long["|IG|"] = np.abs(df_long["IG"])

    heatmap_data = df_long.groupby(["Feature", "Timestep"])["|IG|"].mean().unstack()

    if scaled:
        heatmap_data_transformed = np.sqrt(heatmap_data)
    else:
        heatmap_data_transformed = heatmap_data

    plt.figure(figsize=(14, 10))
    vmax = np.quantile(heatmap_data.values.flatten(), 0.95)

    if scaled:
        sns.heatmap(heatmap_data_transformed, cmap="Reds", cbar_kws={"label": "√(Mean |IG|)"})
        plt.title("Mean IG Value per Feature and Timestep (Square Root Scaled)")
    else:
        sns.heatmap(heatmap_data, cmap="coolwarm", vmin=0, vmax=vmax, cbar_kws={"label": "Mean |IG|"}, center=0)
        plt.title("Mean IG Value per Feature and Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"IG Temporal Heatmap saved at: {save_file}")


def plot_ig_beeswarm(ig_values, input_tensor, feature_names, model_id, model_type, base_path, logger=None):
    save_file = os.path.join(base_path, f"ig_beeswarm_plot_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if ig_values.ndim == 3:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

    # Optional abs verwenden
    # ig_values = np.abs(ig_values)

    static_base_names = [
        "dem", "roads_distance", "slope",
        "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement",
        "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies",
        "lc_wetland", "population"
    ]

    base_names = [
        name.split("_t-")[0] if "_t-" in name else name
        for name in feature_names
    ]
    name_to_indices = defaultdict(list)
    for i, base in enumerate(base_names):
        name_to_indices[base].append(i)

    new_ig = []
    new_input = []
    new_names = []

    for name, indices in name_to_indices.items():
        if name in static_base_names:
            # Mittelwert über Zeitachse für statische Features
            new_ig.append(np.mean(ig_values[:, indices], axis=1))
            new_input.append(np.mean(input_tensor[:, indices], axis=1))
            new_names.append(name)
        else:
            # Dynamische Features einzeln behalten
            for i in indices:
                new_ig.append(ig_values[:, i])
                new_input.append(input_tensor[:, i])
                new_names.append(feature_names[i])

    ig_values = np.stack(new_ig, axis=1)
    input_tensor = np.stack(new_input, axis=1)
    feature_names = new_names

    expl = shap.Explanation(values=ig_values,
                            data=input_tensor,
                            feature_names=feature_names)

    shap.plots.beeswarm(expl, max_display=25, show=False)
    plt.title(f"IG Beeswarm Plot – Model: {model_type}")
    plt.xlim(-4, 4)
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"IG Beeswarm Plot saved at: {save_file}")

def plot_ig_beeswarm_only_once_each_feature(ig_values, input_tensor, feature_names, model_id, model_type, base_path, logger=None):
    save_file = os.path.join(base_path, f"ig_beeswarm_plot_only_once_each_feature_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if ig_values.ndim == 3:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

    base_names_seen = set()
    selected_indices = []
    selected_feature_names = []

    for i, name in enumerate(feature_names):
        base_name = name.split("_t-")[0] if "_t-" in name else name
        if base_name not in base_names_seen:
            base_names_seen.add(base_name)
            selected_indices.append(i)
            selected_feature_names.append(name)

    ig_filtered = ig_values[:, selected_indices]
    input_filtered = input_tensor[:, selected_indices]

    expl = shap.Explanation(values=ig_filtered,
                            data=input_filtered,
                            feature_names=selected_feature_names)

    shap.plots.beeswarm(expl, max_display=25, show=False)
    plt.title(f"IG Beeswarm Plot – Model: {model_type}")
    plt.xlim(-4, 4)
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"IG Beeswarm Plot saved at: {save_file}")

def plot_ig_beeswarm_by_feature(ig_files, feature_name, feature_names, model_names, input_files, base_path):
    ig_values_all_models = []
    input_data_all_models = []
    num_samples = None

    for ig_file, model_name, input_file in zip(ig_files, model_names, input_files):
        ig_data = np.load(ig_file)  # shape: (N, T, F) or (N, F)
        input_data = np.load(input_file)

        # Flatten if necessary
        if ig_data.ndim == 3:
            ig_data = ig_data.reshape(ig_data.shape[0], -1)
        if input_data.ndim == 3:
            input_data = input_data.reshape(input_data.shape[0], -1)

        # Index des gesuchten Features (z. B. lst_day_t-1)
        feature_index = [i for i, name in enumerate(feature_names) if name == feature_name]
        if not feature_index:
            raise ValueError(f"Feature {feature_name} not found in IG data for model {model_name}.")
        index = feature_index[0]

        ig_values_all_models.append(ig_data[:, index])
        input_data_all_models.append(input_data[:, index].reshape(-1, 1))

        if num_samples is None:
            num_samples = ig_data.shape[0]
        elif ig_data.shape[0] != num_samples:
            raise ValueError(f"Sample mismatch for model {model_name}")

    # Combine into shape [num_samples, num_models]
    ig_values_all_models = np.stack(ig_values_all_models, axis=1)
    input_data_all_models = np.concatenate(input_data_all_models, axis=1)

    # SHAP-kompatibles Explanation-Objekt
    expl = shap.Explanation(
        values=ig_values_all_models,
        data=input_data_all_models,
        feature_names=model_names
    )

    save_file = os.path.join(base_path, f"ig_beeswarm_by_feature_{feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(expl, max_display=len(model_names), show=False)
    plt.title(f"Integrated Gradients Comparison – Feature: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"IG Comparison Beeswarm Plot for Feature '{feature_name}' saved at: {save_file}")