import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from collections import defaultdict
from utils.util import get_model_name
from shap_local.shap_utils import map_sample_ids_to_indices


def plot_bar(ig_values, feature_names, model_id, model_type, base_path, logger=None):
    save_file = os.path.join(base_path, f"ig_bar_plot_{model_id}_{model_type}.png")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)

    mean_ig = np.mean(np.abs(ig_values), axis=0)
    #mean_ig = np.mean(ig_values, axis=0)


    amount_of_feature = 25

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |IG value|": mean_ig
    })

    df = df.sort_values("Mean |IG value|", ascending=True)

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
    model_type = get_model_name(model_type)
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
            new_ig.append(np.mean(ig_values[:, indices], axis=1))
            new_input.append(np.mean(input_tensor[:, indices], axis=1))
            new_names.append(name)
        else:
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


def plot_ig_beeswarm_grouped(ig_values, input_tensor, feature_names, model_id, model_type, base_path):
    model_type = get_model_name(model_type)
    save_file = os.path.join(base_path, f"ig_beeswarm_grouped_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    grouped_ig_df, grouped_features = compute_grouped_ig_over_time(ig_values, feature_names)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    reordered_input_features = [base_feature_names[i] for i in col_idx]

    print("Grouped IG shape:", grouped_ig_df.shape)
    print("Grouped Input shape:", grouped_input_np.shape)
    print("Reordered input features (matching IG):", reordered_input_features)

    assert grouped_ig_df.shape == grouped_input_np.shape, "Mismatch between IG values and input tensor"

    expl = shap.Explanation(
        values=grouped_ig_df.values,
        data=grouped_input_np,
        feature_names=grouped_features
    )

    shap.plots.beeswarm(expl, max_display=len(grouped_features), show=False)
    plt.title(None)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[✓] Grouped IG Beeswarm Plot saved at: {save_file}")


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

        if ig_data.ndim == 3:
            ig_data = ig_data.reshape(ig_data.shape[0], -1)
        if input_data.ndim == 3:
            input_data = input_data.reshape(input_data.shape[0], -1)

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

def plot_ig_beeswarm_by_feature_grouped(ig_files, feature_name, feature_names, model_names, input_files, base_path):
    """
    Vergleich von IG-Werten für ein gruppiertes Basis-Feature (z. B. 'lst_day') über mehrere Modelle hinweg.
    IG-Werte und Inputs werden dabei über die Zeit aggregiert (mean) pro Feature.

    Args:
        ig_files (List[str]): Pfade zu .npy-Dateien mit IG-Werten
        feature_name (str): Basisname des Features (z. B. 'lst_day')
        feature_names (List[str]): Liste aller Feature-Namen (z. B. ['lst_day_t-1', 'lst_day_t-2', ...])
        model_names (List[str]): Namen der Modelle (z. B. ['transformer', 'cnn', ...])
        input_files (List[str]): Pfade zu .npy-Dateien mit Input-Tensoren
        base_path (str): Speicherpfad für den Plot
    """
    ig_values_all_models = []
    input_data_all_models = []
    num_samples = None

    for ig_file, model_name, input_file in zip(ig_files, model_names, input_files):
        ig_data = np.load(ig_file)  # shape: (N, T, F) or (N, T*F)
        input_data = np.load(input_file)

        grouped_ig_df, base_names_ig = compute_grouped_ig_over_time(ig_data, feature_names, sum=True)
        grouped_input_np, base_names_input = compute_grouped_input_over_time(input_data, feature_names)

        if feature_name not in base_names_ig:
            raise ValueError(f"Feature '{feature_name}' not found in IG values for model {model_name}.")
        if feature_name not in base_names_input:
            raise ValueError(f"Feature '{feature_name}' not found in input data for model {model_name}.")

        ig_values_all_models.append(grouped_ig_df[feature_name].values)
        feature_idx = base_names_input.index(feature_name)
        input_data_all_models.append(grouped_input_np[:, feature_idx].reshape(-1, 1))

        if num_samples is None:
            num_samples = grouped_ig_df.shape[0]
        elif grouped_ig_df.shape[0] != num_samples:
            raise ValueError(f"Sample mismatch for model {model_name}")

    ig_values_all_models = np.stack(ig_values_all_models, axis=1)
    input_data_all_models = np.concatenate(input_data_all_models, axis=1)

    expl = shap.Explanation(
        values=ig_values_all_models,
        data=input_data_all_models,
        feature_names=model_names
    )

    save_file = os.path.join(base_path, f"ig_beeswarm_by_feature_grouped_{feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(expl, max_display=len(model_names), show=False)
    plt.title(f"Integrated Gradients Comparison – Feature: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[✓] Grouped IG Beeswarm Plot for Feature '{feature_name}' saved at: {save_file}")

def plot_ig_waterfall(ig_values, input_tensor, feature_names, sample_ids, sample_idx, base_path, model_type):
    """
    Plot IG waterfall plot for a single instance (sample_idx) and save to file.
    """
    save_file = os.path.join(base_path, f"ig_waterfall_plot_{model_type}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    indices = map_sample_ids_to_indices(sample_ids, sample_idx)
    index = indices[0]

    if ig_values.ndim == 3:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

    sample_input = input_tensor[index]
    sample_ig = ig_values[index]
    base_value = ig_values.mean(0).sum()

    expl = shap.Explanation(
        values=sample_ig,
        data=sample_input,
        base_values=base_value,
        feature_names=feature_names
    )

    model_type = get_model_name(model_type)
    shap.plots.waterfall(expl, max_display=25, show=False)
    plt.title(f"IG Waterfall – {model_type} {sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"IG Waterfall Plot saved at: {save_file}")


def plot_ig_waterfall_grouped(ig_values, input_tensor, feature_names, sample_ids, sample_idx, base_path, model_type):
    """
    Grouped IG waterfall plot for a single instance (sample_idx), aggregating over time-lags per feature.
    """
    save_file = os.path.join(base_path, f"ig_waterfall_grouped_{model_type}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    indices = map_sample_ids_to_indices(sample_ids, sample_idx)
    index = indices[0]

    grouped_ig_df, grouped_features = compute_grouped_ig_over_time(ig_values, feature_names, sum=True)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    sample_input = grouped_input_np[index]
    sample_ig = grouped_ig_df.iloc[index].values
    base_value = grouped_ig_df.mean().sum()

    expl = shap.Explanation(
        values=sample_ig,
        data=sample_input,
        base_values=base_value,
        feature_names=grouped_features
    )

    model_type = get_model_name(model_type)
    shap.plots.waterfall(expl, max_display=25, show=False)
    plt.title(f"Grouped IG Waterfall – {model_type} {sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"Grouped IG Waterfall Plot saved at: {save_file}")

    df_export = pd.DataFrame({
        "feature": grouped_features,
        "input_value": sample_input,
        "ig_value": sample_ig
    })
    csv_path = os.path.join(base_path, f"ig_waterfall_grouped_{model_type}_sample{sample_idx}.csv")
    df_export.to_csv(csv_path, index=False)


def compute_grouped_ig_over_time(ig_values, feature_names, sum=False):
    """
    Grouped IG-Values over time for base features.
    """
    if ig_values.ndim == 3:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)  # (N, T*F)

    ig_df = pd.DataFrame(ig_values, columns=feature_names)
    base_names = [name.split("_t-")[0] for name in feature_names]
    ig_df.columns = base_names
    if sum:
        grouped_df = ig_df.groupby(axis=1, level=0).sum()
    else:
        grouped_df = ig_df.groupby(axis=1, level=0).mean()
    return grouped_df, grouped_df.columns.tolist()


def compute_grouped_input_over_time(input_tensor, feature_names):
    """
    Aggregates Input-Tensor over time, to match grouped IG values.

    Args:
        input_tensor (np.ndarray): Input data, typically shape (B, seq_len, F) or (B, T*F)
        feature_names (List[str]): e.g. ["t2m_t-1", "t2m_t-2", ..., "rh_t-1"]

    Returns:
        np.ndarray: Aggregated Input (B, n_base_features)
        List[str]: Base-Feature-Names
    """

    if input_tensor.ndim == 2 and "_t-" in feature_names[0]:
        seq_len = len(set(name.split("_t-")[1] for name in feature_names))
        n_features = len(feature_names) // seq_len
        input_tensor = input_tensor.reshape(-1, seq_len, n_features)

    B, T, F = input_tensor.shape
    input_agg = input_tensor.transpose(0, 2, 1).mean(axis=2)

    base_feature_names = []
    seen = set()
    for name in feature_names:
        base = name.split("_t-")[0]
        if base not in seen:
            base_feature_names.append(base)
            seen.add(base)

    return input_agg, base_feature_names
