import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import itertools
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from utils.util import get_model_name
import torch


def plot_beeswarm(shap_values, shap_class, input_tensor, feature_names, model_id, base_path, model_type, logger=None):
    save_file = os.path.join(base_path, f"shap_beeswarm_plot_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")

    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        if input_tensor.dim() == 3:
            input_for_plot = input_tensor.view(input_tensor.shape[0], -1)
        else:
            input_for_plot = input_tensor
    else:
        input_for_plot = input_tensor
    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")

    #in case we used less samples for SHAP computation than for the input tensor (=whole test input)
    #input_for_plot = input_for_plot[:shap_values.shape[0]]

    expl = shap.Explanation(values=shap_values, data=input_for_plot.cpu().numpy(), feature_names=feature_names)
    shap.plots.beeswarm(expl, max_display=25, show=False)

    class_label = f"Fire Danger (Class {shap_class})"
    plt.title(f"{model_type} - SHAP Beeswarm Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Beeswarm Plot stored at: {save_file}")

def plot_beeswarm_grouped(shap_values, shap_class, input_tensor, feature_names, model_id, base_path, model_type, logger=None):
    save_file = os.path.join(base_path, f"shap_beeswarm_grouped_{model_id}_{model_type}_{shap_class}_sum.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names, sum=True)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)
    print(f"Grouped feature names: {grouped_features}")
    print(f"Base feature names: {base_feature_names}")
    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    print(f"Coloum index of grouped features: {col_idx}")
    reordered_input_features = [base_feature_names[i] for i in col_idx]
    print("Reordered input features (matching SHAP):", reordered_input_features)

    assert grouped_shap_df.shape == grouped_input_np.shape, "Mismatch zwischen SHAP und Input"

    expl = shap.Explanation(
        values=grouped_shap_df.values,
        data=grouped_input_np,
        feature_names=grouped_features
    )

    shap.plots.beeswarm(expl, max_display=len(grouped_features), show=False)
    plt.title(f"{model_type} - SHAP Beeswarm Plot Aggregated over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Grouped SHAP Beeswarm Plot saved at: {save_file}")


def plot_grouped_feature_importance(shap_values, feature_names, base_path, model_type):
    save_file = os.path.join(base_path, f"grouped_absolute_feature_importance_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["lstm", "gru", "tft", "transformer"]:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    grouped_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names, sum=True)
    total_effect = grouped_df.abs().sum(axis=0).sort_values(ascending=False)
    print(f"Features sorted by total importance: {model_type}")
    print(total_effect.tolist())

    #mean_effect = grouped_df.abs().mean(axis=0).sort_values(ascending=False)

    if model_type == "transformer":
        model_type = "Transformer"
    else:
        model_type = model_type.upper()

    plt.figure(figsize=(10, 6))
    total_effect.plot(kind="bar")
    plt.ylabel("Summed |SHAP| over Time")
    plt.title(f"Total influence per feature (aggregated over time) - {model_type}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"SHAP Grouped Plot stored at: {save_file}")


def plot_shap_temporal_heatmap(shap_values, shap_class, feature_names, model_id, base_path, model_type, logger=None):
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

    shap_df_long = pd.melt(shap_df, id_vars=["feature", "time"], var_name="instance", value_name="shap_local")
    heatmap_data = shap_df_long.groupby(["feature", "time"])["shap_local"].apply(lambda x: np.mean(np.abs(x))).unstack()

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

def plot_shap_difference_bar(shap_class0, shap_class1, feature_names, model_id, base_path, model_type, logger=None):
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


def plot_shap_difference_aggregated(shap_class0, shap_class1, feature_names, model_id, base_path, model_type, logger=None):

    save_file = os.path.join(base_path, f"shap_difference_aggregated_plot_{model_id}_{model_type}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    grouped_1, base_features = compute_grouped_shap_over_time(shap_class1, feature_names)
    grouped_0, _ = compute_grouped_shap_over_time(shap_class0, feature_names)
    shap_diff = grouped_1.mean() - grouped_0.mean()

    df = pd.DataFrame({
        "BaseFeature": base_features,
        "SHAP_Diff": shap_diff
    }).copy()

    df["abs_diff"] = df["SHAP_Diff"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns="abs_diff")

    plt.figure(figsize=(10, 6))
    plt.barh(df["BaseFeature"], df["SHAP_Diff"], color="darkorange")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Aggregated SHAP-Difference over time (class 1 - class 0)")
    plt.title("Mean Feature Contribution Difference Between Classes: ")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Aggregated SHAP Difference Plot saved at: {save_file}")

def plot_shap_waterfall(shap_values, shap_class, input_tensor, feature_names, sample_ids, sample_idx,model_id, base_path, model_type, logger=None):
    """
    Plot SHAP waterfall plot for a single instance (sample_idx) and save to file.
    """
    save_file = os.path.join(base_path, f"shap_waterfall_plot_{model_type}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    indices = map_sample_ids_to_indices(sample_ids, sample_idx)
    index = indices[0]

    # Prepare input and SHAP values for one sample
    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.view(input_tensor.shape[0], -1)

    sample_input = input_tensor[index].cpu().numpy()
    sample_shap = shap_values[index]
    base_value = shap_values.mean(0).sum()

    expl = shap.Explanation(
        values=sample_shap,
        data=sample_input,
        base_values=base_value,
        feature_names=feature_names
    )
    model_type = get_model_name(model_type)

    shap.plots.waterfall(expl, max_display=25, show=False)
    plt.title(f"SHAP Waterfall – {model_type} {sample_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP Waterfall Plot saved at: {save_file}")


def plot_shap_waterfall_grouped(shap_values, shap_class, input_tensor, feature_names, sample_ids, sample_idx, model_id, base_path, model_type, logger=None):
    """
    Grouped SHAP waterfall plot for a single instance (sample_idx), aggregating over time-lags per feature.
    """
    save_file = os.path.join(base_path, f"shap_waterfall_grouped_{model_type}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Map sample_idx to index in SHAP/input
    indices = map_sample_ids_to_indices(sample_ids, sample_idx)
    index = indices[0]

    # Compute grouped SHAP values and grouped input (aggregated over time)
    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names, sum=True)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    # Filter grouped input to match grouped SHAP column order
    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    sample_input = grouped_input_np[index]
    sample_shap = grouped_shap_df.iloc[index].values
    base_value = grouped_shap_df.mean().sum()

    expl = shap.Explanation(
        values=sample_shap,
        data=sample_input,
        base_values=base_value,
        feature_names=grouped_features
    )

    model_type = get_model_name(model_type)

    shap.plots.waterfall(expl, max_display=25, show=False)
    plt.title(f"Grouped SHAP Waterfall – {model_type} {sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Grouped SHAP Waterfall Plot saved at: {save_file}")

    df_export = pd.DataFrame({
        "feature": grouped_features,
        "input_value": sample_input,
        "shap_value": sample_shap
    })
    csv_path = os.path.join(base_path, f"shap_waterfall_grouped_{model_type}_sample{sample_idx}.csv")
    df_export.to_csv(csv_path, index=False)




def plot_beeswarm_by_grouped_feature(
        shap_files, input_files, feature_names, feature_to_plot,
        model_names, base_path, only_pos=False, only_neg=False
):
    """
    Compares a grouped feature (e.g., 'lst_day') across multiple models.

    Plots the SHAP values of this feature for each model in a vertically stacked beeswarm plot.

    Args:
        shap_files (List[str]): List of .npy or .npz SHAP file paths (one per model).
        input_files (List[str]): List of input file paths (e.g., .npy files, one per model).
        feature_names (List[str]): Original time-dependent feature names such as 'lst_day_t-0', 'lst_day_t-1', etc.
        feature_to_plot (str): Feature to group and plot, e.g., 'lst_day'.
        model_names (List[str]): Model names, in the same order as the file lists.
        base_path (str): Path where the plot will be saved.
        only_neg (bool): If True, only plot negative samples.
        only_pos (bool): If True, only plot positive samples.
    """
    if only_pos:
        samples = "positive"
    elif only_neg:
        samples = "negative"
    else:
        samples = "all"

    all_shap = []
    all_input = []

    for shap_file, input_file, model_name in zip(shap_files, input_files, model_names):
        shap_data = np.load(shap_file)
        shap_values = shap_data['class_1']
        grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names, sum=True)

        if feature_to_plot not in grouped_features:
            raise ValueError(f"Feature '{feature_to_plot}' not found in SHAP for model {model_name}")

        shap_vals_feat = grouped_shap_df[feature_to_plot].values
        all_shap.append(shap_vals_feat)

        input_tensor = torch.tensor(np.load(input_file))
        grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

        if feature_to_plot not in base_feature_names:
            raise ValueError(f"Feature '{feature_to_plot}' not found in Input for model {model_name}")

        feat_idx = base_feature_names.index(feature_to_plot)
        feat_input_vals = grouped_input_np[:, feat_idx].reshape(-1, 1)
        all_input.append(feat_input_vals)
        print(f"[{model_name}] → Samples used: {feat_input_vals.shape[0]}")


    all_shap = np.array(all_shap).T  # (n_samples, n_models)
    all_input = np.concatenate(all_input, axis=1)  # (n_samples, n_models)

    expl = shap.Explanation(
        values=all_shap,
        data=all_input,
        feature_names=model_names
    )


    save_file = os.path.join(base_path, "grouped", f"shap_by_grouped_feature_{feature_to_plot}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(expl, max_display=len(model_names), show=False)

    plt.title(f"SHAP Value Comparison for Grouped Feature: {feature_to_plot}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[✓] SHAP Comparison Plot for Feature '{feature_to_plot}' saved at:\n{save_file}")

def plot_beeswarm_by_feature(
        shap_files, input_files, feature_names, full_feature_name,
        model_names, base_path, only_pos=False, only_neg=False
):
    """
    Compares a specific (non-grouped) feature like 'lst_day_t-1' across multiple models.

    Plots the SHAP values for this feature for each model side by side in a beeswarm plot.

    Args:
        shap_files (List[str]): Paths to SHAP value files (.npz, one per model).
        input_files (List[str]): Paths to input data files (.npy, one per model).
        feature_names (List[str]): List of all time-dependent feature names, e.g. 'lst_day_t-1', 't2m_t-2', etc.
        full_feature_name (str): The exact name of the feature to be plotted, e.g. 'lst_day_t-1'.
        model_names (List[str]): Names of the models, in the same order as the files.
        base_path (str): Directory path where the plot should be saved.
    """

    if only_pos:
        samples = "positive"
    elif only_neg:
        samples = "negative"
    else:
        samples = "all"

    all_shap = []
    all_input = []

    for shap_file, input_file, model_name in zip(shap_files, input_files, model_names):

        shap_data = np.load(shap_file)
        shap_values = shap_data["class_1"]  # Shape: (B, num_features)

        if full_feature_name not in feature_names:
            raise ValueError(f"Feature '{full_feature_name}' not in feature_names.")

        feature_idx = feature_names.index(full_feature_name)
        shap_vals_feat = shap_values[:, feature_idx]
        all_shap.append(shap_vals_feat)

        input_data = np.load(input_file)
        input_tensor = torch.tensor(input_data)

        if input_tensor.dim() == 3:  # Shape: (B, T, F)
            B, T, F = input_tensor.shape
            input_tensor = input_tensor.view(B, -1)  # -> (B, T*F)

        if input_tensor.shape[1] != len(feature_names):
            raise ValueError(f"[{model_name}] Input shape {input_tensor.shape} does not match feature_names length {len(feature_names)}")

        input_vals_feat = input_tensor[:, feature_idx].cpu().numpy()
        all_input.append(input_vals_feat.reshape(-1, 1))  # shape (B, 1)

    all_shap = np.array(all_shap).T  # shape (B, n_models)
    all_input = np.concatenate(all_input, axis=1)  # shape (B, n_models)

    expl = shap.Explanation(
        values=all_shap,
        data=all_input,
        feature_names=model_names
    )

    save_file = os.path.join(base_path, "t-1", f"shap_by_raw_feature_{full_feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(expl, max_display=len(model_names), show=False)
    plt.title(f"SHAP Comparison for Feature: {full_feature_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[✓] Raw SHAP Comparison Plot for '{full_feature_name}' saved at:\n{save_file}")


def map_sample_ids_to_indices(sample_ids, selected_ids):
    """
    Map a list of selected sample IDs from the original positives and negativs csv files
    to their corresponding indices in the data array after the whole dataset is created.

    Args:
        sample_ids (array-like): Full list of sample IDs in SHAP data (e.g. from npz).
        selected_ids (list of int): Sample IDs for which indices should be returned.

    Returns:
        List[int]: Indices corresponding to the selected sample IDs.
    """
    if isinstance(selected_ids, int):
        selected_ids = [selected_ids]

    id_to_index = {sid: i for i, sid in enumerate(sample_ids)}
    indices = []

    for sid in selected_ids:
        if sid in id_to_index:
            indices.append(id_to_index[sid])
        else:
            raise ValueError(f"Sample ID {sid} not found in sample_ids.")
    return indices


def compute_grouped_shap_over_time(shap_values, feature_names, sum=False):
    """
    Grouped SHAP-Values over time for base features
    Returns:
        grouped_df: pd.DataFrame with (n_samples, n_base_features)
        base_features: List with base feature names
        sum: bool, if True, sums the SHAP values over time instead of averaging
    """

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    base_names = [name.split("_t-")[0] for name in feature_names]
    shap_df.columns = base_names
    if sum:
        grouped_df = shap_df.groupby(axis=1, level=0).sum()
    else:
        grouped_df = shap_df.groupby(axis=1, level=0).mean()
    return grouped_df, grouped_df.columns.tolist()


def compute_grouped_input_over_time(input_tensor, feature_names):
    """
    Aggregates Input-Tensor over time, to match grouped shap values.

    Args:
        input_tensor (torch.Tensor): Input-Daten, typically Shape (B, seq_len, F)
        feature_names (List[str]): e.g. ["t2m_t-1", "t2m_t-2", ..., "rh_t-1"]

    Returns:
        np.ndarray: Aggregated Input (B, n_base_features)
        List[str]: Base-Feature-Names
    """

    if input_tensor.dim() == 2 and "_t-" in feature_names[0]:
        seq_len = len(set(name.split("_t-")[1] for name in feature_names))
        n_features = len(feature_names) // seq_len
        input_tensor = input_tensor.view(-1, seq_len, n_features)

    input_np = input_tensor.cpu().numpy()  # (B, T, F)
    B, T, F = input_np.shape
    input_agg = input_np.transpose(0, 2, 1).mean(axis=2)
    base_feature_names = []
    seen = set()
    for name in feature_names:
        base = name.split("_t-")[0]
        if base not in seen:
            base_feature_names.append(base)
            seen.add(base)
    print(f"Aggregated Input shape: {input_agg.shape}, Base Features: {base_feature_names}")

    return input_agg, base_feature_names

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, safe for constant vectors / NaNs."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def plot_grouped_interaction_heatmap(
    shap_values: np.ndarray,
    input_tensor: torch.Tensor,
    feature_names: list,
    base_path: str,
    model_id: str,
    model_type: str,
    logger=None,
    sum_over_time: bool = True,
):
    """
    Creates a global 'interaction strength' heatmap for the 24 base features
    using aggregated SHAP over time and aggregated inputs over time.

    NOTE: This is a proxy (not TreeExplainer interaction tensor).
    Off-diagonal entry(i,j) ~ avg(|corr(shap_i, x_j)|, |corr(shap_j, x_i)|) * 0.5
    Diagonal is mean(|shap_i|) to represent main effect strength.
    """
    save_dir = os.path.join(base_path, "interactions_grouped")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"shap_interaction_heatmap_{model_id}_{model_type}.png")

    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(
        shap_values, feature_names, sum=sum_over_time
    )
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    # Align input columns to grouped SHAP columns
    col_idx = [base_feature_names.index(f) for f in grouped_features]
    X = grouped_input_np[:, col_idx]  # (N, 24)
    S = grouped_shap_df.values        # (N, 24)
    feats = grouped_features

    n = len(feats)
    mat = np.zeros((n, n), dtype=float)

    # diagonal: main effect magnitude
    for i in range(n):
        mat[i, i] = np.nanmean(np.abs(S[:, i]))

    # off-diagonal: symmetric proxy for interaction strength
    for i in range(n):
        for j in range(i + 1, n):
            c1 = _safe_corr(S[:, i], X[:, j])
            c2 = _safe_corr(S[:, j], X[:, i])
            val = np.nanmean([abs(c1), abs(c2)])
            mat[i, j] = val
            mat[j, i] = val

    df = pd.DataFrame(mat, index=feats, columns=feats)

    # mimic Kaggle convention: double off-diagonal
    doubled = df.values.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                doubled[i, j] *= 2.0
    df2 = pd.DataFrame(doubled, index=feats, columns=feats)

    plt.figure(figsize=(14, 10))
    sns.heatmap(df2.round(4), cmap="coolwarm", annot=True, fmt=".4g", cbar=True)
    plt.title(f"Grouped SHAP interaction proxy heatmap (30-day aggregated) – {model_type}", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Grouped interaction heatmap saved at: {save_file}")

def load_tree_shap_interactions(shap_path: str, model_id: str, model_type: str, shap_class: int = 1):
    fp = os.path.join(shap_path, f"shap_interaction_{model_id}_{model_type}.npz")
    data = np.load(fp, allow_pickle=True)
    inter = data[f"class_{shap_class}"]  # shape (N, F, F)
    feature_names = data["feature_names"].tolist()
    return inter, feature_names


def plot_tree_interaction_heatmap_mean(
    shap_interaction: np.ndarray,   # (N, F, F)
    feature_names: list,
    base_path: str,
    model_id: str,
    model_type: str,
    logger=None,
):
    save_dir = os.path.join(base_path, "interactions_tree")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"tree_interaction_heatmap_{model_id}_{model_type}.png")

    mean_abs = np.abs(shap_interaction).mean(axis=0)  # (F,F)
    df = pd.DataFrame(mean_abs, index=feature_names, columns=feature_names)

    # double off diagonal (Kaggle convention)
    vals = df.values
    diag = np.diag(vals).copy()
    vals2 = vals * 2.0
    np.fill_diagonal(vals2, diag)
    df2 = pd.DataFrame(vals2, index=feature_names, columns=feature_names)

    plt.figure(figsize=(16, 12))
    sns.heatmap(df2.round(4), cmap="coolwarm", annot=False, cbar=True)
    plt.title(f"Tree SHAP interaction values (mean |.|) – {model_type}", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Tree interaction heatmap saved at: {save_file}")

def _compute_grouped_interaction_matrix_proxy(
    shap_values: np.ndarray,
    input_tensor: torch.Tensor,
    feature_names: list,
    sum_over_time: bool = True,
):
    """
    Returns:
        df2: DataFrame (n_base_features x n_base_features) with proxy strengths
             (off-diagonal doubled like the Kaggle convention).
        X:   DataFrame (N x n_base_features) aggregated inputs
        S:   np.ndarray (N x n_base_features) aggregated SHAP
        feats: list of base feature names (column order for X and S)
    """
    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(
        shap_values, feature_names, sum=sum_over_time
    )
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    col_idx = [base_feature_names.index(f) for f in grouped_features]
    X = pd.DataFrame(grouped_input_np[:, col_idx], columns=grouped_features)
    S = grouped_shap_df.values
    feats = grouped_features

    n = len(feats)
    mat = np.zeros((n, n), dtype=float)

    # diagonal: main effect magnitude
    for i in range(n):
        mat[i, i] = np.nanmean(np.abs(S[:, i]))

    # off-diagonal: symmetric proxy for interaction strength
    for i in range(n):
        for j in range(i + 1, n):
            c1 = _safe_corr(S[:, i], X.iloc[:, j].values)
            c2 = _safe_corr(S[:, j], X.iloc[:, i].values)
            val = np.nanmean([abs(c1), abs(c2)])  # symmetric
            mat[i, j] = val
            mat[j, i] = val

    df = pd.DataFrame(mat, index=feats, columns=feats)

    # Kaggle-style: double off-diagonal
    doubled = df.values.copy()
    for i in range(n):
        for j in range(n):
            if i != j:
                doubled[i, j] *= 2.0
    df2 = pd.DataFrame(doubled, index=feats, columns=feats)

    return df2, X, S, feats


def plot_topk_grouped_dependence_interactions(
    shap_values: np.ndarray,
    input_tensor: torch.Tensor,
    feature_names: list,
    base_path: str,
    model_id: str,
    model_type: str,
    logger=None,
    sum_over_time: bool = True,
    top_k: int = 20,
):
    """
    1) Compute grouped interaction proxy matrix (24x24).
    2) Rank strongest off-diagonal pairs.
    3) Save dependence plots only for the top_k pairs:
       x = f1, y = SHAP(f1), color = f2.
    Also saves a CSV ranking for paper reporting.
    """
    save_dir = os.path.join(base_path, "interactions_grouped", f"top{top_k}_pairs")
    os.makedirs(save_dir, exist_ok=True)

    df2, X, S, feats = _compute_grouped_interaction_matrix_proxy(
        shap_values=shap_values,
        input_tensor=input_tensor,
        feature_names=feature_names,
        sum_over_time=sum_over_time,
    )

    # rank pairs i<j by off-diagonal score
    pairs = []
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(df2.iloc[i, j])
            if np.isfinite(score):
                pairs.append((feats[i], feats[j], score))

    pairs_sorted = sorted(pairs, key=lambda t: t[2], reverse=True)[:top_k]

    # save ranking csv
    rank_df = pd.DataFrame(pairs_sorted, columns=["feature_1", "feature_2", "interaction_score"])
    rank_csv = os.path.join(save_dir, f"top{top_k}_interaction_pairs_{model_id}_{model_type}.csv")
    rank_df.to_csv(rank_csv, index=False)

    # plot dependence for each top pair
    for f1, f2, score in pairs_sorted:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            ind=f1,
            shap_values=S,
            features=X,
            interaction_index=f2,
            show=False
        )
        plt.title(f"{model_type} – {f1} (colored by {f2}) | score={score:.4g}", fontsize=11)
        plt.tight_layout()
        out = os.path.join(save_dir, f"dep_{model_type}_{model_id}_{f1}__by__{f2}.png")
        plt.savefig(out, dpi=300)
        plt.close()

    if logger:
        logger.info(f"Saved top-{top_k} dependence interactions in: {save_dir}")
        logger.info(f"Top-{top_k} ranking CSV saved at: {rank_csv}")


def plot_topk_tree_interaction_dependence(
    shap_interaction: np.ndarray,  # (N,F,F) for class_1
    X: np.ndarray,                 # (N,F) raw input (must match features)
    feature_names: list,
    base_path: str,
    model_id: str,
    model_type: str,
    logger=None,
    top_k: int = 20,
):
    """
    Rank pairs by mean_abs interaction (off-diagonal doubled convention),
    then produce dependence plots for those top_k pairs.

    Saves:
      - CSV ranking
      - dependence plots using shap.dependence_plot((f1,f2), shap_interaction, X_df)
    """
    save_dir = os.path.join(base_path, "interactions_tree", f"top{top_k}_pairs")
    os.makedirs(save_dir, exist_ok=True)

    X_df = pd.DataFrame(X, columns=feature_names)

    mean_abs = np.abs(shap_interaction).mean(axis=0)  # (F,F)

    # build pair scores i<j, using Kaggle doubling off-diagonal
    pairs = []
    F = len(feature_names)
    for i in range(F):
        for j in range(i + 1, F):
            score = float(mean_abs[i, j] * 2.0)
            if np.isfinite(score):
                pairs.append((feature_names[i], feature_names[j], score))

    pairs_sorted = sorted(pairs, key=lambda t: t[2], reverse=True)[:top_k]

    # save ranking
    rank_df = pd.DataFrame(pairs_sorted, columns=["feature_1", "feature_2", "interaction_score"])
    rank_csv = os.path.join(save_dir, f"top{top_k}_tree_interaction_pairs_{model_id}_{model_type}.csv")
    rank_df.to_csv(rank_csv, index=False)

    # dependence plots
    for f1, f2, score in pairs_sorted:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            (f1, f2),
            shap_interaction,   # must be (N,F,F)
            X_df,
            show=False
        )
        plt.title(f"{model_type} – interaction ({f1},{f2}) | score={score:.4g}", fontsize=11)
        plt.tight_layout()
        out = os.path.join(save_dir, f"dep_inter_{model_type}_{model_id}_{f1}__{f2}.png")
        plt.savefig(out, dpi=300)
        plt.close()

    if logger:
        logger.info(f"Saved top-{top_k} TRUE tree interaction plots at: {save_dir}")
        logger.info(f"Ranking CSV saved at: {rank_csv}")

def compute_grouped_physical_consistency_score_0_treshold(
        shap_values: np.ndarray,
        input_tensor: torch.Tensor,
        feature_names: list,
        physical_signs: dict,
        save_path: str,
        model_type: str,
        threshold: float = 0.0
):
    import pandas as pd
    import os
    import numpy as np


    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)


    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    assert grouped_shap_df.shape == grouped_input_np.shape, "Mismatch between grouped SHAP and Input"

    df_shap = pd.DataFrame(grouped_shap_df.values, columns=grouped_features)
    df_input = pd.DataFrame(grouped_input_np, columns=grouped_features)

    results = []
    for feat in grouped_features:
        base_feat = feat
        print(f"base features names: {base_feat}")
        if base_feat not in physical_signs:
            continue

        sign = physical_signs[base_feat]
        shap_vals = df_shap[feat]
        inputs = df_input[feat]

        mask_high_input = inputs > threshold
        mask_low_input = inputs < -threshold

        if sign == "+":
            mask_consistent = (mask_high_input & (shap_vals > 0)) | (mask_low_input & (shap_vals < 0))
        elif sign == "-":
            mask_consistent = (mask_high_input & (shap_vals < 0)) | (mask_low_input & (shap_vals > 0))
        else:
            continue

        n_consistent = mask_consistent.sum()
        n_considered = (mask_high_input | mask_low_input).sum()
        score = n_consistent / n_considered if n_considered > 0 else np.nan

        results.append({
            "feature": feat,
            "physical_sign": sign,
            "n_considered": n_considered,
            "n_consistent": n_consistent,
            "consistency_score": score
        })

    df_results = pd.DataFrame(results)
    save_file = os.path.join(save_path, f"{model_type}_grouped_physical_consistency_0.0.csv")
    df_results.to_csv(save_file, index=False)
    print(f"feature names: {grouped_features}")
    print(f"feature names: {feature_names}")
    print(f"Grouped Physical consistency scores saved to: {save_file}")

def compute_grouped_physical_consistency_score(
        shap_values: np.ndarray,
        input_tensor: torch.Tensor,
        feature_names: list,
        physical_signs: dict,
        save_path: str,
        model_type: str,
        threshold: float = 0.1
):

    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)

    col_idx = [base_feature_names.index(f) for f in grouped_features]
    grouped_input_np = grouped_input_np[:, col_idx]
    assert grouped_shap_df.shape == grouped_input_np.shape, "Mismatch between grouped SHAP and Input"

    df_shap = pd.DataFrame(grouped_shap_df.values, columns=grouped_features)
    df_input = pd.DataFrame(grouped_input_np, columns=grouped_features)

    results = []
    for feat in grouped_features:
        base_feat = feat
        if base_feat not in physical_signs:
            continue

        sign = physical_signs[base_feat]
        shap_vals = df_shap[feat]
        inputs = df_input[feat]

        lower_q = np.nanpercentile(inputs, 33)
        upper_q = np.nanpercentile(inputs, 66)

        mask_low_input = inputs < lower_q
        mask_high_input = inputs > upper_q

        if sign == "+":
            mask_consistent = (mask_high_input & (shap_vals > 0)) | (mask_low_input & (shap_vals < 0))
        elif sign == "-":
            mask_consistent = (mask_high_input & (shap_vals < 0)) | (mask_low_input & (shap_vals > 0))
        else:
            continue

        n_consistent = mask_consistent.sum()
        n_considered = (mask_high_input | mask_low_input).sum()
        score = n_consistent / n_considered if n_considered > 0 else np.nan

        results.append({
            "feature": feat,
            "physical_sign": sign,
            "lower_q": round(lower_q, 3),
            "upper_q": round(upper_q, 3),
            "n_considered": n_considered,
            "n_consistent": n_consistent,
            "consistency_score": score
        })

    df_results = pd.DataFrame(results)
    save_file = os.path.join(save_path, f"{model_type}_grouped_physical_consistency_quantiles.csv")
    df_results.to_csv(save_file, index=False)
    print(f"Grouped Physical consistency (quantile-based) saved to: {save_file}")
    return df_results, save_file

