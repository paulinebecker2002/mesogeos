import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
    plt.title(f"SHAP Summary Plot – {class_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Beeswarm Plot stored at: {save_file}")

def plot_beeswarm_grouped(shap_values, shap_class, input_tensor, feature_names, model_id, base_path, model_type, logger=None):
    save_file = os.path.join(base_path, f"shap_beeswarm_grouped_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    grouped_shap_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names)
    grouped_input_np, base_feature_names = compute_grouped_input_over_time(input_tensor, feature_names)
    grouped_input_np = normalize_input_per_feature(grouped_input_np)

    assert grouped_shap_df.shape == grouped_input_np.shape, "Mismatch zwischen SHAP und Input"

    expl = shap.Explanation(
        values=grouped_shap_df.values,
        data=grouped_input_np,
        feature_names=grouped_features
    )

    shap.plots.beeswarm(expl, max_display=len(grouped_features), show=False)
    plt.title(f"Grouped SHAP Beeswarm (Aggregated over Time) Class {shap_class}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"Grouped SHAP Beeswarm Plot saved at: {save_file}")


def plot_grouped_feature_importance(shap_values, shap_class, feature_names, model_id, base_path, model_type, logger=None):
    save_file = os.path.join(base_path, f"grouped_shap_plot_{model_id}_{model_type}_{shap_class}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    if model_type in ["lstm", "gru", "tft", "transformer"]:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    grouped_df, grouped_features = compute_grouped_shap_over_time(shap_values, feature_names)
    mean_effect = grouped_df.abs().mean(axis=0).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mean_effect.plot(kind="bar")
    plt.ylabel("Mean |SHAP value|")
    plt.title("Total influence per feature (aggregated over time)")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Grouped Plot stored at: {save_file}")


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

def plot_shap_waterfall(shap_values, shap_class, input_tensor, feature_names, sample_ids, sample_idx,
                        model_id, base_path, model_type, logger=None):
    """
    Plot SHAP waterfall plot for a single instance (sample_idx) and save to file.
    """
    save_file = os.path.join(base_path, f"shap_waterfall_plot_{model_id}_{model_type}_class{shap_class}_sample{sample_idx}.png")
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

    shap.plots.waterfall(expl, max_display=25, show=False)
    plt.title(f"SHAP Waterfall – Sample {sample_idx} (Class {shap_class})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP Waterfall Plot saved at: {save_file}")


def plot_shap_comparison_by_feature(shap_files, feature_name, feature_names, model_names, base_path, logger=None):
    """
    old version of comparison plot, self-build but not really appealing
    """
    shap_values_all_models = []
    num_samples = None

    for shap_file, model_name in zip(shap_files, model_names):
        shap_data = np.load(shap_file)

        feature_index = [i for i, name in enumerate(feature_names) if name == feature_name]
        if not feature_index:
            raise ValueError(f"Feature {feature_name} not found in SHAP data for model {model_name}.")

        shap_values = shap_data['class_1'][:, feature_index[0]]
        shap_values_all_models.append(shap_values)

        if num_samples is None:
            num_samples = len(shap_values)
        elif len(shap_values) != num_samples:
            raise ValueError(f"Mismatch in number of samples between models. Expected {num_samples}, but got {len(shap_values)} for model {model_name}.")

    shap_values_all_models = np.array(shap_values_all_models).T

    # Define the custom red_blue color map
    red_blue = mcolors.LinearSegmentedColormap.from_list(
        "red_blue", ["blue", "white", "red"], N=256
    )

    norm = plt.Normalize(vmin=np.min(shap_values_all_models), vmax=np.max(shap_values_all_models))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    for i, model_name in enumerate(model_names):
        shap_values_model = shap_values_all_models[:, i]
        colors = red_blue(norm(shap_values_model))

        ax.scatter(shap_values_model, [i + 0.1 * i] * len(shap_values_model), label=model_name, alpha=0.7, s=20, c=colors)

    ax.set_xlabel("SHAP Values")
    ax.set_ylabel("Models")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title(f"SHAP Value Comparison for Feature: {feature_name}")

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=red_blue), ax=ax)
    cbar.set_label('SHAP Value (Blue: Low, Red: High)')

    save_file = os.path.join(base_path, f"shap_comparison_by_feature_{feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP Comparison Plot for Feature {feature_name} saved at: {save_file}")


def plot_beeswarm_by_feature(shap_files, feature_name, feature_names, model_names, input_files, base_path):
    """
    comparison of all models for one specific feature - new version with SHAP Beeswarn plot
    """
    shap_values_all_models = []
    input_data_all_models = []
    num_samples = None

    # Loop through the models
    for shap_file, model_name, input_file in zip(shap_files, model_names, input_files):
        shap_data = np.load(shap_file)

        # Find the index of the specified feature in the feature names
        feature_index = [i for i, name in enumerate(feature_names) if name == feature_name]
        if not feature_index:
            raise ValueError(f"Feature {feature_name} not found in SHAP data for model {model_name}.")

        # Extract SHAP values for the specified feature
        shap_values = shap_data['class_1'][:, feature_index[0]]
        shap_values_all_models.append(shap_values)

        # Load input data for the current model
        input_data = np.load(input_file)

        if model_name in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
            # If it's a 3D tensor, flatten it into 2D
            if input_data.ndim == 3:
                input_data_flat = input_data.reshape(input_data.shape[0], -1)
            else:
                input_data_flat = input_data
        else:
            input_data_flat = input_data  # No reshaping needed for other models

        feature_data = input_data_flat[:, feature_index[0]]  # Extract only the specified feature
        feature_data = feature_data.reshape(-1, 1)

        input_data_all_models.append(feature_data)

        #print(f"Shape of input data for {model_name}: {input_data.shape}")
        #print(f"Shape of extracted feature data for {model_name}: {feature_data.shape}")
        #print(f"Shape of SHAP values for {model_name}: {shap_values.shape}")


        if num_samples is None:
            num_samples = len(shap_values)
        elif len(shap_values) != num_samples:
            raise ValueError(f"Mismatch in number of samples between models. Expected {num_samples}, but got {len(shap_values)} for model {model_name}.")

    input_data_all_models = np.concatenate(input_data_all_models, axis=-1)
    shap_values_all_models = np.array(shap_values_all_models).T

    expl = shap.Explanation(values=shap_values_all_models, data=input_data_all_models, feature_names=model_names)

    save_file = os.path.join(base_path, f"beeswarm_by_feature_{feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(expl, max_display=8, show=False, order=np.argsort(model_names))


    plt.title(f"SHAP Value Comparison for Feature: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"SHAP Comparison Plot for Feature {feature_name} saved at: {save_file}")

def plot_beeswarm_by_feature1(shap_files, feature_name, feature_names, model_names, input_files, base_path):
    """
    Vergleicht SHAP-Werte eines bestimmten Features über mehrere Modelle hinweg mithilfe eines Beeswarm-Plots.

    Args:
        shap_files (list): Liste von .npz-Dateien mit SHAP-Werten für jedes Modell
        feature_name (str): Name des zu analysierenden Features (z. B. "lst_day_t-1")
        feature_names (list): Liste aller Feature-Namen (inkl. Zeit)
        model_names (list): Liste der Modellnamen
        input_files (list): Liste der .npy-Dateien mit Input-Daten
        base_path (str): Speicherort für den Plot
    """
    shap_values_all_models = []
    input_data_all_models = []
    num_samples = None

    # extrahiere Basis-Feature und Zeitstempel
    if "_t-" not in feature_name:
        raise ValueError(f"Feature {feature_name} enthält keinen Zeitindex (_t-).")

    base_feature = feature_name.split("_t-")[0]
    timestep = int(feature_name.split("_t-")[1])

    # Finde alle Basisfeatures (einmalig)
    base_feature_names = sorted(set([name.split("_t-")[0] for name in feature_names]))
    if base_feature not in base_feature_names:
        raise ValueError(f"Base Feature {base_feature} nicht in Featureliste gefunden.")

    base_index = base_feature_names.index(base_feature)

    for shap_file, model_name, input_file in zip(shap_files, model_names, input_files):
        shap_data = np.load(shap_file)
        feature_index = [i for i, name in enumerate(feature_names) if name == feature_name]
        if not feature_index:
            raise ValueError(f"Feature {feature_name} nicht in SHAP-Daten von {model_name} gefunden.")
        feature_index = feature_index[0]

        shap_values = shap_data['class_1'][:, feature_index]
        shap_values_all_models.append(shap_values)

        input_data = np.load(input_file)

        if input_data.ndim == 3:
            B, T, F = input_data.shape
            if timestep >= T or base_index >= F:
                raise IndexError(f"Zeitindex {timestep} oder Featureindex {base_index} zu groß für {model_name}")
            feature_data = input_data[:, timestep, base_index].reshape(-1, 1)
        elif input_data.ndim == 2:
            feature_data = input_data[:, feature_index].reshape(-1, 1)
        else:
            raise ValueError(f"Unerwartete Input-Shape: {input_data.shape}")

        input_data_all_models.append(feature_data)

        if num_samples is None:
            num_samples = len(shap_values)
        elif len(shap_values) != num_samples:
            raise ValueError(f"Mismatch in der Anzahl der Samples für {model_name}")

    # Combine across models
    shap_values_all_models = np.array(shap_values_all_models).T  # Shape: (n_samples, n_models)
    input_data_all_models = np.concatenate(input_data_all_models, axis=1)  # Shape: (n_samples, n_models)

    # SHAP expects feature_names to match columns (axis=1)
    expl = shap.Explanation(
        values=shap_values_all_models,
        data=input_data_all_models,
        feature_names=model_names
    )

    save_file = os.path.join(base_path, f"beeswarm_by_feature_{feature_name}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(expl, max_display=len(model_names), show=False)

    plt.title(f"SHAP Value Comparison for Feature: {feature_name}")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"✅ SHAP Beeswarm Plot gespeichert unter: {save_file}")

def map_sample_ids_to_indices(sample_ids, selected_ids):
    """
    Map a list of selected sample IDs to their corresponding indices in the data array.

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

def compute_grouped_shap_over_time(shap_values, feature_names):
    """
    Grouped SHAP-Werte over time for base features
    Returns:
        grouped_df: pd.DataFrame with (n_samples, n_base_features)
        base_features: List with base feature names
    """
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    base_names = [name.split("_t-")[0] for name in feature_names]
    shap_df.columns = base_names
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
    base_feature_names = sorted(set([name.split("_t-")[0] for name in feature_names]))

    return input_agg, base_feature_names

def normalize_input_per_feature(input_np):
    input_norm = np.zeros_like(input_np)
    for i in range(input_np.shape[1]):
        col = input_np[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        if max_val > min_val:
            input_norm[:, i] = (col - min_val) / (max_val - min_val)
        else:
            input_norm[:, i] = 0.0
    return input_norm


