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


def plot_grouped_feature_importance(shap_values, shap_class, feature_names, model_id, base_path, model_type, logger=None):
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

def plot_shap_waterfall(shap_values, shap_class, input_tensor, feature_names, sample_idx,
                        model_id, base_path, model_type, logger=None):
    """
    Plot SHAP waterfall plot for a single instance (sample_idx) and save to file.
    """
    save_file = os.path.join(base_path, f"shap_waterfall_plot_{model_id}_{model_type}_class{shap_class}_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Prepare input and SHAP values for one sample
    if model_type in ["lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.view(input_tensor.shape[0], -1)

    sample_input = input_tensor[sample_idx].cpu().numpy()
    sample_shap = shap_values[sample_idx]
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
