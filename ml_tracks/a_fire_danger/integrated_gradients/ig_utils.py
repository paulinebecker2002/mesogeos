import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def plot_bar(ig_values, feature_names, model_id, model_type, base_path, logger=None):
    save_file = os.path.join(base_path, f"ig_bar_plot_{model_id}_{model_type}.png")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

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

    # Flach machen, falls [B, T, F]
    if ig_values.ndim == 3:
        ig_values = ig_values.reshape(ig_values.shape[0], -1)

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

    # Optional abs verwenden
    # ig_values = np.abs(ig_values)

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


