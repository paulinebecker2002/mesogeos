import argparse
import collections
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pandas as pd
import os
import seaborn as sns

import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device

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

def plot_shap_summary(shap_values, input_tensor, feature_names, checkpoint_path, base_path, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_path = os.path.join(base_path, model_id, f"shap_summary_plot_{model_id}_lstm.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    shap.summary_plot(
        shap_values,
        input_tensor.cpu().numpy(),
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Summary Plot stored at: {save_path}")

def plot_grouped_feature_importance(shap_values, feature_names, checkpoint_path, base_path, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_path = os.path.join(base_path, model_id, f"grouped_shap_plot_{model_id}_lstm.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    plt.savefig(save_path, dpi=300)
    plt.close()
    if logger:
        logger.info(f"SHAP Summary Plot stored at: {save_path}")

def plot_shap_temporal_heatmap(shap_values, feature_names, checkpoint_path, base_path, logger=None):
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    save_path = os.path.join(base_path, model_id, f"shap_temporal_heatmap_{model_id}_lstm.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    shap_df = pd.DataFrame(shap_values).T
    shap_df.columns = [f"instance_{i}" for i in range(shap_df.shape[1])]
    shap_df["feature"] = [name.split("_t-")[0] for name in feature_names]
    shap_df["time"] = [int(name.split("_t-")[1]) for name in feature_names]

    shap_df_long = pd.melt(shap_df, id_vars=["feature", "time"], var_name="instance", value_name="shap")
    heatmap_data = shap_df_long.groupby(["feature", "time"])["shap"].mean().unstack()

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=False)
    plt.title("Mean SHAP-Value per Feature und Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    if logger:
        logger.info(f"SHAP-Heatmap saved at: {save_path}")

def main(config):
    SEED = config['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    checkpoint_path = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/models/lala/0423_105818/model_best.pth"
    base_save_path = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/shap-plots/"

    logger = config.get_logger('shap')

    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]
    input_dim = len(dynamic_features) + len(static_features)

    dataset = config.init_obj('dataset', module_data,
                              dynamic_features=dynamic_features,
                              static_features=static_features,
                              train_val_test='val')
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])

    assert config["model_type"] == "lstm", "This script is for LSTM models only"
    model = config.init_obj('arch', module_arch,
                            input_dim=input_dim,
                            output_lstm=config['model_args']['dim'],
                            dropout=config['model_args']['dropout'])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    batch = next(iter(dataloader))
    dynamic, static, _, labels = batch
    static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
    input_ = torch.cat([dynamic, static], dim=2).to(device).to(torch.float32)

    explainer = shap.GradientExplainer(model, input_[:100])
    shap_values = explainer.shap_values(input_[:100])

    feature_names = get_feature_names(config)
    plot_shap_summary(shap_values[1], input_[:100], feature_names, checkpoint_path, base_save_path, logger)
    plot_grouped_feature_importance(shap_values[1], feature_names, checkpoint_path, base_save_path, logger)
    plot_shap_temporal_heatmap(shap_values[1], feature_names, checkpoint_path, base_save_path, logger)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='SHAP Explanation Script for LSTM')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--dr', '--dropout'], type=float, target='model_args;dropout'),
        CustomArgs(['--hd', '--hidden-dims'], type=lambda s: [int(x) for x in s.split(',')], target='model_args;hidden_dims'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
