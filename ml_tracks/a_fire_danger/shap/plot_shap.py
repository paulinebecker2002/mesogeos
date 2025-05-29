import argparse
import collections
import os
import numpy as np
import torch
from parse_config import ConfigParser
from shap_utils import (
    get_feature_names, plot_shap_summary,
    plot_grouped_feature_importance, plot_shap_temporal_heatmap, plot_shap_difference_bar,
    plot_shap_difference_aggregated
)


def main(config):
    logger = config.get_logger('shap')

    checkpoint_path = config["shap"]["checkpoint_path"]
    shap_class = config["shap"]["class"]
    model_type = config["model_type"]
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/{model_type}/"
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    shap_path = os.path.join(base_save_path, model_id, f"shap_values_{model_id}_{model_type}.npz")

    shap_data = np.load(shap_path)
    input_tensor_path = shap_path.replace(".npz", "_input.npy")
    input_tensor_np = np.load(input_tensor_path)
    input_tensor = torch.tensor(input_tensor_np)
    feature_names = get_feature_names(config)

    if shap_class == 0:
        shap_values = shap_data['class_0']
    else:
        shap_values = shap_data['class_1']

    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")
    plot_grouped_feature_importance(shap_values, shap_class, feature_names, checkpoint_path, base_save_path, model_type, logger)
    plot_shap_summary(shap_values, shap_class, input_tensor[:100], feature_names, checkpoint_path, base_save_path, model_type, logger)
    #plot_shap_difference_bar( shap_data['class_0'], shap_data['class_1'], feature_names, checkpoint_path, base_save_path, model_type, logger)
    #plot_shap_difference_aggregated( shap_data['class_0'], shap_data['class_1'], feature_names, checkpoint_path, base_save_path, model_type, logger)

    #plot_shap_temporal_heatmap(shap_values, feature_names, checkpoint_path, base_save_path, model_type, logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Compute SHAP values')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--cl', '--class'], type=int, target='shap;class'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
