import argparse
import collections
import os
import numpy as np
import torch
from parse_config import ConfigParser
from utils.util import get_feature_names
from shap_utils import (plot_beeswarm,
                        plot_grouped_feature_importance, plot_shap_temporal_heatmap, plot_shap_difference_bar,
                        plot_shap_difference_aggregated, plot_shap_waterfall, plot_shap_comparison_by_feature,
                        plot_beeswarm_by_feature)


def main(config):
    logger = config.get_logger('shap')

    checkpoint_path = config["shap"]["checkpoint_path"]
    shap_class = config["shap"]["class"]
    model_type = config["model_type"]
    only_pos = config["XAI"]["only_positive"]
    only_neg = config["XAI"]["only_negative"]
    shap_path = config["shap"]["shap_path"]
    all_model_path = '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/all_model_comparison'
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    feature_names = get_feature_names(config)

    shap_file = os.path.join(shap_path, f"shap_values_{model_id}_{model_type}.npz")
    shap_data = np.load(shap_file)
    labels_file = os.path.join(shap_path, f"shap_values_{model_id}_{model_type}_labels.npy")
    labels = np.load(labels_file)


    input_tensor_path = os.path.join(shap_path, f"shap_values_{model_id}_{model_type}_input.npy")
    input_tensor_np = np.load(input_tensor_path)
    input_tensor = torch.tensor(input_tensor_np)
    print("input_tensor shape:", input_tensor.shape)
    print(f"labels shape: {labels.shape}, labels: {np.unique(labels)}")

    if shap_class == 0:
        shap_values = shap_data['class_0']
    else:
        shap_values = shap_data['class_1']

    if only_pos:
        shap_values = shap_values[labels == 1]
        input_tensor = input_tensor[labels == 1]
        model_id += "_positive"
        print("Only positive IG values selected with shape:", shap_values.shape)
    elif only_neg:
        shap_values = shap_values[labels == 0]
        input_tensor = input_tensor[labels == 0]
        model_id += "_negative"
        print("Only negative IG values selected with shape:", shap_values.shape)
    else:
        print("Using all IG values with shape:", shap_values.shape)


    shap_files = [
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/cnn/0530_160107/shap_values_0517_181322_cnn.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/mlp/0530_150819/shap_values_0517_175347_mlp.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gru/0530_155939/shap_values_0514_140125_gru.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/lstm/0530_160506/shap_values_0513_230004_lstm.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/transformer/0530_164841/shap_values_0519_125059_transformer.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gtn/0530_201817/shap_values_0520_203840_gtn.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/rf/0604_161105/shap_values_0604_052725_rf.npz',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/tft/0607_040620/shap_values_0529_193434_tft.npz'
    ]

    input_files = [
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/cnn/0530_160107/shap_values_0517_181322_cnn_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/mlp/0530_150819/shap_values_0517_175347_mlp_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gru/0530_155939/shap_values_0514_140125_gru_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/lstm/0530_160506/shap_values_0513_230004_lstm_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/transformer/0530_164841/shap_values_0519_125059_transformer_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/gtn/0530_201817/shap_values_0520_203840_gtn_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/rf/0604_161105/shap_values_0604_052725_rf_input.npy',
        '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/tft/0607_040620/shap_values_0529_193434_tft_input.npy'
    ]

    model_names = ['cnn', 'mlp', 'gru', 'lstm', 'transformer', 'gtn', 'rf', 'tft']
    #plot_shap_comparison_by_feature(shap_files, 'd2m_t-1', feature_names, model_names, shap_path, logger)
    #plot_beeswarm_by_feature(shap_files, 'lst_day_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'lst_day_t-2', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'rh_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 't2m_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'd2m_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'lst_night_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'ndvi_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 't2m_t-2', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'tp_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'wind_speed_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'lai_t-1', feature_names, model_names, input_files, all_model_path)
    #plot_beeswarm_by_feature(shap_files, 'lst_day_t-5', feature_names, model_names, input_files, all_model_path)

    print(f"Shape input: {input_tensor.shape}, SHAP: {np.array(shap_values).shape}")
    #plot_grouped_feature_importance(shap_values, shap_class, feature_names, model_id, shap_path, model_type, logger)
    #plot_beeswarm(shap_values, shap_class, input_tensor, feature_names, model_id, shap_path, model_type, logger)

    #plot_shap_difference_bar( shap_data['class_0'], shap_data['class_1'], feature_names, model_id, shap_path, model_type, logger)
    #plot_shap_difference_aggregated( shap_data['class_0'], shap_data['class_1'], feature_names, model_id, shap_path, model_type, logger)

    #plot_shap_temporal_heatmap(shap_values, shap_class, feature_names, model_id, shap_path, model_type, logger)

    sample_idx = [479, 3984, 3505, 4091, 3594]


    for idx in sample_idx:
        print(f"Plotting SHAP waterfall for sample index: {idx}")
        print(f"\n--- Sample {idx} ---")
        print(f"Label: {labels[idx]}")
        #plot_shap_waterfall(shap_values, shap_class, input_tensor, feature_names, idx, model_id, shap_path, model_type, logger)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Compute SHAP values')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--cl', '--class'], type=int, target='shap;class'),
        CustomArgs(['--only_positive', '--op'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_positive'),
        CustomArgs(['--only_negative', '--on'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_negative'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
