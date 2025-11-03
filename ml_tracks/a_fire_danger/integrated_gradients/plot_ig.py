import argparse
import collections
import os
import numpy as np
from parse_config import ConfigParser
from utils.util import get_feature_names
from ig_utils import plot_bar, plot_temporal_heatmap, plot_ig_beeswarm, plot_ig_beeswarm_only_once_each_feature, plot_ig_beeswarm_by_feature, plot_ig_beeswarm_grouped, plot_ig_beeswarm_by_feature_grouped, plot_ig_waterfall_grouped, plot_ig_waterfall

def load_ig_inputs_from_combined_npz(ig_path, model_id, model_type):
    """
    Loads IG-Values, Labels and Sample-IDs from combined .npz-Datei.
    """
    combined_npz_path = os.path.join(ig_path, f"ig_values_{model_id}_{model_type}_combined.npz")
    if not os.path.exists(combined_npz_path):
        raise FileNotFoundError(f"Combined IG file not found: {combined_npz_path}")

    data = np.load(combined_npz_path, allow_pickle=True)

    ig_values = data["class_1"]
    labels = data["label"]
    sample_ids = data["sample_id"]

    return ig_values, labels, sample_ids

def main(config):
    logger = config.get_logger('ig')

    checkpoint_path = config["XAI"]["checkpoint_path"]
    model_type = config["model_type"]
    only_pos = config["XAI"]["only_positive"]
    only_neg = config["XAI"]["only_negative"]
    feature_names = get_feature_names(config)
    ig_path = config["XAI"]["ig_path"]
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    all_model_path = '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/all_model_comparison'

    ig_data, labels, sample_ids = load_ig_inputs_from_combined_npz(ig_path, model_id, model_type)

    input_file_path = os.path.join(ig_path, f"ig_values_{model_id}_{model_type}_input.npy")
    input_data = np.load(input_file_path)

    if only_neg and only_pos:
        raise ValueError("Both only_positive and only_negative cannot be True at the same time.")

    if only_pos:
        ig_data = ig_data[labels == 1]
        input_data = input_data[labels == 1]
        model_id += "_positive"
        print("Only positive IG values selected with shape:", ig_data.shape)
    elif only_neg:
        ig_data = ig_data[labels == 0]
        input_data = input_data[labels == 0]
        model_id += "_negative"
        print("Only negative IG values selected with shape:", ig_data.shape)
    else:
        print("Using all IG values with shape:", ig_data.shape)

    print("input_data shape:", input_data.shape)     # z. B. (4107, 30, 24)
    print("ig_data shape:", ig_data.shape)           # z. B. (4107, 30, 24)
    print("len(feature_names):", len(feature_names))

    plot_bar(ig_data, feature_names, model_id, model_type, ig_path, logger)
    plot_temporal_heatmap(ig_data, feature_names, model_id, model_type, ig_path, logger, scaled=True)
    plot_temporal_heatmap(ig_data, feature_names, model_id, model_type, ig_path, logger, scaled=False)
    plot_ig_waterfall_grouped(ig_values=ig_data, input_tensor=input_data, feature_names=feature_names, sample_ids=sample_ids, sample_idx=679, base_path=ig_path, model_type=model_type)
    plot_ig_waterfall_grouped(ig_values=ig_data, input_tensor=input_data, feature_names=feature_names, sample_ids=sample_ids, sample_idx=1645, base_path=ig_path, model_type=model_type)
    plot_ig_beeswarm(ig_data, input_data, feature_names, model_id, model_type, ig_path, logger)
    plot_ig_beeswarm_only_once_each_feature(ig_data, input_data, feature_names, model_id, model_type, ig_path, logger)
    plot_ig_beeswarm_grouped(ig_values=ig_data, input_tensor=input_data, feature_names=feature_names, model_id=model_id, model_type=model_type, base_path=ig_path)

    shap_files = [
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/cnn/0606_191829/ig_values_cnn.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/mlp/0606_103457/ig_values_mlp.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/gru/0606_191651/ig_values_gru.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/lstm/0606_191651/ig_values_lstm.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/transformer/0606_191656/ig_values_transformer.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/gtn/0624_084137/ig_values_gtn.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/tft/0607_112458/ig_values_tft.npy'
    ]

    input_files = [
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/cnn/0606_191829/ig_input_tensor_cnn.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/mlp/0606_103457/ig_input_tensor_mlp.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/gru/0606_191651/ig_input_tensor_gru.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/lstm/0606_191651/ig_input_tensor_lstm.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/transformer/0606_191656/ig_input_tensor_transformer.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/gtn/0624_084137/ig_input_tensor_gtn.npy',
        '/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/tft/0607_112458/ig_input_tensor_tft.npy'
    ]

    model_names = ['cnn', 'mlp', 'gru', 'lstm', 'transformer', 'gtn', 'tft']
    #features = ["lst_day_t-1", "lst_day_t-30", "rh_t-1", "rh_t-30", "smi_t-1", "smi_t-30", "lc_forest_t-1", "lc_forest_t-30", "wind_speed_t-1", "wind_speed_t-30"]
    features = ["lst_day", "rh", "smi", "lc_forest", "wind_speed"]
    #for feature in features:
        #plot_ig_beeswarm_by_feature_grouped(shap_files, feature, feature_names, model_names, input_files, all_model_path)

    bigFire_sample_idx = [4792, 679, 8418, 1645, 1676]
    for idx in bigFire_sample_idx:
        print(f"Plotting SHAP waterfall for Sample: {idx}")
        plot_ig_waterfall_grouped(ig_values=ig_data, input_tensor=input_data, feature_names=feature_names, sample_ids=sample_ids, sample_idx=idx, base_path=f"{ig_path}/big_fire_waterfall", model_type=model_type)





if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Plot IG results')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = [
        CustomArgs(['--only_positive', '--op'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_positive', nargs=None),
        CustomArgs(['--only_negative', '--on'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='XAI;only_negative', nargs=None),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
