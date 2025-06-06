import argparse
import collections
import os
import numpy as np
from parse_config import ConfigParser
from utils.util import get_feature_names
from ig_utils import plot_bar, plot_temporal_heatmap, plot_ig_beeswarm

def main(config):
    logger = config.get_logger('ig')

    checkpoint_path = config["shap"]["checkpoint_path"]
    model_type = config["model_type"]
    only_pos = config["ig"]["only_positive"]
    only_neg = config["ig"]["only_negative"]
    feature_names = get_feature_names(config)
    ig_path = config["ig"]["ig_path"]
    model_id = os.path.basename(os.path.dirname(checkpoint_path))

    ig_file = os.path.join(ig_path, f"ig_values_{model_type}.npy")
    ig_data = np.load(ig_file)
    label_file = os.path.join(ig_path, f"ig_labels_{model_type}.npy")
    labels = np.load(label_file)
    input_file = os.path.join(ig_path, f"ig_input_tensor_{model_type}.npy")
    input_data = np.load(input_file)

    if only_neg and only_pos:
        raise ValueError("Both only_positive and only_negative cannot be True at the same time.")

    if only_pos:
        ig_data = ig_data[labels == 1]
        model_id += "_positive"
        print("Only positive IG values selected with shape:", ig_data.shape)
    elif only_neg:
        ig_data = ig_data[labels == 0]
        model_id += "_negative"
        print("Only negative IG values selected with shape:", ig_data.shape)
    else:
        print("Using all IG values with shape:", ig_data.shape)

    #plot_bar(ig_data, feature_names, model_id, model_type, ig_path, logger)
    #plot_temporal_heatmap(ig_data, feature_names, model_id, model_type, ig_path, logger, scaled=True)
    #plot_temporal_heatmap(ig_data, feature_names, model_id, model_type, ig_path, logger, scaled=False)
    plot_ig_beeswarm(ig_data, input_data, feature_names, model_id, model_type, ig_path, logger)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Plot IG results')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--only_positive', '--op'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='ig;only_positive'),
        CustomArgs(['--only_negative', '--on'], type=lambda x: x.lower() in ['true', '1', 'yes'], target='ig;only_negative'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
