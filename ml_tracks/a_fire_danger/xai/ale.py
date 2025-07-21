import argparse
import collections
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PyALE import ale
from itertools import combinations

from parse_config import ConfigParser
from utils.util import extract_numpy, get_feature_names, build_model, get_dataloader


class ALEModelWrapper:
    def __init__(self, model, model_type, device, dynamic_features, static_features, seq_len=30):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.seq_len = seq_len

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert DataFrame to NumPy array

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        num_dyn = len(self.dynamic_features)
        num_stat = len(self.static_features)

        if self.model_type == 'mlp':
            input_ = X_tensor.view(X_tensor.shape[0], -1)
            outputs = self.model(input_)

        elif self.model_type in ['gru', 'lstm', 'cnn']:
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
            outputs = self.model(input_)

        elif self.model_type in ['gtn', 'transformer']:
             input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
             input_ = input_.transpose(0, 1)
             outputs = self.model(input_)

        elif self.model_type == 'tft':
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
            dyn = input_[:, :, :num_dyn]
            stat = input_[:, 0, num_dyn:]
            outputs = self.model(dyn, stat)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        probs = torch.softmax(outputs, dim=1)[:, 1]
        return probs.detach().cpu().numpy()


def plot_feature_histograms(X_df, features_to_plot, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for feature in features_to_plot:
        if feature not in X_df.columns:
            print(f"Feature {feature} not found in DataFrame columns.")
            continue

        plt.figure()
        plt.hist(X_df[feature], bins=30, edgecolor='black')
        plt.xlabel(feature)
        plt.ylabel("Häufigkeit")
        plt.title(f"Verteilung: {feature}")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"hist_{feature}.png")
        plt.savefig(save_path)
        plt.close()


def plot_second_order_interactions(X_df, model_wrapper, feature1, feature2, base_save_path, model_type, logger):
    logger.info(f"Generating second order ALE plot for: {feature1} and {feature2}")
    plot_path = os.path.join(base_save_path, "second_order", f"second_order_ale_{feature1}_{feature2}_{model_type}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    ale(
        X=X_df,
        model=model_wrapper,
        feature=[feature1, feature2],
        grid_size=100,
        plot=True,
    )

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_first_order_interactions(X_df, model_wrapper, feature, base_save_path, model_type, logger):
    logger.info(f"Generating ALE plot for: {feature}")
    plot_path = os.path.join(base_save_path, "first_order_0.15_y", f"ale_{feature}_{model_type}_0.15.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    ale(
        X=X_df,
        model=model_wrapper,
        feature=[feature],
        grid_size=20,
        include_CI=True,
        C=0.95,
        plot=True,
    )

    # Anpassungen am Plot
    plt.title("")
    plt.ylabel("")
    plt.xlabel(feature, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(-0.15, 0.15)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def main(config):
    logger = config.get_logger('ale')
    model_type = config["model_type"]
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]
    checkpoint_path = config["XAI"]["checkpoint_path"]
    seq_len = config["dataset"]["args"]["lag"]
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/ale-plots/{model_type}/"

    dataloader = get_dataloader(config, static_features, dynamic_features, mode='test')

    X_test, y_test = extract_numpy(dataloader)
    feature_names = get_feature_names(config)
    X_df = pd.DataFrame(X_test, columns=feature_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, dynamic_features, static_features)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    model_wrapper = ALEModelWrapper(
        model=model,
        model_type=model_type,
        device=device,
        dynamic_features=dynamic_features,
        static_features=static_features,
        seq_len=seq_len
    )

    #features_to_plot = ["ndvi_t-1", "lai_t-1", "population_t-1", "sp_t-1", "dem_t-1"]

    features_to_plot = [
        "d2m_t-1", "lai_t-1",  "lst_day_t-1", "lst_night_t-1", "ndvi_t-1", "rh_t-1", "smi_t-1", "sp_t-1", "ssrd_t-1",
        "t2m_t-1", "tp_t-1",  "wind_speed_t-1",
        "dem_t-1",  "population_t-1",  "roads_distance_t-1", "slope_t-1", "lc_agriculture_t-1", "lc_forest_t-1",
        "lc_grassland_t-1", "lc_settlement_t-1",  "lc_shrubland_t-1",  "lc_sparse_vegetation_t-1",
        "lc_water_bodies_t-1", "lc_wetland_t-1"]


    #plot_feature_histograms(X_df=X_df, features_to_plot=features_to_plot, save_dir=os.path.join(base_save_path, "feature_distributions"))

    #plot_second_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature1='ndvi_t-1', feature2='tp_t-1', base_save_path=base_save_path, model_type=model_type, logger=logger)
    #plot_second_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature1='wind_speed_t-1', feature2='slope_t-1', base_save_path=base_save_path, model_type=model_type, logger=logger)
    #plot_second_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature1='t2m_t-1', feature2='rh_t-1', base_save_path=base_save_path, model_type=model_type, logger=logger)
    #plot_second_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature1='smi_t-1', feature2='lc_forest_t-1', base_save_path=base_save_path, model_type=model_type, logger=logger)
    #plot_second_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature1='lst_day_t-1', feature2='dem_t-1', base_save_path=base_save_path, model_type=model_type, logger=logger)

    #for feature1, feature2 in combinations(features_to_plot, 2):
        #plot_second_order_interactions( X_df=X_df, model_wrapper=model_wrapper, feature1=feature1, feature2=feature2, base_save_path=base_save_path, model_type=model_type, logger=logger)

    for feature in features_to_plot:
        plot_first_order_interactions(X_df=X_df, model_wrapper=model_wrapper, feature=feature, base_save_path=base_save_path, model_type=model_type, logger=logger)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ALE Plot Generator')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
