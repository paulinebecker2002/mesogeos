import argparse
import collections
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from alibi.explainers import ALE, plot_ale

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

    def __call__(self, X):  # alibi expects __call__, not .predict
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        num_dyn = len(self.dynamic_features)
        num_stat = len(self.static_features)

        if self.model_type == 'mlp':
            input_ = X_tensor.view(X_tensor.shape[0], -1)
        elif self.model_type in ['transformer', 'gtn', 'gru', 'lstm', 'cnn']:
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
            input_ = input_.transpose(0, 1)
        elif self.model_type == 'tft':
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
            dyn = input_[:, :, :num_dyn]
            stat = input_[:, 0, num_dyn:]
            outputs = self.model(dyn, stat)
        else:
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)

        if self.model_type != 'tft':
            outputs = self.model(input_)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        return probs.detach().cpu().numpy()


def plot_feature_histograms(X_df, features_to_plot, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for feature in features_to_plot:
        if feature not in X_df.columns:
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

    wrapped_model = ALEModelWrapper(
        model=model,
        model_type=model_type,
        device=device,
        dynamic_features=dynamic_features,
        static_features=static_features,
        seq_len=seq_len
    )

    features_to_plot = ["ndvi_t-1", "lai_t-1", "population_t-1", "sp_t-1", "dem_t-1"]

    plot_feature_histograms(
        X_df=X_df,
        features_to_plot=features_to_plot,
        save_dir=os.path.join(base_save_path, "feature_distributions")
    )

    ale_explainer = ALE(predict_fn=wrapped_model, feature_names=feature_names)
    explanation = ale_explainer.explain(X_df, features=[feature_names.index(f) for f in features_to_plot])

    for i, feature in enumerate(features_to_plot):
        plt.figure()
        plot_ale(explanation, features=[i])
        path = os.path.join(base_save_path, "first_order_alibi", f"ale_{feature}_{model_type}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved ALE plot to: {path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ALE Plot Generator')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
