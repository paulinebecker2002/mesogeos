import argparse
import collections
import os
import pandas as pd
import torch
import numpy as np
from PyALE import ale

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
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        num_dyn = len(self.dynamic_features)
        num_stat = len(self.static_features)

        if self.model_type == 'mlp':
            input_ = X_tensor.view(X_tensor.shape[0], -1)
            outputs = self.model(input_)

        elif self.model_type in ['gtn', 'gru', 'lstm', 'cnn']:
            input_ = X_tensor.view(-1, self.seq_len, num_dyn + num_stat)
            outputs = self.model(input_)

        elif self.model_type == 'transformer':
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


def main(config):
    logger = config.get_logger('ale')
    model_type = config["model_type"]
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]
    checkpoint_path = config["XAI"]["checkpoint_path"]
    seq_len = config["dataset"]["args"]["lag"]
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/ale-values/{model_type}/"

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

    base_feature = "ndvi"
    lags = list(range(1, 30))  # t-1 bis t-29
    feature_lag_names = [f"{base_feature}_t-{i}" for i in lags]

    os.makedirs(base_save_path, exist_ok=True)
    grid_ref = None
    ale_values_all = []

    # 1. Einheitliche Binning-Kanten definieren (z. B. 20 Bins)
    reference_feature = feature_lag_names[0]
    bin_edges = np.linspace(X_df[reference_feature].min(), X_df[reference_feature].max(), 21)  # 20 Bins = 21 Edges

    # 2. Alle Lag-Spalten (ndvi_t-1 bis ndvi_t-29) auf dieses Grid bringen
    for col in feature_lag_names:
        # binned: enthält Indexe von 0 bis 19 (für 20 Bins), aber auch NaN für Werte außerhalb
        binned = pd.cut(X_df[col], bins=bin_edges, labels=False, include_lowest=True)

        # neue Spalte = linker Rand des Bin-Intervalls
        X_df[col + "_binned"] = binned.map(lambda i: bin_edges[int(i)] if pd.notnull(i) else np.nan)


# 3. Dann ALE berechnen für die binned-Spalten:
    ale_values_all = []
    grid_ref = bin_edges[:-1]  # das sind die mittleren Werte der Bins (linke Kanten)
    for feat in feature_lag_names:
        feat_binned = feat + "_binned"
        logger.info(f"Berechne ALE für {feat_binned}")

        feature_names_wo_lag = [f for f in feature_names if f != feat]
        X_input = X_df[feature_names_wo_lag + [feat_binned]]

        ale_result = ale(
            X=X_input,
            model=model_wrapper,
            feature=[feat_binned],
            grid_size=20,  # nötig, aber es wird ignoriert, da wir manuell gebinned haben
            include_CI=False,
            plot=False
        )

        eff_col = "eff"
        ale_values_all.append(ale_result[eff_col].values)



    ale_values_all = np.stack(ale_values_all, axis=0)  # [n_lags, n_bins]
    mean_ale = np.mean(ale_values_all, axis=0)
    print("grid_ref:", grid_ref.shape)
    print("mean_ale:", mean_ale.shape)


    result_df = pd.DataFrame({
        "feature_value": grid_ref,
        f"mean_ale_{base_feature}": mean_ale
    })

    save_path = os.path.join(base_save_path,"average_ALE", f"ale_mean_{base_feature}_lags1-29.csv")
    result_df.to_csv(save_path, index=False)
    logger.info(f"ALE-Mittelwert gespeichert unter: {save_path}")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(grid_ref, mean_ale, marker='o')
    plt.xlabel(f"{base_feature} value")
    plt.ylabel("Average ALE")
    plt.title(f"Averaged ALE over time ({base_feature}_t-1 bis t-29)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_path,"average_ALE", f"plot_ale_mean_{base_feature}.png"))
    plt.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Average ALE over time-lags')
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
