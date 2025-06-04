import argparse
import collections
import os
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from captum.attr import IntegratedGradients
from parse_config import ConfigParser
from utils.util import set_seed, build_model, get_dataloader, prepare_device
from utils.util import get_feature_names

def get_baseline(input_tensor):
    return torch.zeros_like(input_tensor)

def compute_ig_for_model(model, input_tensor, model_type, static_tensor=None, target_class=1, seq_len=None):


    if model_type in ["mlp", "lstm", "gru", "cnn"]:
        input_tensor.requires_grad_()
        ig = IntegratedGradients(model)
        baseline = get_baseline(input_tensor)
        attributions = ig.attribute(input_tensor, baseline, target=target_class)

    elif model_type in ["transformer", "gtn"]:
        input_tensor.requires_grad_()
        baseline = get_baseline(input_tensor)

        def forward_fn(x):
            return model(x.permute(1, 0, 2))

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(input_tensor, baseline, target=target_class, n_steps=50)

    elif model_type == "tft":
        static_tensor.requires_grad_()
        input_tensor.requires_grad_()
        baseline_dyn = get_baseline(input_tensor)
        baseline_stat = get_baseline(static_tensor)

        def forward_fn(dyn, stat):
            return model(dyn, stat)

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(inputs=(input_tensor, static_tensor),
                                    baselines=(baseline_dyn, baseline_stat),
                                    target=target_class)
        return attributions[0], attributions[1]

    else:
        raise NotImplementedError(f"IG für Modelltyp {model_type} nicht implementiert.")

    return attributions

def main(config):
    SEED = config['seed']
    set_seed(SEED)
    logger = config.get_logger('ig')

    checkpoint_path = config["shap"]["checkpoint_path"]
    model_type = config["model_type"]
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]
    feature_names = get_feature_names(config)
    seq_len = config["dataset"]["args"]["lag"]

    if model_type not in ["mlp", "lstm", "gru", "tft", "transformer", "gtn", "cnn"]:
        raise ValueError(f"Model type {model_type} not supported for IG computation.")

    dataloader = get_dataloader(config, static_features, dynamic_features, mode='test')
    device, _ = prepare_device(config['n_gpu'], config['gpu_id'])

    model = build_model(config, dynamic_features, static_features)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    if model_type in ["lstm", "gru", "tft"]:
        model.train()

    ig_results, coord_x, coord_y = [], [], []

    logger.info(f"Computing IG in mini-batches for model_type={model_type}")

    for batch_idx, batch in enumerate(dataloader):
        dynamic, static, _, _, x, y = batch[:6]
        if model_type == "tft":
            input_ = dynamic  # Nur dynamische Features fürs Modell
        else:
            static_expanded = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            input_ = torch.cat([dynamic, static_expanded], dim=2)

        if model_type == "mlp":
            input_ = input_.view(input_.shape[0], -1)
        else:
            pass

        input_ = input_.float().to(device)
        static = static.float().to(device)

        if model_type == "tft":
            ig_dyn, ig_stat = compute_ig_for_model(model, input_, model_type, static_tensor=static,
                                                   target_class=1, seq_len=seq_len)
            ig_batch = torch.cat([ig_dyn, ig_stat.unsqueeze(1).repeat(1, seq_len, 1)], dim=-1)
        else:
            ig_batch = compute_ig_for_model(model, input_, model_type, target_class=1, seq_len=seq_len)

        ig_results.append(ig_batch.detach().cpu())
        coord_x.extend(x.cpu().numpy())
        coord_y.extend(y.cpu().numpy())

        if (batch_idx + 1) % 2 == 0:
            logger.info(f"Processed {batch_idx + 1} batches")

    ig_all = torch.cat(ig_results, dim=0)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/ig-plots/{model_type}/"
    save_path = os.path.join(base_save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    ig_np = ig_all.numpy()
    print("[DEBUG] ig_all shape:", ig_all.shape)
    print("[DEBUG] ig_all.reshape: ", ig_np.reshape(ig_np.shape[0], -1).shape)
    print("[DEBUG] len(feature_names):", len(feature_names))
    np.save(os.path.join(save_path, "ig_values.npy"), ig_np)

    base_names = [f.split("_t-")[0] for f in feature_names]
    df = pd.DataFrame(ig_np.reshape(ig_np.shape[0], -1), columns=feature_names)
    df.columns = base_names
    df["x"] = coord_x
    df["y"] = coord_y
    df.to_csv(os.path.join(save_path, f"ig_map_{model_type}.csv"), index=False)

    logger.info(f"IG saved to: {save_path}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Compute Integrated Gradients')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
