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

    checkpoint_path = config["XAI"]["checkpoint_path"]
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
    logger.info(f"Computing IG in mini-batches for model_type={model_type}")
    if model_type in ["lstm", "gru", "tft"]:
        model.train()

    ig_results, coord_x, coord_y, labels_all, input_all, sample_ids_all, probs_all = [], [], [], [], [], [], []

    for batch_idx, batch in enumerate(dataloader):
        dynamic, static, _, labels, x, y, sample_id = batch[:7]

        if model_type == "tft":
            input_ = dynamic
        else:
            static_expanded = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            input_ = torch.cat([dynamic, static_expanded], dim=2)

        if model_type == "mlp":
            input_ = input_.view(input_.shape[0], -1)

        input_ = input_.float().to(device)
        static = static.float().to(device)
        with torch.no_grad():
            if model_type == "tft":
                output = model(input_, static)
            elif model_type in ["transformer", "gtn"]:
                reshaped = input_.permute(1, 0, 2)  # [seq_len, B, F]
                output = model(reshaped)
            elif model_type in ["lstm", "gru", "cnn", "mlp"]:
                output = model(input_)
            else:
                raise NotImplementedError

            probs = torch.softmax(output, dim=1)[:, 1]
            probs_all.extend(probs.cpu().numpy())

        labels_all.extend(labels.cpu().numpy().astype(int))
        sample_ids_all.extend(sample_id.cpu().numpy().astype(int))
        coord_x.extend(x.cpu().numpy())
        coord_y.extend(y.cpu().numpy())

        if model_type == "tft":
            static_expanded = static.unsqueeze(1).repeat(1, seq_len, 1)
            input_full = torch.cat([input_, static_expanded], dim=2)
            input_all.append(input_full.detach().cpu())
            ig_dyn, ig_stat = compute_ig_for_model(model, input_, model_type, static_tensor=static,
                                                   target_class=1, seq_len=seq_len)
            ig_batch = torch.cat([ig_dyn, ig_stat.unsqueeze(1).repeat(1, seq_len, 1)], dim=-1)
        else:
            ig_batch = compute_ig_for_model(model, input_, model_type, target_class=1, seq_len=seq_len)
            input_all.append(input_.detach().cpu())

        ig_results.append(ig_batch.detach().cpu())

        if (batch_idx + 1) % 2 == 0:
            logger.info(f"Processed {batch_idx + 1} batches")

    ig_all = torch.cat(ig_results, dim=0).numpy()
    input_all = torch.cat(input_all, dim=0).numpy()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-mesogeos2/code/ml_tracks/a_fire_danger/saved/ig-plots/{model_type}/"
    save_path = os.path.join(base_save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    model_id = os.path.basename(os.path.dirname(checkpoint_path))

    ig_save_path = os.path.join(save_path, f"ig_values_{model_id}_{model_type}.npz")
    np.savez(
        ig_save_path,
        class_1=ig_all
    )

    combined_npz_path = ig_save_path.replace(".npz", "_combined.npz")
    np.savez(
        combined_npz_path,
        class_1=ig_all,
        sample_id=np.array(sample_ids_all),
        label=np.array(labels_all),
        feature_names=np.array(feature_names)
    )

    df_combined = pd.DataFrame({
        'sample_id': sample_ids_all,
        'label': labels_all,
        'x': coord_x,
        'y': coord_y,
        'probs': probs_all
    })

    ig_df = pd.DataFrame(ig_all.reshape(ig_all.shape[0], -1), columns=feature_names)
    ig_df = ig_df.add_prefix("ig_class_1_")
    df_combined = pd.concat([df_combined, ig_df], axis=1)

    combined_csv_path = ig_save_path.replace(".npz", "_combined.csv")
    df_combined.to_csv(combined_csv_path, index=False)

    # Aggregierte Map nach Basis-Feature-Namen
    base_feature_names = [f.split("_t-")[0] for f in feature_names]
    ig_df = pd.DataFrame(ig_all.reshape(ig_all.shape[0], -1), columns=feature_names)
    ig_df.columns = base_feature_names
    ig_agg = ig_df.groupby(axis=1, level=0).mean()

    df_map = pd.DataFrame({
        'sample': sample_ids_all,
        'label': labels_all,
        'x': coord_x,
        'y': coord_y,
        'probs': probs_all
    })
    df_map = pd.concat([df_map, ig_agg], axis=1)
    map_csv_path = os.path.join(save_path, f"ig_map_{model_type}.csv")
    df_map.to_csv(map_csv_path, index=False)

    np.save(ig_save_path.replace(".npz", "_input.npy"), input_all)

    logger.info(f"Saved IG NPZ to: {ig_save_path}")
    logger.info(f"Saved combined CSV to: {combined_csv_path}")
    logger.info(f"Saved IG map CSV to: {map_csv_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Compute Integrated Gradients')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
