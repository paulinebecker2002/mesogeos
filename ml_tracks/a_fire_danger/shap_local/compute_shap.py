import argparse
import collections
import torch
import shap
import numpy as np
from datetime import datetime
import os
import pandas as pd
import joblib
from parse_config import ConfigParser
from utils.util import (set_seed, build_model, get_dataloader, prepare_device, get_feature_names)
def get_shap_explanation(model, model_type, input_all, device, seq_len, static_features, dynamic_features, logger=None):
    """
    Select the appropriate SHAP explainer for the given model type and compute SHAP values.
    """
    n_features = len(static_features) + len(dynamic_features)
    print(f"Dimensionen von input_all: {input_all.shape}")
    #if model_type in ['mlp']:
     #   explainer = shap.DeepExplainer(model, input_all[:100])
      #  shap_values = explainer.shap_values(input_all[:100])

    if model_type in ['mlp']:
        if logger:
            logger.info(f"Using KernelExplainer for model_type='{model_type}'")
            print(f"input {input_all.shape}, seq_len={seq_len}, n_features={n_features}")
        # Wrapper to convert input to PyTorch tensor, run the model and convert output back to numpy
        def model_wrapper(x_numpy):
            x_tensor = torch.from_numpy(x_numpy).float().to(device)
            with torch.no_grad():
                output = model(x_tensor)
                output = torch.softmax(output, dim=1)
            return output.cpu().numpy()

        np.random.seed(42)
        indices = np.random.choice(len(input_all), 100, replace=False)
        background = input_all[indices].detach().cpu().numpy()
        test_input = input_all.detach().cpu().numpy()
        #test_input = input_all[:10].detach().cpu().numpy()

        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_input, nsamples=1000)

    elif model_type in ['tft']:
        n_background = 50
        batch_size = 1000

        def model_wrapper(x_numpy_flat):
            x_numpy = x_numpy_flat.reshape(-1, seq_len, len(dynamic_features) + len(static_features))
            dyn = x_numpy[:, :, :-len(static_features)]
            stat = x_numpy[:, 0, -len(static_features):]  # -> [B, F_static]

            dyn_tensor = torch.from_numpy(dyn).float().to(device)
            stat_tensor = torch.from_numpy(stat).float().to(device)
            with torch.no_grad():
                output = model(dyn_tensor, stat_tensor)
                output = torch.softmax(output, dim=1)
            return output.cpu().numpy()

        background = input_all[:n_background].detach().cpu().numpy().reshape(n_background, -1)
        test_input = input_all.detach().cpu().numpy().reshape(input_all.shape[0], -1)
        #test_input = input_all[:batch_size].detach().cpu().numpy().reshape(batch_size, -1)

        explainer = shap.KernelExplainer(model_wrapper, background)

        shap_values = [[], []]
        for i in range(0, test_input.shape[0], batch_size):
            batch = test_input[i:i + batch_size]
            sv = explainer.shap_values(batch, nsamples=1000)
            shap_values[0].append(sv[0])
            shap_values[1].append(sv[1])

        # Zu einer Matrix zusammenfügen
        shap_values[0] = np.concatenate(shap_values[0], axis=0)
        shap_values[1] = np.concatenate(shap_values[1], axis=0)

    elif model_type in ['lstm', 'gru', 'cnn', 'transformer', 'gtn']:
        if logger:
            logger.info(f"Using KernelExplainer for model_type='{model_type}'")

        def model_wrapper(x_numpy):
            x_tensor = torch.from_numpy(x_numpy).float().to(device)
            if model_type in ['lstm', 'gru', 'cnn']:
                # Reshape for LSTM/GRU: [batch_size, seq_len, n_features]
                x_tensor = x_tensor.view(x_tensor.shape[0], seq_len, n_features)
            if model_type in ['transformer', 'gtn']:
                x_tensor = x_tensor.reshape(-1, seq_len, len(dynamic_features) + len(static_features))
                x_tensor = x_tensor.permute(1, 0, 2)  # -> [seq_len, batch_size, features]
            with torch.no_grad():
                output = model(x_tensor)
                output = torch.softmax(output, dim=1)
            return output.cpu().numpy()

        np.random.seed(42)
        indices = np.random.choice(len(input_all), 100, replace=False)
        background = input_all[indices].detach().cpu().numpy().reshape(100, -1)
        #background = input_all[:10].detach().cpu().numpy().reshape(10, -1)
        test_input = input_all.detach().cpu().numpy().reshape(input_all.shape[0], -1)
        #test_input = input_all[:10].detach().cpu().numpy().reshape(10, -1)

        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_input, nsamples=1000) #test only nsamples of feature-kombinations
        print(f"SHAP: {np.array(shap_values).shape}")
    else:
        raise NotImplementedError(f"SHAP explanation not implemented for model_type='{model_type}'")

    return shap_values

def generate_rf_shap_values(model, dataloader, base_save_path, model_id, feature_names, logger=None):
    """
    Compute and save SHAP values for a Random Forest model using TreeExplainer,
    and output the same files as for deep learning models.
    """
    X_val_list, y_val_list, x_coords, y_coords, sample_ids = [], [], [], [], []

    for batch in dataloader:
        dynamic, static, bas_size, labels, x, y, sample_id = batch[:7]
        static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
        input_ = torch.cat([dynamic, static], dim=2)
        input_ = input_.view(input_.shape[0], -1)

        X_val_list.append(input_.cpu().numpy())
        y_val_list.append(labels.cpu().numpy())
        x_coords.extend(x.cpu().numpy())
        y_coords.extend(y.cpu().numpy())
        sample_ids.extend(sample_id.cpu().numpy().astype(int))

    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
    coords_x = np.array(x_coords)
    coords_y = np.array(y_coords)
    sample_ids = np.array(sample_ids)

    # SHAP Werte berechnen
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)  # [class_0, class_1]

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    shap_dir = os.path.join(base_save_path, timestamp)
    os.makedirs(shap_dir, exist_ok=True)

    # Save .npz (class only)
    shap_save_path = os.path.join(shap_dir, f"shap_values_{model_id}_rf.npz")
    np.savez(shap_save_path, class_0=shap_values[0], class_1=shap_values[1])
    if logger:
        logger.info(f"[RF] SHAP values saved at: {shap_save_path}")

    # Save combined .npz
    combined_npz_path = shap_save_path.replace(".npz", "_combined.npz")
    np.savez(
        combined_npz_path,
        class_0=shap_values[0],
        class_1=shap_values[1],
        sample_id=sample_ids,
        label=y_val,
        feature_names=np.array(feature_names)
    )
    if logger:
        logger.info(f"[RF] Combined SHAP NPZ saved at: {combined_npz_path}")

    # Save combined CSV (only class 1)
    df_combined = pd.DataFrame({
        'sample_id': sample_ids,
        'label': y_val,
        'x': coords_x,
        'y': coords_y
    })
    shap_class1_df = pd.DataFrame(shap_values[1], columns=feature_names)
    shap_class1_df = shap_class1_df.add_prefix("shap_")
    df_combined = pd.concat([df_combined, shap_class1_df], axis=1)

    combined_csv_path = shap_save_path.replace(".npz", "_combined.csv")
    df_combined.to_csv(combined_csv_path, index=False)
    if logger:
        logger.info(f"[RF] Combined SHAP CSV saved at: {combined_csv_path}")

    # Aggregierte SHAP Map
    base_feature_names = [name.split("_t-")[0] for name in feature_names]
    shap_agg_df = shap_class1_df.copy()
    shap_agg_df.columns = base_feature_names
    shap_agg = shap_agg_df.groupby(axis=1, level=0).mean()

    df_map = pd.DataFrame({
        'sample': sample_ids,
        'label': y_val,
        'x': coords_x,
        'y': coords_y
    })
    df_map = pd.concat([df_map, shap_agg], axis=1)

    map_csv_path = os.path.join(shap_dir, f"shap_map_rf.csv")
    df_map.to_csv(map_csv_path, index=False)
    if logger:
        logger.info(f"[RF] Aggregated SHAP map CSV saved at: {map_csv_path}")


def main(config):
    SEED = config['seed']
    set_seed(SEED)
    logger = config.get_logger('shap')

    checkpoint_path = config["shap"]["checkpoint_path"]
    static_features = config["features"]["static"]
    dynamic_features = config["features"]["dynamic"]
    seq_len = config["dataset"]["args"]["lag"]
    model_type = config["model_type"]
    base_save_path = f"/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a_fire_danger/saved/shap-plots/{model_type}/"
    feature_names = get_feature_names(config)

    dataloader = get_dataloader(config, static_features, dynamic_features, mode='test')
    device, _ = prepare_device(config['n_gpu'], config['gpu_id'])

    if model_type == 'rf':
        model = joblib.load(checkpoint_path)
        generate_rf_shap_values(
            model=model,
            dataloader=dataloader,
            base_save_path=base_save_path,
            model_id=os.path.basename(os.path.dirname(checkpoint_path)),
            feature_names=feature_names,
            logger=logger
        )
        return
    else:
        model = build_model(config, dynamic_features, static_features)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        all_inputs, coords_x, coords_y, all_labels, all_samples = [], [], [], [], []
        for batch in dataloader:
            dynamic, static, bas_size, labels, x, y, sample_id = batch[:7]
            static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            input_ = torch.cat([dynamic, static], dim=2)
            if config["model_type"] == "mlp":
                input_ = input_.view(input_.shape[0], -1)
            all_inputs.append(input_)
            coords_x.extend(x.cpu().numpy())
            coords_y.extend(y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_samples.extend(sample_id.cpu().numpy().astype(int))


        input_all = torch.cat(all_inputs, dim=0).to(device).float()
        print(type(input_all), input_all.shape)
        logger.info(f"Computing SHAP values on {input_all.shape[0]} samples...")

    shap_values = get_shap_explanation(model, model_type, input_all, device, seq_len, static_features, dynamic_features, logger)

    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    shap_save_path = os.path.join(base_save_path, timestamp, f"shap_values_{model_id}_{model_type}.npz")
    os.makedirs(os.path.dirname(shap_save_path), exist_ok=True)

    np.savez(
        shap_save_path,
        class_0=shap_values[0],
        class_1=shap_values[1]
    )


    combined_npz_path = shap_save_path.replace(".npz", "_combined.npz")
    np.savez(
        combined_npz_path,
        class_0=shap_values[0],
        class_1=shap_values[1],
        sample_id=np.array(all_samples),
        label=np.array(all_labels),
        feature_names=np.array(feature_names)
    )
    logger.info(f"Saved combined SHAP NPZ to: {combined_npz_path}")

    df_combined = pd.DataFrame({
        'sample_id': all_samples,
        'label': all_labels,
        'x': coords_x,
        'y': coords_y
    })

    for i, class_name in enumerate(['class_0', 'class_1']):
        class_df = pd.DataFrame(shap_values[i], columns=feature_names)
        class_df = class_df.add_prefix(f'shap_{class_name}_')
        df_combined = pd.concat([df_combined, class_df], axis=1)

    combined_csv_path = shap_save_path.replace(".npz", "_combined.csv")
    df_combined.to_csv(combined_csv_path, index=False)
    logger.info(f"Saved combined SHAP CSV to: {combined_csv_path}")

    base_feature_names = [name.split("_t-")[0] for name in feature_names]
    shap_df = pd.DataFrame(shap_values[1], columns=feature_names)
    shap_df.columns = base_feature_names
    shap_agg = shap_df.groupby(axis=1, level=0).mean()  # → (n_samples, n_base_features)
    df_shap = shap_agg.copy()
    df_shap['x'] = coords_x
    df_shap['y'] = coords_y
    df_shap['label'] = all_labels
    df_shap['sample'] = all_samples
    csv_save_path = os.path.join(base_save_path, timestamp, f"shap_map_{model_type}.csv")
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df_shap.to_csv(csv_save_path, index=False)

    logger.info(f"Saved SHAP value + coordinate map CSV to: {csv_save_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Compute SHAP values')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)