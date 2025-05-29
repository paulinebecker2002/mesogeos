import argparse
import collections
import torch
import shap
import numpy as np
import os
import pandas as pd
from parse_config import ConfigParser
from utils.util import (set_seed, build_model, get_dataloader, prepare_device)
from shap_utils import get_feature_names

def get_shap_explanation(model, model_type, input_all, device, seq_len, static_features, dynamic_features, logger=None):
    """
    Select the appropriate SHAP explainer for the given model type and compute SHAP values.
    """
    n_features = len(static_features) + len(dynamic_features)
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

        background = input_all[:100].detach().cpu().numpy()
        test_input = input_all[:100].detach().cpu().numpy()

        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_input, nsamples=1000)

    elif model_type == 'tft':
        if logger:
            logger.info("Using KernelExplainer for TFT (two-input model)")

        # Split input_all into dynamic and static
        dynamic_input = input_all[:, :, :-len(static_features)]
        static_input = input_all[:, 0, -len(static_features):].unsqueeze(1).repeat(1, seq_len, 1)

        # Define wrapper for SHAP
        def model_wrapper(inputs):
            dyn, stat = inputs
            dyn_tensor = torch.from_numpy(dyn).float().to(device)
            stat_tensor = torch.from_numpy(stat).float().to(device)
            with torch.no_grad():
                return model(dyn_tensor, stat_tensor).cpu().numpy()

        # Prepare background and test data
        background = (
            dynamic_input[:100].detach().cpu().numpy(),
            static_input[:100].detach().cpu().numpy()
        )
        test_input = (
            dynamic_input[:10].detach().cpu().numpy(),
            static_input[:10].detach().cpu().numpy()
        )

        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_input, nsamples="auto")

    elif model_type in ['lstm', 'gru', 'cnn', 'rf', 'transformer']:
        if logger:
            logger.info(f"Using KernelExplainer for model_type='{model_type}'")

        def model_wrapper(x_numpy):
            x_tensor = torch.from_numpy(x_numpy).float().to(device)
            if model_type in ['lstm', 'gru', 'cnn']:
                # Reshape for LSTM/GRU: [batch_size, seq_len, n_features]
                x_tensor = x_tensor.view(x_tensor.shape[0], seq_len, n_features)
            elif model_type == 'transformer':
                x_tensor = x_tensor.transpose(0, 1)


            with torch.no_grad():
                output = model(x_tensor)
                output = torch.softmax(output, dim=1)
            return output.cpu().numpy()

        background = input_all[:100].detach().cpu().numpy().reshape(100, -1)
        test_input = input_all[:100].detach().cpu().numpy().reshape(100, -1)

        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_input, nsamples=1000) #test only nsamples of feature-kombinations
        print(f"SHAP: {np.array(shap_values).shape}")

    else:
        raise NotImplementedError(f"SHAP explanation not implemented for model_type='{model_type}'")

    return shap_values
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

    model = build_model(config, dynamic_features, static_features)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    all_inputs = []
    for batch in dataloader:
        dynamic, static, bas_size, labels = batch[:4]
        static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
        input_ = torch.cat([dynamic, static], dim=2)
        if config["model_type"] == "mlp":
            input_ = input_.view(input_.shape[0], -1)
        all_inputs.append(input_)

    input_all = torch.cat(all_inputs, dim=0).to(device).float()
    print(type(input_all), input_all.shape)
    logger.info(f"Computing SHAP values on {input_all.shape[0]} samples...")

    shap_values = get_shap_explanation(model, model_type, input_all, device, seq_len, static_features, dynamic_features, logger)

    #store SHAP values
    model_id = os.path.basename(os.path.dirname(checkpoint_path))
    shap_save_path = os.path.join(base_save_path, model_id, f"shap_values_{model_id}_{model_type}.npz")
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)

    np.savez(
        shap_save_path,
        class_0=shap_values[0],
        class_1=shap_values[1]
    )

    pd.DataFrame(shap_values[0], columns=feature_names).to_csv(shap_save_path.replace(".npz", "_class0.csv"), index=False)
    pd.DataFrame(shap_values[1], columns=feature_names).to_csv(shap_save_path.replace(".npz", "_class1.csv"), index=False)
    np.save(shap_save_path.replace(".npz", "_input.npy"), input_all.cpu().numpy())

    if logger:
        logger.info(f"SHAP values saved at: {shap_save_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Compute SHAP values')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to trained checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)