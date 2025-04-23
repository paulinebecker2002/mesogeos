import argparse
import collections
import torch
import numpy as np
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import matplotlib.pyplot as plt
import shap


def main(config):
    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    logger = config.get_logger('train')

    # setup datasets instances
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = {}
    dataloader = {}
    for mode in ['train', 'val']:
        dataset[mode] = config.init_obj('dataset', module_data,
                                        dynamic_features=dynamic_features, static_features=static_features,
                                        train_val_test=mode)
        dataloader[mode] = config.init_obj('dataloader', module_dataloader, dataset=dataset[mode]).dataloader()

    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    # build models architecture, then print to console
    if config["model_type"] == "lstm":
        model = config.init_obj('arch', module_arch, input_dim=len(dynamic_features) + len(static_features),
                                output_lstm=config['model_args']['dim'], dropout=config['model_args']['dropout'])

    elif config["model_type"] == "transformer":
        model = config.init_obj('arch', module_arch, seq_len=config["dataset"]["args"]["lag"],
                                input_dim=len(dynamic_features) + len(static_features),
                                d_model=config['model_args']['model_dim'],
                                nhead = config['model_args']['nheads'],
                                dim_feedforward=config['model_args']['ff_dim'],
                                num_layers=config['model_args']['num_layers'],
                                channel_attention=False)

    elif config["model_type"] == "gtn":
        model = config.init_obj('arch', module_arch, seq_len=config["dataset"]["args"]["lag"],
                                input_dim=len(dynamic_features) + len(static_features),
                                d_model=config['model_args']['model_dim'],
                                nhead=config['model_args']['nheads'],
                                dim_feedforward=config['model_args']['ff_dim'],
                                num_layers=config['model_args']['num_layers'],
                                channel_attention=True)
    elif config["model_type"] == "mlp":
        model = config.init_obj('arch', module_arch,
                                input_dim=(len(dynamic_features) + len(static_features))*(config["dataset"]["args"]["lag"]),
                                dropout=config['model_args']['dropout'],
                                hidden_dims=config['model_args']['hidden_dims'],
                                output_dim=config['model_args']['output_dim'])

    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=dataloader['train'],
                      valid_data_loader=dataloader['val'],
                      lr_scheduler=lr_scheduler)

    trainer.train()


    #SHAP values
    if config["model_type"] == "mlp":
        model.eval()
        batch = next(iter(dataloader['val']))
        dynamic, static, _, labels = batch
        static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
        input_ = torch.cat([dynamic, static], dim=2)
        input_ = input_.view(input_.shape[0], -1).to(device)

        def model_predict(x_numpy):
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(x_tensor)
                return out.cpu().numpy()

        explainer = shap.GradientExplainer(model, input_[:100])
        shap_values = explainer.shap_values(input_[:10])
        shap.summary_plot(shap_values[0], input_[:10].cpu().numpy(), feature_names=[f'f{i}' for i in range(input_.shape[1])])
        plt.tight_layout()
        plt.savefig("/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/shap_summary_plot.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--dr', '--dropout'], type=float, target='model_args;dropout'),
        CustomArgs(['--hd', '--hidden-dims'], type=lambda s: [int(x) for x in s.split(',')], target='model_args;hidden_dims'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)