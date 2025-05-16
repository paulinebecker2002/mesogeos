import argparse
import collections
import optuna
import copy
import json
import torch
from parse_config import ConfigParser
from utils import prepare_device
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.model as module_arch
import models.loss as module_loss
import models.metric as module_metric
from trainer import Trainer

def objective(trial, base_config):
    # deepcopy to avoid side effects
    config = copy.deepcopy(base_config)

    # Sample hyperparameters
    hidden_dims = trial.suggest_categorical('hidden_dims', [[128, 64], [256, 128], [512, 256]])
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)


# Update config dynamically
    config['optimizer']['args']['lr'] = lr
    config['model_args']['dropout'] = dropout
    config['model_args']['hidden_dims'] = hidden_dims
    config['dataloader']['args']['batch_size'] = batch_size
    config['optimizer']['args']['weight_decay'] = weight_decay
    config['lr_scheduler']['args']['gamma'] = gamma

    # Data setup
    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = {}
    dataloader = {}
    for mode in ['train', 'val']:
        dataset[mode] = config.init_obj('dataset', module_data,
                                        dynamic_features=dynamic_features, static_features=static_features,
                                        train_val_test=mode)
        dataloader[mode] = config.init_obj('dataloader', module_dataloader, dataset=dataset[mode]).dataloader()

    # Device setup
    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])

    # Model selection based on config["model_type"]
    if config["model_type"] == "lstm":
        model = config.init_obj('arch', module_arch,
                                input_dim=len(dynamic_features) + len(static_features),
                                output_lstm=config['model_args']['dim'],
                                dropout=config['model_args']['dropout'])

    elif config["model_type"] == "transformer":
        model = config.init_obj('arch', module_arch, seq_len=config["dataset"]["args"]["lag"],
                                input_dim=len(dynamic_features) + len(static_features),
                                d_model=config['model_args']['model_dim'],
                                nhead=config['model_args']['nheads'],
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
                                input_dim=(len(dynamic_features) + len(static_features)) * (config["dataset"]["args"]["lag"]),
                                dropout=config['model_args']['dropout'],
                                hidden_dims=config['model_args']['hidden_dims'],
                                output_dim=config['model_args']['output_dim'])

    elif config["model_type"] == "gru":
        model = config.init_obj('arch', module_arch,
                                input_dim=len(dynamic_features) + len(static_features),
                                output_gru=config['model_args']['dim'],
                                dropout=config['model_args']['dropout'])

    elif config["model_type"] == "cnn":
        model = config.init_obj('arch', module_arch,
                                input_channels=config["model_args"]["input_channels"],
                                seq_len=config["dataset"]["args"]["lag"],
                                num_features=len(dynamic_features) + len(static_features),
                                dim=config["model_args"]["dim"],
                                dropout=config["model_args"]["dropout"])

    elif config["model_type"] == "rf":
        from trainer.trainer_rf import train_rf
        train_rf(config, dataloader['train'], dataloader['val'])
        return 0.0  # Dummy value for Optuna

    # Device handling
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Loss, metrics, optimizer, scheduler
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Trainer setup
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=dataloader['train'],
                      valid_data_loader=dataloader['val'],
                      lr_scheduler=lr_scheduler)

    # Train
    log = trainer.train()

    # Report val_aucpr to Optuna (maximize it)
    #val_aucpr = log.get('val_aucpr', 0.0)
    #val_f1 = log.get('val_f1_score', 0.0)
    #val_f1 = trainer.best_val_f1
    val_aucpr = trainer.best_val_aucpr

    # Pruning support
    trial.report(val_aucpr, step=trainer.epochs)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_aucpr



if __name__ == "__main__":
    # Argumente wie in train.py
    args = argparse.ArgumentParser(description='Optuna Tuning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--dr', '--dropout'], type=float, target='model_args;dropout'),
        CustomArgs(['--hd', '--hidden-dims'], type=lambda s: [int(x) for x in s.split(',')], target='model_args;hidden_dims'),
        CustomArgs(['--ft', '--finetune'], type=str, target='finetune;sklearn_tune'),
    ]

    config = ConfigParser.from_args(args, options)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, config), n_trials=30)


    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    with open('best_trial.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
