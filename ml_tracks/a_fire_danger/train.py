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
import optuna
from trainer.trainer_rf import optuna_rf
#from trainer.trainer_tune_rf import train_rf
from trainer.trainer_rf import train_rf
from utils.util import set_seed, build_model, get_dataloader


def main(config):

    SEED = config['seed']
    set_seed(SEED)

    logger = config.get_logger('train')
    logger.info(f"Save directory: {config.save_dir}")
    logger.info(f"Learning rate:       {config['optimizer']['args'].get('lr')}")
    logger.info(f"Dropout:             {config['model_args'].get('dropout')}")
    logger.info(f"Batch size:          {config['dataloader']['args'].get('batch_size')}")
    logger.info(f"Hidden dims:         {config['model_args'].get('hidden_dims')}")
    logger.info(f"Epochs:              {config['trainer'].get('epochs')}")

    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataloader = {
        "train": get_dataloader(config, static_features, dynamic_features, mode='train'),
        "val": get_dataloader(config, static_features, dynamic_features, mode='val'),
    }

    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])

    if config["model_type"] == "rf":
        # separate training process as Random Forest is not a torch mowas isdel
        train_rf(config, dataloader['train'], dataloader['val'])
        return

        def optuna_objective(trial):
            return optuna_rf(trial, config, dataloader['train'], dataloader['val'])

        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=30)

        logger.info("Best trial:")
        logger.info(study.best_trial)
        logger.info(f"Best params: {study.best_trial.params}")
        return

    # Model setup
    model = build_model(config, dynamic_features, static_features)
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
        CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;weight_decay'),
        CustomArgs(['--ft', '--finetune'], type=str, target='finetune;sklearn_tune'),

    ]
    config = ConfigParser.from_args(args, options)
    main(config)