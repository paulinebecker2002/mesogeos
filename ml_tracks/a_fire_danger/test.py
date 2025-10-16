import argparse
import collections
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from utils import MetricTracker, build_model
from logger import TensorboardWriter
from tester.test_rf import test_rf
import torch.nn as nn
from utils import grouped_classification_metrics, calculate_metrics


def main(config):


    logger = config.get_logger('test')

    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = config.init_obj('dataset', module_data,
                              dynamic_features=dynamic_features, static_features=static_features,
                              train_val_test='test')
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    # device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    device = 'cpu'
    # build models architecture
    if config["model_type"] == "rf":
        test_rf(config)
        return
    else:
        model = build_model(config, dynamic_features, static_features)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config["model_path"]))
    logger.info(f"costal_only: {config['dataset']['args'].get('coastal_only', False)}")
    logger.info(f"inland_only: {config['dataset']['args'].get('inland_only', False)}")
    logger.info(f"Seed:          {config['dataset']['args'].get('seed')}")
    logger.info(f"Pos source:          {config['dataset']['args'].get('pos_source', 'all')}")
    logger.info(f"Neg source:          {config['dataset']['args'].get('neg_source', 'all')}")

    checkpoint = torch.load(config["model_path"], map_location=device)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    e = 0.000001
    cfg_trainer = config['trainer']
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns], writer=writer)
    test_metrics.reset()

    all_probs, all_lats, all_lons, all_samples, all_labels, all_bas, all_preds, all_coastal = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            (dynamic, static, bas_size, labels, x, y, sample_id, coastal) = batch[:8]
            if config['model_type'] == 'tft':
                dynamic = dynamic.to(device, dtype=torch.float32)
                static = static.to(device, dtype=torch.float32)
                input_ = (dynamic, static)
            else:
                static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
                input_ = torch.cat([dynamic, static], dim=2)
                input_ = input_.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            bas_size = bas_size.to(device, dtype=torch.float32)
            # bas_size=1

            if config['model_type'] in ['transformer', 'gtn']:
                input_ = torch.transpose(input_, 0, 1)
            elif config['model_type'] == 'mlp':
                input_ = input_.view(input_.shape[0], -1)
            if config['model_type'] == 'tft':
                outputs = model(dynamic, static)
            else:
                outputs = model(input_)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)

            softmax_probs = outputs[:, 1].detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            bas_np = bas_size.detach().cpu().numpy()
            sample_ids = sample_id.cpu().numpy().astype(int)
            coastal_np = coastal.detach().cpu().numpy().astype(int)

            output = torch.argmax(outputs, dim=1)
            predictions_np = output.detach().cpu().numpy()

            all_preds.extend(predictions_np)
            all_probs.extend(softmax_probs)
            all_lats.extend(x)
            all_lons.extend(y)
            all_samples.extend(sample_ids)
            all_labels.extend(labels_np)
            all_bas.extend(bas_np)
            all_coastal.extend(coastal_np)

            loss = criterion(torch.log(outputs + e), labels)
            loss = torch.mean(loss * bas_size)



            writer.set_step(batch_idx)
            test_metrics.update('loss', loss.item() * dynamic.size(0), dynamic.size(0))

            for met in metric_ftns:
                if met.__name__ not in ['aucpr']:
                    test_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                elif met.__name__ == 'aucpr':
                    test_metrics.aucpr_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                    met(outputs[:, 1], labels)[1])

    log = test_metrics.result()
    logger.info(log)

    y_true  = np.asarray(all_labels, dtype=int)
    y_pred  = np.asarray(all_preds, dtype=int)
    y_proba = np.asarray(all_probs, dtype=float)
    g       = np.asarray(all_coastal, dtype=int)  # 0 = inland, 1 = coastal

    metrics = grouped_classification_metrics(y_true, y_pred, y_proba, g, positive_group=1)

    # Schönes Logging
    logger.info(
        "[OVERALL] acc={acc:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f} auprc={auprc:.4f}"
        .format(**metrics['overall'])
    )
    for name in ['coastal', 'inland']:
        m = metrics[name]
        logger.info(
            f"[{name.upper()}] acc={m['acc']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"f1={m['f1']:.4f} auprc={m['auprc']:.4f}"
        )

    summary_path = Path(config.save_dir) / f"group_metrics_{config['model_type']}.csv"
    (pd.DataFrame(metrics).T).to_csv(summary_path)
    logger.info(f"Saved grouped metrics to: {summary_path}")


    df = pd.DataFrame({
        'prob': all_probs,
        'lat': all_lats,
        'lon': all_lons,
        'sample_id': all_samples,
        'label': all_labels,
        'pred_label': all_preds,
        'log_burned_area': all_bas,
        'coastal': all_coastal,
    })
    output_path = Path(config.save_dir) / f"test_softmax_outputs_{config['model_type']}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved test softmax predictions with coordinates to: {output_path}")
    samples_path = Path(config.save_dir) / f"samples.npy"
    np.save(samples_path, np.array(all_samples, dtype=np.int32))
    logger.info(f"Saved test sample IDs to: {samples_path}")



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target nargs')
    options = [
        CustomArgs(['--model_path', '--mp'], type=str, target='model_path', nargs=None),
        CustomArgs(['--tlag', '--last_n_timesteps'], type=int, target='dataset;args;last_n_timesteps', nargs=None),
        CustomArgs(['--train_year'], type=str, nargs='+', target='dataset;args;train_year'),
        CustomArgs(['--val_year'], type=str, nargs='+', target='dataset;args;val_year'),
        CustomArgs(['--test_year'], type=str, nargs='+', target='dataset;args;test_year'),
        CustomArgs(['--seed'], type=int, target='dataset;args;seed', nargs=None),
        CustomArgs(['--pos_source'], type=str, target='dataset;args;pos_source', nargs=None),
        CustomArgs(['--neg_source'], type=str, target='dataset;args;neg_source', nargs=None),
        CustomArgs(['--coastal_only'], type=bool, target='dataset;args;coastal_only', nargs=None),
        CustomArgs(['--inland_only'], type=bool, target='dataset;args;inland_only', nargs=None),
        CustomArgs(['--n_test_pos', '--ntestp'], type=int, target='dataset;args;n_test_pos', nargs=None),

    ]
    config = ConfigParser.from_args(args, options)
    main(config)
