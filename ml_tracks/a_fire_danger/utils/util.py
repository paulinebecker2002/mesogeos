import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import models.model as module_arch
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use, num_device):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:' + str(num_device) if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def extract_numpy(dataloader):
    X_all, y_all = [], []
    for batch in dataloader:
        dynamic, static, bas_size, labels = batch[:4]
        static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
        input_ = torch.cat([dynamic, static], dim=2)
        input_ = input_.view(input_.shape[0], -1).numpy()
        X_all.append(input_)
        y_all.append(labels.numpy().astype(int))
    return np.vstack(X_all), np.concatenate(y_all)

def calculate_metrics(y_values, y_pred, y_proba):
    acc = accuracy_score(y_values, y_pred)
    prec = precision_score(y_values, y_pred)
    rec = recall_score(y_values, y_pred)
    f1 = f1_score(y_values, y_pred)
    auprc = average_precision_score(y_values, y_proba)
    return acc, prec, rec, f1, auprc

def build_model(config, dynamic_features, static_features):
    dynamic_dim = len(dynamic_features)
    static_dim = len(static_features)
    model_type = config["model_type"]
    only_last_five = config["dataset"]["args"].get("only_last_five", False)
    seq_len = 5 if only_last_five else config["dataset"]["args"].get("lag", 30)


    if model_type == "mlp":
        return config.init_obj('arch', module_arch,
                               input_dim=(dynamic_dim + static_dim) * seq_len,
                               dropout=config['model_args']['dropout'],
                               hidden_dims=config['model_args']['hidden_dims'],
                               output_dim=config['model_args']['output_dim'])

    elif model_type == "lstm":
        return config.init_obj('arch', module_arch,
                               input_dim=dynamic_dim + static_dim,
                               output_lstm=config['model_args']['dim'],
                               dropout=config['model_args']['dropout'])

    elif model_type == "transformer":
        return config.init_obj('arch', module_arch,
                               seq_len=seq_len,
                               input_dim=dynamic_dim + static_dim,
                               d_model=config['model_args']['model_dim'],
                               nhead=config['model_args']['nheads'],
                               dim_feedforward=config['model_args']['ff_dim'],
                               num_layers=config['model_args']['num_layers'],
                               channel_attention=False)

    elif model_type == "gtn":
        return config.init_obj('arch', module_arch,
                               seq_len=seq_len,
                               input_dim=dynamic_dim + static_dim,
                               d_model=config['model_args']['model_dim'],
                               nhead=config['model_args']['nheads'],
                               dim_feedforward=config['model_args']['ff_dim'],
                               num_layers=config['model_args']['num_layers'],
                               channel_attention=True)

    elif model_type == "cnn":
        return config.init_obj('arch', module_arch,
                               input_channels=config['model_args']['input_channels'],
                               seq_len=seq_len,
                               num_features=dynamic_dim + static_dim,
                               dim=config['model_args']['dim'],
                               dropout=config['model_args']['dropout'])

    elif model_type == "gru":
        return config.init_obj('arch', module_arch,
                               input_dim=dynamic_dim + static_dim,
                               output_gru=config['model_args']['dim'],
                               dropout=config['model_args']['dropout'])

    elif model_type == "tft":
        return config.init_obj('arch', module_arch,
                               input_dim=dynamic_dim,
                               static_dim=static_dim,
                               seq_len=seq_len,
                               d_model=config['model_args']['model_dim'],
                               nhead=config['model_args']['nheads'],
                               num_layers=config['model_args']['num_layers'],
                               dropout=config['model_args']['dropout'])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_dataloader(config, static_features, dynamic_features, mode='val'):
    dataset = config.init_obj('dataset', module_data,
                              dynamic_features=dynamic_features,
                              static_features=static_features,
                              train_val_test=mode)
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()
    return dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_feature_names(config):
    dynamic = config['features']['dynamic']
    static = config['features']['static']
    lag = config['dataset']['args']['lag']

    feature_names = []
    for t in range(lag):
        for name in dynamic:
            feature_names.append(f"{name}_t-{lag - t}")
        for name in static:
            feature_names.append(f"{name}_t-{lag - t}")

    return feature_names



class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for ind in self._data.index:
            for col in self._data.columns:
                if ind not in ['auc', 'aucpr', 'ece', 'spread_skill']:
                    self._data.loc[ind][col] = 0
                else:
                    self._data.loc[ind][col] = []

    def update(self, key, nominator, denominator):
        if self.writer is not None:
            self.writer.add_scalar(key, (nominator / (denominator + 1e-8)), denominator) #self.writer.add_scalar(key, nominator, denominator)
        self._data.total[key] += nominator
        self._data.counts[key] += denominator
        self._data.average[key] = self._data.total[key] / self._data.counts[key]


    def aucpr_update(self, key, preds, labels):
        ap = average_precision_score(labels, preds) #new
        if self.writer is not None:
            self.writer.add_scalar(key, ap) #old: self.writer.add_scalar(key, preds, labels)
        self._data.total[key].extend(preds)
        self._data.counts[key].extend(labels)
        self._data.average[key] = average_precision_score(self._data.counts[key], self._data.total[key])

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    @property
    def data(self):
        return self._data


