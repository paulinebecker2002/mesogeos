import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import pandas as pd
from pathlib import Path


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.e = 0.000001
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.best_val_f1 = 0.0
        self.best_val_aucpr = 0.0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, batch in enumerate(self.data_loader):
            (dynamic, static, bas_size, labels) = batch[:4]
            if self.config['model_type'] == 'tft':
                dynamic = dynamic.to(self.device, dtype=torch.float32)
                static = static.to(self.device, dtype=torch.float32)
                input_ = (dynamic, static)
            else:
                static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
                input_ = torch.cat([dynamic, static], dim=2)
                input_ = input_.to(self.device, dtype=torch.float32)

            labels = labels.to(self.device, dtype=torch.long)
            bas_size = bas_size.to(self.device, dtype=torch.float32)
            # bas_size=1
            self.optimizer.zero_grad()
            if self.config['model_type'] in ['transformer', 'gtn']:
                input_ = torch.transpose(input_, 0, 1)
            elif self.config['model_type'] == 'mlp':
                input_ = input_.view(input_.shape[0], -1)
            if self.config['model_type'] == 'tft':
                outputs = self.model(*input_)
            else:
                outputs = self.model(input_)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)

            loss = self.criterion(torch.log(outputs + self.e), labels)
            loss = torch.mean(loss * bas_size)

            loss.backward()
            self.optimizer.step()

            output = torch.argmax(outputs, dim=1)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item()*dynamic.size(0), dynamic.size(0))

            for met in self.metric_ftns:
                if met.__name__ not in ['aucpr']:
                    self.train_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                elif met.__name__ == 'aucpr':
                    self.train_metrics.aucpr_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                        met(outputs[:, 1], labels)[1])

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.val_outputs = []

        with torch.no_grad():
            for batch_idx, (dynamic, static, bas_size, labels, x, y) in enumerate(self.valid_data_loader):
                if self.config['model_type'] == 'tft':
                    dynamic = dynamic.to(self.device, dtype=torch.float32)
                    static = static.to(self.device, dtype=torch.float32)
                    input_ = (dynamic, static)
                else:
                    static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
                    input_ = torch.cat([dynamic, static], dim=2)
                    input_ = input_.to(self.device, dtype=torch.float32)

                labels = labels.to(self.device, dtype=torch.long)
                bas_size = bas_size.to(self.device, dtype=torch.float32)
                # bas_size=1

                if self.config['model_type'] in ['transformer', 'gtn']:
                    input_ = torch.transpose(input_, 0, 1)
                elif self.config['model_type'] == 'mlp':
                    input_ = input_.view(input_.shape[0], -1)
                if self.config['model_type'] == 'tft':
                    outputs = self.model(*input_)  # dynamic, static getrennt übergeben
                else:
                    outputs = self.model(input_)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)

                softmax_probs = outputs[:, 1].detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                self.val_outputs.append((softmax_probs, x, y))

                loss = self.criterion(torch.log(outputs + self.e), labels)
                loss = torch.mean(loss * bas_size)

                output = torch.argmax(outputs, dim=1)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item()*dynamic.size(0), dynamic.size(0))


                for met in self.metric_ftns:
                    if met.__name__ not in ['aucpr']:
                        self.valid_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                    elif met.__name__ == 'aucpr':
                        self.valid_metrics.aucpr_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                        met(outputs[:, 1], labels)[1])


        current_val_f1 = self.valid_metrics.result().get('f1_score', 0.0)
        if current_val_f1 > self.best_val_f1:
            self.best_val_f1 = current_val_f1

        current_val_aucpr = self.valid_metrics.result().get('aucpr', 0.0)
        if current_val_aucpr > self.best_val_aucpr:
            self.best_val_aucpr = current_val_aucpr

        # add histogram of models parameters to the tensorboard
        #and self.config["model_type"] != "transformer"
        if epoch % 2 == 0 and (self.config["model_type"] != "tft" and self.config["model_type"] != "transformer") and self.config["model_type"] != "gtn":
            for name, p in self.model.named_parameters():
                if p is not None and p.numel() > 0:
                    self.writer.add_histogram(name, p, bins='auto')

        all_probs, all_lats, all_lons = zip(*self.val_outputs)
        df = pd.DataFrame({
            'prob': np.concatenate(all_probs),
            'lat': np.concatenate(all_lats),
            'lon': np.concatenate(all_lons)
        })
        output_path = Path(self.config.save_dir) / f"val_softmax_outputs_epoch{epoch}.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved softmax predictions with coordinates to: {output_path}")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
