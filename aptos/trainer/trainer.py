import numpy as np
import torch
from torchvision.utils import make_grid

from aptos.base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, resume, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_loader)))

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            acc_metrics = []
            for i, metric in enumerate(self.metrics):
                score = metric(output, target)
                acc_metrics.append(score)
                if not hasattr(score, '__len__'):  # hacky way to avoid logging conf matrix
                    self.writer.add_scalar(f'{metric.__name__}', acc_metrics[-1])
            return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        outputs = np.zeros(self.data_loader.n_samples)
        targets = np.zeros(self.data_loader.n_samples)
        for bidx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) + (bidx / len(self.data_loader)))
            # self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            bs = target.size(0)
            outputs[bidx * bs:(bidx + 1) * bs] = output.cpu().squeeze(1).detach().numpy()
            targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy()

            if bidx % self.log_step == 0:
                self._log_batch(epoch, bidx, bs, len(self.data_loader), loss.item())

        # tensorboard logging
        self.writer.set_step(epoch - 1)
        # self.writer.add_scalar('model/lr', self.optimizer.model_lr)
        # self.writer.add_scalar('loss/lr', self.optimizer.loss_lr)
        # self.writer.add_scalar('loss/alpha', self.loss.alpha())
        # self.writer.add_scalar('loss/scale', self.loss.scale())
        # for name, param in self.loss.named_parameters():
        #     if param.requires_grad:
        #         self.writer.add_scalar(f'loss/{name}', param[0, 0])
        if epoch == 1:  # only log images once to save time
            self.writer.add_image(
                'input', make_grid(data.cpu(), nrow=8, normalize=True))

        total_metrics = self._eval_metrics(outputs, targets)
        total_loss /= len(self.data_loader)
        self.writer.add_scalar('total_loss', total_loss)
        log = {
            'loss': total_loss,
            'metrics': total_metrics
        }

        if self.do_validation:
            self.logger.debug('Starting validation...')
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _log_batch(self, epoch, batch_idx, batch_size, n_batches, loss):
        n_complete = batch_idx * batch_size
        n_samples = batch_size * n_batches
        percent = 100.0 * batch_idx / n_batches
        msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] Loss: {loss:.6f}'
        self.logger.debug(msg)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        outputs = np.zeros(self.data_loader.n_samples)
        targets = np.zeros(self.data_loader.n_samples)
        with torch.no_grad():
            for bidx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
                bs = target.size(0)
                outputs[bidx * bs:(bidx + 1) * bs] = output.cpu().squeeze(1).detach().numpy()
                targets[bidx * bs:(bidx + 1) * bs] = target.cpu().detach().numpy()

        self.writer.set_step((epoch - 1), 'valid')
        total_val_metrics = self._eval_metrics(outputs, targets)
        total_val_loss /= len(self.valid_data_loader)
        self.writer.add_scalar('total_loss', total_val_loss)
        return {
            'val_loss': total_val_loss,
            'val_metrics': total_val_metrics
        }
