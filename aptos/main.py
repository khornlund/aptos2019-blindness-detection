import os
import random

from tqdm import tqdm
#from apex import amp
import numpy as np
import pandas as pd
import torch

import aptos.data_loader.data_loaders as module_data
import aptos.model.loss as module_loss
import aptos.model.metric as module_metric
import aptos.model.model as module_arch
import aptos.model.optimizer as module_optim
import aptos.model.scheduler as module_sched
from aptos.trainer import Trainer
from aptos.data_loader import PngDataLoader
from aptos.utils import setup_logger, setup_logging


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Runner:

    def train(self, config, resume):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config)
        valid_data_loader = data_loader.split_validation()

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        self.logger.debug('Getting loss and metric function handles')
        loss = get_instance(module_loss, 'loss', config, device)
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optim, 'optimizer', config, trainable_params)
        lr_scheduler = get_instance(module_sched, 'lr_scheduler', config, optimizer)

        # opt_level = config['apex']
        # self.logger.debug(f'Setting apex mixed precision to level: {opt_level}')
        # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        self.logger.debug('Initialising trainer')
        trainer = Trainer(model, loss, metrics, optimizer,
                          resume=resume,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()
        self.logger.debug('Finished!')

    def test(self, config, model_checkpoint):
        setup_logging(config)
        self.logger = setup_logger(self, config['testing']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.info(f'Using config:\n{config}')

        self.logger.debug('Getting data_loader instance')
        data_loader = PngDataLoader(
            config['testing']['data_dir'],
            batch_size=config['testing']['batch_size'],
            validation_split=0.0,
            train=True,
            alpha=None,
            img_size=config['testing']['img_size'],
            num_workers=config['testing']['num_workers'],
            verbose=config['testing']['verbose']
        )

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        opt_level = config['apex']
        self.logger.debug(f'Setting apex mixed precision to level: {opt_level}')
        model = amp.initialize(model, opt_level=opt_level)

        self.logger.debug(f'Loading checkpoint {model_checkpoint}')
        checkpoint = torch.load(model_checkpoint)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        model.eval()

        ensemble_size = config['testing']['ensemble_size']
        pred_df = pd.DataFrame({'id_code': data_loader.ids})

        self.logger.debug(f'Generating {ensemble_size} predictions for {pred_df.shape[0]} samples')
        with torch.no_grad():
            for e in range(ensemble_size):  # perform N sets of predictions and average results
                preds = torch.zeros(len(data_loader.dataset))
                for i, (data, _) in enumerate(tqdm(data_loader)):
                    data = data.to(device)
                    output = model(data).cpu()
                    batch_size = output.shape[0]
                    batch_preds = output.squeeze(1).clamp(min=0, max=4)
                    preds[i * batch_size:(i + 1) * batch_size] = batch_preds

                # add column for this iteration of predictions
                pred_df[str(e)] = preds.numpy()

        # wrangle predictions
        pred_df.set_index('id_code', inplace=True)
        pred_df['diagnosis'] = pred_df.apply(lambda row: int(np.round(row.mean())), axis=1)
        pred_df['target'] = data_loader.dataset.df.set_index('id_code')['diagnosis']
        self.logger.info(pred_df.head(5))

        preds = pred_df['diagnosis'].values
        targets = pred_df['target'].values

        metrics = [getattr(module_metric, met) for met in config['metrics']]

        for metric in metrics:
            result = metric(preds, targets)
            self.logger.info(f'{metric.__name__}: {result}')

        # pred_df.to_csv('preds.csv')
        pred_df.to_csv('test_predictions.csv')
        self.logger.info('Finished saving test predictions!')

    def predict(self, config, model_checkpoint):
        setup_logging(config)
        self.logger = setup_logger(self, config['testing']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.info(f'Using config:\n{config}')

        self.logger.debug('Getting data_loader instance')
        data_loader = PngDataLoader(
            config['testing']['data_dir'],
            batch_size=config['testing']['batch_size'],
            validation_split=0.0,
            train=False,
            alpha=None,
            img_size=config['testing']['img_size'],
            num_workers=config['testing']['num_workers'],
            verbose=config['testing']['verbose']
        )

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        self.logger.debug(f'Loading checkpoint {model_checkpoint}')
        checkpoint = torch.load(model_checkpoint)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        # prepare model for testing
        model.eval()

        ensemble_size = config['testing']['ensemble_size']
        pred_df = pd.DataFrame({'id_code': data_loader.ids})

        self.logger.debug(f'Generating {ensemble_size} predictions for {pred_df.shape[0]} samples')
        with torch.no_grad():
            for e in range(ensemble_size):  # perform N sets of predictions and average results
                preds = torch.zeros(len(data_loader.dataset))
                for i, data in enumerate(tqdm(data_loader)):
                    data = data.to(device)
                    if torch.isnan(data).any().item():
                        msg = f'NaN input: {data}'
                        raise Exception(msg)
                    output = model(data)
                    if torch.isnan(output).any().item():
                        msg = f'NaN output: {output}'
                        raise Exception(msg)
                    output = output.detach().cpu()
                    batch_size = output.shape[0]
                    preds[i * batch_size:(i + 1) * batch_size] = output.squeeze(1)

                # add column for this iteration of predictions
                pred_df[str(e)] = preds.numpy()

        # wrangle predictions
        pred_df.set_index('id_code', inplace=True)
        self.logger.info(pred_df.head(100))
        pred_df['diagnosis'] = pred_df.apply(lambda row: int(np.round(row.mean())), axis=1)
        self.logger.info(pred_df.head(5))

        # pred_df.to_csv('preds.csv')
        pred_df[['diagnosis']].to_csv('submission.csv')
        self.logger.info('Finished saving predictions!')

    def _prepare_device(self, model, n_gpu_use):
        device, device_ids = self._get_device(n_gpu_use)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model, device

    def _get_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                                f"but only {n_gpu} are available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self.logger.info(f'Using device: {device}, {list_ids}')
        return device, list_ids

    def _seed_everything(self, seed):
        self.logger.info(f'Using random seed: {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
