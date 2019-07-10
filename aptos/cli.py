import click
import os
import yaml

import torch

from aptos.main import Runner
from aptos.utils import CodeExtractor, ImportExtractor


@click.group()
def cli():
    """CLI for aptos"""


@cli.command()
def flatten():
    """Producing flat version of package"""
    CodeExtractor().start()
    ImportExtractor().start()


@cli.command()
@click.option('-c', '--config-filename', default=None, type=str, multiple=True,
              help='config file path (default: None)')
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def train(config_filename, resume, device):
    if config_filename:
        configs = [load_config(filename) for filename in config_filename]
    elif resume:
        configs = [torch.load(resume)['config']]
    else:
        raise AssertionError('Configuration file need to be specified. '
                             'Add "-c experiments/config.yaml", for example.')
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    for config in configs:
        Runner().train(config, resume)


@cli.command()
@click.option('-c', '--config-filename', default=None, type=str,
              help='config file path (default: None)')
@click.option('-m', '--model-checkpoint', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def test(config_filename, model_checkpoint, device):
    config = load_config(config_filename)
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    Runner().test(config, model_checkpoint)


@cli.command()
@click.option('-c', '--config-filename', default=None, type=str,
              help='config file path (default: None)')
@click.option('-m', '--model-checkpoint', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def predict(config_filename, model_checkpoint, device):
    config = load_config(config_filename)
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    Runner().predict(config, model_checkpoint)


def load_config(filename):
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    config['name'] = verbose_config_name(config)
    return config


def verbose_config_name(config):
    short_name = config['short_name']
    arch = config['arch']['type']
    loss = config['loss']['type']
    optim = config['optimizer']['type']
    mlr = config['optimizer']['args']['model_config']['args']['lr']
    llr = config['optimizer']['args']['loss_config']['args']['lr']
    return '-'.join([short_name, arch, loss, optim, f'mlr{mlr}', f'llr{llr}'])


if __name__ == '__main__':
    config = load_config('experiments/config.yml')
    Runner().train(config, None)
