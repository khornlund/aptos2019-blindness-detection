import torch
import torch.nn as nn
import torch.nn.functional as F

from robust_loss_pytorch import AdaptiveLossFunction, lossfun


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def mse_loss(output, target):
    return F.mse_loss(output.squeeze(1), target.to(torch.float))


def l1_loss(output, target):
    return F.l1_loss(output.squeeze(1), target.to(torch.float))


class RobustLossAdaptive:

    def __init__(self, device, reduction='mean'):
        self.loss = AdaptiveLossFunction(num_dims=1, device=device, float_dtype=torch.float32)
        self.reduction = reduction

    def __call__(self, output, target):
        target = target.float().unsqueeze(1)
        if self.reduction == 'mean':
            return self.loss.lossfun(target - output).mean()
        if self.reduction == 'sum':
            return self.loss.lossfun(target - output).sum()
        raise

    def parameters(self):
        return self.loss.parameters()

    def named_parameters(self):
        return self.loss.named_parameters()

    def alpha(self):
        return self.loss.alpha()

    def scale(self):
        return self.loss.scale()


class RobustLossGeneral:

    def __init__(self, device, alpha=0.0, scale=0.5, reduction='mean'):
        self.reduction = reduction
        self.alpha = torch.Tensor([alpha]).to(device)
        self.scale = torch.Tensor([scale]).to(device)
        self.loss = lossfun

    def __call__(self, output, target):
        target = target.float().unsqueeze(1)
        if self.reduction == 'mean':
            return self.loss(target - output, alpha=self.alpha, scale=self.scale).mean()
        if self.reduction == 'sum':
            return self.loss(target - output, alpha=self.alpha, scale=self.scale).sum()
        raise


class WassersteinLoss(nn.Module):
    """
    Implements the `Wassertein metric <https://en.wikipedia.org/wiki/Wasserstein_metric>`_
    AKA `Earth Mover's Distance <https://en.wikipedia.org/wiki/Earth_mover%27s_distance>`_ using
    `cumulative distribution functions <https://stats.stackexchange.com/a/299391>`_.

    In theory, this addresses the problem of KL divergence not taking into acount the ordinal
    nature of classes (ratings).

    Code has been adapted from:
    `<https://github.com/truskovskiyk/nima.pytorch/blob/master/nima/emd_loss.py>`
    """
    def __init__(self, n_classes, reduction='mean', quadratic=True):
        self.n_classes = n_classes
        self.reduction = reduction

        if self.reduction is not None and \
           self.reduction != 'mean' and \
           self.reduction != 'sum':
            raise Exception(f'`reduction` must be either `None`, "mean", or "sum"')

        if quadratic:
            self.loss = self.cdf_distance_batch_quadratic
        else:
            self.loss = self.cdf_distance_batch_linear

        super().__init__()

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        target = F.one_hot(target, num_classes=self.n_classes).float()

        if self.reduction == 'mean':
            return self.loss(output, target).mean()
        if self.reduction == 'sum':
            return self.loss(output, target).sum()
        return self.loss(output, target)

    # -- batch operations, used by forward pass ---------------------------------------------------

    def cdf_distance_batch_linear(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=1)
        cdf_estimate = torch.cumsum(output, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.abs(cdf_diff), dim=1)

    def cdf_distance_batch_quadratic(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=1)
        cdf_estimate = torch.cumsum(output, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.pow(torch.abs(cdf_diff), 2), dim=1)

    # -- 1D operations, used for testing ----------------------------------------------------------

    def cdf_distance_single_linear(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=0)
        cdf_estimate = torch.cumsum(output, dim=0)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.abs(cdf_diff))

    def cdf_distance_single_quadratic(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=0)
        cdf_estimate = torch.cumsum(output, dim=0)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.pow(torch.abs(cdf_diff), 2))