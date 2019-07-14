import torch
import torch.nn.functional as F

from .robust_loss_pytorch import AdaptiveLossFunction, lossfun


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

    def __init__(self, alpha=[0.0], scale=[0.5], reduction='mean'):
        self.reduction = reduction
        self.alpha = torch.Tensor(alpha)
        self.scale = torch.Tensor(scale)
        self.loss = lossfun()

    def __call__(self, output, target):
        target = target.float().unsqueeze(1)
        if self.reduction == 'mean':
            return self.loss(target - output, alpha=self.alpha, scale=self.scale).mean()
        if self.reduction == 'sum':
            return self.loss(target - output, alpha=self.alpha, scale=self.scale).sum()
        raise
