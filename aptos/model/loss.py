import torch
import torch.nn.functional as F

from .robust_loss_pytorch import AdaptiveLossFunction


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def mse_loss(output, target):
    return F.mse_loss(output.squeeze(1), target.to(torch.float))


def l1_loss(output, target):
    return F.l1_loss(output.squeeze(1), target.to(torch.float))


class RobustLoss:

    def __init__(self):
        device = torch.device('cuda:0')
        self.loss = AdaptiveLossFunction(num_dims=5, device=device)

    def __call__(self, output, target):
        return self.loss.lossfun(self._residual(output, target))

    def _residual(output, target):
        preds = output.max(1)[1]  # argmax
        return preds - target
