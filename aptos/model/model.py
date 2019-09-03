from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from aptos.base import BaseModel


def cycle_channel_layers(weights, n):
    """Repeat channel weights n times. Assumes channels are dim 1."""
    slices = [(c % 3, c % 3 + 1) for c in range(n)]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights


class ResNet18MaxAvg(BaseModel):

    def __init__(self, num_classes, in_channel, dropout=0.5, pretrained=True, verbose=0):
        super(ResNet18MaxAvg, self).__init__(verbose=verbose)
        self.num_classes = num_classes
        self.in_channel = in_channel
        encoder = models.resnet18(pretrained=pretrained)

        # replace first layer and use resnet weights
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        w18c = cycle_channel_layers(encoder.conv1.weight, in_channel)
        self.conv1.weight = nn.Parameter(w18c)

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1, self.relu, self.bn1, self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('drop1', nn.Dropout(p=dropout)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=dropout)),
            ('linear2', nn.Linear(512, num_classes))
        ]))

        nn.init.kaiming_normal_(self.fc._modules['linear1'].weight)
        nn.init.kaiming_normal_(self.fc._modules['linear2'].weight)

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(
                model_name,
                override_params={'num_classes': num_classes})
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EffNetMaxAvg(BaseModel):

    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)

        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNetMaxAvg.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNetMaxAvg.from_name(
                model_name,
                override_params={'num_classes': num_classes})

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()


class EfficientNetMaxAvg(EfficientNet):
    """
    Modified EfficientNet to use concatenated Max + Avg pooling
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)

        fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.model._bn1.num_features * 2)),
            ('drop1', nn.Dropout(p=self._dropout)),
            ('linear1', nn.Linear(self.model._bn1.num_features * 2, 512)),
            ('mish', Mish()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=self._dropout / 2)),
            ('linear2', nn.Linear(512, self._global_params.num_classes))
        ]))

        nn.init.kaiming_normal_(fc._modules['linear1'].weight)
        nn.init.kaiming_normal_(fc._modules['linear2'].weight)

        self._bn1 = AdaptiveMaxAvgPool()
        self._fc = fc

    def forward(self, x):
        x = self.extract_features(x)
        x = self._bn1(x)
        x = self._fc(x)
        return x


class AdaptiveMaxAvgPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        return x


class Mish(nn.Module):
    """
    https://github.com/lessw2020/mish/blob/master/mish.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


# class SimpleSelfAttention(nn.Module):

#     def __init__(self, n_in, ks=1, sym=False):
#         super().__init__()
#         self.conv = nn.conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
#         self.gamma = nn.Parameter(torch.tensor([0.]))
#         self.sym = sym
#         self.n_in = n_in

#     def forward(self, x):
#         if self.sym:
#             # symmetry hack by https://github.com/mgrankin
#             c = self.conv.weight.view(self.n_in, self.n_in)
#             c = (c + c.t()) / 2
#             self.conv.weight = c.view(self.n_in, self.n_in, 1)

#         size = x.size()
#         x = x.view(*size[:2], -1)   # (C,N)

#         # changed the order of mutiplication to avoid O(N^2) complexity
#         # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
#         convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
#         xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
#         o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
#         o = self.gamma * o + x
#         return o.view(*size).contiguous()


# def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
#     "Create and initialize a `nn.Conv1d` layer with spectral normalization."
#     conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
#     nn.init.kaiming_normal_(conv.weight)
#     if bias:
#         conv.bias.data.zero_()
#     return nn.utils.spectral_norm(conv)
