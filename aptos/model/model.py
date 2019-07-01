from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
