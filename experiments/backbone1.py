import torch.nn.functional as F
from torch import nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision.models import resnet

class ResNet(nn.Module):
    def __init__(self, num_channel=3, num_class=8, pretrained=True, model='resnet18'):
        super(ResNet, self).__init__()
        self.num_channel = num_channel
        self.num_class = num_class
        self.model = model
        self.pretrained = pretrained
        if self.model == 'resnet18':
            base = resnet.resnet18(pretrained=self.pretrained)
            self.resnet_expansion = 1
            print("ResNet18 is used.")
        elif self.model == 'resnet34':
            base = resnet.resnet34(pretrained=self.pretrained)
            self.resnet_expansion = 1
            print("ResNet34 is used.")
        elif self.model == 'resnet50':
            base = resnet.resnet50(pretrained=self.pretrained)
            self.resnet_expansion = 4
            print("ResNet50 is used.")
        elif self.model == 'resnet101':
            base = resnet.resnet101(pretrained=self.pretrained)
            self.resnet_expansion = 4
            print("ResNet101 is used.")
        elif self.model == 'resnet152':
            base = resnet.resnet152(pretrained=self.pretrained)
            self.resnet_expansion = 4
            print("ResNet152 is used.")
        else:
            raise NotImplemented('Requested model is not supported.')

        self.in_block = nn.Sequential(
            nn.Conv2d(self.num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            base.bn1,
            base.relu,
            base.maxpool)
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.avgpool = base.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*self.resnet_expansion, self.num_class, bias=True)

    def forward(self, x):
        h = self.in_block(x)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = self.encoder4(h)
        y = self.fc(self.flatten(self.avgpool(h)))
        return y