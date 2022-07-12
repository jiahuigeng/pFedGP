import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import resnet

class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, embedding_dim=84):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.embed_dim = nn.Linear(84, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.embed_dim(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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



# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py

def get_feature_extractor(ft="cnn", input_size=32, embedding_dim=84):
    if input_size == 32:
        return CNNTarget(3, 16, 84)

    elif ft == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, embedding_dim)
        return model_ft

    elif ft == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, embedding_dim)
        return model_ft

    elif ft == "efficientnetb3":
        model_ft = models.efficientnet_b3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, embedding_dim)
        return model_ft

    elif ft == "efficientnetb5":
        model_ft = models.efficientnet_b3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, embedding_dim)
        return model_ft

    # elif ft == "presnet18":
    #     return model_ft



# def get_model(num_classes, num_channels, input_size=512, model="cnn", pretrained=True):
#     if input_size == 32:
#         return CNN()
#
#     elif model in ["resnet18", "resnet50"]:
#         return ResNet(num_channel=num_channels, num_class=num_classes, pretrained=True, model=model)
#
#     elif model in ["efficientnetb3", "efficientnetb5"]:
#         if model == "efficientnetb3"
#             model = models.efficientnet_b3(pretrained=pretrained)
#             num
#             model.fc = nn.Linear()
#             return models.efficientnet_b3
#
