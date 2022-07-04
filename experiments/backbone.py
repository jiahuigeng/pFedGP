import torch.nn.functional as F
from torch import nn
from torchvision import models


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

class CNNTargetlarge(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, embedding_dim=84, input_size=64):
        super(CNNTargetlarge, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.hidden_size = int((input_size)/4 -3)
        self.fc1 = nn.Linear(2 * n_kernels * self.hidden_size * self.hidden_size, 120)
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


# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py

def get_feature_extractor(ft="cnn", input_size=32):
    if input_size == 32:
        return CNNTarget(3, 16, 84)

    elif ft == 'cnn':
        return CNNTargetlarge(3, 16, 84, input_size)

    elif ft == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 84)
        return model_ft

    elif ft == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 84)
        return model_ft

    elif ft == "efficientnetb3":
        model_ft = models.efficientnet_b3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 84)
        return model_ft

    elif ft == "efficientnetb5":
        model_ft = models.efficientnet_b3(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 84)
        return model_ft

