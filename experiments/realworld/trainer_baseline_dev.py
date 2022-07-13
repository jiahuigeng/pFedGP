import os
import os.path as osp
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


fix_all_seeds(2021)


class Sicapv2Dataset(Dataset):
    def __init__(self, df):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.path = SICAPV2_PATH
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        onehot_targets = self.data_df[['NC', 'G3', 'G4', 'G5']].values
        self.targets = np.argmax(onehot_targets, axis=1)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.data_df.iloc[index]

        image = Image.open(osp.join(self.path, "images", row["image_name"]))
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]
        return image, target


SICAPV2_PATH = "/data/BasesDeDatos/SICAP/SICAPv2/"

train_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Validation/Val1", "Train.xlsx"))
val_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Validation/Val1", "Test.xlsx"))
test_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Test", "Test.xlsx"))

train_set = Sicapv2Dataset(train_df_raw)
val_set = Sicapv2Dataset(val_df_raw)
test_set = Sicapv2Dataset(test_df_raw)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(train_set, batch_size=16, shuffle=False)
test_loader = DataLoader(train_set, batch_size=16, shuffle=False)


class ResNet(nn.Module):
    def __init__(self, num_channel=3, num_class=4, pretrained=True, model='resnet18'):
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
        self.fc = nn.Linear(512 * self.resnet_expansion, self.num_class, bias=True)

    def forward(self, x):
        h = self.in_block(x)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = self.encoder4(h)
        y = self.fc(self.flatten(self.avgpool(h)))
        return y


net = ResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
cuda = torch.cuda.is_available()

if cuda:
    print('Cuda is available and used!!!')
    net = net.cuda()

for epoch in range(10):
    train_loss, val_loss = 0.0, 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        train_loss += running_loss
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss:.4f}')
    train_loss = train_loss / len(train_loader)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    print(f'{epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Acc: {(correct / total)}')

print('Finished Training')
