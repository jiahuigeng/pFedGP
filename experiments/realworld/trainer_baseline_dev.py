import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from experiments.backbone import get_feature_extractor
from experiments.realworld.clients import RealClients
from experiments.backbone import get_feature_extractor
from utils import get_device, set_logger, set_seed, detach_to_numpy, save_experiment, \
    print_calibration, calibration_search, offset_client_classes, calc_metrics


import torch.nn.functional as F
from torchvision.models import resnet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

parser.add_argument('-n', '--data-name', default=['sicapv2'], nargs='+')
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")

parser.add_argument("--ft", '-ft', default="resnet18",
                    choices=['cnn', 'resnet18', 'resnet50', 'efficientnetb3', 'efficientnetb5'])
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--input-size", type=int, default=512, help="input size")
parser.add_argument("--mini", type=bool, default=True, help="use mini size")
parser.add_argument("--batch-size", type=int, default=512)
args = parser.parse_args()

set_logger()
# set_seed(args.seed)


num_classes = {
    'cifar10': 10,
    'cifar100':100,
    'cinic': 10,
    'sicapv2': 4,
    'radboud': 4,
    'karolinska': 2
}

args.num_clients = len(args.data_name)

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


fix_all_seeds(2021)

def get_optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


# def broadcast(global_model, Feds):
#     return

# def aggregate(global_model, Feds):
#     return

# global_model = None
clients = RealClients(args.data_name, args.data_path, args.num_clients,
                      batch_size=args.batch_size, input_size=args.input_size, mini=args.mini)


Feds = []
for data in args.data_name:
    local_model = get_feature_extractor(ft=args.ft, input_size=args.input_size, embedding_dim=num_classes[data], pretrained=False)
    local_model = local_model.cuda()
    Feds.append(local_model)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizers = [get_optimizer(net) for net in Feds]
cuda = torch.cuda.is_available()

if cuda:
    print('Cuda is available and used!!!')
    # net = net.cuda()
    Feds = [client_model.cuda() for client_model in Feds]

for epoch in range(10):
    for client_id in range(clients.n_clients):
        train_loss, val_loss = 0.0, 0.0
        for i, data in enumerate(clients.train_loaders[0], 0):
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizers[client_id].zero_grad()
            outputs = Feds[client_id](inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[client_id].step()
            running_loss = loss.item()
            train_loss += running_loss
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss:.4f}')
        train_loss = train_loss / len(clients.train_loaders[0])
        correct = 0
        total = 0
        with torch.no_grad():
            for data in clients.val_loaders[0]:
                images, labels = data
                if cuda:
                    images, labels = images.cuda(), labels.cuda()
                outputs = Feds[client_id](images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(clients.val_loaders[0])
        print(f'{epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Acc: {(correct / total)}')

print('Finished Training')
