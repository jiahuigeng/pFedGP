import copy
import os
import logging
from pathlib import Path
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
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
from experiments.calibrate import ECELoss

import torch.nn.functional as F
from torchvision.models import resnet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

parser.add_argument('-n', '--data-name', default=['sicapv2'], nargs='+')
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument('--objective', type=str, default='predictive_likelihood',
                    choices=['predictive_likelihood', 'marginal_likelihood'])
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")

parser.add_argument('--predict-ratio', type=float, default=0.5,
                    help='ratio of samples to make predictions for when using predictive_likelihood objective')
parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--ft", '-ft', default="resnet18",
                    choices=['cnn', 'resnet18', 'resnet50', 'efficientnetb3', 'efficientnetb5'])
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--input-size", type=int, default=512, help="input size")
parser.add_argument("--mini", type=bool, default=True, help="use mini size")
parser.add_argument("--seed", type=int, default=2021, help="seed for reproduction")
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

set_logger()

args.num_clients = len(args.data_name)
exp_name = f'pFedGP-Full_{args.data_name}_num_clients_{args.num_clients}_seed_{args.seed}_' \
           f'lr_{args.lr}_num_steps_{args.num_steps}' \
           f'_objective_{args.objective}_predict_ratio_{args.predict_ratio}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

ECE_module = ECELoss()


num_classes = {
    'cifar10': 10,
    'cifar100':100,
    'cinic': 10,
    'sicapv2': 4,
    'radboud': 4,
    'karolinska': 2
}



def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


fix_all_seeds(2021)

# def get_optimizer(network):
#     return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
#         if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)

def init_system():
    Feds = []
    for data in args.data_name:
        local_model = get_feature_extractor(ft=args.ft, input_size=args.input_size, embedding_dim=num_classes[data],
                                            pretrained=False)
        local_model = local_model.cuda()
        Feds.append(local_model)

    # global_state_dict = OrderedDict()
    # for key in Feds[0].state_dict().keys():
    #     if 'bn' not in key and 'fc' not in key:
    #         global_state_dict[key] = Feds[0].state_dict()[key].data
    #
    # for key in global_state_dict:
    #     for local_model in Feds:
    #         local_model.state_dict()[key].data = global_state_dict[key]
    return Feds


def aggregate_broadcast(Feds):
    num_clients = len(Feds)
    global_state_dict = OrderedDict()
    for client_model in Feds:
        for key in client_model.state_dict().keys():
            if 'bn' not in key and 'fc' not in key:
                global_state_dict[key] += 1/num_clients * client_model.state_dict()[key].data

    for key in global_state_dict:
        for local_model in Feds:
            local_model.state_dict()[key].data = global_state_dict[key]
    return Feds


clients = RealClients(args.data_name, args.data_path, args.num_clients,
                      batch_size=args.batch_size, input_size=args.input_size, mini=args.mini)



criterion = nn.CrossEntropyLoss()
optimizers = [optim.SGD(net.parameters(), lr=0.001, momentum=0.9) for net in Feds]

cuda = torch.cuda.is_available()

if cuda:
    print('Cuda is available and used!!!')
    # net = net.cuda()
    Feds = [client_model.cuda() for client_model in Feds]

@torch.no_grad()
def eval_model(global_model, Feds, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    targets = []
    preds = []


    for client_id in range(args.num_clients):
        is_first_iter = True
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        # Feds[client_id].eval()


        for batch_count, batch in enumerate(curr_data):
            # print(batch_count)
            # img, label = tuple(t.to(device) for t in batch)
            img, label = batch
            if cuda:
                img, label =img.cuda(), label.cuda()
            pred = Feds[client_id](img)
            running_loss += criterion(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

            targets.append(label)
            preds.append(pred)


        results[client_id]['loss'] = running_loss / (batch_count + 1)
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples

    if global_model:
        pass

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return results, labels_vs_preds


Feds = init_system()
for epoch in range(args.num_steps):
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
            print(f'client id {client_id} [{epoch + 1}, {i + 1}] loss: {running_loss:.4f}')

    Feds = aggregate_broadcast(Feds)



        # train_loss = train_loss / len(clients.train_loaders[0])
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for data in clients.val_loaders[0]:
        #         images, labels = data
        #         if cuda:
        #             images, labels = images.cuda(), labels.cuda()
        #         outputs = Feds[client_id](images)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        # val_loss = val_loss / len(clients.val_loaders[0])
        # print(f'{epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Acc: {(correct / total)}')


print('Finished Training')
