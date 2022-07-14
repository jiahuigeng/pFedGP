import copy
import os
import json
import logging
from pathlib import Path
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import random
from tqdm import trange
from time import time, ctime

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
from experiments.backbone import get_feature_extractor
from experiments.realworld.clients import RealClients
from torchmetrics import F1Score
from torchmetrics.functional import cohen_kappa
from experiments.backbone import get_feature_extractor
from utils import get_device, set_logger, set_seed, detach_to_numpy, save_experiment, \
    print_calibration, calibration_search, offset_client_classes, calc_metrics
from experiments.calibrate import ECELoss

import torch.nn.functional as F
from torchvision.models import resnet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

parser.add_argument('-n', '--data-name', default=['sicapv2', 'radboud','karolinska'], nargs='+') #'
parser.add_argument('--color', type=str, default='darkblue', help='color for calibration plot')
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")

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
parser.add_argument("--num-steps", type=int, default=2)
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
    logging.info("start aggregate and broadcast!")
    num_clients = len(Feds)
    global_state_dict = OrderedDict()
    for client_model in Feds:
        for key in client_model.state_dict().keys():
            if 'bn' not in key and 'fc' not in key:
                if key not in global_state_dict:
                    global_state_dict[key] = torch.zeros_like(client_model.state_dict()[key].data)
                    try:
                        global_state_dict[key] += 1.0/num_clients * client_model.state_dict()[key].data
                    except:
                        print(key)

    for key in global_state_dict:
        for local_model in Feds:
            local_model.state_dict()[key].data = global_state_dict[key]
    return global_state_dict, Feds




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
            img, label = batch
            if cuda:
                img, label =img.cuda(), label.cuda()
            pred = Feds[client_id](img)
            running_loss += criterion(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

            targets.append(label)
            if pred.shape[1] == 2:
                pred = torch.concat((pred, torch.zeros_like(pred)), dim=1)
            preds.append(pred)
            if batch_count > 3:
                break


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
# Feds = []
# for data in args.data_name:
#     local_model = get_feature_extractor(ft=args.ft, input_size=args.input_size, embedding_dim=num_classes[data],
#                                         pretrained=False)
#     local_model = local_model.cuda()
#     Feds.append(local_model)

clients = RealClients(args.data_name, args.data_path, args.num_clients,
                      batch_size=args.batch_size, input_size=args.input_size, mini=args.mini)
criterion = nn.CrossEntropyLoss()
optimizers = [optim.SGD(net.parameters(), lr=0.001, momentum=0.9) for net in Feds]
cuda = torch.cuda.is_available()

global_state, Feds = aggregate_broadcast(Feds)
if cuda:
    print('Cuda is available and used!!!')
    # net = net.cuda()
    Feds = [client_model.cuda() for client_model in Feds]

#======================
last_eval = -1
best_step = -1
best_acc = -1
test_best_based_on_step, test_best_min_based_on_step = -1, -1
test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
step_iter = trange(args.num_steps)

results = defaultdict(list)

best_model = copy.deepcopy(global_state)
best_labels_vs_preds_val = None
best_val_loss = -1
#======================


for step in step_iter:
    for client_id in range(clients.n_clients):
        train_loss, val_loss = 0.0, 0.0
        for i, data in enumerate(clients.train_loaders[client_id], 0):
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
            print(f'client id {client_id} [{step + 1}, {i + 1}] loss: {running_loss:.4f}')
            if i % 10 == 9:
                break
    global_state, Feds = aggregate_broadcast(Feds)

    val_results, labels_vs_preds_val = eval_model([], Feds, clients, split="val")
    val_avg_loss, val_avg_acc = calc_metrics(val_results)
    logging.info(f"Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")

    if best_acc < val_avg_acc:
        best_val_loss = val_avg_loss
        best_acc = val_avg_acc
        best_step = step
        best_labels_vs_preds_val = labels_vs_preds_val
        best_model = copy.deepcopy(global_state)

    results['val_avg_loss'].append(val_avg_loss)
    results['val_avg_acc'].append(val_avg_acc)
    results['best_step'].append(best_step)
    results['best_val_acc'].append(best_acc)

print("end training time:", ctime(time()))
net = best_model

test_results, labels_vs_preds_test = eval_model([], Feds, clients, split="test")
avg_test_loss, avg_test_acc = calc_metrics(test_results)

logging.info(f"\nStep: {step + 1}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_acc:.4f}")
logging.info(f"\nStep: {step + 1}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

best_temp = calibration_search(ECE_module, out_dir, best_labels_vs_preds_val, args.color, 'calibration_val.png')
logging.info(f"best calibration temp: {best_temp}")
print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_temp1.png', args.color, temp=1.0)
print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_best.png', args.color, temp=best_temp)

results['best_step'].append(best_step)
results['best_val_acc'].append(best_acc)
results['test_loss'].append(avg_test_loss)
results['test_acc'].append(avg_test_acc)

lbls_preds = torch.tensor(labels_vs_preds_test)
probs = lbls_preds[:, 1:]
targets = lbls_preds[:, 0].type(torch.long)
v_cohen_kappa = cohen_kappa(probs, targets, num_classes=4)
print(v_cohen_kappa)


f1 = F1Score(num_classes=4)
f1(probs, targets)





with open(str(out_dir / f"results_seed_{args.seed}.json"), "w") as file:
    json.dump(results, file, indent=4)
    # Feds = aggregate_broadcast(Feds)



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
