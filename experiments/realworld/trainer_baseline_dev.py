import argparse
import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from time import time, ctime
import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import copy


from experiments.realworld.clients import RealClients
# from experiments.backbone1 import ResNet
from experiments.backbone import get_feature_extractor
from utils import get_device, set_logger, set_seed, detach_to_numpy, save_experiment, \
    print_calibration, calibration_search, offset_client_classes, calc_metrics

from experiments.calibrate import ECELoss

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

#############################
#       Dataset Args        #
#############################
# parser.add_argument(
#     "--data-name", type=str, default="cifar10",
#     choices=['cifar10', 'cifar100', 'cinic10', 'panda', 'sicapv2', "minipanda"],
# )
parser.add_argument('-n', '--data-name', default=['sicapv2', 'radboud', 'karolinska'], nargs='+')
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
# parser.add_argument("--num-clients", type=int, default=50, help="number of simulated clients")
parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", 'efficientnetb3', 'efficientnetb5'])
parser.add_argument("--ft", '-ft', default='cnn',
                    choices=['cnn', 'resnet18', 'resnet50', 'efficientnetb3', 'efficientnetb5'])

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
parser.add_argument("--num-client-agg", type=int, default=1, help="number of kernels")

parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

################################
#       GP args        #
################################
parser.add_argument('--embed-dim', type=int, default=84)
parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                    help='kernel function')
parser.add_argument('--objective', type=str, default='predictive_likelihood',
                    choices=['predictive_likelihood', 'marginal_likelihood'])
parser.add_argument('--predict-ratio', type=float, default=0.5,
                    help='ratio of samples to make predictions for when using predictive_likelihood objective')
parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
parser.add_argument('--outputscale', type=float, default=8., help='output scale')
parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--outputscale-increase', type=str, default='constant',
                    choices=['constant', 'increase', 'decrease'],
                    help='output scale increase/decrease/constant along tree')

#############################
#       General args        #
#############################
parser.add_argument('--color', type=str, default='darkblue', help='color for calibration plot')
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument("--eval-every", type=int, default=25, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")
parser.add_argument("--seed", type=int, default=42, help="seed value")
parser.add_argument("--input-size", type=int, default=32, help="input size")
parser.add_argument("--mini", type=bool, default=False, help="use mini size")
parser.add_argument("--classes-per-client", type=int, help="classes per client")

args = parser.parse_args()

set_logger()
set_seed(args.seed)
cuda = torch.cuda.is_available()
device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
print("device:", device)


num_classes = {
    'cifar10': 10,
    'cifar100':100,
    'cinic': 10,
    'sicapv2': 4,
    'radboud': 4,
    'karolinska': 2
}

classes_per_client = {
    'cifar10': 10,
    'cifar100':100,
    'cinic': 10,
    'sicapv2': 4,
    'radboud': 4,
    'karolinska': 2
}

args.num_clients = len(args.data_name)

exp_name = f'pFedGP-Full_{args.data_name}_num_clients_{args.num_clients}_seed_{args.seed}_' \
           f'lr_{args.lr}_num_steps_{args.num_steps}_inner_steps_{args.inner_steps}_' \
           f'_objective_{args.objective}_predict_ratio_{args.predict_ratio}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

def get_optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


clients = RealClients(args.data_name, args.data_path, args.num_clients,
                      batch_size=args.batch_size, input_size=args.input_size, mini=args.mini)


# Feds = []
# for data in args.data_name:
#     local_model = get_feature_extractor(ft=args.ft, input_size=args.input_size, embedding_dim=num_classes[data], pretrained=False)
#     local_model = local_model.to(device)
#     Feds.append(local_model)


# optimizers = [get_optimizer(Feds[client_id]) for client_id in range(args.num_clients)]


criteria = torch.nn.CrossEntropyLoss()
results = defaultdict(list)


first_name = args.data_name[0]
net = get_feature_extractor(args.ft, input_size=args.input_size, embedding_dim=num_classes[first_name])
if cuda:
    print('Cuda is available and used!!!')
    net = net.cuda()
optimizer = get_optimizer(net)
print("start training time:", ctime(time()))


for epoch in range(10):
    train_loss, val_loss = 0.0, 0.0
    for i, data in enumerate(clients.train_loaders[0], 0):
        inputs, labels = data
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()
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
            outputs = net(images)
            loss = criteria(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(clients.val_loaders[0])
    print(f'{epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Acc: {(correct/total)}')

print('Finished Training')

