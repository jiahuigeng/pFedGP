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


from experiments.heterogeneous_class_dist.clients import BaseClients
from experiments.backbone1 import ResNet
from utils import get_device, set_logger, set_seed, detach_to_numpy, save_experiment, \
    print_calibration, calibration_search, offset_client_classes, calc_metrics

from experiments.calibrate import ECELoss

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

#############################
#       Dataset Args        #
#############################
parser.add_argument(
    "--data-name", type=str, default="cifar10",
    choices=['cifar10', 'cifar100', 'cinic10', 'panda', 'sicapv2', "minipanda"],
)
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--num-clients", type=int, default=50, help="number of simulated clients")
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
parser.add_argument("--classes-per-client", type=int, help="classes per client")

args = parser.parse_args()

set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
# num_classes = 10 if args.data_name in ('cifar10', 'cinic10') else 100
if args.data_name in ("cifar10", "cinic10"):
    num_classes = 10
    classes_per_client = 2
elif args.data_name in ("panda", "minipanda"):
    num_classes = 5
    classes_per_client = 5
elif args.data_name in ("cifar100"):
    num_classes = 100
    classes_per_client = 10

# classes_per_client = 2 if args.data_name == 'cifar10' else 10 if args.data_name == 'cifar100' else 4
if args.classes_per_client:
    classes_per_client = args.classes_per_client

exp_name = f'pFedGP-Full_{args.data_name}_num_clients_{args.num_clients}_seed_{args.seed}_' \
           f'lr_{args.lr}_num_steps_{args.num_steps}_inner_steps_{args.inner_steps}_' \
           f'_objective_{args.objective}_predict_ratio_{args.predict_ratio}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

@torch.no_grad()
def eval_model(global_model, Feds, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    targets = []
    preds = []

    if global_model:
        global_model.eval()

    for client_id in range(args.num_clients):
        is_first_iter = True
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        Feds[client_id].eval()


        for batch_count, batch in enumerate(curr_data):
            # print(batch_count)
            img, label = tuple(t.to(device) for t in batch)

            pred = Feds[client_id](img)
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

            targets.append(label)
            preds.append(pred)


        # erase tree (no need to save it)
        # GPs[client_id].tree = None

        results[client_id]['loss'] = running_loss / (batch_count + 1)
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return results, labels_vs_preds


###############################
# init net and GP #
###############################
clients = BaseClients(args.data_name, args.data_path, args.num_clients,
                      classes_per_client=classes_per_client,
                      batch_size=args.batch_size, input_size=args.input_size)

# NN

# net = CNNTarget(n_kernels=args.n_kernels, embedding_dim=args.embed_dim)
# if args.data_name in ['cifar10', 'cifar100', 'cinic10', 'minipanda']:
#     net = get_feature_extractor(args.ft, input_size=32)
# elif args.data_name in ['panda', 'sicapv2']:
#     net = get_feature_extractor(args.ft, input_size=512)

# num_channel = 3, num_class = 8, pretrained = True, model = 'resnet18'



# GPs = torch.nn.ModuleList([])
Feds = torch.nn.ModuleList([])
for client_id in range(args.num_clients):
    cur_net = ResNet(num_channel=3, num_class=num_classes, pretrained=True, model=args.model)
    cur_net = cur_net.to(device)
    Feds.append(cur_net)
    # GPs.append(pFedGPFullLearner(args, classes_per_client))  # GP instances


def get_optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
        if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)


criteria = torch.nn.CrossEntropyLoss()
results = defaultdict(list)

################
# init metrics #
################
last_eval = -1
best_step = -1
best_acc = -1
test_best_based_on_step, test_best_min_based_on_step = -1, -1
test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
step_iter = trange(args.num_steps)

global_net = ResNet(num_channel=3, num_class=num_classes, pretrained=True, model=args.model)
best_model = copy.deepcopy(global_net)
best_labels_vs_preds_val = None
best_val_loss = -1

print("start training time:", ctime(time()))
# for step in range(4):
#     for client_id in range(args.num_clients):
#         Feds[client_id].train()
#         train_loss = 0.0
#         num_data = 0
#         correct = 0
#         optimizer = get_optimizer(Feds[client_id])
#         for batch_idx, (img, label) in enumerate(clients.train_loaders[client_id]):
#             num_data += label.size(0)
#             # TODO: 'cuda' must be defined in the given 'params'
#
#             # data, target = data.cuda(), target.cuda()
#             # data, target = Variable(data), Variable(target)
#             optimizer.zero_grad()
#             score = Feds[client_id](img)
#             loss = criteria(score, label)
#             loss_data = loss.data.item()
#             train_loss += loss_data
#             if np.isnan(loss_data):
#                 raise ValueError('loss is nan while training')
#             loss.backward()
#             optimizer.step()
#             pred = score.data.max(1)[1]
#             correct += pred.eq(label.view(-1)).sum().item()


params = OrderedDict()
for n in global_net.state_dict().keys():
    params[n] = torch.zeros_like(global_net.state_dict()[n].data)

for step in step_iter:
    # print tree stats every 100 epochs
    to_print = True if step % 100 == 0 else False

    # select several clients
    client_ids = np.random.choice(range(args.num_clients), size=args.num_client_agg, replace=False)

    # initialize global model params


    # iterate over each client
    train_avg_loss = 0
    num_samples = 0

    for j, client_id in enumerate(client_ids):
        # curr_global_net = copy.deepcopy(net)
        # curr_global_net.train()
        # optimizer = get_optimizer(curr_global_net)

        Feds[client_id].train()
        optimizer = get_optimizer(Feds[client_id])

        for k, batch in enumerate(clients.train_loaders[client_id]):

            batch = (t.to(device) for t in batch)
            img, label = batch
            num_samples += label.size(0)
            optimizer.zero_grad()
            loss = criteria(Feds[client_id](img), label)

            train_avg_loss += loss


            # propagate loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Feds[client_id].parameters(), 50)
            optimizer.step()

            if k % 2 == 1:
                logging.info(f"batch: {k}, training loss: {(train_avg_loss/num_samples+1):.4f}")
                train_avg_loss =0
                num_samples = 0
                val_results, labels_vs_preds_val = eval_model(Feds[client_id], Feds, clients, split="val")
                val_avg_loss, val_avg_acc = calc_metrics(val_results)
                logging.info(f"Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")


        eval_model(None, Feds, clients, split="val")

        for n in Feds[client_id].state_dict().keys():
            params[n] += Feds[client_id].state_dict()[n].data
#

#
#     # average parameters
    for n, p in params.items():
        params[n] = p / args.num_client_agg
#
#
#     # update new parameters
    global_net.load_state_dict(params)
#
    if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
        val_results, labels_vs_preds_val = eval_model(global_net, Feds, clients, split="val")
        val_avg_loss, val_avg_acc = calc_metrics(val_results)
        logging.info(f"Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")

        if best_acc < val_avg_acc:
            best_val_loss = val_avg_loss
            best_acc = val_avg_acc
            best_step = step
            best_labels_vs_preds_val = labels_vs_preds_val
            best_model = copy.deepcopy(global_net)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)

print("end training time:", ctime(time()))
net = best_model

test_results, labels_vs_preds_test = eval_model(net, Feds, clients, split="test")
avg_test_loss, avg_test_acc = calc_metrics(test_results)

logging.info(f"\nStep: {step + 1}, Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_acc:.4f}")
logging.info(f"\nStep: {step + 1}, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

# best_temp = calibration_search(ECE_module, out_dir, best_labels_vs_preds_val, args.color, 'calibration_val.png')
# logging.info(f"best calibration temp: {best_temp}")
# print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_temp1.png', args.color, temp=1.0)
# print_calibration(ECE_module, out_dir, labels_vs_preds_test, 'calibration_test_best.png', args.color, temp=best_temp)

results['best_step'].append(best_step)
results['best_val_acc'].append(best_acc)
results['test_loss'].append(avg_test_loss)
results['test_acc'].append(avg_test_acc)

with open(str(out_dir / f"results_{args.inner_steps}_inner_steps_seed_{args.seed}.json"), "w") as file:
    json.dump(results, file, indent=4)