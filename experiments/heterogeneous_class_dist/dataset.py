import random
from collections import defaultdict
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torchvision.datasets import CIFAR10, CIFAR100
import pandas as pd
import os.path as osp
import torchvision
from PIL import Image

SICAPV2_PATH = "/data/BasesDeDatos/SICAP/SICAPv2/"
PANDA_PATH = "/data/BasesDeDatos/Panda/Panda_patches_resized/"

MINI_PANDA_PATH = "/home/jiahui/data/minipanda"
MINI_SICAPV2_PATH = "/home/jiahui/data/minisicap"

RADBOUN_CSV_PATH = "/data/BasesDeDatos/Panda/Panda_patches_resized/radb_only"
KAROLIN_CSV_PATH = "/home/jiahui/data/karolinska"



panda_stats = {"norm_mean":  (0.4914, 0.4822, 0.4465), "norm_std": (0.2023, 0.1994, 0.2010)}


def get_instance_classes(dataframe, dataframe_raw):
    class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]]
    dataframe["class"] = np.argmax(class_columns, axis=0).astype(str)
    dataframe["wsi_contains_unlabeled"] = False
    return dataframe



def get_datasets(data_name, dataroot, normalize=True, val_size=10000, input_size=32, mini=False):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of datafolder, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch datafolder/subset)
    """

    norm_map = {
        "cifar10": [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            CIFAR10
        ],
        "cifar100": [
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            CIFAR100
        ]
    }
    if "cifar" in data_name:
        normalization, data_obj = norm_map[data_name]

        trans = [transforms.ToTensor()]

        if normalize:
            trans.append(normalization)

        transform = transforms.Compose(trans)

        dataset = data_obj(
            dataroot,
            train=True,
            download=True,
            transform=transform
        )

        test_set = data_obj(
            dataroot,
            train=False,
            download=True,
            transform=transform
        )

        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    elif data_name == 'cinic10':
        train_set, val_set, test_set = get_cinic_dataset(dataroot)

    elif data_name == 'panda':
        train_set, val_set, test_set = get_panda_dataset(mini=False)

    elif data_name == "minipanda":
        train_set, val_set, test_set = get_panda_dataset(mini=mini, input_size=input_size)

    elif data_name == 'sicapv2':
        train_set, val_set, test_set = get_sicapv2_dataset(mini=mini, input_size=input_size)

    elif data_name == "radboud":
        train_set, val_set, test_set = get_radboud_dataset(mini=mini, input_size=input_size)

    elif data_name == "karolinska":
        train_set, val_set, test_set = get_karolinska_dataset(mini=mini, input_size=input_size)

    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100', 'cinic10]")

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain datafolder
    :param dataset: pytorch datafolder object
    :return: datafolder info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    elif hasattr(dataset, "dataset"):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:  # tensorDataset Object
        data_labels_list = np.array(dataset.tensors[1])
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch datafolder object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            # avoid selected classes
            max_class_counts = list(set(max_class_counts) - set(c))
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch datafolder object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user, normalize=True, input_size=32):
    """
    generates train/val/test loaders of each client
    :param data_name: name of datafolder, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 4}
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=normalize, input_size=input_size)

    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        # create dataloaders from subsets
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))
        # do not shuffle at eval and test
        loader_params['shuffle'] = False

    return dataloaders


def get_dataset_split(pkl_path, split):
    if not isinstance(pkl_path, Path):
        pkl_path = Path(pkl_path)
    data = []
    for i in ("x", "y"):
        file = pkl_path / "_".join([i, split, "dataset.pkl"])
        with open(file, "rb") as file:
            data.append(pickle.load(file))
    x, y = data
    x = x / 255.0
    x = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)
    y = torch.from_numpy(y.astype(np.long))
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


def get_cinic_dataset(pkl_path):
    datasets = []
    for split in ("train", "valid", "test"):
        datasets.append(get_dataset_split(pkl_path, split))
    return datasets



def get_sicapv2_dataset(mini, input_size=512):
    train_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Validation/Val1", "Train.xlsx"))
    val_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Validation/Val1", "Test.xlsx"))
    test_df_raw = pd.read_excel(osp.join(SICAPV2_PATH, "partition/Test", "Test.xlsx"))

    train_set = Sicapv2Dataset(train_df_raw, mini, input_size)
    val_set = Sicapv2Dataset(val_df_raw, mini, input_size)
    test_set = Sicapv2Dataset(test_df_raw, mini, input_size)

    return train_set, val_set, test_set




class PandaDatast(data.Dataset):
    def __init__(self, df, mini, input_size):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.path = PANDA_PATH
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                             ])
        if mini==True:
            self.path = MINI_PANDA_PATH
            self.transform = transforms.Compose([
                                                transforms.Resize(input_size),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(panda_stats["norm_mean"], panda_stats["norm_std"])
                                                 ])
        self.data = []
        # onehot_targets = self.data_df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
        onehot_targets = self.data_df[['NC','G3','G4','G5']].values
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

class RadBoudDataset(data.Dataset):
    def __init__(self, df, mini, input_size=512):
        self.data_df = df
        self.path = PANDA_PATH
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if mini == True:
            self.data_df = self.data_df[:1000]
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])

        onehot_targets = self.data_df[['NC','G3','G4','G5']].values
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

class KarolinskaDataset(data.Dataset):
    def __init__(self, df, mini, input_size=512):
        self.data_df = df
        self.path = PANDA_PATH
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if mini == True:
            self.data_df = self.data_df[:1000]
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])

        onehot_targets = self.data_df[['NC', "unlabeled"]].values
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

class Sicapv2Dataset(data.Dataset):
    def __init__(self, df, mini, input_size=512):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.path = SICAPV2_PATH
        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                             ])
        if mini==True:
            self.path = SICAPV2_PATH
            self.transform = transforms.Compose([
                                                transforms.Resize(input_size),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(panda_stats["norm_mean"], panda_stats["norm_std"])
                                                 ])
        self.data = []
        # onehot_targets = self.data_df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
        onehot_targets = self.data_df[['NC','G3','G4','G5']].values
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


def get_panda_dataset(mini=False, input_size=32):
    if mini:
        train_df_raw = pd.read_csv(osp.join(MINI_PANDA_PATH, "train_patches.csv"))
        val_df_raw = pd.read_csv(osp.join(MINI_PANDA_PATH, "val_patches.csv"))
        test_df_raw = pd.read_csv(osp.join(MINI_PANDA_PATH, "test_patches.csv"))

    else:
        train_df_raw = pd.read_csv(osp.join(PANDA_PATH, "train_patches.csv"))
        val_df_raw = pd.read_csv(osp.join(PANDA_PATH, "val_patches.csv"))
        test_df_raw = pd.read_csv(osp.join(PANDA_PATH, "test_patches.csv"))

    train_set, val_set, test_set = PandaDatast(train_df_raw, mini, input_size), PandaDatast(val_df_raw,mini, input_size), PandaDatast(test_df_raw, mini, input_size)
    return train_set, val_set, test_set

def get_radboud_dataset(mini, input_size=512):
    train_df_raw = pd.read_csv(osp.join(RADBOUN_CSV_PATH, "train_patches.csv"))
    val_df_raw = pd.read_csv(osp.join(RADBOUN_CSV_PATH, "val_patches.csv"))
    test_df_raw = pd.read_csv(osp.join(RADBOUN_CSV_PATH, "test_patches.csv"))

    train_set = RadBoudDataset(train_df_raw, mini, input_size)
    val_set = RadBoudDataset(val_df_raw, mini, input_size)
    test_set = RadBoudDataset(test_df_raw, mini, input_size)
    return train_set, val_set, test_set

def get_karolinska_dataset(mini, input_size=512):
    train_df_raw = pd.read_csv(osp.join(KAROLIN_CSV_PATH, "train.csv"))
    val_df_raw = pd.read_csv(osp.join(KAROLIN_CSV_PATH, "val.csv"))
    test_df_raw = pd.read_csv(osp.join(KAROLIN_CSV_PATH, "test.csv"))
    train_set = KarolinskaDataset(train_df_raw, mini, input_size)
    val_set = KarolinskaDataset(val_df_raw, mini, input_size)
    test_set = KarolinskaDataset(test_df_raw, mini, input_size)

    return train_set, val_set, test_set


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader

    for ds in ['sicapv2', 'radboud', 'karolinska']:
        train_set, val_set, test_set = get_datasets(ds, dataroot="")
        train_loader, val_loader, test_loader = DataLoader(train_set, batch_size=32), DataLoader(val_set,
                                                                                                 batch_size=32), DataLoader(
            test_set, batch_size=32)
        train_sample, val_sample, test_sample = next(iter(train_loader)), next(iter(val_loader)), next(
            iter(test_loader))
        print("train sample:", train_sample.shape)
        print("val sample:", val_sample.shape)
        print("test sample:", test_sample.shape)