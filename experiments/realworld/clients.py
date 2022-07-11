from experiments.realworld.dataset import get_datasets
from torch.utils.data import DataLoader


class RealClients:
    def __init__(
            self,
            data_name,
            data_path,
            n_clients,
            classes_per_client=2,
            batch_size=128,
            input_size=32,
            mini=False
    ):

        self.data_name = data_name
        self.data_path = data_path
        self.n_clients = n_clients
        self.classes_per_client = classes_per_client

        self.batch_size = batch_size
        self.input_size = input_size
        self.mini = mini
        self.train_loaders, self.val_loaders, self.test_loaders = [], [], []
        self._init_dataloaders()

    def _init_dataloaders(self):
        for data in self.data_name:
            train_set, val_set, test_set = get_datasets(
                data_name=data,
                dataroot=self.data_path,
                input_size=self.input_size,
                mini=self.mini
            )
            self.train_loaders.append(DataLoader(train_set, batch_size=self.batch_size))
            self.val_loaders.append(DataLoader(val_set, batch_size=self.batch_size))
            self.test_loaders.append(DataLoader(test_set, batch_size=self.batch_size))

    def __len__(self):
        return self.n_clients