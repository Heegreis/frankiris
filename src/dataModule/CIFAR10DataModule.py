from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import pytorch_lightning as pl
import hydra


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, transform=None, download=True, dataloader=None):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.download = download
        self.dataloader = dataloader

    def setup(self, stage=None):
        self.test = CIFAR10(self.data_dir, train=False, download=self.download, transform=self.transform['test']['vision'])
        cifar10_full = CIFAR10(self.data_dir, train=True, download=self.download, transform=self.transform['train']['vision'])
        train_set_size = int(len(cifar10_full) * 0.8)
        valid_set_size = len(cifar10_full) - train_set_size
        self.train, self.val = random_split(cifar10_full, [train_set_size, valid_set_size])

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test)