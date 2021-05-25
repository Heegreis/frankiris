from src.utils.datasets import AudioFolder
from src.utils.collate_fn import *
import pytorch_lightning as pl
import hydra
import torch


class AudioFolderDataModule(pl.LightningDataModule):
    def __init__(self, train: str = None, val: str = None, test: str = None, transform=None, dataloader=None):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.dataloader = dataloader
        self.set_collate_fn()

    def setup(self, stage=None):
        if self.train is not None:
            self.train = AudioFolder(self.train, self.transform['train']['audio'])
        if self.val is not None:
            self.val = AudioFolder(self.val, self.transform['val']['audio'])
        if self.test is not None:
            self.test = AudioFolder(self.test, self.transform['test']['audio'])

    def set_collate_fn(self):
        if 'collate_fn' in self.dataloader:
            print(self.dataloader['collate_fn'])
            self.collate_fn = collate_fn_dict[self.dataloader['collate_fn']]
        else:
            self.collate_fn = None

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test, collate_fn=self.collate_fn)