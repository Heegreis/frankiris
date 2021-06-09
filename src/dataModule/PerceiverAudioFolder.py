from src.utils.datasets.PerceiverAudioFolder import PerceiverAudioFolder
from src.utils.collate_fn import *
import pytorch_lightning as pl
import hydra
import torch


class PerceiverAudioFolderDataModule(pl.LightningDataModule):
    def __init__(self, train: str = None, val: str = None, test: str = None, transform=None, dataloader=None, audio_cfg=None):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.dataloader = dataloader
        self.audio_cfg = audio_cfg
        self.set_collate_fn()

    def setup(self, stage=None):
        if self.train is not None:
            self.train = PerceiverAudioFolder(self.train, transform=self.transform['train']['audio'], audio_cfg=self.audio_cfg)
        if self.val is not None:
            self.val = PerceiverAudioFolder(self.val, transform=self.transform['val']['audio'], audio_cfg=self.audio_cfg)
        if self.test is not None:
            self.test = PerceiverAudioFolder(self.test, transform=self.transform['test']['audio'], audio_cfg=self.audio_cfg)

    def set_collate_fn(self):
        self.collate_fn = None
        if 'collate_fn' in self.dataloader:
            if self.dataloader['collate_fn'] is not None:
                self.collate_fn = collate_fn_dict[self.dataloader['collate_fn']]

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test, collate_fn=self.collate_fn)