from src.utils.datasets import AudioFolder
import pytorch_lightning as pl
import hydra


class AudioFolderDataModule(pl.LightningDataModule):
    def __init__(self, train: str = None, val: str = None, test: str = None, transform=None, dataloader=None):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.dataloader = dataloader

    def setup(self, stage=None):
        if self.train is not None:
            self.train = AudioFolder(self.train, self.transform['train']['audio'])
        if self.val is not None:
            self.val = AudioFolder(self.val, self.transform['val']['audio'])
        if self.test is not None:
            self.test = AudioFolder(self.test, self.transform['test']['audio'])

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test)