from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import hydra


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, train: str = None, val: str = None, test: str = None, transform=None, dataloader=None):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.dataloader = dataloader

    def setup(self, stage=None):
        if self.train is not None:
            self.train = ImageFolder(self.train, self.transform['train']['vision'])
        if self.val is not None:
            self.val = ImageFolder(self.val, self.transform['val']['vision'])
        if self.test is not None:
            self.test = ImageFolder(self.test, self.transform['test']['vision'])

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test)