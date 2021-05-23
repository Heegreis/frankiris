from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, train: str = None, val: str = None, test: str = None, transform=None, batch_size: int = 32):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.train is not None:
            self.train = ImageFolder(self.train, self.transform['train']['vision'])
        if self.val is not None:
            self.val = ImageFolder(self.val, self.transform['val']['vision'])
        if self.test is not None:
            self.test = ImageFolder(self.test, self.transform['test']['vision'])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)