from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 32, download=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download

    def setup(self, stage=None):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mnist_test = CIFAR10(self.data_dir, train=False, download=self.download, transform=transform)
        cifar10_full = CIFAR10(self.data_dir, train=True, download=self.download, transform=transform)
        train_set_size = int(len(cifar10_full) * 0.8)
        valid_set_size = len(cifar10_full) - train_set_size
        self.mnist_train, self.mnist_val = random_split(cifar10_full, [train_set_size, valid_set_size])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)