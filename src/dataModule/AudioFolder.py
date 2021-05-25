from src.utils.datasets import AudioFolder
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

    def setup(self, stage=None):
        if self.train is not None:
            self.train = AudioFolder(self.train, self.transform['train']['audio'])
        if self.val is not None:
            self.val = AudioFolder(self.val, self.transform['val']['audio'])
        if self.test is not None:
            self.test = AudioFolder(self.test, self.transform['test']['audio'])

    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for tensor, target in batch:
            tensors += [tensor]
            targets += [torch.tensor(target)]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    def train_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['train'], self.train, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['val'], self.val, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return hydra.utils.instantiate(self.dataloader['test'], self.test, collate_fn=self.collate_fn)