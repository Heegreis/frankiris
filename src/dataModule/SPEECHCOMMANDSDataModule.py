from typing import Tuple
from torch import Tensor
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.speechcommands import load_speechcommands_item
import torch
import pytorch_lightning as pl
import hydra


class SPEECHCOMMANDSwithTransform(SPEECHCOMMANDS):
    def __init__(self, root, download = False, subset = None, transform = None):
        super().__init__(root, download=download, subset=subset)
        self.transform = transform
    
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]
        waveform, sample_rate, label, speaker_id, utterance_number = load_speechcommands_item(fileid, self._path)
        new_waveform = self.transform['audio'](waveform)
        return new_waveform, sample_rate, label, speaker_id, utterance_number

class SPEECHCOMMANDSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, transform=None, download=True, dataloader=None):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.download = download
        self.dataloader = dataloader

    def setup(self, stage=None):
        self.train = SPEECHCOMMANDSwithTransform(self.data_dir, download=self.download, subset="training", transform=self.transform['train'])
        self.val = SPEECHCOMMANDSwithTransform(self.data_dir, download=self.download, subset="validation", transform=self.transform['val'])
        self.test = SPEECHCOMMANDSwithTransform(self.data_dir, download=self.download, subset="testing", transform=self.transform['test'])

        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train)))

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

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
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

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
