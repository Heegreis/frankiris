import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from perceiver_pytorch import Perceiver
from pytorch_lightning.metrics.functional import accuracy
import torch
import hydra
from dlcliche.torch_utils import IntraBatchMixup


class LightningModule(pl.LightningModule):
    def __init__(self, torch_module=None, learning_rate=3e-4, mixup_alpha=0.4, transpose_tfm=True, dataModule=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.torch_module = hydra.utils.instantiate(torch_module)
        dataModule.setup()
        weight = dataModule.train.get_class_weight()
        print(weight)
        self.batch_mixer = IntraBatchMixup(torch.nn.NLLLoss(weight=torch.Tensor(weight).to('cuda:0')), alpha=mixup_alpha)
        # self.batch_mixer = IntraBatchMixup(torch.nn.NLLLoss(), alpha=mixup_alpha)
        self.transpose_tfm = transpose_tfm

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        if self.transpose_tfm:
            x = x.squeeze(1).transpose(-1, -2) # (B, 1, F, T) -> (B, T, F)
        x = self.torch_module(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch 
        x, stacked_ys = self.batch_mixer.transform(x, y, train=True)
        preds = self(x)
        loss = self.batch_mixer.criterion(preds, stacked_ys)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, stacked_ys = self.batch_mixer.transform(x, y, train=False)
        preds = self(x)
        loss = self.batch_mixer.criterion(preds, stacked_ys)
        yhat = torch.argmax(preds, dim=1)
        acc = accuracy(yhat, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
