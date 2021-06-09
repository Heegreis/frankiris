import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from src.torch_module.M5 import M5 as Net
from pytorch_lightning.metrics.functional import accuracy
import torch


class M5(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.torch_module = Net()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.torch_module(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        loss = F.nll_loss(self(x).squeeze(), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.nll_loss(outputs.squeeze(), y)

        preds = torch.argmax(outputs.squeeze(), dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        return optimizer
