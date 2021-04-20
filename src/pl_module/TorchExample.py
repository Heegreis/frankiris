import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from src.torch_module.TorchExample import Net
from pytorch_lightning.metrics.functional import accuracy
import torch

class TorchExample(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.torch_module = Net()
        

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.torch_module(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)

        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
