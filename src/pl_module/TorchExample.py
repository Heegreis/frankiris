import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from src.torch_module.TorchExample import Net


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

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
