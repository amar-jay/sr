from torch import nn
import torch
import pytorch_lightning as L
from torchmetrics.classification.accuracy import Accuracy

class ExampleConfig:
    out_channel = 5
    lr=1e-3
    pass

class ExampleModel(nn.Module):
    def __init__(self,config:ExampleConfig) -> None:
        self.config=config
        pass
    def forward(self, x, target=None):
        logits = x
        loss = None
        if target is not None:
            loss = False
        return logits, loss

class LitExample(L.LightningModule, ExampleModel):
    def __init__(self, config: ExampleConfig):
        super().__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=config.out_channel)
        self.config = config

    def training_step(self, batch, _):
        x, target = batch
        _, loss = self(x, target)
        return loss

    def validation_step(self, batch, _):
        x, target = batch
        logits, loss = self(x, target)
        self.log('val_loss', loss)

        acc = self.accuracy(logits, target)
        self.log('val_accuracy', acc)
        return loss

    def test_step(self, batch, _):
        x, target = batch
        logits, loss = self(x, target)
        self.log('test_loss', loss)

        acc = self.accuracy(logits, target)
        self.log('test_accuracy', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer


