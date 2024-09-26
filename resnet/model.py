from dataclasses import dataclass
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as L
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

@dataclass
class SRResnetConfig:
    out_channels: int = 3
    in_channels: int = 3
    hidden_channel: int = 16 # beginning channel size of th hidden block
    lr: float = 1e-3
    num_blocks: int = 16
    is_training: bool = True

    def default(self):
        return self

class SRResnetModel(nn.Module):
    def initialize(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def __init__(self,config:SRResnetConfig) -> None:
        self.config=config

        modules = []

        self.in_channel = nn.Sequential(
                nn.Conv2d(config.in_channels, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
        )

        hidden_channel = config.hidden_channel

        # encoder
        for i in range(1, config.num_blocks+1):
            modules += [
                    nn.Conv2d(hidden_channel, hidden_channel*i, 3, 2, 1),
                    nn.BatchNorm2d(hidden_channel*i),
            ]
            hidden_channel *= i

        # no bottleneck
        #
        # decoder
        for i in range(1, config.num_blocks+1):
            modules += [
                    nn.Conv2d(hidden_channel, hidden_channel//i, 3, 2, 1),
                    nn.BatchNorm2d(hidden_channel//i),
            ]
            hidden_channel //= i

        self.block = nn.ModuleList(modules)

        self.out_channel = nn.Sequential(
            nn.Conv2d(16, 3, 3, 2, 1),
            nn.BatchNorm2d(3),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
            )


    @staticmethod
    def loss(x, target):
        # Reconstruction loss (binary cross-entropy for image data)
        recon_loss = F.binary_cross_entropy(x, target, reduction='sum')
        return recon_loss

    def forward(self, x, target=None):
        x = x + F.relu(self.in_channel(x)) # pre-activation skip connection
        _x = x
        for layer in self.block:
            x = layer(x)
            x = x + F.relu(x) if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) else _x
        x = _x + x

        x = self.out_channel(x)

        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x



class LitSRResnet(L.LightningModule, SRResnetModel):
    def __init__(self, config: SRResnetConfig):
        super().__init__()
        self.pnsr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.config = config

    def training_step(self, batch, _):
        x, target = batch
        _, loss = self(x, target)
        return loss

    def validation_step(self, batch, _):
        x, target = batch
        logits, loss = self(x, target)

        psnr = self.pnsr(logits, target)
        ssim = self.ssim(logits, target)

        self.log('val_loss', loss)
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer


