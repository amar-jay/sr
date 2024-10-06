from dataclasses import dataclass
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as L
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


@dataclass
class SRResnetConfig:
    out_channels: int = 3
    in_channels: int = 3
    hidden_channel: int = 16  # beginning channel size of th hidden block
    lr: float = 1e-3
    num_blocks: int = 16
    is_training: bool = True
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
    stride: int = 2
    padding: int = 1  # best to be half or less than the kernel size
    kernel_size: int = 8

    def default(self):
        return self


class SRResnetModelv1(nn.Module):
    def initialize(self):
        for layer in self.parameters():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def __init__(self, config: SRResnetConfig) -> None:
        super().__init__()
        self.config = config

        self.conv1 = nn.ConvTranspose2d(
            config.in_channels, config.hidden_channel, kernel_size=config.kernel_size, stride=config.stride)
        # self.bn1= nn.BatchNorm2d(config.hidden_channel)
        # self.conv2 = nn.ConvTranspose2d(config.hidden_channel, config.in_channels, kernel_size=config.kernel_size, stride=config.stride)
        self.out_ch = nn.Tanh()

        self.initialize()

    @staticmethod
    def loss(x, target):
        # Reconstruction loss (binary cross-entropy for image data)
        recon_loss = F.binary_cross_entropy(x, target, reduction='sum')
        return recon_loss

    def forward(self, x, target=None):
        _x = self.conv1(x)  # pre-activation skip connection
        # x = F.leaky_relu(self.bn1(_x))
        # x = self.conv2(x)
        x = self.out_ch(_x)

        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x


class SRResnetModel(nn.Module):
    def initialize(self):
        for layer in self.parameters():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def __init__(self, config: SRResnetConfig) -> None:
        super().__init__()
        self.config = config

        modules = []

        self.in_channel = nn.Sequential(
            nn.Conv2d(config.in_channels, config.hidden_channel,
                      kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            nn.BatchNorm2d(config.hidden_channel)
        )

        hidden_channel = config.hidden_channel

        # encoder
        for i in range(2, config.num_blocks+2):
            modules += [
                nn.Conv2d(hidden_channel, hidden_channel*i,
                          kernel_size=config.kernel_size, stride=config.stride),
                nn.BatchNorm2d(hidden_channel*i),
            ]
            hidden_channel *= i
        self.encoder = nn.ModuleList(modules)

        # no bottleneck
        #
        # decoder
        modules = []
        for i in range(config.num_blocks):
            modules += [
                nn.ConvTranspose2d(hidden_channel, hidden_channel//(config.num_blocks+1-i),
                                   kernel_size=config.kernel_size, stride=config.stride),
                nn.BatchNorm2d(hidden_channel//(config.num_blocks+1-i)),
            ]
            hidden_channel //= config.num_blocks+1-i

        self.decoder = nn.ModuleList(modules)

        self.out_channel = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_channel, config.out_channels,
                               kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            nn.BatchNorm2d(config.out_channels),
        )
        self.sr_channel = nn.Sequential(
            nn.ConvTranspose2d(config.out_channels, config.out_channels,
                               kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
            nn.Tanh()
        )
        self.initialize()

    @staticmethod
    def loss(x, target):
        # Reconstruction loss (binary cross-entropy for image data)
        recon_loss = F.binary_cross_entropy(x, target, reduction='sum')
        return recon_loss

    def forward(self, _x, target=None):
        x = self.in_channel(_x)  # pre-activation skip connection
        skip_connections = [x]
        x = F.relu(x)
        for layer in self.encoder:
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                x = layer(x)
                x = x + F.leaky_relu(x)
                skip_connections.append(x)
            else:
                x = layer(x)

        skip_connections.pop()  # remove the last layer

        for layer in self.decoder:
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                prev = skip_connections.pop()
                x = layer(prev+x)
                x = x + F.leaky_relu(x)
            else:
                x = layer(x)

        # empty the skip connections
        skip_connections.clear()
        x = _x + F.leaky_relu(self.out_channel(x))
        x = self.sr_channel(x)

        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x


class LitSRResnet(SRResnetModel, L.LightningModule):
    def __init__(self, config: SRResnetConfig):
        super().__init__(config)
        self.pnsr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.config = config

    def training_step(self, batch, _):
        x, target = batch
        _, loss = self.model(x, target)
        return loss

    def validation_step(self, batch, _):
        x, target = batch
        logits, loss = self.model(x, target)

        psnr = self.pnsr(logits, target)
        ssim = self.ssim(logits, target)

        self.log('val_loss', loss)
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)

        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.adamw.AdamW(
            self.parameters(), lr=self.config.lr)
        return optimizer


if __name__ == "__main__":
    # test code
    """
    config = SRResnetConfig(
        out_channels=1,
        in_channels=1,
        hidden_channel=2,
        lr=1e-3,
        num_blocks=2,
        is_training=True
    )
    """
    config = SRResnetConfig(
        out_channels=3,
        in_channels=3,
        hidden_channel=3,
        lr=1e-3,
        num_blocks=5,
        kernel_size=3,
        is_training=True
    )
    model = LitSRResnet(config)
    print(model.config)
    print("-"*50)
    print(model)
    print("-"*50)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print("\nresults = ", y.shape,
          f"resolution upscale size= {y.size(2)/x.size(2)}x")
    _model = SRResnetModelv1(config)
    print(_model)
    print("-"*50)
    y = _model(x)
    print("\nresults = ", y.shape,
          f"resolution upscale size= {y.size(2)/x.size(2)}x")
