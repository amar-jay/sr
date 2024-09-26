from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as L
from torchmetrics.classification.accuracy import Accuracy

class SRVaeConfig:
    out_channel = 5
    lr=1e-3
    pass

class SRVaeModel(nn.Module):
    def __init__(self,config:SRVaeConfig) -> None:
        self.config=config

        self.encoder = nn.ModuleList([
                nn.Conv2d(1, 16, 3, 2, 1),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(64*7*7, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 64),
        ])

        self.z_mean = nn.Linear(64, config.out_channel)
        self.z_log_var = nn.Linear(64, config.out_channel)

        self.decoder = nn.Sequential(
                nn.Linear(config.out_channel, 64),
                nn.Linear(64, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 64*7*7),
                nn.Unflatten(1, (64, 7, 7)),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                nn.BatchNorm2d(16),
                nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
                nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
                nn.Sigmoid()
        )

    @staticmethod
    def reparameterize(z_mean, z_log_var):
        """ z ~ N(z| z_mu, z_logvar) """
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon


    @staticmethod
    def loss(x, x_recon, mu, logvar, beta):
        # Reconstruction loss (binary cross-entropy for image data)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_div
        
        return total_loss, recon_loss, kl_div

    def forward(self, x, target=None):
        for layer in self.encoder:
            _x = layer(x)
            x = F.relu(_x) if isinstance(layer, nn.Conv2d) else _x

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        z = self.reparameterize(z_mean, z_log_var)

        for layer in self.decoder:
            _z = layer(z)
            if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                z = z + F.leaky_relu(_z)
            else:
                z = _z
        if target is not None:
            loss, recon_loss, kl_div = self.loss(x, z, z_mean, z_log_var, beta=1)
            return z, recon_loss, kl_div, loss
        return z



class LitSRVae(L.LightningModule, SRVaeModel):
    def __init__(self, config: SRVaeConfig):
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


