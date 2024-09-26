import torch
from torch.nn import functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

class SuperResolutionVAEConfig:
    testing = False
    rgb_channels = 3
    n_embd = 64 # number of channels in the intermediate representation
    latent_dim = 4 # size of the latent space. Assuming input is 32x32 and latent_dim = 4, the latent space dimension is 4x8x8=256

class SuperResolutionVAEModel(nn.Module):
    def __init__(self, config: SuperResolutionVAEConfig):
        super().__init__()
        self.encoder = nn.ModuleDict(dict(
            conv1 = nn.Conv2d(config.rgb_channels, config.n_embd, kernel_size=3, stride=2, padding=1), 
            conv2 = nn.Conv2d(config.n_embd, config.latent_dim * 2, kernel_size=3, stride=2, padding=1),
            norm = nn.BatchNorm2d(config.latent_dim * 2)
        ))

        self.decoder = nn.ModuleDict(dict(
            conv2 = nn.ConvTranspose2d(config.latent_dim, config.n_embd * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm2 = nn.BatchNorm2d(config.n_embd * 2),
            conv1 = nn.ConvTranspose2d(config.n_embd * 2, config.n_embd, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm1 = nn.BatchNorm2d(config.n_embd),
            conv0 = nn.ConvTranspose2d(config.n_embd, config.rgb_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        ))
        self.testing = False
        if config.testing:
            self.testing = True
            self.activations = {}


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


    def loss_fn(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def test_forward(self, x, targets=None):
        B, _, H, W = x.size()
        # encoder
        y = F.relu(self.encoder.conv1(x)) # (B, n_embd, 16, 16)
        self.activations['encoder_relu1'] = y
        y = self.encoder.conv2(y) # (B, n_embd * 2, 8, 8)
        y = self.encoder.norm(y) 
        y = y.view(B, -1) # (B, latent_dim * 2 * 8 * 8) Flatten the output for the latent space


        # reparameterize
        mu, log_var = y.chunk(2, dim=1) # [(B, latent_dim * 8 * 8), (B, latent_dim * 8 * 8)]
        self.activations['mu'] = mu
        self.activations['log_var'] = log_var
        y = self.reparameterize(mu, log_var) # (B, latent_dim * 8 * 8)
        self.activations['latent_space'] = y

        # decoder
        # y = self.decoder.norm(y)
        y = y.view(B, -1, H // 4, W // 4) # (B, latent_dim, 8, 8)
        y = self.decoder.conv2(y) # (B, n_embd, 16, 16)
        y = y + F.relu(self.decoder.norm2(y)) 
        self.activations['decoder_relu2'] = y
        y = self.decoder.conv1(y) # (B, rgb_channels, 32, 32)
        y = y + F.relu(self.decoder.norm1(y)) 
        self.activations['decoder_relu1'] = y
        y = F.sigmoid(self.decoder.conv0(y)) # (B, rgb_channels, 64, 64)
        self.activations['decoder_sigmoid'] = y

        if targets is not None:
            loss = self.loss_fn(y, targets, mu, log_var)
            return y, loss
        return y

    def forward(self, x, targets=None):
        if self.testing:
            return self.test_forward(x, targets)

        B, _, H, W = x.size()
        # encoder
        y = F.relu(self.encoder.conv1(x)) # (B, n_embd, 16, 16)
        y = self.encoder.conv2(y) # (B, n_embd * 2, 8, 8)
        y = self.encoder.norm(y) 
        y = y.view(B, -1) # (B, latent_dim * 2 * 8 * 8) Flatten the output for the latent space


        # reparameterize
        mu, log_var = y.chunk(2, dim=1) # [(B, latent_dim * 8 * 8), (B, latent_dim * 8 * 8)]
        y = self.reparameterize(mu, log_var) # (B, latent_dim * 8 * 8)

        # decoder
        # y = self.decoder.norm(y)
        y = y.view(B, -1, H // 4, W // 4) # (B, latent_dim, 8, 8)
        y = self.decoder.conv2(y) # (B, n_embd, 16, 16)
        y = y + F.relu(self.decoder.norm2(y)) 
        y = self.decoder.conv1(y) # (B, rgb_channels, 32, 32)
        y = y + F.relu(self.decoder.norm1(y)) 
        y = F.sigmoid(self.decoder.conv0(y)) # (B, rgb_channels, 64, 64)

        if targets is not None:
            loss = self.loss_fn(y, targets, mu, log_var)
            return y, loss
        return y


if __name__ == "__main__":
    from gpt_dataset import get_dataloader
    model = SuperResolutionVAEModel(SuperResolutionVAEConfig())
    x = torch.randn((64, 3, 32, 32))
    y = model(x)
    print(f"{x.shape=}\t {y.shape=}\n{model=}")
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    for batch_idx, batch in enumerate(train_dataloader):
        x, target = batch
        print(f"{x.shape=}\t{target.shape=}")
        _, loss = model(x, target)
        print('val_loss', loss.item())
        break

