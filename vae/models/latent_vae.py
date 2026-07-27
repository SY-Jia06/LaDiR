import torch
import torch.nn as nn


class LatentEncoder(nn.Module):
    """Compress LLM hidden states into continuous thought latents."""

    def __init__(self, hidden_dim=4096, latent_dim=512, num_latents=4,
                 depth=4, heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden_states):
        batch = hidden_states.size(0)
        queries = self.latents.unsqueeze(0).expand(batch, -1, -1)

        x = torch.cat([queries, hidden_states], dim=1)
        x = self.encoder(x)

        latent_states = x[:, :self.num_latents]
        mu = self.to_mu(latent_states)
        logvar = self.to_logvar(latent_states)

        return mu, logvar


class LatentDecoder(nn.Module):
    """Project latent thought tokens back to LLM hidden space."""

    def __init__(self, latent_dim=512, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z):
        return self.proj(z)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )


class LatentVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = LatentEncoder(**kwargs)
        self.decoder = LatentDecoder(
            latent_dim=kwargs.get("latent_dim", 512),
            hidden_dim=kwargs.get("hidden_dim", 4096),
        )

    def forward(self, hidden_states):
        mu, logvar = self.encoder(hidden_states)
        z = reparameterize(mu, logvar)
        return {
            "latent": z,
            "hidden": self.decoder(z),
            "mu": mu,
            "logvar": logvar,
        }
