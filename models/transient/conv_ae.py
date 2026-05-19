import torch
import torch.nn as nn

from models.transient.laplace_ae import SinusoidalFreqEncoding, _to_f_vec


class ConvEncoder(nn.Module):
    """
    Encodeur générique : (B, in_channels, N, N) → (B, latent_dim)
    conditionné sur un ratio scalaire ∈ [0, 1] via FiLM sinusoïdal.

    in_channels=1 : frame temporelle réelle U(t)
    in_channels=2 : frame fréquentielle complexe Û(s_k) — canaux Re et Im
    N doit être multiple de 8.
    """

    def __init__(self, in_channels: int, N: int, latent_dim: int, cond_L: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 4, 2, 1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.cond_enc = SinusoidalFreqEncoding(L=cond_L, hidden_dim=64, out_dim=64)
        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)
        conv_out = 64 * (N // 8) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, cond_ratio=0.0) -> torch.Tensor:
        """U : (B, in_channels, N, N) → z : (B, latent_dim)"""
        B = U.shape[0]
        emb = self.cond_enc(_to_f_vec(cond_ratio, B, U.dtype, U.device))
        x = self._film(self.conv1(U), self.film1, emb)
        x = self._film(self.conv2(x), self.film2, emb)
        x = self._film(self.conv3(x), self.film3, emb)
        return self.fc(x.flatten(1))


class ConvDecoder(nn.Module):
    """
    Décodeur générique : (B, latent_dim) → (B, out_channels, N, N)
    conditionné sur un ratio scalaire ∈ [0, 1] via FiLM sinusoïdal.

    out_channels=1 : frame temporelle réelle
    out_channels=2 : frame fréquentielle complexe — canaux Re et Im

    Backbone : FC + 3 deconv stride-2 → 32 canaux.
    Tête     : out_channels blocs de raffinement séparés Conv(32→32→32→1).
    N doit être multiple de 8.
    """

    def __init__(self, out_channels: int, N: int, latent_dim: int, cond_L: int = 8):
        super().__init__()
        self.base = N // 8
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.base ** 2), nn.ReLU())
        self.cond_enc = SinusoidalFreqEncoding(L=cond_L, hidden_dim=64, out_dim=64)
        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 1,  1),
            )
            for _ in range(out_channels)
        ])

    def _film(self, x: torch.Tensor, proj: nn.Linear, emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, cond_ratio=0.0) -> torch.Tensor:
        """z : (B, latent_dim) → (B, out_channels, N, N)"""
        B = z.shape[0]
        emb = self.cond_enc(_to_f_vec(cond_ratio, B, z.dtype, z.device))
        x = self.fc(z).view(B, 128, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, emb)
        x = self._film(self.deconv2(x), self.film2, emb)
        x = self._film(self.deconv3(x), self.film3, emb)
        return torch.cat([head(x) for head in self.heads], dim=1)
