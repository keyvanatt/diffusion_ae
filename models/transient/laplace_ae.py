import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseAutoEncoder
import math


def _to_f_vec(freq_ratio, B: int, dtype, device) -> torch.Tensor:
    """Convertit freq_ratio (float scalaire ou tenseur (B,)) en (B, 1)."""
    if isinstance(freq_ratio, (float, int)):
        return torch.full((B, 1), float(freq_ratio), dtype=dtype, device=device)
    return freq_ratio.to(dtype=dtype, device=device).view(B, 1)


class SinusoidalFreqEncoding(nn.Module):
    """
    Encode un scalaire f ∈ [0, 1] en un vecteur de dimension 2·L via un
    positional encoding sinusoïdal (style NeRF), avant de le passer dans
    un petit MLP pour obtenir l'embedding de conditionnement FiLM.

    Pour le ième niveau de fréquence (i = 0, …, L-1) :
        sin(2^i · π · f),  cos(2^i · π · f)

    Les fréquences couvrent [2^0 · π, 2^(L-1) · π], ce qui donne au réseau
    une représentation dense de la position spectrale : basses fréquences pour
    la structure globale, hautes fréquences pour discriminer des fréquences
    de Laplace voisines.

    Paramètres
    ----------
    L          : nombre de niveaux de fréquence → vecteur de dim 2·L
    hidden_dim : dimension cachée du MLP qui projette l'encoding
    out_dim    : dimension de l'embedding de sortie (utilisée pour FiLM)
    """

    def __init__(self, L: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.L = L
        # Fréquences 2^i · π, i = 0, …, L-1  (shape : (L,))
        freqs = math.pi * (2.0 ** torch.arange(L).float())
        self.register_buffer('freqs', freqs)   # non-trainable

        # MLP : 2L → hidden_dim → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * L, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f : (B, 1) — freq_ratio normalisé ∈ [0, 1]
        retourne : (B, out_dim)
        """
        # (B, 1) × (L,) → (B, L)  (broadcasting)
        angles = f * self.freqs                           # (B, L)
        enc = torch.cat([angles.sin(), angles.cos()], dim=1)  # (B, 2L)
        return self.mlp(enc)                              # (B, out_dim)


class LaplaceEncoder(nn.Module):
    """
    U → z, conditionné sur freq_ratio = k / K ∈ [0, 1].

    U : (B, 2, N, N)  champ spatial complexe (Re, Im), normalisé
    """

    def __init__(self, N: int, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N = N

        # 3 downsampling steps → résolution de sortie : N // 8
        self.conv1 = nn.Sequential(nn.Conv2d(2,   16,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,  32,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))

        # Encodeur de fréquence sinusoïdal
        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        # Projections FiLM (gamma, beta) par bloc conv
        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)

        conv_out = 64 * (N // 8) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        """FiLM : x ← x · (1 + γ) + β,  γ et β conditionnés sur freq_ratio."""
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)   # borne gamma dans [-1, 1] → évite l'explosion fp16
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne z : (B, latent_dim)."""
        B = U.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, U.dtype, U.device))  # (B, 64)

        x = self._film(self.conv1(U), self.film1, f_emb)
        x = self._film(self.conv2(x), self.film2, f_emb)
        x = self._film(self.conv3(x), self.film3, f_emb)
        h = x.flatten(start_dim=1)                               # (B, conv_out)
        return self.fc(h)                                         # (B, latent_dim)

class LaplaceDecoder(nn.Module):
    """
    z → Û (complexe, normalisé), conditionné sur freq_ratio = k / K ∈ [0, 1].

    z : (B, latent_dim)
    retourne : (B, 2, N, N)  — canaux Re et Im

    Architecture :
    - base = N // 8 → 3 étapes d'upsampling.
    - FiLM conditioning (sinusoïdal) après chaque bloc deconv.
    - Deux blocs de raffinement séparés (Re / Im) pour les détails fins.
    """

    def __init__(self, N: int = 64, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N    = N
        self.base = N // 8  # résolution de départ avant 3 upsampling

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.base ** 2),
            nn.ReLU(),
        )

        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        # Projections FiLM par bloc deconv
        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)

        # Blocs deconv séparés pour intercaler le FiLM
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())

        # Blocs de raffinement séparés Re / Im
        # Chacun produit un seul canal de sortie (1, pas 2 — corrige le bug original)
        self.refine_re = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),            # → (B, 1, N, N)
        )
        self.refine_im = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),            # → (B, 1, N, N)
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)   # borne gamma dans [-1, 1] → évite l'explosion fp16
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne Û : (B, 2, N, N)."""
        B = z.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, z.dtype, z.device))  # (B, 64)

        x = self.fc(z).view(B, 128, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, f_emb)
        x = self._film(self.deconv2(x), self.film2, f_emb)
        x = self._film(self.deconv3(x), self.film3, f_emb)

        re = self.refine_re(x)    # (B, 1, N, N)
        im = self.refine_im(x)    # (B, 1, N, N)
        return torch.cat([re, im], dim=1)   # (B, 2, N, N)


class LaplaceAE(BaseAutoEncoder):
    """
    Autoencoder déterministe conditionné sur la fréquence de Laplace k/K.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 64)
    latent_dim : dimension de l'espace latent z
    beta       : poids de la régularisation ridge (L2 sur la sortie)
    freq_L     : nombre de niveaux de fréquence pour le sinusoidal encoding
    """

    def __init__(
        self,
        N          : int   = 64,
        latent_dim : int   = 32,
        beta       : float = 1e-3,
        freq_L     : int   = 8,
    ):
        super().__init__()

        self.beta       = beta
        self.latent_dim = latent_dim

        self.encoder = LaplaceEncoder(N, latent_dim, freq_L=freq_L)
        self.decoder = LaplaceDecoder(N, latent_dim, freq_L=freq_L)

    def forward(
        self, U: torch.Tensor, freq_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne : (Û, z)

        Û : (B, 2, N, N)  reconstruction normalisée
        z : (B, latent_dim)  code latent
        """
        z     = self.encoder(U, freq_ratio)
        U_hat = self.decoder(z, freq_ratio)
        return U_hat, z

    def loss(
        self,
        U     : torch.Tensor,
        U_hat : torch.Tensor,
        z     : torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(U_hat, U)
        ridge      = z.pow(2).mean()   # régularisation L2 sur le latent
        total      = recon_loss + self.beta * ridge
        return total, {
            'recon_loss': recon_loss.detach(),
            'ridge'     : ridge.detach(),
        }


if __name__ == "__main__":
    ae = LaplaceAE(N=128, latent_dim=64, freq_L=8)
    total_params   = sum(p.numel() for p in ae.parameters())
    encoder_params = sum(p.numel() for p in ae.encoder.parameters())
    decoder_params = sum(p.numel() for p in ae.decoder.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"Encoder parameters : {encoder_params:,}")
    print(f"Decoder parameters : {decoder_params:,}")
    print(f"Decoder / Total ratio : {decoder_params / total_params:.2%}")
    print(ae)

    U = torch.randn(2, 2, 128, 128)
    U_hat, z = ae(U, freq_ratio=0.75)
    assert U_hat.shape == U.shape,  f"U_hat : attendu {U.shape}, obtenu {U_hat.shape}"
    assert z.shape == (2, 64),      f"z : attendu (2, 64), obtenu {z.shape}"
    print(f"U {U.shape} → U_hat {U_hat.shape}, z {z.shape} ")

    loss, metrics = ae.loss(U, U_hat, z)
    print(f"Loss : {loss.item():.4f}  |  metrics : {metrics}")

    enc = SinusoidalFreqEncoding(L=8, out_dim=64)
    f_test = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    emb = enc(f_test)
    assert emb.shape == (5, 64)
    print(f"SinusoidalFreqEncoding : {f_test.flatten().tolist()} → embeddings shape {emb.shape}")
