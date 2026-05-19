import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.base import BaseAutoEncoder
from models.transient.conv_ae import ConvEncoder, ConvDecoder
from models.transient.learnable_laplace import LearnableLaplace


# ---------------------------------------------------------------------------
# Modèle principal
# ---------------------------------------------------------------------------

class LatentLaplaceAE(BaseAutoEncoder):
    """
    Autoencoder dont la transformée de Laplace opère dans l'espace latent,
    avec conditionnement FiLM sur t_ratio = t / (Nt-1) dans l'encodeur et le décodeur.

    Pipeline :
        U(t) ──encoder(t)──► z(t) ──Laplace──► ẑ(s_k) ──Laplace⁻¹──► z̃(t) ──decoder(t)──► Û(t)

    Paramètres
    ----------
    N           : résolution spatiale (multiple de 8)
    Nt          : nombre de pas de temps
    latent_dim  : dimension de l'espace latent z
    K           : nombre de fréquences de Laplace apprenables
    dt          : pas de temps (fixe)
    beta        : poids ridge sur z (régularisation latente)
    beta_latent : poids de la MSE de reconstruction dans l'espace latent
    gamma_init  : valeur initiale de l'amortissement γ
    time_L      : niveaux de fréquence pour l'encoding sinusoïdal du temps
    """

    def __init__(
        self,
        N          : int   = 64,
        Nt         : int   = 150,
        latent_dim : int   = 64,
        K          : int   = 64,
        dt         : float = 1.0,
        beta       : float = 1e-3,
        beta_latent: float = 0.1,
        gamma_init : float = 0.5,
        time_L     : int   = 8,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.Nt          = Nt
        self.beta        = beta
        self.beta_latent = beta_latent

        self.encoder = ConvEncoder(in_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)
        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)
        self.laplace  = LearnableLaplace(K, dt, Nt, gamma_init)

    # ------------------------------------------------------------------
    # Utilitaires séquence
    # ------------------------------------------------------------------

    def _make_t_ratios(self, Nt: int, B: int, dtype, device) -> torch.Tensor:
        """Retourne (B*Nt, 1) avec t_ratio = t / max(Nt-1, 1)."""
        t = torch.arange(Nt, dtype=dtype, device=device) / max(Nt - 1, 1)  # (Nt,)
        t = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)                 # (B*Nt, 1)
        return t

    def _encode_seq(self, U: torch.Tensor) -> torch.Tensor:
        """U : (B, Nt, N, N) → z : (B, Nt, latent_dim)"""
        B, Nt, N, _ = U.shape
        frames   = U.reshape(B * Nt, 1, N, N)              # (B*Nt, 1, N, N)
        t_ratios = self._make_t_ratios(Nt, B, U.dtype, U.device)  # (B*Nt, 1)

        if self.training:
            chunks_f = frames.split(256)
            chunks_t = t_ratios.split(256)
            z = torch.cat([
                checkpoint(self.encoder, f, t, use_reentrant=False)
                for f, t in zip(chunks_f, chunks_t)
            ])
        else:
            z = self.encoder(frames, t_ratios)
        return z.view(B, Nt, self.latent_dim)

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, Nt, latent_dim) → U : (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._make_t_ratios(Nt, B, z.dtype, z.device)  # (B*Nt, 1)

        if self.training:
            chunks_z = flat.split(256)
            chunks_t = t_ratios.split(256)
            U = torch.cat([
                checkpoint(self.decoder, zc, t, use_reentrant=False)
                for zc, t in zip(chunks_z, chunks_t)
            ])
        else:
            U = self.decoder(flat, t_ratios)
        N = U.shape[-1]
        return U.view(B, Nt, N, N)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        U : (B, Nt, N, N)
        retourne : (U_rec, z_hat, z, z_rec)
          U_rec : (B, Nt, N, N)  reconstruction spatiale
          z_hat : (B, K, latent_dim)  latents transformés via Laplace
          z     : (B, Nt, latent_dim)  latents encodés
          z_rec : (B, Nt, latent_dim)  latents reconstruits via Laplace
        """
        Nt    = U.shape[1]
        z     = self._encode_seq(U)
        z_hat = self.laplace.forward_transform(z)            # (B, K, D) complex
        z_rec = self.laplace.inverse_transform(z_hat, Nt)   # (B, Nt, D)
        U_rec = self._decode_seq(z_rec)
        return U_rec, z_hat, z, z_rec

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        U    : torch.Tensor,
        U_rec: torch.Tensor,
        z_hat: torch.Tensor,
        z    : torch.Tensor,
        z_rec: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        recon    : MSE spatiale U_rec vs U
        lat_rec  : MSE latente z_rec vs z  (qualité du roundtrip Laplace)
        ridge    : L2 sur z
        """
        recon   = F.mse_loss(U_rec, U)
        lat_rec = F.mse_loss(z_rec, z)
        ridge   = z_hat.abs().pow(2).mean()
        total   = recon + self.beta_latent * lat_rec + self.beta * ridge
        return total, {
            'recon'  : recon.detach(),
            'lat_rec': lat_rec.detach(),
            'ridge'  : ridge.detach(),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, Nt, N, D, K = 2, 20, 64, 32, 20
    model = LatentLaplaceAE(N=N, Nt=Nt, latent_dim=D, K=K, dt=0.1, time_L=8)

    n_total = sum(p.numel() for p in model.parameters())
    n_enc   = sum(p.numel() for p in model.encoder.parameters())
    n_dec   = sum(p.numel() for p in model.decoder.parameters())
    n_lap   = sum(p.numel() for p in model.laplace.parameters())
    print(f"Params — total: {n_total:,}  encoder: {n_enc:,}  decoder: {n_dec:,}  laplace: {n_lap:,}")

    U = torch.randn(B, Nt, N, N)
    U_rec, z_hat, z, z_rec = model(U)
    assert U_rec.shape == (B, Nt, N, N), f"U_rec shape {U_rec.shape}"
    assert z_hat.shape == (B, K, D),     f"z_hat shape {z_hat.shape}"
    assert z.shape     == (B, Nt, D),    f"z shape {z.shape}"
    assert z_rec.shape == (B, Nt, D),    f"z_rec shape {z_rec.shape}"
    print(f"U {tuple(U.shape)} → U_rec {tuple(U_rec.shape)}, z_hat {tuple(z_hat.shape)}, z {tuple(z.shape)}, z_rec {tuple(z_rec.shape)}")

    loss, metrics = model.loss(U, U_rec, z_hat, z, z_rec)
    loss.backward()
    print(f"Loss: {loss.item():.4f} | {metrics}")

    print(f"grad s_re: {model.laplace.s_re.grad is not None}")
    print(f"grad s_im: {model.laplace.s_im.grad is not None}")
    print(f"grad log_alpha_t: {model.laplace.log_alpha_t.grad is not None}")
    print(f"grad log_lam: {model.laplace.log_lam.grad is not None}")
