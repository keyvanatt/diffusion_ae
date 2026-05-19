import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.base import BaseAutoEncoder
from models.transient.conv_ae import ConvEncoder, ConvDecoder
from models.transient.learnable_laplace import LearnableLaplace


class SpatialLaplaceAE(BaseAutoEncoder):
    """
    AE dont la transformée de Laplace opère directement dans le domaine temporel (par pixel).

    Pipeline :
        U(t) ──LearnableLaplace(pixel-wise)──► Û(s_k)
             ──LaplaceEncoder(freq_ratio)──► z(s_k)
             ──LaplaceDecoder(freq_ratio)──► Ũ(s_k)
             ──LearnableLaplace⁻¹──► U_rec(t)

    Contrairement à LatentLaplaceAE, la transformée de Laplace opère directement sur
    les champs U(t) (chaque pixel est une série temporelle), et l'AE (LaplaceEncoder
    + LaplaceDecoder) compresse les K frames fréquentielles résultantes.
    Les K points s_k sont apprenables via LearnableLaplace.

    Paramètres
    ----------
    N          : résolution spatiale (multiple de 8)
    Nt         : nombre de pas de temps
    K          : nombre de fréquences de Laplace apprenables
    latent_dim : dimension de l'espace latent z
    dt         : pas de temps
    beta       : poids ridge sur Û (régularisation spectrale)
    beta_freq  : poids MSE dans le domaine de Laplace (qualité AE)
    freq_L     : niveaux sinusoïdaux pour l'encoding fréquentiel (FiLM)
    gamma_init : valeur initiale de l'amortissement γ
    """

    def __init__(
        self,
        N          : int   = 128,
        Nt         : int   = 150,
        K          : int   = 32,
        latent_dim : int   = 64,
        dt         : float = 1.0,
        beta       : float = 1e-3,
        beta_freq  : float = 1.0,
        freq_L     : int   = 8,
        gamma_init : float = 0.0,
    ):
        super().__init__()
        self.N         = N
        self.Nt        = Nt
        self.K         = K
        self.beta      = beta
        self.beta_freq = beta_freq

        self.laplace = LearnableLaplace(K, dt, Nt, gamma_init)
        self.encoder = ConvEncoder(in_channels=2, N=N, latent_dim=latent_dim, cond_L=freq_L)
        self.decoder = ConvDecoder(out_channels=2, N=N, latent_dim=latent_dim, cond_L=freq_L)

    def _make_freq_ratios(self, B: int, device: torch.device, dtype) -> torch.Tensor:
        """(B*K, 1) avec freq_ratio = k / max(K-1, 1) pour chaque fréquence."""
        fr = torch.arange(self.K, device=device, dtype=dtype) / max(self.K - 1, 1)  # (K,)
        return fr.unsqueeze(0).expand(B, -1).reshape(B * self.K, 1)                  # (B*K, 1)

    def _encode_freq_frames(self, U_hat: torch.Tensor) -> torch.Tensor:
        """
        U_hat : (B, K, N²) complex64
        → z   : (B, K, latent_dim)
        """
        B, K, NN = U_hat.shape
        N = self.N
        freq_ratios = self._make_freq_ratios(B, U_hat.device, torch.float32)  # (B*K, 1)

        frames = torch.stack([
            U_hat.real.reshape(B * K, N, N),
            U_hat.imag.reshape(B * K, N, N),
        ], dim=1)  # (B*K, 2, N, N)

        if self.training:
            chunks_f = frames.split(128)
            chunks_r = freq_ratios.split(128)
            z = torch.cat([
                checkpoint(self.encoder, f, r, use_reentrant=False)
                for f, r in zip(chunks_f, chunks_r)
            ])
        else:
            z = self.encoder(frames, freq_ratios)   # (B*K, latent_dim)

        return z.view(B, K, -1)  # (B, K, latent_dim)

    def _decode_freq_frames(self, z: torch.Tensor) -> torch.Tensor:
        """
        z     : (B, K, latent_dim)
        → Ũ   : (B, K, N²) complex64
        """
        B, K, D = z.shape
        N = self.N
        freq_ratios = self._make_freq_ratios(B, z.device, z.dtype)  # (B*K, 1)
        flat = z.reshape(B * K, D)

        if self.training:
            chunks_z = flat.split(128)
            chunks_r = freq_ratios.split(128)
            out = torch.cat([
                checkpoint(self.decoder, zc, r, use_reentrant=False)
                for zc, r in zip(chunks_z, chunks_r)
            ])
        else:
            out = self.decoder(flat, freq_ratios)  # (B*K, 2, N, N)

        re = out[:, 0].reshape(B, K, N * N)
        im = out[:, 1].reshape(B, K, N * N)
        return torch.complex(re, im)  # (B, K, N²) complex64

    def forward(
        self, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        U : (B, Nt, N, N) float32, normalisé
        retourne : (U_rec, U_hat, U_rec_hat, z)
          U_rec     : (B, Nt, N, N)        reconstruction temporelle
          U_hat     : (B, K, N²) complex   transformée de Laplace de U
          U_rec_hat : (B, K, N²) complex   reconstruction dans le domaine de Laplace
          z         : (B, K, latent_dim)   codes latents par fréquence
        """
        B, Nt, N, _ = U.shape
        assert N == self.N, f"N mismatch: got {N}, expected {self.N}"

        # 1. Transformée de Laplace pixel-par-pixel : (B, Nt, N²) → (B, K, N²) complex
        U_flat = U.reshape(B, Nt, N * N)
        U_hat  = self.laplace.forward_transform(U_flat)   # (B, K, N²) complex

        # 2. Encoder chaque frame fréquentielle
        z = self._encode_freq_frames(U_hat)               # (B, K, latent_dim)

        # 3. Décoder
        U_rec_hat = self._decode_freq_frames(z)           # (B, K, N²) complex

        # 4. Transformée inverse
        U_rec_flat = self.laplace.inverse_transform(U_rec_hat, Nt)  # (B, Nt, N²)
        U_rec      = U_rec_flat.reshape(B, Nt, N, N)

        return U_rec, U_hat, U_rec_hat, z

    def loss(
        self,
        U        : torch.Tensor,
        U_rec    : torch.Tensor,
        U_hat    : torch.Tensor,
        U_rec_hat: torch.Tensor,
        z        : torch.Tensor,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict]:
        """
        recon    : MSE temporelle U_rec vs U
        freq_rec : MSE dans le domaine de Laplace (qualité de l'AE fréquentiel)
        ridge    : L2 sur Û (régularisation spectrale)
        """
        recon    = F.mse_loss(U_rec, U)
        freq_rec = (U_hat - U_rec_hat).abs().pow(2).mean()
        ridge    = U_hat.abs().pow(2).mean()
        total    = recon + self.beta_freq * freq_rec + self.beta * ridge
        return total, {
            'recon'   : recon.detach(),
            'freq_rec': freq_rec.detach(),
            'ridge'   : ridge.detach(),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, Nt, N, K, D = 2, 20, 64, 16, 32
    model = SpatialLaplaceAE(N=N, Nt=Nt, K=K, latent_dim=D, dt=0.1, freq_L=8)

    n_total = sum(p.numel() for p in model.parameters())
    n_enc   = sum(p.numel() for p in model.encoder.parameters())
    n_dec   = sum(p.numel() for p in model.decoder.parameters())
    n_lap   = sum(p.numel() for p in model.laplace.parameters())
    print(f"Params — total: {n_total:,}  encoder: {n_enc:,}  decoder: {n_dec:,}  laplace: {n_lap:,}")

    U = torch.randn(B, Nt, N, N)
    U_rec, U_hat, U_rec_hat, z = model(U)
    assert U_rec.shape    == (B, Nt, N, N),    f"U_rec    {U_rec.shape}"
    assert U_hat.shape    == (B, K, N * N),    f"U_hat    {U_hat.shape}"
    assert U_rec_hat.shape == (B, K, N * N),   f"U_rec_hat {U_rec_hat.shape}"
    assert z.shape         == (B, K, D),        f"z        {z.shape}"
    print(f"U {tuple(U.shape)} → U_rec {tuple(U_rec.shape)}, U_hat {tuple(U_hat.shape)}, z {tuple(z.shape)}")

    loss, metrics = model.loss(U, U_rec, U_hat, U_rec_hat, z)
    loss.backward()
    print(f"Loss: {loss.item():.4f} | {metrics}")
    print(f"grad s_re: {model.laplace.s_re.grad is not None}")
    print(f"grad s_im: {model.laplace.s_im.grad is not None}")
