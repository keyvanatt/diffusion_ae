"""
freq_latent_surrogate.py — Surrogate θ → latents pour LatentLaplaceAE et SpatialLaplaceAE.

Architecture commune : FreqLatentSurrogate (trunk MLP partagé + K FFN heads par fréquence)
                       + décodeur trainable (initialisé depuis l'AE)
                       + encodeur gelé + Laplace gelée (cibles latentes pendant l'entraînement).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from models.transient.conv_ae import ConvDecoder, ConvEncoder
from models.transient.learnable_laplace import LearnableLaplace
from models.transient.latent_laplace_ae import LatentLaplaceAE
from models.transient.spatial_laplace_ae import SpatialLaplaceAE


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """
    n_layers=2 → Linear(in→hidden) + LayerNorm + GELU → Linear(hidden→out)
    n_layers=3 → idem avec une couche cachée supplémentaire
    """
    layers = []
    for i in range(n_layers):
        d_in  = in_dim    if i == 0            else hidden_dim
        d_out = out_dim   if i == n_layers - 1 else hidden_dim
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# FreqLatentSurrogate
# ---------------------------------------------------------------------------

class FreqLatentSurrogate(nn.Module):
    """
    Trunk MLP partagé (θ → h) + K FFN heads par fréquence (h → z_k).

    Conditionnement fréquentiel : encoding sinusoïdal de k/(K-1) concaténé à h
    avant chaque head — plus simple que FiLM car les heads sont déjà spécialisés.

    Paramètres
    ----------
    theta_dim  : dimension de θ (entrée)
    out_dim    : dimension de sortie par fréquence
                   LatentLaplaceAE → 2·latent_dim  (Re + Im de ẑ)
                   SpatialLaplaceAE → latent_dim    (z réel)
    K          : nombre de fréquences de Laplace
    shared_dim : largeur du trunk
    head_dim   : largeur cachée de chaque head
    n_trunk    : nombre de couches Linear dans le trunk
    n_head     : nombre de couches Linear dans chaque head (≥ 2)
    freq_L     : niveaux sinusoïdaux pour l'encoding fréquentiel
    """

    def __init__(
        self,
        theta_dim  : int,
        out_dim    : int,
        K          : int,
        shared_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        self.K      = K
        self.freq_L = freq_L
        freq_cond   = 2 * freq_L

        trunk_layers = []
        for i in range(n_trunk):
            d_in = theta_dim if i == 0 else shared_dim
            trunk_layers += [nn.Linear(d_in, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()]
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            _make_mlp(shared_dim + freq_cond, head_dim, out_dim, n_head)
            for _ in range(K)
        ])

        freq_ratios = torch.arange(K).float() / max(K - 1, 1)
        self.register_buffer('_freq_ratios', freq_ratios)

    def _sinenc(self, ratios: torch.Tensor) -> torch.Tensor:
        """ratios: (K,) → (K, 2·freq_L)  encoding sinusoïdal style NeRF."""
        freqs = (2.0 ** torch.arange(self.freq_L, device=ratios.device, dtype=ratios.dtype)) * torch.pi
        x = ratios[:, None] * freqs[None, :]          # (K, freq_L)
        return torch.cat([x.sin(), x.cos()], dim=1)   # (K, 2·freq_L)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """theta: (B, theta_dim) → (B, K, out_dim)"""
        h        = self.trunk(theta)                   # (B, shared_dim)
        freq_enc = self._sinenc(self._freq_ratios)     # (K, 2·freq_L)
        B = h.shape[0]
        outs = []
        for k, head in enumerate(self.heads):
            e_k = freq_enc[k].unsqueeze(0).expand(B, -1)          # (B, 2·freq_L)
            outs.append(head(torch.cat([h, e_k], dim=1)))          # (B, out_dim)
        return torch.stack(outs, dim=1)                            # (B, K, out_dim)


# ---------------------------------------------------------------------------
# LatentLaplaceSurrogateModel
# ---------------------------------------------------------------------------

class LatentLaplaceSurrogateModel(nn.Module):
    """
    Surrogate θ → U_rec pour LatentLaplaceAE.

    Trainable : FreqLatentSurrogate (θ → ẑ_pred) + ConvDecoder(1) (z̃ → U)
    Gelé      : ConvEncoder(1) + LearnableLaplace (cibles latentes pendant le train)

    Pipeline forward :
        θ         ──surrogate──► ẑ_pred (B, K, D) complex
        U  ──enc(gelé)──lap(gelé)──► ẑ_true              [supervision latente]
        ẑ_pred ──lap⁻¹(gelé)──► z̃(B, Nt, D) ──decoder──► U_rec

    Pipeline generate :
        θ ──surrogate──► ẑ_pred ──lap⁻¹──► z̃ ──decoder──► U_rec
    """

    def __init__(
        self,
        ae         : LatentLaplaceAE,
        theta_dim  : int,
        shared_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        D  = ae.latent_dim
        K  = ae.laplace.K
        self.latent_dim = D
        self.K          = K
        self.Nt         = ae.Nt

        # Trainable
        self.surrogate = FreqLatentSurrogate(
            theta_dim, 2 * D, K, shared_dim, head_dim, n_trunk, n_head, freq_L,
        )
        self.decoder = copy.deepcopy(ae.decoder)

        # Gelés (cibles latentes + inversion Laplace)
        self.encoder = _freeze(copy.deepcopy(ae.encoder))
        self.laplace  = _freeze(copy.deepcopy(ae.laplace))

    def train(self, mode: bool = True):
        super().train(mode)
        # Toujours en eval pour activer le cache de LearnableLaplace
        self.encoder.eval()
        self.laplace.eval()
        return self

    # ------------------------------------------------------------------ internals

    @torch.no_grad()
    def _encode_targets(self, U: torch.Tensor) -> torch.Tensor:
        """U: (B, Nt, N, N) → ẑ_true: (B, K, D) complex"""
        B, Nt, N, _ = U.shape
        frames   = U.reshape(B * Nt, 1, N, N)
        t        = torch.arange(Nt, dtype=U.dtype, device=U.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        # Chunking pour la mémoire
        chunks_f = frames.split(256)
        chunks_t = t_ratios.split(256)
        z = torch.cat([self.encoder(f, t) for f, t in zip(chunks_f, chunks_t)])
        z = z.view(B, Nt, self.latent_dim)
        return self.laplace.forward_transform(z)   # (B, K, D) complex

    def _predict_z_hat(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """theta_norm: (B, θ_dim) → ẑ_pred: (B, K, D) complex"""
        D   = self.latent_dim
        out = self.surrogate(theta_norm)            # (B, K, 2D)
        return torch.complex(out[..., :D], out[..., D:])

    def _decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Nt, D) → U_rec: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t        = torch.arange(Nt, dtype=z.dtype, device=z.device) / max(Nt - 1, 1)
        t_ratios = t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)
        if self.training:
            chunks_z = flat.split(256)
            chunks_t = t_ratios.split(256)
            U = torch.cat([
                grad_ckpt(self.decoder, zc, tc, use_reentrant=False)
                for zc, tc in zip(chunks_z, chunks_t)
            ])
        else:
            U = self.decoder(flat, t_ratios)   # (B*Nt, 1, N, N)
        N = U.shape[-1]
        return U.view(B, Nt, N, N)

    # ------------------------------------------------------------------ API publique

    def forward(
        self, theta_norm: torch.Tensor, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (U_rec, ẑ_pred, ẑ_true).
          ẑ_pred : (B, K, D) complex  — sortie du surrogate
          ẑ_true : (B, K, D) complex  — cible encodeur+Laplace
          U_rec  : (B, Nt, N, N)
        """
        z_hat_true = self._encode_targets(U)
        z_hat_pred = self._predict_z_hat(theta_norm)
        z_tilde    = self.laplace.inverse_transform(z_hat_pred, self.Nt)
        U_rec      = self._decode_seq(z_tilde)
        return U_rec, z_hat_pred, z_hat_true

    def loss(
        self,
        U          : torch.Tensor,
        U_rec      : torch.Tensor,
        z_hat_pred : torch.Tensor,
        z_hat_true : torch.Tensor,
        alpha_lat  : float = 1.0,
        alpha_spat : float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        lat_loss  = (z_hat_pred - z_hat_true.detach()).abs().pow(2).mean()
        spat_loss = F.mse_loss(U_rec, U)
        total     = alpha_lat * lat_loss + alpha_spat * spat_loss
        return total, {'lat': lat_loss.detach(), 'spat': spat_loss.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """theta_norm: (B, θ_dim) → U_rec: (B, Nt, N, N)"""
        z_hat   = self._predict_z_hat(theta_norm)
        z_tilde = self.laplace.inverse_transform(z_hat, self.Nt)
        return self._decode_seq(z_tilde)


# ---------------------------------------------------------------------------
# SpatialLaplaceSurrogateModel
# ---------------------------------------------------------------------------

class SpatialLaplaceSurrogateModel(nn.Module):
    """
    Surrogate θ → U_rec pour SpatialLaplaceAE.

    Trainable : FreqLatentSurrogate (θ → z_pred) + ConvDecoder(2) (z → Ũ)
    Gelé      : ConvEncoder(2) + LearnableLaplace (cibles latentes pendant le train)

    Pipeline forward :
        θ         ──surrogate──► z_pred (B, K, D) réel
        U  ──lap(gelé)──enc(gelé)──► z_true              [supervision latente]
        z_pred ──decoder──► Ũ(s_k) ──lap⁻¹(gelé)──► U_rec

    Pipeline generate :
        θ ──surrogate──► z_pred ──decoder──► Ũ ──lap⁻¹──► U_rec
    """

    def __init__(
        self,
        ae         : SpatialLaplaceAE,
        theta_dim  : int,
        latent_dim : int | None = None,
        shared_dim : int = 256,
        head_dim   : int = 128,
        n_trunk    : int = 4,
        n_head     : int = 2,
        freq_L     : int = 6,
    ):
        super().__init__()
        D  = latent_dim if latent_dim is not None else ae.encoder.fc[-1].out_features
        K  = ae.K
        N  = ae.N
        self.latent_dim = D
        self.K          = K
        self.N          = N
        self.Nt         = ae.Nt

        # Trainable
        self.surrogate = FreqLatentSurrogate(
            theta_dim, D, K, shared_dim, head_dim, n_trunk, n_head, freq_L,
        )
        self.decoder = copy.deepcopy(ae.decoder)

        # Gelés
        self.encoder = _freeze(copy.deepcopy(ae.encoder))
        self.laplace  = _freeze(copy.deepcopy(ae.laplace))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.laplace.eval()
        return self

    # ------------------------------------------------------------------ internals

    def _freq_ratios(self, B: int, device, dtype) -> torch.Tensor:
        fr = torch.arange(self.K, device=device, dtype=dtype) / max(self.K - 1, 1)
        return fr.unsqueeze(0).expand(B, -1).reshape(B * self.K, 1)

    @torch.no_grad()
    def _encode_targets(self, U: torch.Tensor) -> torch.Tensor:
        """U: (B, Nt, N, N) → z_true: (B, K, D)"""
        B, Nt, N, _ = U.shape
        U_flat = U.reshape(B, Nt, N * N)
        U_hat  = self.laplace.forward_transform(U_flat)   # (B, K, N²) complex
        freq_r = self._freq_ratios(B, U.device, torch.float32)
        frames = torch.stack([
            U_hat.real.reshape(B * self.K, N, N),
            U_hat.imag.reshape(B * self.K, N, N),
        ], dim=1)  # (B*K, 2, N, N)
        chunks_f = frames.split(128)
        chunks_r = freq_r.split(128)
        z = torch.cat([self.encoder(f, r) for f, r in zip(chunks_f, chunks_r)])
        return z.view(B, self.K, self.latent_dim)

    def _decode_freq_frames(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, K, D) → Ũ: (B, K, N²) complex"""
        B, K, D = z.shape
        N  = self.N
        fr = self._freq_ratios(B, z.device, z.dtype)
        flat = z.reshape(B * K, D)
        if self.training:
            chunks_z = flat.split(128)
            chunks_r = fr.split(128)
            out = torch.cat([
                grad_ckpt(self.decoder, zc, r, use_reentrant=False)
                for zc, r in zip(chunks_z, chunks_r)
            ])
        else:
            out = self.decoder(flat, fr)          # (B*K, 2, N, N)
        re = out[:, 0].reshape(B, K, N * N)
        im = out[:, 1].reshape(B, K, N * N)
        return torch.complex(re, im)              # (B, K, N²) complex

    # ------------------------------------------------------------------ API publique

    def forward(
        self, theta_norm: torch.Tensor, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (U_rec, z_pred, z_true).
          z_pred : (B, K, D) réel  — sortie du surrogate
          z_true : (B, K, D) réel  — cible encodeur (gelé)
          U_rec  : (B, Nt, N, N)
        """
        z_true  = self._encode_targets(U)
        z_pred  = self.surrogate(theta_norm)
        U_tilde = self._decode_freq_frames(z_pred)   # (B, K, N²) complex
        U_rec   = self.laplace.inverse_transform(U_tilde, self.Nt)   # (B, Nt, N²)
        U_rec   = U_rec.reshape(U.shape[0], self.Nt, self.N, self.N)
        return U_rec, z_pred, z_true

    def loss(
        self,
        U          : torch.Tensor,
        U_rec      : torch.Tensor,
        z_pred     : torch.Tensor,
        z_true     : torch.Tensor,
        alpha_lat  : float = 1.0,
        alpha_spat : float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        lat_loss  = F.mse_loss(z_pred, z_true.detach())
        spat_loss = F.mse_loss(U_rec, U)
        total     = alpha_lat * lat_loss + alpha_spat * spat_loss
        return total, {'lat': lat_loss.detach(), 'spat': spat_loss.detach()}

    @torch.no_grad()
    def generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """theta_norm: (B, θ_dim) → U_rec: (B, Nt, N, N)"""
        z_pred  = self.surrogate(theta_norm)
        U_tilde = self._decode_freq_frames(z_pred)
        U_rec   = self.laplace.inverse_transform(U_tilde, self.Nt)
        return U_rec.reshape(z_pred.shape[0], self.Nt, self.N, self.N)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, Nt, N, D, K, theta_dim = 2, 20, 64, 32, 12, 3

    print("=== LatentLaplaceSurrogateModel ===")
    ae_lat = LatentLaplaceAE(N=N, Nt=Nt, latent_dim=D, K=K, dt=0.1, time_L=6)
    model_lat = LatentLaplaceSurrogateModel(ae_lat, theta_dim=theta_dim,
                                             shared_dim=128, head_dim=64, freq_L=4)
    n_sur = sum(p.numel() for p in model_lat.surrogate.parameters())
    n_dec = sum(p.numel() for p in model_lat.decoder.parameters())
    print(f"  surrogate: {n_sur:,}  decoder: {n_dec:,}")

    theta = torch.randn(B, theta_dim)
    U     = torch.randn(B, Nt, N, N)

    model_lat.train()
    U_rec, z_hat_pred, z_hat_true = model_lat(theta, U)
    assert U_rec.shape    == (B, Nt, N, N),   f"U_rec {U_rec.shape}"
    assert z_hat_pred.shape == (B, K, D),     f"z_hat_pred {z_hat_pred.shape}"
    assert z_hat_true.shape == (B, K, D),     f"z_hat_true {z_hat_true.shape}"

    loss, metrics = model_lat.loss(U, U_rec, z_hat_pred, z_hat_true)
    loss.backward()
    print(f"  loss={loss.item():.4f} | {metrics}")
    assert model_lat.surrogate.trunk[0].weight.grad is not None
    assert model_lat.decoder.fc[0].weight.grad is not None
    assert model_lat.encoder.fc[0].weight.grad is None   # gelé

    model_lat.eval()
    U_gen = model_lat.generate(theta)
    assert U_gen.shape == (B, Nt, N, N)
    print(f"  generate OK → {tuple(U_gen.shape)}")

    print("\n=== SpatialLaplaceSurrogateModel ===")
    ae_spa = SpatialLaplaceAE(N=N, Nt=Nt, K=K, latent_dim=D, dt=0.1, freq_L=6)
    model_spa = SpatialLaplaceSurrogateModel(ae_spa, theta_dim=theta_dim,
                                              shared_dim=128, head_dim=64, freq_L=4)
    n_sur = sum(p.numel() for p in model_spa.surrogate.parameters())
    n_dec = sum(p.numel() for p in model_spa.decoder.parameters())
    print(f"  surrogate: {n_sur:,}  decoder: {n_dec:,}")

    model_spa.train()
    U_rec2, z_pred, z_true = model_spa(theta, U)
    assert U_rec2.shape == (B, Nt, N, N),  f"U_rec {U_rec2.shape}"
    assert z_pred.shape == (B, K, D),      f"z_pred {z_pred.shape}"
    assert z_true.shape == (B, K, D),      f"z_true {z_true.shape}"

    loss2, metrics2 = model_spa.loss(U, U_rec2, z_pred, z_true)
    loss2.backward()
    print(f"  loss={loss2.item():.4f} | {metrics2}")
    assert model_spa.surrogate.trunk[0].weight.grad is not None
    assert model_spa.decoder.fc[0].weight.grad is not None
    assert model_spa.encoder.fc[0].weight.grad is None   # gelé

    model_spa.eval()
    U_gen2 = model_spa.generate(theta)
    assert U_gen2.shape == (B, Nt, N, N)
    print(f"  generate OK → {tuple(U_gen2.shape)}")

    print("\nTous les smoke tests OK.")
