"""
correction_ae.py — AE de correction frame-par-frame.

Prend U_pred(t) (sortie légèrement erronée du surrogate) et prédit U_corrected(t).
La sortie est résiduelle : U_corrected = U_pred + decoder(z).

Architecture légère pour N=128 (base = N//8 = 16) :
  Encoder : 3 Conv2d (stride 2) → 64 × 16 × 16 → z ∈ R^latent_dim
  Decoder : fc + 2 ConvTranspose2d → (1, N//2, N//2) → Linear → (N, N)

Usage :
    ae = CorrectionAE(N=128, latent_dim=32)
    U_corr = ae(U_pred)          # (B, N, N) → (B, N, N)
    U_corr = ae.correct(U_pred)  # inférence no_grad
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionAE(nn.Module):
    """
    AE résiduel de correction frame-par-frame.

    Paramètres
    ----------
    N         : résolution spatiale (multiple de 8)
    latent_dim: dimension du code latent z
    """

    def __init__(self, N: int = 128, latent_dim: int = 32):
        super().__init__()
        self.N          = N
        self.latent_dim = latent_dim
        base = N // 8
        half = N // 2

        # --- Encodeur ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        flat = 64 * base * base
        self.fc_enc = nn.Linear(flat, latent_dim)

        # --- Décodeur (prédit le résidu) ---
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, flat),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  1, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            # → (B, 1, N//2, N//2)
        )
        self.out_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(half * half, N * N),
        )
        self._base = base

    def encode(self, U: torch.Tensor) -> torch.Tensor:
        """U : (B, N, N) ou (B, 1, N, N) → z : (B, latent_dim)"""
        if U.dim() == 3:
            U = U.unsqueeze(1)
        return self.fc_enc(self.encoder(U).flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, latent_dim) → résidu : (B, N, N)"""
        h = self.fc_dec(z).view(-1, 64, self._base, self._base)
        return self.out_fc(self.deconv(h)).view(-1, self.N, self.N)

    def forward(self, U_pred: torch.Tensor) -> torch.Tensor:
        """U_pred : (B, N, N) → U_corrected : (B, N, N)"""
        z = self.encode(U_pred)
        return U_pred + self.decode(z)

    @torch.no_grad()
    def correct(self, U_pred: torch.Tensor) -> torch.Tensor:
        """U_pred : (B, N, N) ou (N, N) → U_corrected même shape."""
        single = U_pred.dim() == 2
        if single:
            U_pred = U_pred.unsqueeze(0)
        out = self.forward(U_pred)
        return out.squeeze(0) if single else out

    def loss(self, U_corrected: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_corrected, U_true)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"CorrectionAE(N={self.N}, latent_dim={self.latent_dim}) — {n:,} params"
