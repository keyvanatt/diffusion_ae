"""
latent_svd_laplace.py — Surrogate θ→U(t) via SVD espace latent + Laplace.

Pipeline :
  Phase 1 — LatentLaplaceAE pré-entraîné (inchangé).
  Phase 2 — offline :
      z(t) = encoder(U(t))         [n, Nt, latent_dim]
      Z_flat = reshape(z, [n·Nt, D])  →  SVD tronquée  →  V [D, k_svd]
      G(t)  = Z_flat @ V           →  reshape [n, Nt, k_svd]
      Ĝ(k) = rfft(G · quad_w)[:K] →  [n, K, k_svd] complex
  Phase 3 — entraîne LatentSVDLaplaceModel :
      θ → proj → Ĝ_norm [B, K, k_svd, 2]
      → dénorm → irfft → G̃(t) [B, Nt, k_svd]
      → @ V.T → z̃(t) [B, Nt, D] → decoder(t_ratio) → Û(t) [B, Nt, N, N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseDecoder
from models.transient.conv_ae import ConvDecoder


class LatentSVDLaplaceModel(BaseDecoder):
    """
    Surrogate θ → Û(t) via base SVD espace latent + transformée de Laplace.

    Buffers gelés (calculés offline depuis un LatentLaplaceAE) :
      V          [latent_dim, k_svd]   base orthonormée SVD tronquée
      G_hat_mean [K, k_svd, 2]         moyenne de Ĝ (Re/Im) sur le train set
      G_hat_std  [K, k_svd, 2]         écart-type de Ĝ sur le train set
      _quad_w    [Nt]                   dt · w_trap · exp(-γ·t)
      U_mean     [N, N]                 moyenne pixel U train
      U_std      [N, N]                 std pixel U train

    Module appris :
      proj  MLP  θ_norm → Ĝ_norm [B, K, k_svd, 2]  (Re et Im séparés)

    Décodeur (gelé, copié depuis LatentLaplaceAE) :
      decoder  ConvDecoder

    Paramètres
    ----------
    N, Nt, theta_dim, latent_dim : doivent correspondre au LatentLaplaceAE source
    k_svd  : nombre de modes SVD retenus
    K      : nombre de fréquences Laplace (≤ Nt//2+1)
    dt     : pas de temps
    gamma  : amortissement Laplace (0 = Fourier pur)
    time_L : niveaux FiLM (doit correspondre au LatentLaplaceAE source)
    """

    def __init__(
        self,
        N          : int   = 128,
        Nt         : int   = 150,
        theta_dim  : int   = 3,
        latent_dim : int   = 64,
        k_svd      : int   = 16,
        K          : int   = 75,
        dt         : float = 1.0,
        gamma      : float = 0.0,
        time_L     : int   = 8,
    ):
        super().__init__()
        self.N          = N
        self.Nt         = Nt
        self.k_svd      = k_svd
        self.latent_dim = latent_dim
        self.K          = K
        self.dt         = dt
        self.gamma      = gamma

        # ── Buffers (remplis via set_svd_basis / set_normalization) ──────
        self.register_buffer('V',          torch.zeros(latent_dim, k_svd))
        self.register_buffer('G_hat_mean', torch.zeros(K, k_svd, 2))
        self.register_buffer('G_hat_std',  torch.ones( K, k_svd, 2))

        t = torch.arange(Nt, dtype=torch.float32) * dt
        w = torch.ones(Nt, dtype=torch.float32); w[0] = 0.5; w[-1] = 0.5
        self.register_buffer('_quad_w',
            dt * w * torch.exp(torch.tensor(-gamma, dtype=torch.float32) * t))

        self.register_buffer('U_mean',     torch.zeros(N, N))
        self.register_buffer('U_std',      torch.ones( N, N))
        self.register_buffer('theta_mean', torch.zeros(theta_dim))
        self.register_buffer('theta_std',  torch.ones( theta_dim))

        # ── MLP surrogate ─────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(theta_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),                          nn.ReLU(),
            nn.Linear(256, K * k_svd * 2),
        )

        # ── Décodeur (gelé après load_ae_decoder) ────────────────────────
        self.decoder = ConvDecoder(out_channels=1, N=N, latent_dim=latent_dim, cond_L=time_L)

    # ── Setters ──────────────────────────────────────────────────────────────

    def set_svd_basis(self, V):
        """V : Tensor ou ndarray (latent_dim, k_svd)."""
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=torch.float32)
        self.V.copy_(V)

    def set_normalization(self, G_hat_mean, G_hat_std, U_mean, U_std, theta_mean, theta_std):
        def _t(x):
            return x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.G_hat_mean.copy_(_t(G_hat_mean))
        self.G_hat_std.copy_( _t(G_hat_std))
        self.U_mean.copy_(    _t(U_mean))
        self.U_std.copy_(     _t(U_std))
        self.theta_mean.copy_(_t(theta_mean))
        self.theta_std.copy_( _t(theta_std))

    def load_ae_decoder(self, ae):
        """Copie le décodeur depuis un LatentLaplaceAE et le gèle."""
        self.decoder.load_state_dict(ae.decoder.state_dict())
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    # ── Forward (entraînement sur loss spectrale) ─────────────────────────────

    def forward(self, theta_norm):
        """theta_norm: (B, θ_dim) → Ĝ_norm: (B, K, k_svd, 2)"""
        B = theta_norm.shape[0]
        return self.proj(theta_norm).view(B, self.K, self.k_svd, 2)

    def loss(self, G_hat_pred_norm, G_hat_true_norm):
        """MSE sur le spectre de Laplace normalisé."""
        mse = F.mse_loss(G_hat_pred_norm, G_hat_true_norm)
        return mse, {'spec': mse.detach()}

    # ── Inférence ────────────────────────────────────────────────────────────

    def _t_ratios(self, Nt, B, dtype, device):
        t = torch.arange(Nt, dtype=dtype, device=device) / max(Nt - 1, 1)
        return t.unsqueeze(0).expand(B, -1).reshape(B * Nt, 1)

    def _decode_seq(self, z):
        """z: (B, Nt, latent_dim) → U_norm: (B, Nt, N, N)"""
        B, Nt, D = z.shape
        flat     = z.reshape(B * Nt, D)
        t_ratios = self._t_ratios(Nt, B, z.dtype, z.device)
        return self.decoder(flat, t_ratios).view(B, Nt, self.N, self.N)

    def _inverse_laplace(self, G_hat_phys):
        """
        G_hat_phys : (B, K, k_svd) complex
        → G_tilde  : (B, Nt, k_svd) float

        Inverse via irfft (zero-pad si K < Nt//2+1) + correction quadrature.
        Convention : Ĝ = rfft(G · dt · w_trap · exp(-γ·t)).
        """
        device = G_hat_phys.device
        # (B, k_svd, K) → irfft sur dim=-1 → (B, k_svd, Nt)
        a_rec   = torch.fft.irfft(G_hat_phys.permute(0, 2, 1), n=self.Nt, dim=-1)
        G_tilde = a_rec / self._quad_w.to(device)[None, None, :]
        return G_tilde.permute(0, 2, 1).float()                # (B, Nt, k_svd)

    def _generate(self, theta_norm):
        """
        theta_norm : (B, θ_dim)  normalisé
        → Û        : (B, Nt, N, N)  valeurs physiques
        """
        G_hat_norm = self.forward(theta_norm)                                   # (B, K, k_svd, 2)
        G_hat_ri   = G_hat_norm * self.G_hat_std + self.G_hat_mean
        G_hat_phys = torch.complex(G_hat_ri[..., 0], G_hat_ri[..., 1])         # (B, K, k_svd)

        G_tilde = self._inverse_laplace(G_hat_phys)                             # (B, Nt, k_svd)
        z_tilde = G_tilde @ self.V.T                                            # (B, Nt, latent_dim)
        U_norm  = self._decode_seq(z_tilde)                                     # (B, Nt, N, N)
        return U_norm * self.U_std[None, None] + self.U_mean[None, None]


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, Nt, N, D, k_svd, K = 2, 20, 64, 32, 8, 10

    model = LatentSVDLaplaceModel(N=N, Nt=Nt, theta_dim=3, latent_dim=D,
                                   k_svd=k_svd, K=K, dt=0.1)
    V_fake = torch.linalg.qr(torch.randn(D, k_svd))[0]
    model.set_svd_basis(V_fake)
    model.set_normalization(
        G_hat_mean=torch.zeros(K, k_svd, 2), G_hat_std=torch.ones(K, k_svd, 2),
        U_mean=torch.zeros(N, N),            U_std=torch.ones(N, N),
        theta_mean=torch.zeros(3),           theta_std=torch.ones(3),
    )

    theta = torch.randn(B, 3)
    G_hat_pred = model(theta)
    G_hat_true = torch.randn(B, K, k_svd, 2)
    loss, m = model.loss(G_hat_pred, G_hat_true)
    loss.backward()
    print(f"Ĝ {tuple(G_hat_pred.shape)}  spec_loss={loss.item():.4f}")

    U_pred = model.generate(theta)
    print(f"θ {tuple(theta.shape)} → Û {tuple(U_pred.shape)}")
