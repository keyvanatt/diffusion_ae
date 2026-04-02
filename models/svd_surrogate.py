import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.base import BaseDecoder


class SVDSurrogate(BaseDecoder):
    """
    Surrogate basé sur la décomposition Tucker SVD : theta_norm → U_pred.

    forward(theta_norm)  → G_pred_norm (B, nf_eff)  — coefficients normalisés
    generate(theta_norm, G_mean, G_std) → U_pred (B, Nt, Hsub, Wsub)

    Les bases SVD (F, P, alph) sont stockées comme buffers dans le modèle.
    G_mean/G_std viennent du checkpoint (comme target_mean/std pour LaplaceModel).
    """

    def __init__(self, nf_eff: int = 20, theta_dim: int = 4):
        super().__init__()
        self.nf_eff    = nf_eff
        self.theta_dim = theta_dim
        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, nf_eff),
        )
        # Bases SVD + stats normalisation G — initialisées vides, remplies via set_bases()
        self.register_buffer('F',      torch.zeros(1, nf_eff))  # (nr, nf_eff)
        self.register_buffer('P',      torch.zeros(1, nf_eff))  # (Nt, nf_eff)
        self.register_buffer('alph',   torch.zeros(nf_eff))     # (nf_eff,)
        self.register_buffer('G_mean', torch.zeros(nf_eff))     # (nf_eff,)
        self.register_buffer('G_std',  torch.ones(nf_eff))      # (nf_eff,)

    def set_bases(self, F: np.ndarray, P: np.ndarray, alph: np.ndarray,
                  G_mean: np.ndarray, G_std: np.ndarray):
        """Charge les bases SVD et les stats de normalisation G."""
        def _t(x): return torch.tensor(x, dtype=torch.float32)
        self.F      = _t(F)
        self.P      = _t(P)
        self.alph   = _t(alph)
        self.G_mean = _t(G_mean)
        self.G_std  = _t(G_std)

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """theta_norm (B, theta_dim) → G_pred_norm (B, nf_eff)."""
        return self.fc(theta_norm)

    def loss(self, G_hat: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(G_hat, G)

    def _generate(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """
        theta_norm : (B, theta_dim) — déjà normalisé
        Retourne U_pred (B, Nt, Hsub, Wsub).
        """
        G_pred_n = self.forward(theta_norm)                            # (B, nf_eff)
        G_pred   = (G_pred_n * self.G_std + self.G_mean) / self.alph  # (B, nf_eff)

        F_np = self.F.cpu().numpy()                                    # (nr, nf_eff)
        P_np = self.P.cpu().numpy()                                    # (Nt, nf_eff)
        G_np = G_pred.cpu().numpy()                                    # (B, nf_eff)
        a_np = self.alph.cpu().numpy()                                 # (nf_eff,)

        from utils.SVD_Amine_3D import svd_inverse_3d
        fields = svd_inverse_3d(F_np, G_np, P_np, a_np)               # (nr, B, Nt)

        nr   = F_np.shape[0]
        Nt   = P_np.shape[0]
        Hsub = Wsub = int(np.round(np.sqrt(nr)))
        B    = G_np.shape[0]
        U_pred = fields.transpose(1, 2, 0).reshape(B, Nt, Hsub, Wsub).astype(np.float32)
        return torch.from_numpy(U_pred).to(theta_norm.device)          # (B, Nt, Hsub, Wsub)
