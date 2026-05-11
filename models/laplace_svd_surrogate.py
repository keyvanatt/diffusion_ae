import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base import BaseDecoder


class LaplaceSVDSurrogate(nn.Module):
    """
    Surrogate pour une fréquence et une composante (Re ou Im) dans l'espace de Laplace.
    Prédit k_svd coefficients SVD depuis theta_norm.

    Buffers (remplis via set_svd avant sauvegarde) :
      Vt         : (k_svd, N*N) — base spatiale SVD (lignes orthonormées)
      coeff_mean : (k_svd,)
      coeff_std  : (k_svd,)

    N doit être fourni à la construction pour que les buffers aient la bonne forme
    et que load_state_dict() fonctionne correctement.
    """

    def __init__(self, k_svd: int, theta_dim: int, N: int):
        super().__init__()
        self.k_svd     = k_svd
        self.theta_dim = theta_dim
        self.N         = N
        self.mlp = nn.Sequential(
            nn.Linear(theta_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, k_svd),
        )
        # Placeholder avec la bonne forme → load_state_dict() peut les remplir
        self.register_buffer('Vt',         torch.zeros(k_svd, N * N))
        self.register_buffer('coeff_mean', torch.zeros(k_svd))
        self.register_buffer('coeff_std',  torch.ones(k_svd))

    def set_svd(self, Vt: torch.Tensor,
                coeff_mean: torch.Tensor,
                coeff_std:  torch.Tensor):
        """Charge la base SVD et les stats de normalisation (appelé avant sauvegarde)."""
        self.register_buffer('Vt',         Vt.float())
        self.register_buffer('coeff_mean', coeff_mean.float())
        self.register_buffer('coeff_std',  coeff_std.float())

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """(B, theta_dim) → (B, k_svd) coefficients normalisés."""
        return self.mlp(theta_norm)

    def get_field(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """(B, theta_dim) → (B, N*N) champ spatial dénormalisé."""
        coeff_norm = self(theta_norm)                              # (B, k_svd)
        coeff      = coeff_norm * self.coeff_std + self.coeff_mean # (B, k_svd)
        return coeff @ self.Vt                                     # (B, N*N)

    def loss(self, theta_norm: torch.Tensor,
             coeff_norm: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(theta_norm), coeff_norm)


class LaplaceSVDModel(BaseDecoder):
    """
    Modèle complet : k_freq surrogates SVD (Re + Im) + inverse Laplace.

    Pour chaque fréquence k = 0 .. k_freq-1 :
      surrogates_re[k] : theta_norm → Re(Û_k)  via coefficients SVD
      surrogates_im[k] : theta_norm → Im(Û_k)  via coefficients SVD

    Les fréquences k_freq .. K-1 restent à zéro.
    La symétrie conjuguée reconstruit les k dernières fréquences du spectre complet.

    Interface compatible avec transient/main.py :
      model.generate(theta_norm, dt=..., gamma=..., rule=...) → (B, Nt, N, N)
    """

    def __init__(self, k_freq: int, K: int, Nt: int, N: int,
                 theta_dim: int, k_svd: int):
        super().__init__()
        self.k_freq    = k_freq
        self.K         = K
        self.Nt        = Nt
        self.N         = N
        self.theta_dim = theta_dim
        self.k_svd     = k_svd

        self.surrogates_re = nn.ModuleList([
            LaplaceSVDSurrogate(k_svd, theta_dim, N) for _ in range(k_freq)
        ])
        self.surrogates_im = nn.ModuleList([
            LaplaceSVDSurrogate(k_svd, theta_dim, N) for _ in range(k_freq)
        ])
        self.register_buffer('s_real', torch.zeros(K, dtype=torch.float64))
        self.register_buffer('s_imag', torch.zeros(K, dtype=torch.float64))

    @property
    def s_list(self) -> torch.Tensor:
        return torch.complex(self.s_real, self.s_imag)

    def set_s_list(self, s_list):
        s = np.asarray(s_list, dtype=np.complex128)
        self.s_real.copy_(torch.tensor(s.real, dtype=torch.float64))
        self.s_imag.copy_(torch.tensor(s.imag, dtype=torch.float64))

    def _forward_k(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """Spectre complexe (B, N*N, K). Fréquences k >= k_freq restent à zéro."""
        B, NN = theta_norm.shape[0], self.N * self.N
        M = torch.zeros((B, NN, self.K), dtype=torch.complex64, device=theta_norm.device)
        for k in range(self.k_freq):
            re = self.surrogates_re[k].get_field(theta_norm)
            im = self.surrogates_im[k].get_field(theta_norm)
            M[:, :, k] = torch.complex(re, im)
        return M

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """Spectre normalisé : (B, K, N, N) complexe."""
        B = theta_norm.shape[0]
        M = self._forward_k(theta_norm)            # (B, N*N, K)
        return M.reshape(B, self.N, self.N, self.K).permute(0, 3, 1, 2)

    def _generate(self, theta_norm: torch.Tensor,
                  dt: float = 1.0, alpha_t: float = 0.0, lam: float = 1e-6,
                  rule: str = 'trap', **kwargs) -> torch.Tensor:
        """Inférence CPU float64 via laplace_inverse_tik."""
        from utils.laplace import laplace_inverse_tik
        B, NN  = theta_norm.shape[0], self.N ** 2
        device = theta_norm.device

        M      = self._forward_k(theta_norm)       # (B, NN, K) complex64
        M_flat = M.detach().cpu().cdouble().reshape(B * NN, self.K)
        U_flat = laplace_inverse_tik(M_flat, self.s_list, dt, self.Nt, alpha_t, lam, rule)

        return (U_flat.float()
                .reshape(B, NN, self.Nt).permute(0, 2, 1)
                .reshape(B, self.Nt, self.N, self.N)
                .to(device))

    def loss(self, theta_norm: torch.Tensor,
             coeff_re_norm: torch.Tensor,
             coeff_im_norm: torch.Tensor,
             k: int) -> torch.Tensor:
        """MSE sur les coefficients SVD de la fréquence k."""
        loss_re = self.surrogates_re[k].loss(theta_norm, coeff_re_norm)
        loss_im = self.surrogates_im[k].loss(theta_norm, coeff_im_norm)
        return loss_re + loss_im

    def __repr__(self) -> str:
        return (f"LaplaceSVDModel(k_freq={self.k_freq}, K={self.K}, Nt={self.Nt}, "
                f"N={self.N}, theta_dim={self.theta_dim}, "
                f"k_svd={self.k_svd})\n  surrogates_re[0]: {self.surrogates_re[0].mlp}")


if __name__ == '__main__':
    k_freq, K, Nt, N, theta_dim, k_svd = 5, 10, 150, 32, 3, 8
    model  = LaplaceSVDModel(k_freq, K, Nt, N, theta_dim, k_svd)
    theta  = torch.randn(2, theta_dim)
    print(model)
    print(f"K={model.K}  Nt={model.Nt}")
