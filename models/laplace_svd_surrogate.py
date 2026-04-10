import torch
import torch.nn as nn
import torch.nn.functional as F
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

    Les fréquences k_freq .. N_half-1 restent à zéro.
    La symétrie conjuguée reconstruit les k dernières fréquences du spectre complet.

    Interface compatible avec transient/main.py :
      model.generate(theta_norm, dt=..., gamma=..., rule=...) → (B, Nt, N, N)
    """

    def __init__(self, k_freq: int, N_freq: int, N_half: int, N: int,
                 theta_dim: int, k_svd: int):
        super().__init__()
        self.k_freq    = k_freq
        self.N_freq    = N_freq   # = Nt, passé explicitement comme dans LaplaceModel
        self.N_half    = N_half
        self.N         = N
        self.theta_dim = theta_dim
        self.k_svd     = k_svd

        self.surrogates_re = nn.ModuleList([
            LaplaceSVDSurrogate(k_svd, theta_dim, N) for _ in range(k_freq)
        ])
        self.surrogates_im = nn.ModuleList([
            LaplaceSVDSurrogate(k_svd, theta_dim, N) for _ in range(k_freq)
        ])

    def _forward_half(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """
        Calcule le demi-spectre complexe (B, N*N, N_half).
        Les fréquences k >= k_freq sont à zéro.
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N
        M_half = torch.zeros((B, NN, self.N_half),
                              dtype=torch.complex64, device=theta_norm.device)
        for k in range(self.k_freq):
            re = self.surrogates_re[k].get_field(theta_norm)  # (B, N*N)
            im = self.surrogates_im[k].get_field(theta_norm)  # (B, N*N)
            M_half[:, :, k] = torch.complex(re, im)
        return M_half

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """Spectre complet (B, N_freq, N, N) complexe."""
        B      = theta_norm.shape[0]
        M_half = self._forward_half(theta_norm)
        n_tail = self.N_freq - self.N_half
        if n_tail > 0:
            tail   = torch.conj(M_half[:, :, 1:n_tail + 1]).flip(dims=[2])
            M_full = torch.cat([M_half, tail], dim=2)
        else:
            M_full = M_half
        return M_full.reshape(B, self.N, self.N, self.N_freq).permute(0, 3, 1, 2)

    def _generate(self, theta_norm: torch.Tensor,
                  dt: float = 1.0, gamma: float = 0.0,
                  rule: str = 'trap', **kwargs) -> torch.Tensor:
        """
        theta_norm : (B, theta_dim) — déjà normalisé
        Retourne U_pred (B, Nt, N, N) en valeurs physiques.
        """
        B  = theta_norm.shape[0]
        Nt = self.N_freq

        M_half = self._forward_half(theta_norm)  # (B, N*N, N_half)

        # Symétrie conjuguée → spectre complet
        n_tail = self.N_freq - self.N_half
        if n_tail > 0:
            tail   = torch.conj(M_half[:, :, 1:n_tail + 1]).flip(dims=[2])
            M_full = torch.cat([M_half, tail], dim=2)   # (B, N*N, N_freq)
        else:
            M_full = M_half

        # Inverse Laplace différentiable (même logique que LaplaceModel._generate_diff)
        t     = torch.arange(Nt, dtype=torch.float32, device=M_full.device) * dt
        w     = torch.ones(Nt,  dtype=torch.float32, device=M_full.device)
        if rule == 'trap':
            w[0] = 0.5; w[-1] = 0.5
        denom = dt * w * torch.exp(torch.tensor(-gamma, device=M_full.device) * t)

        a_rec  = torch.fft.ifft(M_full, n=Nt, dim=-1)  # (B, N*N, Nt) complex
        U_pred = a_rec.real / denom                      # (B, N*N, Nt)
        return U_pred.permute(0, 2, 1).reshape(B, Nt, self.N, self.N)

    def loss(self, theta_norm: torch.Tensor,
             coeff_re_norm: torch.Tensor,
             coeff_im_norm: torch.Tensor,
             k: int) -> torch.Tensor:
        """MSE sur les coefficients SVD de la fréquence k."""
        loss_re = self.surrogates_re[k].loss(theta_norm, coeff_re_norm)
        loss_im = self.surrogates_im[k].loss(theta_norm, coeff_im_norm)
        return loss_re + loss_im

    def __repr__(self) -> str:
        return (f"LaplaceSVDModel(k_freq={self.k_freq}, N_freq={self.N_freq}, "
                f"N_half={self.N_half}, N={self.N}, theta_dim={self.theta_dim}, "
                f"k_svd={self.k_svd})\n  surrogates_re[0]: {self.surrogates_re[0].mlp}")


if __name__ == '__main__':
    k_freq, N_freq, N_half, N, theta_dim, k_svd = 5, 19, 10, 32, 3, 8
    model  = LaplaceSVDModel(k_freq, N_freq, N_half, N, theta_dim, k_svd)
    theta  = torch.randn(2, theta_dim)
    # Sans set_svd les buffers sont None → get_field plantera ;
    # ce bloc vérifie seulement la construction.
    print(model)
    print(f"N_freq={model.N_freq}")
