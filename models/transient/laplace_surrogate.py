import torch
import torch.nn.functional as F
import torch.nn as nn
from models.base import BaseDecoder
import numpy as np


class LaplaceSurrogate(nn.Module):
    """
    Surrogate pour une fréquence de Laplace : prédit (Re(Û), Im(Û)) pour un s_k donné.

    Entrée : theta_norm (B, theta_dim)
    Sortie : (B, 2, N, N) — Re et Im de Û(theta, s_k), valeurs normalisées
    """

    def __init__(self, N, theta_dim: int = 4, s: complex | None = None,
                 freq_ratio: float = 0.0):
        """
        freq_ratio : k / K ∈ [0, 1] — position normalisée dans le spectre.
        """
        super().__init__()
        self.s          = s
        self.freq_ratio = freq_ratio
        self.N          = N
        self.base       = N // 16

        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 256 * self.base ** 2),
            nn.ReLU(),
        )

        # Encodeur de fréquence : scalaire k/K → vecteur de conditionnement
        self.freq_enc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Projections FiLM (gamma, beta) par bloc deconv
        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)
        self.film4 = nn.Linear(64, 2 * 32)

        # Blocs deconv séparés pour intercaler le FiLM
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())

        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2,  kernel_size=1),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        """FiLM : x ← x * (1 + γ) + β,  γ et β conditionnés sur freq_ratio."""
        gamma, beta = proj(f_emb).chunk(2, dim=1)          # (B, C) chacun
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        B = theta.shape[0]
        f_vec = torch.full((B, 1), self.freq_ratio,
                           dtype=theta.dtype, device=theta.device)
        f_emb = self.freq_enc(f_vec)                        # (B, 64)

        x = self.fc(theta).view(B, 256, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, f_emb)
        x = self._film(self.deconv2(x), self.film2, f_emb)
        x = self._film(self.deconv3(x), self.film3, f_emb)
        x = self._film(self.deconv4(x), self.film4, f_emb)
        return self.refine(x).view(B, 2, self.N, self.N)

    def loss(self, U_hat: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceModel(BaseDecoder):
    """
    Modèle complet dans l'espace de Laplace : encapsule K LaplaceSurrogate.

    forward(theta_norm)  → spectre normalisé (B, K, N, N) complexe
    generate(theta_norm, dt, alpha_t, lam, rule)
                         → U_pred (B, Nt, N, N) via laplace_inverse_tik
                           (symétrie conjuguée gérée dans laplace_inverse_tik)
    """

    def __init__(self, K: int, Nt: int, N: int, theta_dim: int = 4,
                 k_max: int | None = None):
        super().__init__()
        self.surrogates = nn.ModuleList([
            LaplaceSurrogate(N, theta_dim, freq_ratio=k / max(K - 1, 1))
            for k in range(K)
        ])
        self.K         = K
        self.Nt        = Nt
        self.N         = N
        self.theta_dim = theta_dim
        self.k_max     = k_max

        self.register_buffer('target_mean', torch.zeros(K, 2, N, N))
        self.register_buffer('target_std',  torch.ones(K,  2, N, N))
        # Points s stockés en deux buffers float64 (complex128 non universel)
        self.register_buffer('s_real', torch.zeros(K, dtype=torch.float64))
        self.register_buffer('s_imag', torch.zeros(K, dtype=torch.float64))

    @property
    def s_list(self) -> torch.Tensor:
        """Points s complexes : (K,) complex128 sur CPU."""
        return torch.complex(self.s_real, self.s_imag)

    def set_s_list(self, s_list):
        """Charge les points s (numpy array ou liste). Appeler avant la sauvegarde."""
        s = np.asarray(s_list, dtype=np.complex128)
        self.s_real.copy_(torch.tensor(s.real, dtype=torch.float64))
        self.s_imag.copy_(torch.tensor(s.imag, dtype=torch.float64))

    def _forward_k(self, theta_norm: torch.Tensor, k_max: int | None = None) -> torch.Tensor:
        """(B, N*N, K) complexe normalisé. Fréquences > k_max restent à 0."""
        B     = theta_norm.shape[0]
        M     = torch.zeros((B, self.N * self.N, self.K),
                             dtype=torch.complex64, device=theta_norm.device)
        limit    = k_max if k_max is not None else self.k_max
        n_active = min(limit + 1, self.K) if limit is not None else self.K
        for k in range(n_active):
            pred = self.surrogates[k](theta_norm)              # (B, 2, N, N)
            M[:, :, k] = (pred[:, 0] + 1j * pred[:, 1]).reshape(B, self.N * self.N)
        return M

    def _forward_k_diff(self, theta_norm: torch.Tensor, n_active: int) -> torch.Tensor:
        """Version différentiable pour n_active fréquences. (B, N*N, n_active) complex64.
        Surchargé dans LaplaceLatentModel pour un appel batché au shared_decoder."""
        B, NN = theta_norm.shape[0], self.N * self.N
        preds = [self.surrogates[k](theta_norm) for k in range(n_active)]
        return torch.stack(
            [torch.complex(p[:, 0].reshape(B, NN).float(),
                           p[:, 1].reshape(B, NN).float()) for p in preds],
            dim=2,
        )

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """Spectre normalisé : (B, K, N, N) complexe."""
        B = theta_norm.shape[0]
        M = self._forward_k(theta_norm)            # (B, N*N, K)
        return M.reshape(B, self.N, self.N, self.K).permute(0, 3, 1, 2)

    def set_normalization(self, target_mean, target_std):
        """Charge les stats de dénormalisation (appelé avant la sauvegarde)."""
        def _t(x): return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.target_mean.copy_(_t(target_mean))
        self.target_std.copy_(_t(target_std))

    def loss(self, *args, **kwargs):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceSurrogate.loss()."
        )

    def _denorm_k(self, M: torch.Tensor) -> torch.Tensor:
        """Dénormalise (B, N*N, K) complexe (modifie en place)."""
        NN = self.N * self.N
        for k in range(self.K):
            tm, ts = self.target_mean[k], self.target_std[k]
            re = M[:, :, k].real * ts[0].reshape(NN) + tm[0].reshape(NN)
            im = M[:, :, k].imag * ts[1].reshape(NN) + tm[1].reshape(NN)
            M[:, :, k] = torch.complex(re, im)
        return M

    def _generate(self, theta_norm: torch.Tensor,
                  dt: float = 1.0, alpha_t: float = 0.0, lam: float = 1e-6,
                  rule: str = 'trap', k_max: int | None = None) -> torch.Tensor:
        """
        Inférence CPU float64 via laplace_inverse_tik.
        La symétrie conjuguée est gérée dans laplace_inverse_tik.
        """
        from utils.laplace import laplace_inverse_tik
        B, NN  = theta_norm.shape[0], self.N ** 2
        device = theta_norm.device

        M = self._forward_k(theta_norm, k_max=k_max)  # (B, NN, K) complex64
        M = self._denorm_k(M)

        # CPU float64 pour la précision
        M_flat = M.detach().cpu().cdouble().reshape(B * NN, self.K)
        U_flat = laplace_inverse_tik(M_flat, self.s_list, dt, self.Nt, alpha_t, lam, rule)
        # (B*NN, Nt) float64

        return (U_flat.float()
                .reshape(B, NN, self.Nt).permute(0, 2, 1)
                .reshape(B, self.Nt, self.N, self.N)
                .to(device))

    def _generate_diff(self, theta_norm: torch.Tensor,
                       dt: float = 1.0, alpha_t: float = 0.0, lam: float = 1e-6,
                       rule: str = 'trap', k_max: int | None = None) -> torch.Tensor:
        """
        Version différentiable GPU float32 via laplace_inverse_tik.
        Gradients traversent θ → surrogates → spectre → laplace_inverse_tik (linalg.solve).
        """
        from utils.laplace import laplace_inverse_tik
        B, NN  = theta_norm.shape[0], self.N ** 2
        device = theta_norm.device

        limit    = k_max if k_max is not None else self.k_max
        n_active = min(limit + 1, self.K) if limit is not None else self.K
        M_active = self._forward_k_diff(theta_norm, n_active)  # (B, NN, n_active)

        if n_active < self.K:
            pad = torch.zeros(B, NN, self.K - n_active, dtype=torch.complex64, device=device)
            M   = torch.cat([M_active, pad], dim=2)
        else:
            M = M_active

        # Dénorm vectorisée (B, NN, K)
        tm     = self.target_mean.to(device)
        ts     = self.target_std.to(device)
        tm_re  = tm[:, 0].reshape(self.K, NN).T.unsqueeze(0)   # (1, NN, K)
        tm_im  = tm[:, 1].reshape(self.K, NN).T.unsqueeze(0)
        ts_re  = ts[:, 0].reshape(self.K, NN).T.unsqueeze(0)
        ts_im  = ts[:, 1].reshape(self.K, NN).T.unsqueeze(0)
        M = torch.complex(M.real * ts_re + tm_re, M.imag * ts_im + tm_im)

        # GPU float32 (différentiable via linalg.solve)
        s = torch.complex(
            self.s_real.to(device=device, dtype=torch.float32),
            self.s_imag.to(device=device, dtype=torch.float32),
        )
        U_flat = laplace_inverse_tik(M.reshape(B * NN, self.K), s, dt, self.Nt, alpha_t, lam, rule)
        # (B*NN, Nt) float32

        return U_flat.reshape(B, NN, self.Nt).permute(0, 2, 1).reshape(B, self.Nt, self.N, self.N)

    def __repr__(self) -> str:
        return (f"LaplaceModel(K={self.K}, Nt={self.Nt}, "
                f"N={self.N}, theta_dim={self.theta_dim})\n"
                f"Surrogate par fréquence :\n{self.surrogates[0].__repr__()}")


if __name__ == "__main__":
    K  = 21
    Nt = 150
    N  = 64
    model = LaplaceModel(K=K, Nt=Nt, N=N, theta_dim=4)
    print(model)
    theta = torch.randn(2, 4)
    out   = model(theta)
    print(f"theta {theta.shape} → forward {out.shape}")   # (2, K, 64, 64)
