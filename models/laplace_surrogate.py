import torch
import torch.nn.functional as F
import torch.nn as nn
from models.base import BaseDecoder
from tqdm import tqdm
import numpy as np
from utils.laplace import laplace_inverse


class LaplaceSurrogate(nn.Module):
    """
    Surrogate pour une fréquence de Laplace : prédit (Re(Û), Im(Û)) pour un s_k donné.

    Entrée : theta_norm (B, theta_dim)
    Sortie : (B, 2, N, N) — Re et Im de Û(theta, s_k), valeurs normalisées
    """

    def __init__(self, N, theta_dim: int = 4, s: complex | None = None,
                 freq_ratio: float = 0.0):
        """
        freq_ratio : k / N_half ∈ [0, 1] — position normalisée dans le spectre.
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

        # Encodeur de fréquence : scalaire k/N_half → vecteur de conditionnement
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
    Modèle complet dans l'espace de Laplace : encapsule N_half LaplaceSurrogate.

    forward(theta_norm)  → spectre normalisé (B, N_freq, N, N) complexe
    generate(theta_norm, target_mean, target_std, dt, gamma, rule)
                         → U_pred (B, Nt, N, N) en valeurs physiques

    La normalisation est gérée en dehors du modèle (dans main.py / checkpoint).
    """

    def __init__(self, N_freq: int, N_half: int, N: int, theta_dim: int = 4):
        super().__init__()
        self.surrogates = nn.ModuleList([
            LaplaceSurrogate(N, theta_dim, freq_ratio=k / max(N_half - 1, 1))
            for k in range(N_half)
        ])
        self.N_freq    = N_freq
        self.N_half    = N_half
        self.N         = N
        self.theta_dim = theta_dim

        # Stats de dénormalisation target — remplies via set_normalization()
        self.register_buffer('target_mean', torch.zeros(N_half, 2, 1, 1))
        self.register_buffer('target_std',  torch.ones(N_half,  2, 1, 1))

    def _forward_half(self, theta_norm: torch.Tensor, k_max: int | None = None) -> torch.Tensor:
        """(B, N*N, N_half) complexe, valeurs normalisées par fréquence.
        Si k_max est donné, seules les fréquences k <= k_max sont calculées, le reste vaut 0."""
        B = theta_norm.shape[0]
        M_half = torch.zeros((B, self.N * self.N, self.N_half),
                              dtype=torch.complex64, device=theta_norm.device)
        n_active = min(k_max + 1, self.N_half) if k_max is not None else self.N_half
        for k in range(n_active):
            pred = self.surrogates[k](theta_norm)                      # (B, 2, N, N)
            M_half[:, :, k] = (pred[:, 0] + 1j * pred[:, 1]).reshape(B, self.N * self.N)
        return M_half

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        """Spectre normalisé complet : (B, N_freq, N, N) complexe."""
        B      = theta_norm.shape[0]
        M_half = self._forward_half(theta_norm)
        M_full = torch.zeros((B, self.N * self.N, self.N_freq),
                              dtype=torch.complex64, device=theta_norm.device)
        M_full[:, :, :self.N_half] = M_half
        n_tail = self.N_freq - self.N_half
        if n_tail > 0:
            M_full[:, :, self.N_half:] = torch.conj(M_half[:, :, 1:n_tail + 1]).flip(dims=[2])
        return M_full.reshape(B, self.N, self.N, self.N_freq).permute(0, 3, 1, 2)

    def set_normalization(self, target_mean, target_std):
        """Charge les stats de dénormalisation target (appelé avant la sauvegarde)."""
        def _t(x): return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        self.target_mean.copy_(_t(target_mean))
        self.target_std.copy_(_t(target_std))

    def loss(self, *args, **kwargs):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceSurrogate.loss()."
        )

    def _generate_diff(self, theta_norm: torch.Tensor,
                       dt: float = 1.0, gamma: float = 0.0,
                       rule: str = 'trap', k_max: int | None = None) -> torch.Tensor:
        """
        Inverse Laplace différentiable via torch.fft.ifft.
        Le gradient traverse tout le chemin θ → surrogates → spectre → U(t).

        theta_norm : (B, theta_dim) — déjà normalisé
        Retourne U_pred (B, Nt, N, N) en valeurs physiques.
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N

        # Forward tous les surrogates, empilé avec torch.stack (pas d'in-place)
        n_active = min(k_max + 1, self.N_half) if k_max is not None else self.N_half
        preds  = [self.surrogates[k](theta_norm) for k in range(n_active)]   # n_active × (B, 2, N, N)
        M_active = torch.stack(
            [torch.complex(p[:, 0].reshape(B, NN).float(),
                           p[:, 1].reshape(B, NN).float()) for p in preds],
            dim=2,
        )  # (B, N*N, n_active) complex64
        if n_active < self.N_half:
            pad = torch.zeros(B, NN, self.N_half - n_active,
                              dtype=torch.complex64, device=theta_norm.device)
            M_half = torch.cat([M_active, pad], dim=2)
        else:
            M_half = M_active

        # Dénormalisation vectorisée
        tm_re = self.target_mean[:, 0, 0, 0].view(1, 1, -1)   # (1, 1, N_half)
        tm_im = self.target_mean[:, 1, 0, 0].view(1, 1, -1)
        ts_re = self.target_std[:,  0, 0, 0].view(1, 1, -1)
        ts_im = self.target_std[:,  1, 0, 0].view(1, 1, -1)
        M_half = torch.complex(
            M_half.real * ts_re + tm_re,
            M_half.imag * ts_im + tm_im,
        )

        # Symétrie conjuguée → spectre complet (B, N*N, N_freq)
        n_tail = self.N_freq - self.N_half
        if n_tail > 0:
            tail   = torch.conj(M_half[:, :, 1:n_tail + 1]).flip(dims=[2])
            M_full = torch.cat([M_half, tail], dim=2)
        else:
            M_full = M_half

        # Inverse Laplace différentiable
        Nt    = self.N_freq
        t     = torch.arange(Nt, dtype=torch.float32, device=M_full.device) * dt
        w     = torch.ones(Nt,  dtype=torch.float32, device=M_full.device)
        if rule == 'trap':
            w[0] = 0.5; w[-1] = 0.5
        denom = dt * w * torch.exp(torch.tensor(-gamma, device=M_full.device) * t)  # (Nt,)

        a_rec  = torch.fft.ifft(M_full, n=Nt, dim=-1)       # (B, N*N, Nt) complex
        U_pred = a_rec.real / denom                           # (B, N*N, Nt)
        return U_pred.permute(0, 2, 1).reshape(B, Nt, self.N, self.N)

    def _generate(self, theta_norm: torch.Tensor,
                  dt: float = 1.0, gamma: float = 0.0,
                  rule: str = 'trap', k_max: int | None = None) -> torch.Tensor:
        """
        theta_norm : (B, theta_dim) — déjà normalisé
        k_max      : si donné, seules les fréquences k <= k_max sont calculées
        Retourne U_pred (B, Nt, N, N) en valeurs physiques.
        """
        B = theta_norm.shape[0]
        M_half = self._forward_half(theta_norm, k_max=k_max)          # (B, N*N, N_half)

        # Dénormalisation par fréquence
        for k in range(self.N_half):
            tm = self.target_mean[k]                                   # (2, 1, 1)
            ts = self.target_std[k]
            re = M_half[:, :, k].real * ts[0, 0, 0] + tm[0, 0, 0]
            im = M_half[:, :, k].imag * ts[1, 0, 0] + tm[1, 0, 0]
            M_half[:, :, k] = torch.complex(re, im)

        # Symétrie conjuguée → spectre complet
        n_tail = self.N_freq - self.N_half
        M_full = torch.zeros((B, self.N * self.N, self.N_freq),
                              dtype=torch.complex64, device=theta_norm.device)
        M_full[:, :, :self.N_half] = M_half
        if n_tail > 0:
            M_full[:, :, self.N_half:] = torch.conj(M_half[:, :, 1:n_tail + 1]).flip(dims=[2])

        # Transformée inverse de Laplace
        M_np   = M_full.cpu().numpy()                                  # (B, N*N, N_freq)
        U_pred = np.zeros((B, self.N_freq, self.N, self.N), dtype=np.float32)
        for b in tqdm(range(B), desc="Inverse Laplace", leave=True):
            C_b, _ = laplace_inverse(M_np[b], dt, self.N_freq, rule=rule, gamma=gamma)
            U_pred[b] = C_b.reshape(self.N, self.N, self.N_freq).transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(U_pred).to(theta_norm.device)          # (B, Nt, N, N)

    def __repr__(self) -> str:
        return (f"LaplaceModel(N_freq={self.N_freq}, N_half={self.N_half}, "
                f"N={self.N}, theta_dim={self.theta_dim})\n"
                f"Surrogate par fréquence :\n{self.surrogates[0].__repr__()}")


if __name__ == "__main__":
    N_freq = 16
    N_half = N_freq // 2 + 1
    N      = 64
    model  = LaplaceModel(N_freq=N_freq, N_half=N_half, N=N, theta_dim=4)
    print(model)
    theta = torch.randn(2, 4)
    out   = model(theta)
    print(f"theta {theta.shape} → forward {out.shape}")
