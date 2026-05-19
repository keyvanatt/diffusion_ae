import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class LearnableLaplace(nn.Module):
    """
    Transformée de Laplace discrète avec K points s_k libres.

    Paramètres apprenables
    ----------------------
    s_re        : (K,) parties réelles σ_k des pôles s_k = σ_k + iω_k
    s_im        : (K,) parties imaginaires ω_k
    log_alpha_t : ln(α_t), poids du lissage temporel dans l'inversion
    log_lam     : ln(λ), ridge dans l'inversion

    Initialisation : σ_k = gamma_init, ω_k uniformément sur [0, π/dt].

    La matrice DtTDt (terme de lissage) est constante → buffer pré-calculé.
    En mode eval, les matrices A et F_full sont mises en cache après le premier
    appel (A ne change plus dès que les paramètres sont gelés).
    """

    def __init__(self, K: int, dt: float, Nt: int, gamma_init: float = 0.0):
        super().__init__()
        self.K   = K
        self.dt  = dt
        self.Nt  = Nt

        self.s_re        = nn.Parameter(torch.full((K,), gamma_init))
        self.s_im        = nn.Parameter(torch.linspace(0.0, math.pi / dt, K))
        self.log_alpha_t = nn.Parameter(torch.tensor(-2.0))
        self.log_lam     = nn.Parameter(torch.tensor(-2.0))

        # DtTDt est entièrement constant (pas de paramètre appris dedans)
        Dt = (torch.diag(torch.ones(Nt - 1), 1) - torch.eye(Nt))[:Nt - 1, :]
        self.register_buffer('_DtTDt', Dt.T @ Dt)   # (Nt, Nt) float32

        # Position initiale des s_k — pour le scatter de suivi
        self.register_buffer('_s_init_re', torch.full((K,), gamma_init))
        self.register_buffer('_s_init_im', torch.linspace(0.0, math.pi / dt, K))

        # Cache des matrices d'inversion — valide uniquement en mode eval
        self._eval_cache: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Gestion du cache
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        if mode:
            self._eval_cache = None   # les params vont changer → cache périmé
        return super().train(mode)

    # ------------------------------------------------------------------

    @property
    def s_list(self) -> torch.Tensor:
        """s_k = σ_k + iω_k,  (K,) complex64."""
        return torch.complex(self.s_re, self.s_im)

    # ------------------------------------------------------------------
    # Construction des matrices d'inversion
    # ------------------------------------------------------------------

    def _build_inv_matrices(self, device: torch.device):
        """
        Construit s_full, F_full (K_full, Nt) et A (Nt, Nt).
        Utilise le buffer _DtTDt pour éviter de recalculer le terme de lissage.
        """
        s = self.s_list.to(dtype=torch.complex64, device=device)
        c_mask = s.imag > 0
        s_full = torch.cat([s, torch.conj(s[c_mask])])           # symétrie conjuguée

        t = torch.arange(self.Nt, dtype=torch.float32, device=device) * self.dt
        w = torch.ones(self.Nt, dtype=torch.float32, device=device)
        w[0] = 0.5; w[-1] = 0.5

        F_full = self.dt * w[None, :] * torch.exp(-s_full[:, None] * t[None, :])  # (K_full, Nt)
        FtF    = torch.real(torch.conj(F_full).T @ F_full)        # (Nt, Nt)

        alpha_t = self.log_alpha_t.exp()
        lam     = self.log_lam.exp()
        A = (FtF
             + alpha_t * self._DtTDt.to(device=device)
             + lam * torch.eye(self.Nt, dtype=torch.float32, device=device))

        return s_full, F_full, A, c_mask

    def _get_inv_matrices(self, device: torch.device):
        """Retourne les matrices d'inversion (depuis le cache en eval, recalculées en train)."""
        if not self.training and self._eval_cache is not None:
            return self._eval_cache
        matrices = self._build_inv_matrices(device)
        if not self.training:
            self._eval_cache = matrices
        return matrices

    # ------------------------------------------------------------------
    # Transformées
    # ------------------------------------------------------------------

    def forward_transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, Nt, latent_dim)
        → ẑ : (B, K, latent_dim) complex64
        """
        B, Nt, D = z.shape
        device = z.device
        z_flat = z.permute(0, 2, 1).reshape(B * D, Nt).float()   # (B*D, Nt)

        s = self.s_list.to(dtype=torch.complex64, device=device)
        t = torch.arange(Nt, dtype=torch.float32, device=device) * self.dt
        w = torch.ones(Nt, dtype=torch.float32, device=device)
        w[0] = 0.5; w[-1] = 0.5
        F = self.dt * w[None, :] * torch.exp(-s[:, None] * t[None, :])  # (K, Nt)

        z_hat_flat = z_flat.to(dtype=F.dtype) @ F.T              # (B*D, K)
        return z_hat_flat.view(B, D, self.K).permute(0, 2, 1)    # (B, K, D)

    def inverse_transform(self, z_hat: torch.Tensor, Nt: int) -> torch.Tensor:
        """
        ẑ : (B, K, latent_dim) complex64
        → z_rec : (B, Nt, latent_dim) float32
        """
        assert Nt == self.Nt, f"Nt mismatch: got {Nt}, expected {self.Nt}"
        B, K, D = z_hat.shape
        device  = self.s_re.device

        s_full, F_full, A, c_mask = self._get_inv_matrices(device)

        z_hat_flat  = z_hat.permute(0, 2, 1).reshape(B * D, K)   # (B*D, K)
        U_hat_full  = torch.cat(
            [z_hat_flat, torch.conj(z_hat_flat[:, c_mask])], dim=1
        ).to(torch.complex64)                                      # (B*D, K_full)

        RHS = torch.real(U_hat_full @ torch.conj(F_full))         # (B*D, Nt)
        z_rec_flat = torch.linalg.solve(A, RHS.T).T               # (B*D, Nt)
        return z_rec_flat.view(B, D, self.Nt).permute(0, 2, 1).float()  # (B, Nt, D)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def log_scatter(self, epoch: int):
        """
        Retourne un wandb.Image du scatter s_k (initial → courant).
        Importer wandb dans le script appelant ; cette méthode ne l'importe
        pas au niveau module pour ne pas créer de dépendance obligatoire.
        """
        import wandb

        s_cur = self.s_list.detach().cpu().numpy()
        s_ini = np.vectorize(complex)(
            self._s_init_re.cpu().numpy(),
            self._s_init_im.cpu().numpy(),
        )
        K    = len(s_cur)
        cmap = plt.cm.plasma

        fig, ax = plt.subplots(figsize=(5, 4))
        for k, (a, b) in enumerate(zip(s_ini, s_cur)):
            col = cmap(k / max(K - 1, 1))
            ax.plot([a.real, b.real], [a.imag, b.imag], color=col, lw=0.6, alpha=0.5)
        ax.scatter(s_ini.real, s_ini.imag, c=np.arange(K), cmap='plasma',
                   s=50, marker='o', zorder=3, alpha=0.4, label='initial')
        ax.scatter(s_cur.real, s_cur.imag, c=np.arange(K), cmap='plasma',
                   s=80, marker='*', zorder=4, label='current')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Re(s)')
        ax.set_ylabel('Im(s)')
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_title(f's-points  epoch {epoch}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    def log_text(self):
        """Retourne un wandb.Html listant les s_k courants."""
        import wandb
        s = self.s_list.detach().cpu().numpy().tolist()
        body = ", ".join(f"{z.real:.4f}+{z.imag:.4f}j" for z in s)
        return wandb.Html(
            f"<pre style='font-family:monospace;white-space:pre-wrap;'>[{body}]</pre>"
        )
