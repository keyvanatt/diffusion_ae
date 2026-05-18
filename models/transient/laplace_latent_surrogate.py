import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseDecoder
from models.transient.laplace_surrogate import LaplaceModel
from models.transient.laplace_ae import LaplaceAE, LaplaceDecoder


class LaplaceLatentSurrogate(nn.Module):
    """
    MLP θ → z → Û, conditionné sur freq_ratio = k / K.

    Le décodeur partagé reçoit freq_ratio à chaque appel forward.
    """

    def __init__(self, latent_dim: int, theta_dim: int, freq_ratio: float = 0.0,
                 hidden_dim: int = 256):
        super().__init__()
        self.freq_ratio = freq_ratio
        mid = hidden_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(theta_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Linear(mid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._decoder: nn.Module | None = None  # référence non-enregistrée

    def set_decoder(self, decoder: nn.Module):
        """Injecte le décodeur partagé sans le ré-enregistrer comme sous-module."""
        self.__dict__['_decoder'] = decoder

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.proj(theta)                   # (B, latent_dim)
        assert self._decoder is not None, "Appeler set_decoder() avant forward()"
        return self._decoder(z, self.freq_ratio)  # (B, 2, N, N)

    def loss(self, U_hat: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceLatentModel(LaplaceModel):
    """
    Modèle complet AE :
      - K LaplaceLatentSurrogate (MLP θ→z, un par fréquence)
      - un unique shared_decoder (LaplaceDecoder gelé, partagé par référence)

    Pipeline en 2 étapes :
      1. Entraîner LaplaceAE sur tous les champs Laplace (avec freq_ratio)
      2. Pour chaque fréquence k, entraîner LaplaceLatentSurrogate.proj
    """

    def __init__(self, K: int, Nt: int, N: int,
                 theta_dim: int = 4, latent_dim: int = 64, freq_L: int = 8,
                 k_max: int | None = None, hidden_dim: int = 256):
        BaseDecoder.__init__(self)
        self.surrogates = nn.ModuleList([
            LaplaceLatentSurrogate(
                latent_dim=latent_dim,
                theta_dim=theta_dim,
                freq_ratio=k / max(K - 1, 1),
                hidden_dim=hidden_dim,
            )
            for k in range(K)
        ])
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

        self.K          = K
        self.Nt         = Nt
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.k_max      = k_max
        self.hidden_dim = hidden_dim
        self.register_buffer('target_mean', torch.zeros(K, 2, N, N))
        self.register_buffer('target_std',  torch.ones(K,  2, N, N))
        self.register_buffer('s_real', torch.zeros(K, dtype=torch.float64))
        self.register_buffer('s_imag', torch.zeros(K, dtype=torch.float64))

    def _inject_decoder(self):
        for s in self.surrogates:
            assert isinstance(s, LaplaceLatentSurrogate)
            s.set_decoder(self.shared_decoder)

    def set_ae_decoder(self, ae: LaplaceAE):
        """Charge les poids du décodeur AE dans shared_decoder."""
        self.shared_decoder.load_state_dict(ae.decoder.state_dict())
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._inject_decoder()
        return result

    def _forward_k_diff(self, theta_norm: torch.Tensor, n_active: int) -> torch.Tensor:
        """
        Override optimisé : toutes les projections θ→z en n_active appels MLP,
        puis UN SEUL appel au shared_decoder batché.
        Retourne M_active : (B, N*N, n_active) complex64.
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N

        z_all = torch.cat(
            [self.surrogates[k].proj(theta_norm) for k in range(n_active)],  # type: ignore[operator]
            dim=0,
        )  # (n_active*B, latent_dim)

        fr = torch.tensor(
            [self.surrogates[k].freq_ratio for k in range(n_active)],
            device=theta_norm.device, dtype=torch.float32,
        ).repeat_interleave(B)  # (n_active*B,)

        preds = self.shared_decoder(z_all.float(), fr)  # (n_active*B, 2, N, N)
        preds = preds.view(n_active, B, 2, self.N, self.N)
        return torch.complex(
            preds[:, :, 0].reshape(n_active, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(n_active, B, NN).permute(1, 2, 0).float(),
        )  # (B, NN, n_active)

    def loss(self, *_, **__):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceLatentSurrogate.loss()."
        )

    def __repr__(self) -> str:
        return (f"LaplaceLatentModel(K={self.K}, Nt={self.Nt}, "
                f"N={self.N}, theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
                f"Shared decoder : {self.shared_decoder.__class__.__name__}\n"
                f"Surrogate par fréquence : {self.surrogates[0].__repr__()}")
