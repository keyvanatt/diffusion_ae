import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseAutoEncoder, BaseDecoder
from models.laplace_surrogate import LaplaceModel

class LaplaceVariationalEncoder(nn.Module):
    """
    Û → (mu, log var)

    U : (B, 2, N, N)  champ spatial complexe (Re, Im), normalisé

    """

    def __init__(self, N: int, latent_dim: int = 64):
        super().__init__()
        self.N = N

        self.conv = nn.Sequential(
            nn.Conv2d(2,  16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        conv_out = 128 * (N // 16) ** 2

        self.fc_mu     = nn.Sequential(
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, U: torch.Tensor) -> tuple:
        h = self.conv(U).flatten(start_dim=1)              # (B, conv_out)
        return self.fc_mu(h), self.fc_logvar(h)


class LaplaceDecoder(nn.Module):
    """
    z → Û (complexe, normalisé)

    z : (B, latent_dim)
    """

    def __init__(self, N: int = 64, latent_dim: int = 64, lambda_grad: float = 1.0):
        super().__init__()
        self.N           = N
        self.base        = N // 32
        self.lambda_grad = lambda_grad

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.base ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,  16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,  8,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,   8,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.refine = nn.Conv2d(8, 2, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                                 # (B, 128*base**2)
        x = x.view(-1, 128, self.base, self.base)     # (B, 128, base, base)
        x = self.deconv(x)                             # (B, 8, N, N)
        return self.refine(x)                          # (B, 2, N, N)


class LaplaceVAE(BaseAutoEncoder):
    """
    Variational Autoencoder.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 64)
    theta_dim  : dimension du vecteur de paramètres θ (ex. 6)
    latent_dim : dimension de l'espace latent z
    beta       : poids du terme KL dans l'ELBO
    """

    def __init__(
        self,
        N          : int   = 64,
        latent_dim : int   = 32,
        beta       : float = 1.0,
        free_bits  : float = 0.1,   # seuil KL min par dimension latente
    ):
        super().__init__()

        self.beta      = beta
        self.free_bits = free_bits
        self.latent_dim = latent_dim

        self.encoder = LaplaceVariationalEncoder(N, latent_dim)
        self.decoder = LaplaceDecoder(N, latent_dim)


    def reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        z = μ + σ·ε   avec   ε ~ N(0, I)

        À l'entraînement : stochastique (gradient passe via μ et σ).
        À l'évaluation   : déterministe (z = μ).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu


    def forward(self, U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne : (Û, μ, log σ²)"""
        mu, logvar = self.encoder(U)
        z          = self.reparametrize(mu, logvar)
        U_hat      = self.decoder(z)
        return U_hat, mu, logvar


    def elbo(
        self,
        U      : torch.Tensor,
        U_hat  : torch.Tensor,
        mu     : torch.Tensor,
        logvar : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO = E_q[log p(U|z,θ)]  -  β·KL(q(z|U,θ) || p(z))

        Returns
        --------
        neg_elbo   : loss totale à minimiser
        recon_loss : terme reconstruction seul
        kl_loss    : terme KL seul
        """
        recon_loss = F.mse_loss(U_hat, U)

        # KL : free-bits — on ne pénalise pas en dessous de free_bits par dim
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
        kl_per_dim = kl_per_dim.mean(dim=0)                           # (latent,)
        kl_loss    = kl_per_dim.clamp(min=self.free_bits).sum()

        neg_elbo = recon_loss + self.beta * kl_loss

        return neg_elbo, recon_loss, kl_loss
    
    def loss(
        self,
        U      : torch.Tensor,
        U_hat  : torch.Tensor,
        mu     : torch.Tensor,
        logvar : torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        total, recon, kl = self.elbo(U, U_hat, mu, logvar)
        return total, {'recon': recon, 'kl': kl}

class LaplaceLatentSurrogate(nn.Module):
    """
    MLP θ → z → Û. Interface identique à LaplaceSurrogate.

    Le décodeur n'est PAS enregistré comme sous-module ici : il est stocké
    par référence via set_decoder() depuis LaplaceLatentModel.shared_decoder,
    ce qui évite N_half copies en mémoire.
    """
    def __init__(self, latent_dim: int, theta_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self._decoder: nn.Module | None = None  # référence non-enregistrée, injectée plus tard

    def set_decoder(self, decoder: nn.Module):
        """Injecte le décodeur partagé sans le re-enregistrer comme sous-module."""
        self.__dict__['_decoder'] = decoder

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.proj(theta)               # (B, latent_dim)
        assert self._decoder is not None, "Appeler set_decoder() avant forward()"
        return self._decoder(z)            # (B, 2, N, N)

    def loss(self, U_hat: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceLatentModel(LaplaceModel):
    """
    Modèle complet VAE :
      - N_half LaplaceLatentSurrogate (MLP θ→z, un par fréquence)
      - un unique shared_decoder (LaplaceDecoder gelé, partagé par référence)

    Pipeline en 2 étapes :
      1. Entraîner LaplaceVAE sur tous les champs Laplace
      2. Pour chaque fréquence k, entraîner LaplaceLatentSurrogate (fc seulement)
    """

    def __init__(self, N_freq: int, N_half: int, N: int,
                 theta_dim: int = 4, latent_dim: int = 64):
        BaseDecoder.__init__(self)
        self.surrogates = nn.ModuleList([
            LaplaceLatentSurrogate(latent_dim=latent_dim, theta_dim=theta_dim)
            for _ in range(N_half)
        ])
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim)
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()
        self.N_freq     = N_freq
        self.N_half     = N_half
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.register_buffer('target_mean', torch.zeros(N_half, 2, 1, 1))
        self.register_buffer('target_std',  torch.ones(N_half,  2, 1, 1))

    def _inject_decoder(self):
        """Injecte shared_decoder par référence dans chaque surrogate."""
        for s in self.surrogates:
            assert isinstance(s, LaplaceLatentSurrogate)
            s.set_decoder(self.shared_decoder)

    def set_vae_decoder(self, vae: LaplaceVAE):
        """Charge les poids du décodeur VAE dans shared_decoder (une seule copie)."""
        self.shared_decoder.load_state_dict(vae.decoder.state_dict())
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._inject_decoder()   # réinjecte la référence après chargement
        return result

    def loss(self, *_, **__):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceLatentSurrogate.loss()."
        )

    def __repr__(self) -> str:
        return (f"LaplaceLatentModel(N_freq={self.N_freq}, N_half={self.N_half}, "
                f"N={self.N}, theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
                f"Shared decoder : {self.shared_decoder.__class__.__name__}\n"
                f"Surrogate par fréquence : {self.surrogates[0].__repr__()}")

    