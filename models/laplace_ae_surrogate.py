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
            nn.Linear(conv_out + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_out + 1, 64), #+1 pour la fréquence normalisée
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, U: torch.Tensor, freq_norm: torch.Tensor) -> tuple:
        """freq_norm : (B, 1) — k / Nt_half ∈ [0, 1]"""
        h  = self.conv(U).flatten(start_dim=1)              # (B, conv_out)
        hs = torch.cat([h, freq_norm], dim=1)               # (B, conv_out+1)
        return self.fc_mu(hs), self.fc_logvar(hs)


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
            nn.Linear(latent_dim + 1, 128 * self.base ** 2), # +1 pour la fréquence normalisée
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
            nn.ConvTranspose2d(16,  2,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.N//2)**2, 256),
            nn.ReLU(),
            nn.Linear(256, self.N**2),
        )

    def forward(self, z: torch.Tensor, freq_norm: torch.Tensor) -> torch.Tensor:
        """freq_norm : (B, 1) — k / Nt_half ∈ [0, 1]"""
        zs = torch.cat([z, freq_norm], dim=1)          # (B, latent_dim+1)
        x  = self.fc(zs)                               # (B, 128*base**2)
        x  = x.view(-1, 128, self.base, self.base)     # (B, 128, base, base)
        x  = self.deconv(x)                            # (B, 2, N//2, N//2)
        x  = self.out_fc(x)
        return x.view(-1, 2, self.N, self.N)           # (B, 2, N, N)


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


    def forward(self, U: torch.Tensor, freq_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        freq_norm : (B, 1) — k / Nt_half ∈ [0, 1]
        Retourne : (Û, μ, log σ²)
        """
        mu, logvar = self.encoder(U, freq_norm)
        z          = self.reparametrize(mu, logvar)
        U_hat      = self.decoder(z, freq_norm)
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
        mse = F.mse_loss(U_hat, U, reduction='none')
        # Somme sur les dimensions (C, H, W), moyenne sur le batch (B)
        recon_loss = mse.view(U.shape[0], -1).sum(dim=1).mean()

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
        Surrogate dans l'espace latent du VAE.
        Le décodeur est conditionné sur s_k (fréquence de Laplace).
    """
    def __init__(self, latent_dim: int, theta_dim: int, N: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.decoder = LaplaceDecoder(N=N, latent_dim=latent_dim)
        self.decoder.requires_grad_(False)  # on ne met à jour que le fc du surrogate
        self.register_buffer('freq_norm', torch.zeros(1))   # k / Nt_half

    def set_freq(self, k: int, Nt_half: int):
        """Fixe la fréquence normalisée pour ce surrogate."""
        self.freq_norm[0] = k / Nt_half

    def toggle_grad_decoder(self):
        requires_grad = not next(self.decoder.parameters()).requires_grad
        self.decoder.requires_grad_(requires_grad)
        print(f"Decoder requires_grad set to {requires_grad}")

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z       = self.fc(theta)                                         # (B, latent_dim)
        freq_B  = self.freq_norm.unsqueeze(0).expand(z.shape[0], -1)   # (B, 1)
        return self.decoder(z, freq_B)

    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceLatentModel(LaplaceModel):
    """
    Modèle complet VAE : N_half LaplaceLatentSurrogate avec décodeur partagé (gelé).

    Hérite de LaplaceModel → _forward_half / forward / _generate / set_normalization
    sont réutilisés tels quels car chaque surrogate expose la même interface forward(theta).

    Pipeline en 2 étapes :
      1. Entraîner LaplaceVAE sur tous les champs Laplace
      2. Pour chaque fréquence k, entraîner LaplaceLatentSurrogate (theta_proj seulement)
         avec le décodeur VAE gelé
    """

    def __init__(self, N_freq: int, N_half: int, N: int,
                 theta_dim: int = 4, latent_dim: int = 64):
        # Initialise BaseDecoder directement pour éviter de créer des LaplaceSurrogate
        BaseDecoder.__init__(self)
        self.surrogates = nn.ModuleList([
            LaplaceLatentSurrogate(latent_dim=latent_dim, theta_dim=theta_dim, N=N)
            for _ in range(N_half)
        ])
        self.N_freq     = N_freq
        self.N_half     = N_half
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.register_buffer('target_mean', torch.zeros(N_half, 2, 1, 1))
        self.register_buffer('target_std',  torch.ones(N_half,  2, 1, 1))

    def set_vae_decoder(self, vae: LaplaceVAE):
        """Copie les poids du décodeur VAE dans tous les surrogates (appelé après step 1)."""
        for k, surrogate in enumerate(self.surrogates):
            surrogate.decoder.load_state_dict(vae.decoder.state_dict())
            surrogate.decoder.requires_grad_(False)
            surrogate.set_freq(k, self.N_half)

    def loss(self, *args, **kwargs):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceLatentSurrogate.loss()."
        )

    def __repr__(self) -> str:
        return (f"LaplaceLatentModel(N_freq={self.N_freq}, N_half={self.N_half}, "
                f"N={self.N}, theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
                f"Surrogate par fréquence :\n{self.surrogates[0].__repr__()}")
