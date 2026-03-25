import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseAutoEncoder, BaseDecoder

class Encoder(nn.Module):
    """
    U → (mu, log var)

    U : (B, 1, N, N)  champ spatial

    """

    def __init__(self, N: int, latent_dim: int = 64):
        super().__init__()
        self.N = N

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,            64,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,            128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,           256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        conv_out = 256 * (N // 16) ** 2

        self.fc_mu     = nn.Linear(conv_out, latent_dim)
        self.fc_logvar = nn.Linear(conv_out, latent_dim)

    def forward(self, U: torch.Tensor):

        h = self.conv(U).flatten(start_dim=1)   # (B, conv_out)

        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    e → Û

    z : (B, latent_dim)
    """

    def __init__(self, N: int, latent_dim: int = 64):
        super().__init__()
        self.N    = N
        self.base = N // 16

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.base ** 2),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,  1,   kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                             # (B, 256*base**2)
        x = x.view(-1, 256, self.base, self.base)  # (B, 256, base, base)
        return self.deconv(x)                      # (B, 1, N, N)


class VAE(BaseAutoEncoder):
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
        latent_dim : int   = 64,
        beta       : float = 1.0,
        free_bits  : float = 0.5,   # seuil KL min par dimension latente
    ):
        super().__init__()

        self.beta      = beta
        self.free_bits = free_bits
        self.latent_dim = latent_dim

        self.encoder = Encoder(N, latent_dim)
        self.decoder = Decoder(N, latent_dim)


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
        """
        Retourne : (Û, μ, log σ²)
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO = E_q[log p(U|z,θ)]  -  β·KL(q(z|U,θ) || p(z))

        Returns
        --------
        neg_elbo   : loss totale à minimiser
        recon_loss : terme reconstruction seul
        kl_loss    : terme KL seul
        grad_loss  : terme gradient
        """
        recon_loss = F.mse_loss(U_hat, U, reduction='mean')

        # Gradient MSE : penalise les erreurs sur les gradients spatiaux
        def spatial_grads(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        dx_gt,  dy_gt  = spatial_grads(U)
        dx_hat, dy_hat = spatial_grads(U_hat)
        eps = 1e-6
        w_x = dx_gt.abs().detach() + eps
        w_y = dy_gt.abs().detach() + eps
        grad_loss = (
            (w_x * (dx_hat - dx_gt).pow(2)).mean() / w_x.mean() +
            (w_y * (dy_hat - dy_gt).pow(2)).mean() / w_y.mean()
        ) * 0.5

        # KL : free-bits — on ne pénalise pas en dessous de free_bits par dim
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
        kl_per_dim = kl_per_dim.mean(dim=0)                           # (latent,)
        kl_loss    = kl_per_dim.clamp(min=self.free_bits).sum()

        neg_elbo = recon_loss + grad_loss + self.beta * kl_loss

        return neg_elbo, recon_loss, kl_loss, grad_loss
    
    def loss(
        self,
        U      : torch.Tensor,
        U_hat  : torch.Tensor,
        mu     : torch.Tensor,
        logvar : torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        total, recon, kl, grad = self.elbo(U, U_hat, mu, logvar)
        return total, {'recon': recon, 'kl': kl, 'grad': grad}

class IndirectDecoder(BaseDecoder):
    """
    Projete theta dans l'espace latent, puis decode avec le decoder du VAE.
    """

    def __init__(self, trained_AE: BaseAutoEncoder, N: int, theta_dim: int, latent_dim: int = 64):
        super().__init__()
        self.N = N
        self.theta_proj = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = trained_AE.decoder
        self.decoder.requires_grad_(False) # à refléchir : geler le decoder ou pas ?


    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.theta_proj(theta)
        return self.decoder(z)

    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(U_hat, U)

        def spatial_grads(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        dx_gt,  dy_gt  = spatial_grads(U)
        dx_hat, dy_hat = spatial_grads(U_hat)
        grad_loss = (
            F.mse_loss(dx_hat, dx_gt) +
            F.mse_loss(dy_hat, dy_gt)
        ) * 0.5

        total = recon_loss + grad_loss
        return total, recon_loss, grad_loss

if __name__ == "__main__":
    model = VAE()
    rand_U = torch.randn(2, 1, 64, 64)
    U_hat, mu, logvar = model(rand_U)
    print(model)
    print("U_hat shape:", U_hat.shape)