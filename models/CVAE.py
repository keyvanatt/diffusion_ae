import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    (U, theta) → (mu, log var)

    U : (B, 1, N, N)  champ spatial
    theta : (B, 6)        paramètres physiques

    """

    def __init__(self, N: int, theta_dim: int = 6, latent_dim: int = 64):
        super().__init__()
        self.N = N

        self.conv = nn.Sequential(
            nn.Conv2d(1 + theta_dim, 32,  kernel_size=4, stride=2, padding=1),
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

    def forward(self, U: torch.Tensor, theta: torch.Tensor):
        B, _, H, W = U.shape

        # theta : (B, 6) → (B, 6, H, W)
        theta_map = theta[:, :, None, None].expand(B, -1, H, W)

        x = torch.cat([U, theta_map], dim=1)    # (B, 1+6, H, W)
        h = self.conv(x).flatten(start_dim=1)   # (B, conv_out)

        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    (z, theta) → Û

    z : (B, latent_dim)
    theta  : (B, 6)
    """

    def __init__(self, N: int, theta_dim: int = 6, latent_dim: int = 64):
        super().__init__()
        self.N    = N
        self.base = N // 16

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + theta_dim, 256 * self.base ** 2),
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

    def forward(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, theta], dim=1)           # (B, latent+6)
        x = self.fc(x)                             # (B, 256*base**2)
        x = x.view(-1, 256, self.base, self.base)  # (B, 256, base, base)
        return self.deconv(x)                      # (B, 1, N, N)


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 64)
    theta_dim  : dimension du vecteur de paramètres θ (ex. 6)
    latent_dim : dimension de l'espace latent z
    beta       : poids du terme KL dans l'ELBO
    """

    def __init__(
        self,
        N           : int   = 64,
        theta_dim   : int   = 6,
        latent_dim  : int   = 64,
        beta        : float = 1.0,
        free_bits   : float = 0.5,   # seuil KL min par dimension latente
        lambda_grad : float = 1.0,   # poids du terme gradient dans la loss
    ):
        super().__init__()

        self.beta        = beta
        self.free_bits   = free_bits
        self.lambda_grad = lambda_grad
        self.latent_dim = latent_dim

        self.encoder = Encoder(N, theta_dim, latent_dim)
        self.decoder = Decoder(N, theta_dim, latent_dim)


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


    def forward(self, U: torch.Tensor, theta: torch.Tensor):
        """

        Retourne : (Û, μ, log σ²)
        """
        mu, logvar = self.encoder(U, theta)
        z          = self.reparametrize(mu, logvar)
        U_hat      = self.decoder(z, theta)
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
        grad_loss = (
            F.mse_loss(dx_hat, dx_gt) +
            F.mse_loss(dy_hat, dy_gt)
        ) * 0.5

        # KL : free-bits — on ne pénalise pas en dessous de free_bits par dim
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent)
        kl_per_dim = kl_per_dim.mean(dim=0)                           # (latent,)
        kl_loss    = kl_per_dim.clamp(min=self.free_bits).sum()

        neg_elbo = recon_loss + self.lambda_grad * grad_loss + self.beta * kl_loss

        return neg_elbo, recon_loss, kl_loss, grad_loss


    @torch.no_grad()
    def generate(
        self,
        theta    : torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """

        theta : (6,) ou (B, 6)  paramètres normalisés
        """
        self.eval()

        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(n_samples, -1)

        z     = torch.randn(len(theta), self.latent_dim, device=theta.device)
        U_hat = self.decoder(z, theta)

        return U_hat   # (n_samples, 1, N, N)


    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"CVAE(\n"
            f"  N={self.decoder.N}, "
            f"latent_dim={self.latent_dim}, "
            f"beta={self.beta}\n"
            f"  params entraînables : {n:,}\n"
            f"Encoder : {self.encoder}\n"
            f"Decoder : {self.decoder}\n"
            f")"
        )

if __name__ == "__main__":
    model = CVAE()
    print(model)