import torch
import torch.nn as nn
import torch.nn.functional as F


# Indices de bx et by dans le vecteur theta = [D, bx, by, f, x0_x, x0_y]
_BX_BY_IDX = [1, 2]


class CVAELight(nn.Module):
    """
    CVAE light : encodeur et décodeur conditionnés uniquement sur (bx, by).

    theta complet = [D, bx, by, f, x0_x, x0_y]  (6 composantes)
    theta utilisé = [bx, by]                      (indices 1 et 2)

    z doit donc encoder D, f, x0_x, x0_y en plus de la variabilité
    résiduelle — le posterior collapse est naturellement découragé.

    Paramètres
    ----------
    N          : résolution de la grille
    theta_dim  : dimension du theta complet reçu en entrée (6 par défaut)
    latent_dim : dimension de l'espace latent z
    beta       : poids du terme KL dans l'ELBO
    lambda_grad: poids du terme gradient dans la loss
    bx_by_idx  : indices de bx et by dans theta (défaut [1, 2])
    """

    def __init__(
        self,
        N           : int        = 64,
        theta_dim   : int        = 6,
        latent_dim  : int        = 64,
        beta        : float      = 1.0,
        free_bits   : float      = 0.2,
        lambda_grad : float      = 1.0,
        bx_by_idx   : list[int]  = _BX_BY_IDX,
    ):
        super().__init__()

        self.beta        = beta
        self.free_bits   = free_bits
        self.lambda_grad = lambda_grad
        self.latent_dim  = latent_dim
        self.bx_by_idx   = bx_by_idx

        dec_theta_dim = len(bx_by_idx)  # 2

        # Encoder : U + bx,by seulement
        N_base   = N // 16
        conv_out = 256 * N_base ** 2

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1 + dec_theta_dim, 32,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,               64,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,               128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,              256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(conv_out, latent_dim)
        self.fc_logvar = nn.Linear(conv_out, latent_dim)

        # Decoder : z + bx,by seulement
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + dec_theta_dim, 256 * N_base ** 2),
            nn.ReLU(),
        )
        self.dec_deconv = nn.Sequential(
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

        self.N_base = N_base
        self.N      = N

    def _bxby(self, theta: torch.Tensor) -> torch.Tensor:
        return theta[:, self.bx_by_idx]

    def encode(self, U: torch.Tensor, theta: torch.Tensor):
        bxby = self._bxby(theta)
        B, _, H, W = U.shape
        bxby_map = bxby[:, :, None, None].expand(B, -1, H, W)
        x = torch.cat([U, bxby_map], dim=1)
        h = self.enc_conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        bxby = self._bxby(theta)
        x = torch.cat([z, bxby], dim=1)
        x = self.dec_fc(x)
        x = x.view(-1, 256, self.N_base, self.N_base)
        return self.dec_deconv(x)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, U: torch.Tensor, theta: torch.Tensor):
        """Retourne : (Û, μ, log σ²)"""
        mu, logvar = self.encode(U, theta)
        z          = self.reparametrize(mu, logvar)
        U_hat      = self.decode(z, theta)
        return U_hat, mu, logvar

    def elbo(
        self,
        U      : torch.Tensor,
        U_hat  : torch.Tensor,   # logits bruts du décodeur
        mu     : torch.Tensor,
        logvar : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # U dans [-1,1] → cible dans [0,1] pour BCE
        U_01 = (U + 1.0) / 2.0
        recon_loss = F.binary_cross_entropy_with_logits(U_hat, U_01, reduction='mean')

        # Gradient loss sur les probabilités (après sigmoid)
        U_hat_prob = torch.sigmoid(U_hat)

        def spatial_grads(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        dx_gt,  dy_gt  = spatial_grads(U_01)
        dx_hat, dy_hat = spatial_grads(U_hat_prob)
        grad_loss = (
            F.mse_loss(dx_hat, dx_gt) +
            F.mse_loss(dy_hat, dy_gt)
        ) * 0.5

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss    = kl_per_dim.mean(dim=0).clamp(min=self.free_bits).sum()

        neg_elbo = recon_loss + self.lambda_grad * grad_loss + self.beta * kl_loss

        return neg_elbo, recon_loss, kl_loss, grad_loss

    @torch.no_grad()
    def generate(
        self,
        theta    : torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """theta : (6,) ou (B, 6)  paramètres normalisés complets"""
        self.eval()

        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(n_samples, -1)

        z     = torch.randn(len(theta), self.latent_dim, device=theta.device)
        U_hat = torch.sigmoid(self.decode(z, theta))

        return U_hat   # (n_samples, 1, N, N)  valeurs dans [0, 1]

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"CVAELight(\n"
            f"  N={self.N}, latent_dim={self.latent_dim}, beta={self.beta}\n"
            f"  conditionnement : bx, by uniquement (indices {self.bx_by_idx})\n"
            f"  params entraînables : {n:,}\n"
            f")"
        )


if __name__ == "__main__":
    model = CVAELight()
    print(model)
    U     = torch.randn(4, 1, 64, 64)
    theta = torch.randn(4, 6)
    U_hat, mu, logvar = model(U, theta)
    print(f"U_hat : {U_hat.shape}, mu : {mu.shape}")
