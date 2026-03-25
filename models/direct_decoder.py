import torch
import torch.nn.functional as F
import torch.nn as nn

from models.base import BaseDecoder


class DirectDecoder(BaseDecoder):
    """
    theta → Û

    Décodeur direct sans espace latent : prédit U uniquement à partir de θ.

    theta : (B, theta_dim)
    Û     : (B, 1, N, N)
    """

    def __init__(self, N: int = 64, theta_dim: int = 4, lambda_grad: float = 1.0):
        super().__init__()
        self.N           = N
        self.base        = N // 16
        self.lambda_grad = lambda_grad

        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 256 * self.base ** 2),
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

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        x = self.fc(theta)                             # (B, 256*base**2)
        x = x.view(-1, 256, self.base, self.base)      # (B, 256, base, base)
        x = self.deconv(x)                             # (B, 1, N, N)
        return x.view(-1, 1, self.N, self.N)           # (B, 1, N, N)

    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loss mixte : MSE(U) + lambda_grad * MSE(gradients spatiaux)

        Returns
        -------
        total_loss, recon_loss, grad_loss
        """
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

        total = recon_loss + self.lambda_grad * grad_loss
        return total, recon_loss, grad_loss

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"DirectDecoder(N={self.N}, params={n:,})"

class DirectDecoderDenseOut(BaseDecoder):
    """
    theta → Û

    Décodeur direct sans espace latent : prédit U uniquement à partir de θ.

    theta : (B, theta_dim)
    Û     : (B, 1, N, N)
    """

    def __init__(self, N: int = 64, theta_dim: int = 4, lambda_grad: float = 1.0):
        super().__init__()
        self.N           = N
        self.base        = N // 32
        self.lambda_grad = lambda_grad

        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 256 * self.base ** 2),
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
            nn.ReLU(),
        )
        self.out_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((self.N//2)**2, self.N**2),
            nn.ReLU(),
            nn.Linear(self.N**2, self.N**2),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        x = self.fc(theta)                             # (B, 256*base**2)
        x = x.view(-1, 256, self.base, self.base)      # (B, 256, base, base)
        x = self.deconv(x)                             # (B, 1, N, N)
        x = self.out_fc(x)
        return x.view(-1, 1, self.N, self.N)           # (B, 1, N, N)

    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loss mixte : MSE(U) + lambda_grad * MSE(gradients spatiaux)

        Returns
        -------
        total_loss, recon_loss, grad_loss
        """
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

        total = recon_loss + self.lambda_grad * grad_loss
        return total, recon_loss, grad_loss

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"DirectDecoder(N={self.N}, params={n:,})"


if __name__ == "__main__":
    model = DirectDecoder(N=64, theta_dim=4)
    print(model)
    theta = torch.randn(4, 4)
    out   = model(theta)
    print(f"theta {theta.shape} → U_hat {out.shape}")
