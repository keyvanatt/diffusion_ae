import torch
import torch.nn.functional as F
import torch.nn as nn

from models.base import BaseAutoEncoder

class SimpleAutoEncoder(BaseAutoEncoder):
    """
    Autoencodeur simple : encodeur + décodeur linéaires.

    theta : (B, theta_dim)
    z     : (B, latent_dim)
    Û     : (B, 1, N, N)
    """

    def __init__(self, N: int = 64, theta_dim: int = 4, latent_dim: int = 16, lambda_grad: float = 1.0):
        super().__init__()
        self.N           = N
        self.latent_dim  = latent_dim
        self.lambda_grad = lambda_grad

        self.encoder = 
        self.decoder 

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.encoder(theta)                         # (B, latent_dim)
        x = self.decoder(z)                             # (B, N*N)
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