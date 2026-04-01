import torch
import torch.nn.functional as F
import torch.nn as nn

class LaplaceSurrogate(nn.Module):
    """
    Surrogate pour la fonction de laplace : prédit la valeur de U^(theta,s).

    theta : (B, theta_dim)
    out: B(B, 2, N, N) (Re(U), Im(U))
    """

    def __init__(self, s, N, theta_dim: int = 4):
        super().__init__()
        self.N = N
        self.s = s
        self.base = N // 16
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
            nn.ConvTranspose2d(32,  2,   kernel_size=4, stride=2, padding=1),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        x = self.fc(theta)                             # (B, 256*base**2)
        x = x.view(-1, 256, self.base, self.base)      # (B, 256, base, base)
        x = self.deconv(x)                             # (B, 2, N, N)
        return x.view(-1, 2, self.N, self.N)           # (B, 2, N, N)
    
    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE(U_hat, U)
        """
        return F.mse_loss(U_hat, U)
    
if __name__ == "__main__":
    theta_dim = 4
    N = 64
    s = 0.5
    model = LaplaceSurrogate(s, N, theta_dim)
    print(model)
    random_theta = torch.randn(8, theta_dim)
    output = model(random_theta)
    print("Output shape:", output.shape)  # Devrait être (8, 2, 64, 64)
    