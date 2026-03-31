import torch
import torch.nn.functional as F
import torch.nn as nn

class SVDSurrogate(nn.Module):
    """
    Surrogate basé sur la décomposition SVD 3D : prédit la valeur de H(theta).

    theta : (B, theta_dim)
    coeffs : (B, nf_eff)
    """

    def __init__(self, nf_eff: int = 20, theta_dim: int = 4):
        super().__init__()
        self.nf_eff = nf_eff
        self.fc = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, nf_eff),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        coeffs = self.fc(theta)  # (B, nf_eff)
        return coeffs