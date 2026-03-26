import torch
torch.linalg.svd
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseAutoEncoder, BaseDecoder
from models.variationalAutoEncoder import Decoder

class CustomSVDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z):
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        ctx.save_for_backward(Z, U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        Z, U, S, Vh = ctx.saved_tensors
        m, n = Z.shape
        k = S.shape[0]

        # Ut_dU and Vt_dV (k x k)
        Ut_dU = U.T @ dU
        Vt_dV = Vh @ dVh.T

        # Derivative of S (eigenvalues)
        dZ_s = U @ torch.diag(dS) @ Vh

        # Spectral difference matrix
        s_i = S.unsqueeze(0)
        s_j = S.unsqueeze(1)
        #F = 1.0 / (s_i ** 2 - s_j ** 2 + 1e-10)  # Avoid division by zero
        F = torch.where((s_i - s_j).abs() > 1e-9, 1.0 / (s_i ** 2 - s_j ** 2), torch.zeros_like(s_i))
        F.fill_diagonal_(0)

        # Derivative contributions from U and V
        term_U = U @ (F * Ut_dU)
        term_V = Vh.T @ (F.T * Vt_dV)

        dZ_u = term_U @ torch.diag(S) @ Vh
        dZ_v = U @ torch.diag(S) @ term_V.T

        # Final total gradient
        dZ = dZ_s + dZ_u + dZ_v

        return dZ


class CustomSVD(nn.Module):
    def forward(self, Z):
        return CustomSVDFunction.apply(Z)


class SVDProjection(nn.Module):
    def __init__(self, kmax):
        super(SVDProjection, self).__init__()
        self.kmax = kmax
        self.svd = CustomSVD()
        self.eps = 1e-10
        self.register_buffer('fixed_basis_buffer', None)

    def set_fixed_basis(self, U_k):
        self.fixed_basis_buffer = U_k

    def forward(self, z):
        z = z.T  # (latent_dim, batch_size)

        if self.fixed_basis_buffer is not None:
            U_k = self.fixed_basis_buffer  # ya está en el device correcto
        else:
            try:
                
                #pdb.set_trace()  # Debugger breakpoint
                U, S, Vh = self.svd(z)
                U_k = U[:, :self.kmax]
            except RuntimeError as e:
                print(f"[Fallback] SVD failed: {e}")
                Q, _ = torch.linalg.qr(z)
                U_k = Q[:, :self.kmax]

        #pdb.set_trace()  # Debugger breakpoint

        alfa_proj = U_k.T @ z
        z_proj = U_k @ alfa_proj
        z_proj = z_proj.T  # (batch_size, kmax)
        return z_proj

class Encoder(nn.Module):
    """
    U → (mu, log var)

    U : (B, 1, N, N)  champ spatial

    """

    def __init__(self, N: int, latent_dim: int = 64):
        super().__init__()
        self.N = N

        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        conv_out = 128 * (N // 16) ** 2

        self.fc  = nn.Sequential(
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, U: torch.Tensor):

        h = self.conv(U).flatten(start_dim=1)   # (B, conv_out)

        return self.fc(h)

class AutoencoderSVD(BaseAutoEncoder):
    def __init__(self, N, latent_dim, kmax):
        super(AutoencoderSVD, self).__init__()
        self.encoder = Encoder(N, latent_dim)
        
        self.svd_proj = SVDProjection(kmax)
        
        self.decoder = Decoder(N,latent_dim)


    def set_fixed_basis(self, U_k):
        self.svd_proj.set_fixed_basis(U_k)

    def forward(self, x):
        z = self.encoder(x)
        z_k = self.svd_proj(z)  # Proyección con SVD
        x_recon = self.decoder(z_k)
        return (x_recon,)
    
    def loss(
        self,
        U_hat : torch.Tensor,
        U     : torch.Tensor,

    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        return total, {'recon': recon_loss, 'grad': grad_loss}


def compute_fixed_svd_basis(model, x_train, kmax):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # Codificar todo el conjunto de entrenamiento
        x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
        Z = model.encoder(x_tensor).T  # [latent_dim, N]

        # Calcular SVD
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        U_k = U[:, :kmax]  # [latent_dim, kmax]
        print("Base SVD calculada.")

        #return U_k.cpu()
        return U_k.to(device)

class IndirectDecoderSVD(BaseDecoder):
    def __init__(self, N, kmax,theta_dim,latent_dim, trained_autoencoder):
        super(IndirectDecoderSVD, self).__init__()
        self.svd_proj = SVDProjection(kmax)
        self.trained_autoencoder = trained_autoencoder
        self.decoder = trained_autoencoder.decoder
        self.svd_proj.requires_grad_(False)

        self.N = N
        self.theta_proj = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
    def compute_and_set_fixed_basis(self,U_train):
        U_k = compute_fixed_svd_basis(self.trained_autoencoder, U_train, self.svd_proj.kmax)
        self.svd_proj.set_fixed_basis(U_k)

    def forward(self, theta):
        z   = self.theta_proj(theta)  # theta (theta_dim,) → latent (latent_dim,)
        z_k = self.svd_proj(z)        # projection sur la base SVD fixe
        return self.decoder(z_k)

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
    model = AutoencoderSVD(N=64, latent_dim=32, kmax=2)
    rand_U = torch.randn(2, 1, 64, 64)
    U_k = compute_fixed_svd_basis(model, rand_U, kmax=2)
    model.set_fixed_basis(U_k)
    print("Fixed SVD basis set in the model.")