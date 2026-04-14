"""
correction_ae.py — AE de correction frame-par-frame (architecture UNet).

UNet léger avec skip connections : préserve le détail spatial sans goulot
d'étranglement destructeur. Sortie résiduelle : U_corrected = U_pred + UNet(U_pred).

Usage :
    ae = CorrectionAE(N=128)
    U_corr = ae(U_pred)          # (B, N, N) → (B, N, N)
    U_corr = ae.correct(U_pred)  # inférence no_grad
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.laplace_ae_surrogate import LaplaceLatentModel


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
    )


class CorrectionAE(nn.Module):
    """
    UNet résiduel de correction frame-par-frame.

    Paramètres
    ----------
    N      : résolution spatiale (multiple de 8)
    base_ch: canaux de base (×1, ×2, ×4 dans l'encodeur)
    """

    def __init__(self, N: int = 128, base_ch: int = 16):
        super().__init__()
        self.N = N
        c = base_ch

        # --- Encodeur ---
        self.enc1 = _conv_block(1,      c)       # N   → N,   c ch
        self.enc2 = _conv_block(c,  2 * c)       # N/2 → N/2, 2c ch
        self.enc3 = _conv_block(2*c, 4 * c)      # N/4 → N/4, 4c ch

        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = _conv_block(4*c, 8 * c)  # N/8, 8c ch

        # --- Décodeur avec skip connections ---
        self.up3    = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.dec3   = _conv_block(8*c, 4*c)   # 4c (up) + 4c (skip)

        self.up2    = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.dec2   = _conv_block(4*c, 2*c)   # 2c (up) + 2c (skip)

        self.up1    = nn.ConvTranspose2d(2*c, c,   kernel_size=2, stride=2)
        self.dec1   = _conv_block(2*c, c)     # c (up) + c (skip)

        # Projection finale → résidu 1 canal
        # Initialisée à zéro : le modèle part de l'identité (résidu=0) et apprend progressivement
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, U_pred: torch.Tensor) -> torch.Tensor:
        """U_pred : (B, N, N) → U_corrected : (B, N, N)"""
        mean = U_pred.mean(dim=(-2, -1), keepdim=True)   # (B, 1, 1)
        std  = U_pred.std(dim=(-2, -1), keepdim=True) + 1e-8
        x = ((U_pred - mean) / std).unsqueeze(1)          # (B, 1, N, N)

        # Encodeur
        s1 = self.enc1(x)                    # (B, c,   N,   N)
        s2 = self.enc2(self.pool(s1))        # (B, 2c,  N/2, N/2)
        s3 = self.enc3(self.pool(s2))        # (B, 4c,  N/4, N/4)

        # Bottleneck
        b  = self.bottleneck(self.pool(s3))  # (B, 8c,  N/8, N/8)

        # Décodeur
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))  # (B, 4c, N/4, N/4)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))  # (B, 2c, N/2, N/2)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B, c,  N,   N)

        # Résidu re-scalé dans l'espace original
        residual = self.out_conv(d1).squeeze(1) * std  # (B, N, N), std=(B,1,1) broadcast
        return U_pred + residual

    @torch.no_grad()
    def correct(self, U_pred: torch.Tensor) -> torch.Tensor:
        """U_pred : (B, N, N) ou (N, N) → U_corrected même shape."""
        single = U_pred.dim() == 2
        if single:
            U_pred = U_pred.unsqueeze(0)
        out = self.forward(U_pred)
        return out.squeeze(0) if single else out

    def loss(self, U_corrected: torch.Tensor, U_true: torch.Tensor,
             U_pred: torch.Tensor,
             lambda_grad: float = 1.0) -> tuple[torch.Tensor, dict]:
        # Option A : loss dans l'espace résidu, normalisée par l'amplitude de la correction cible
        res_pred  = U_corrected - U_pred
        res_true  = U_true      - U_pred
        res_scale = res_true.std() + 1e-8   # scalaire, amplifie les gradients

        mse = F.mse_loss(res_pred / res_scale, res_true / res_scale)

        dx_pred = res_pred[:, :, 1:] - res_pred[:, :, :-1]
        dy_pred = res_pred[:, 1:, :] - res_pred[:, :-1, :]
        dx_true = res_true[:, :, 1:] - res_true[:, :, :-1]
        dy_true = res_true[:, 1:, :] - res_true[:, :-1, :]
        grad    = (F.mse_loss(dx_pred / res_scale, dx_true / res_scale) +
                   F.mse_loss(dy_pred / res_scale, dy_true / res_scale)) * 0.5

        total = mse + lambda_grad * grad
        return total, {'mse': mse.detach(), 'grad': grad.detach()}

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"CorrectionAE(N={self.N}, UNet) — {n:,} params"


class CorrectedPipeline(LaplaceLatentModel):
    """
    Surrogate LaplaceLatentModel + CorrectionAE enchaînés.

    Hérite de LaplaceLatentModel : compatible avec load_model / run_inference
    sans modification de main.py.

    Construit à partir d'une instance surrogate déjà chargée :
        pipeline = CorrectedPipeline(surrogate, correction_ae)
        U_corr   = pipeline.generate(theta_norm, dt=dt, ...)  # (B, Nt, N, N)
    """

    def __init__(self, surrogate, correction_ae: CorrectionAE):
        # Importer ici pour éviter un import circulaire au niveau module
        from models.laplace_ae_surrogate import LaplaceLatentModel

        if not isinstance(surrogate, LaplaceLatentModel):
            raise TypeError(f"CorrectedPipeline attend un LaplaceLatentModel, "
                            f"reçu {type(surrogate).__name__}")

        # Initialisation nn.Module sans recréer les sous-modules du surrogate :
        # on partage directement ses modules/buffers pour éviter de dupliquer
        # les ~17 GB en mémoire GPU lors de la construction.
        nn.Module.__init__(self)
        for name, mod in surrogate._modules.items():
            self._modules[name] = mod
        for name, buf in surrogate._buffers.items():
            self._buffers[name] = buf
        for attr in ('N_freq', 'N_half', 'N', 'theta_dim',
                     'latent_dim', 'k_max', 'hidden_dim'):
            setattr(self, attr, getattr(surrogate, attr))

        self.correction_ae = correction_ae

    def _generate(self, theta_norm: torch.Tensor,
                  dt: float = 1.0, gamma: float = 0.0,
                  rule: str = 'trap', k_max=None,
                  correction_chunk: int = 64) -> torch.Tensor:
        U_pred = super()._generate(theta_norm, dt=dt, gamma=gamma, rule=rule, k_max=k_max)
        B, Nt, N, _ = U_pred.shape
        frames = U_pred.reshape(B * Nt, N, N)
        chunks = [self.correction_ae(frames[i:i + correction_chunk])
                  for i in range(0, B * Nt, correction_chunk)]
        return torch.cat(chunks, dim=0).reshape(B, Nt, N, N)

    def __repr__(self) -> str:
        n_surr = sum(p.numel() for p in self.surrogates.parameters())
        n_ae   = sum(p.numel() for p in self.correction_ae.parameters())
        return (f"CorrectedPipeline(N={self.N}, N_half={self.N_half})\n"
                f"  surrogate : {n_surr:,} params\n"
                f"  correction: {n_ae:,} params")
