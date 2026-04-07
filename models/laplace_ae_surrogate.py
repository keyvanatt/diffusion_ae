import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseAutoEncoder, BaseDecoder
from models.laplace_surrogate import LaplaceModel


def _to_f_vec(freq_ratio, B: int, dtype, device) -> torch.Tensor:
    """Convertit freq_ratio (float scalaire ou tenseur (B,)) en (B, 1)."""
    if isinstance(freq_ratio, (float, int)):
        return torch.full((B, 1), float(freq_ratio), dtype=dtype, device=device)
    return freq_ratio.to(dtype=dtype, device=device).view(B, 1)


class LaplaceVariationalEncoder(nn.Module):
    """
    Û → (mu, log var), conditionné sur freq_ratio = k / N_half ∈ [0, 1].

    U : (B, 2, N, N)  champ spatial complexe (Re, Im), normalisé
    """

    def __init__(self, N: int, latent_dim: int = 64):
        super().__init__()
        self.N = N

        # 4 downsampling steps → base = N // 16
        self.conv1 = nn.Sequential(nn.Conv2d(2,   16,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,  32,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))

        # Encodeur de fréquence : scalaire k/N_half → vecteur de conditionnement
        self.freq_enc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Projections FiLM (gamma, beta) par bloc conv
        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)
        self.film4 = nn.Linear(64, 2 * 128)

        conv_out = 128 * (N // 16) ** 2

        self.fc_mu = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        """FiLM : x ← x * (1 + γ) + β,  γ et β conditionnés sur freq_ratio."""
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, freq_ratio=0.0) -> tuple:
        B = U.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, U.dtype, U.device))  # (B, 64)

        x = self._film(self.conv1(U), self.film1, f_emb)
        x = self._film(self.conv2(x), self.film2, f_emb)
        x = self._film(self.conv3(x), self.film3, f_emb)
        x = self._film(self.conv4(x), self.film4, f_emb)
        h = x.flatten(start_dim=1)                          # (B, conv_out)
        return self.fc_mu(h), self.fc_logvar(h)


class LaplaceDecoder(nn.Module):
    """
    z → Û (complexe, normalisé), conditionné sur freq_ratio = k / N_half ∈ [0, 1].

    z : (B, latent_dim)

    Architecture :
    - base = N // 16 → 4 étapes d'upsampling (vs N//32 + 5 étapes auparavant).
      Partir d'une résolution spatiale plus élevée réduit le sur-lissage et aide
      à reconstruire les oscillations spatiales de haute fréquence (grand k).
    - FiLM conditioning après chaque bloc deconv.
    - Bloc de raffinement à 3 conv pour mieux capter les détails fins.
    """

    def __init__(self, N: int = 64, latent_dim: int = 64, lambda_grad: float = 1.0):
        super().__init__()
        self.N           = N
        self.base        = N // 16          # 4 upsampling steps
        self.lambda_grad = lambda_grad

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.base ** 2),
            nn.ReLU(),
        )

        # Encodeur de fréquence
        self.freq_enc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Projections FiLM par bloc deconv
        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)
        self.film4 = nn.Linear(64, 2 * 32)

        # Blocs deconv séparés pour intercaler le FiLM
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())

        # Bloc de raffinement renforcé : 3 conv pour mieux capter les détails spatiaux fins
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 2,  kernel_size=1),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        B = z.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, z.dtype, z.device))  # (B, 64)

        x = self.fc(z).view(B, 128, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, f_emb)
        x = self._film(self.deconv2(x), self.film2, f_emb)
        x = self._film(self.deconv3(x), self.film3, f_emb)
        x = self._film(self.deconv4(x), self.film4, f_emb)
        return self.refine(x)                               # (B, 2, N, N)


class LaplaceVAE(BaseAutoEncoder):
    """
    Variational Autoencoder conditionné sur la fréquence de Laplace k/N_half.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 64)
    latent_dim : dimension de l'espace latent z
    beta       : poids du terme KL dans l'ELBO
    free_bits  : seuil KL min par dimension latente
    """

    def __init__(
        self,
        N          : int   = 64,
        latent_dim : int   = 32,
        beta       : float = 1.0,
        free_bits  : float = 0.1,
    ):
        super().__init__()

        self.beta       = beta
        self.free_bits  = free_bits
        self.latent_dim = latent_dim

        self.encoder = LaplaceVariationalEncoder(N, latent_dim)
        self.decoder = LaplaceDecoder(N, latent_dim)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self, U: torch.Tensor, freq_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retourne : (Û, μ, log σ²)"""
        mu, logvar = self.encoder(U, freq_ratio)
        z          = self.reparametrize(mu, logvar)
        U_hat      = self.decoder(z, freq_ratio)
        return U_hat, mu, logvar

    def elbo(
        self,
        U      : torch.Tensor,
        U_hat  : torch.Tensor,
        mu     : torch.Tensor,
        logvar : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(U_hat, U)

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss    = kl_per_dim.clamp(min=self.free_bits).mean()

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
    MLP θ → z → Û, conditionné sur freq_ratio = k / N_half.

    Le décodeur partagé reçoit freq_ratio à chaque appel forward, ce qui
    lui permet d'adapter ses filtres FiLM à la fréquence de Laplace courante.
    """

    def __init__(self, latent_dim: int, theta_dim: int, freq_ratio: float = 0.0):
        super().__init__()
        self.freq_ratio = freq_ratio
        self.proj = nn.Sequential(
            nn.Linear(theta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self._decoder: nn.Module | None = None  # référence non-enregistrée, injectée via set_decoder()

    def set_decoder(self, decoder: nn.Module):
        """Injecte le décodeur partagé sans le re-enregistrer comme sous-module."""
        self.__dict__['_decoder'] = decoder

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.proj(theta)                # (B, latent_dim)
        assert self._decoder is not None, "Appeler set_decoder() avant forward()"
        return self._decoder(z, self.freq_ratio)   # (B, 2, N, N)

    def loss(self, U_hat: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceLatentModel(LaplaceModel):
    """
    Modèle complet VAE :
      - N_half LaplaceLatentSurrogate (MLP θ→z, un par fréquence, conditionné sur k/N_half)
      - un unique shared_decoder (LaplaceDecoder gelé, partagé par référence)

    Pipeline en 2 étapes :
      1. Entraîner LaplaceVAE sur tous les champs Laplace (avec freq_ratio)
      2. Pour chaque fréquence k, entraîner LaplaceLatentSurrogate (fc seulement)
    """

    def __init__(self, N_freq: int, N_half: int, N: int,
                 theta_dim: int = 4, latent_dim: int = 64):
        BaseDecoder.__init__(self)
        self.surrogates = nn.ModuleList([
            LaplaceLatentSurrogate(
                latent_dim=latent_dim,
                theta_dim=theta_dim,
                freq_ratio=k / max(N_half - 1, 1),
            )
            for k in range(N_half)
        ])
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim)
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()
        self.N_freq     = N_freq
        self.N_half     = N_half
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.register_buffer('target_mean', torch.zeros(N_half, 2, 1, 1))
        self.register_buffer('target_std',  torch.ones(N_half,  2, 1, 1))

    def _inject_decoder(self):
        for s in self.surrogates:
            assert isinstance(s, LaplaceLatentSurrogate)
            s.set_decoder(self.shared_decoder)

    def set_vae_decoder(self, vae: LaplaceVAE):
        """Charge les poids du décodeur VAE dans shared_decoder."""
        self.shared_decoder.load_state_dict(vae.decoder.state_dict())
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._inject_decoder()
        return result

    def loss(self, *_, **__):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceLatentSurrogate.loss()."
        )

    def __repr__(self) -> str:
        return (f"LaplaceLatentModel(N_freq={self.N_freq}, N_half={self.N_half}, "
                f"N={self.N}, theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
                f"Shared decoder : {self.shared_decoder.__class__.__name__}\n"
                f"Surrogate par fréquence : {self.surrogates[0].__repr__()}")


if __name__ == "__main__":
    vae = LaplaceVAE(N=128, latent_dim=32)
    total_params   = sum(p.numel() for p in vae.parameters())
    encoder_params = sum(p.numel() for p in vae.encoder.parameters())
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")

    # Vérification forward avec freq_ratio
    U     = torch.randn(2, 2, 128, 128)
    U_hat, mu, logvar = vae(U, freq_ratio=0.75)
    print(f"U {U.shape} → U_hat {U_hat.shape}, mu {mu.shape}")
