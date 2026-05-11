import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseAutoEncoder, BaseDecoder
from models.laplace_surrogate import LaplaceModel
import math


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _to_f_vec(freq_ratio, B: int, dtype, device) -> torch.Tensor:
    """Convertit freq_ratio (float scalaire ou tenseur (B,)) en (B, 1)."""
    if isinstance(freq_ratio, (float, int)):
        return torch.full((B, 1), float(freq_ratio), dtype=dtype, device=device)
    return freq_ratio.to(dtype=dtype, device=device).view(B, 1)


class SinusoidalFreqEncoding(nn.Module):
    """
    Encode un scalaire f ∈ [0, 1] en un vecteur de dimension 2·L via un
    positional encoding sinusoïdal (style NeRF), avant de le passer dans
    un petit MLP pour obtenir l'embedding de conditionnement FiLM.

    Pour le ième niveau de fréquence (i = 0, …, L-1) :
        sin(2^i · π · f),  cos(2^i · π · f)

    Les fréquences couvrent [2^0 · π, 2^(L-1) · π], ce qui donne au réseau
    une représentation dense de la position spectrale : basses fréquences pour
    la structure globale, hautes fréquences pour discriminer des fréquences
    de Laplace voisines.

    Paramètres
    ----------
    L          : nombre de niveaux de fréquence → vecteur de dim 2·L
    hidden_dim : dimension cachée du MLP qui projette l'encoding
    out_dim    : dimension de l'embedding de sortie (utilisée pour FiLM)
    """

    def __init__(self, L: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.L = L
        # Fréquences 2^i · π, i = 0, …, L-1  (shape : (L,))
        freqs = math.pi * (2.0 ** torch.arange(L).float())
        self.register_buffer('freqs', freqs)   # non-trainable

        # MLP : 2L → hidden_dim → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * L, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f : (B, 1) — freq_ratio normalisé ∈ [0, 1]
        retourne : (B, out_dim)
        """
        # (B, 1) × (L,) → (B, L)  (broadcasting)
        angles = f * self.freqs                           # (B, L)
        enc = torch.cat([angles.sin(), angles.cos()], dim=1)  # (B, 2L)
        return self.mlp(enc)                              # (B, out_dim)


class LaplaceEncoder(nn.Module):
    """
    U → z, conditionné sur freq_ratio = k / K ∈ [0, 1].

    U : (B, 2, N, N)  champ spatial complexe (Re, Im), normalisé
    """

    def __init__(self, N: int, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N = N

        # 3 downsampling steps → résolution de sortie : N // 8
        self.conv1 = nn.Sequential(nn.Conv2d(2,   16,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,  32,  kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))

        # Encodeur de fréquence sinusoïdal
        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        # Projections FiLM (gamma, beta) par bloc conv
        self.film1 = nn.Linear(64, 2 * 16)
        self.film2 = nn.Linear(64, 2 * 32)
        self.film3 = nn.Linear(64, 2 * 64)

        conv_out = 64 * (N // 8) ** 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        """FiLM : x ← x · (1 + γ) + β,  γ et β conditionnés sur freq_ratio."""
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)   # borne gamma dans [-1, 1] → évite l'explosion fp16
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, U: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne z : (B, latent_dim)."""
        B = U.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, U.dtype, U.device))  # (B, 64)

        x = self._film(self.conv1(U), self.film1, f_emb)
        x = self._film(self.conv2(x), self.film2, f_emb)
        x = self._film(self.conv3(x), self.film3, f_emb)
        h = x.flatten(start_dim=1)                               # (B, conv_out)
        return self.fc(h)                                         # (B, latent_dim)

class LaplaceDecoder(nn.Module):
    """
    z → Û (complexe, normalisé), conditionné sur freq_ratio = k / K ∈ [0, 1].

    z : (B, latent_dim)
    retourne : (B, 2, N, N)  — canaux Re et Im

    Architecture :
    - base = N // 8 → 3 étapes d'upsampling.
    - FiLM conditioning (sinusoïdal) après chaque bloc deconv.
    - Deux blocs de raffinement séparés (Re / Im) pour les détails fins.
    """

    def __init__(self, N: int = 64, latent_dim: int = 64, freq_L: int = 8):
        super().__init__()
        self.N    = N
        self.base = N // 8  # résolution de départ avant 3 upsampling

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.base ** 2),
            nn.ReLU(),
        )

        self.freq_enc = SinusoidalFreqEncoding(L=freq_L, hidden_dim=64, out_dim=64)

        # Projections FiLM par bloc deconv
        self.film1 = nn.Linear(64, 2 * 128)
        self.film2 = nn.Linear(64, 2 * 64)
        self.film3 = nn.Linear(64, 2 * 32)

        # Blocs deconv séparés pour intercaler le FiLM
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU())

        # Blocs de raffinement séparés Re / Im
        # Chacun produit un seul canal de sortie (1, pas 2 — corrige le bug original)
        self.refine_re = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),            # → (B, 1, N, N)
        )
        self.refine_im = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1,  kernel_size=1),            # → (B, 1, N, N)
        )

    def _film(self, x: torch.Tensor, proj: nn.Linear, f_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = proj(f_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)   # borne gamma dans [-1, 1] → évite l'explosion fp16
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, z: torch.Tensor, freq_ratio=0.0) -> torch.Tensor:
        """Retourne Û : (B, 2, N, N)."""
        B = z.shape[0]
        f_emb = self.freq_enc(_to_f_vec(freq_ratio, B, z.dtype, z.device))  # (B, 64)

        x = self.fc(z).view(B, 128, self.base, self.base)
        x = self._film(self.deconv1(x), self.film1, f_emb)
        x = self._film(self.deconv2(x), self.film2, f_emb)
        x = self._film(self.deconv3(x), self.film3, f_emb)

        re = self.refine_re(x)    # (B, 1, N, N)
        im = self.refine_im(x)    # (B, 1, N, N)
        return torch.cat([re, im], dim=1)   # (B, 2, N, N)


class LaplaceAE(BaseAutoEncoder):
    """
    Autoencoder déterministe conditionné sur la fréquence de Laplace k/K.

    Paramètres
    ----------
    N          : résolution de la grille (ex. 64)
    latent_dim : dimension de l'espace latent z
    beta       : poids de la régularisation ridge (L2 sur la sortie)
    freq_L     : nombre de niveaux de fréquence pour le sinusoidal encoding
    """

    def __init__(
        self,
        N          : int   = 64,
        latent_dim : int   = 32,
        beta       : float = 1e-3,
        freq_L     : int   = 8,
    ):
        super().__init__()

        self.beta       = beta
        self.latent_dim = latent_dim

        self.encoder = LaplaceEncoder(N, latent_dim, freq_L=freq_L)
        self.decoder = LaplaceDecoder(N, latent_dim, freq_L=freq_L)

    def forward(
        self, U: torch.Tensor, freq_ratio: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne : (Û, z)

        Û : (B, 2, N, N)  reconstruction normalisée
        z : (B, latent_dim)  code latent
        """
        z     = self.encoder(U, freq_ratio)
        U_hat = self.decoder(z, freq_ratio)
        return U_hat, z

    def loss(
        self,
        U     : torch.Tensor,
        U_hat : torch.Tensor,
        z     : torch.Tensor,          # non utilisé ici, gardé pour compatibilité
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(U_hat, U)
        ridge      = U_hat.pow(2).mean()   # régularisation L2 sur les sorties
        total      = recon_loss + self.beta * ridge
        return total, {
            'recon_loss': recon_loss.detach(),
            'ridge'     : ridge.detach(),
        }



class LaplaceLatentSurrogate(nn.Module):
    """
    MLP θ → z → Û, conditionné sur freq_ratio = k / K.

    Le décodeur partagé reçoit freq_ratio à chaque appel forward.
    """

    def __init__(self, latent_dim: int, theta_dim: int, freq_ratio: float = 0.0,
                 hidden_dim: int = 256):
        super().__init__()
        self.freq_ratio = freq_ratio
        mid = hidden_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(theta_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Linear(mid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._decoder: nn.Module | None = None  # référence non-enregistrée

    def set_decoder(self, decoder: nn.Module):
        """Injecte le décodeur partagé sans le ré-enregistrer comme sous-module."""
        self.__dict__['_decoder'] = decoder

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        z = self.proj(theta)                   # (B, latent_dim)
        assert self._decoder is not None, "Appeler set_decoder() avant forward()"
        return self._decoder(z, self.freq_ratio)  # (B, 2, N, N)

    def loss(self, U_hat: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(U_hat, U)


class LaplaceLatentModel(LaplaceModel):
    """
    Modèle complet AE :
      - K LaplaceLatentSurrogate (MLP θ→z, un par fréquence)
      - un unique shared_decoder (LaplaceDecoder gelé, partagé par référence)

    Pipeline en 2 étapes :
      1. Entraîner LaplaceAE sur tous les champs Laplace (avec freq_ratio)
      2. Pour chaque fréquence k, entraîner LaplaceLatentSurrogate.proj
    """

    def __init__(self, K: int, Nt: int, N: int,
                 theta_dim: int = 4, latent_dim: int = 64, freq_L: int = 8,
                 k_max: int | None = None, hidden_dim: int = 256):
        BaseDecoder.__init__(self)
        self.surrogates = nn.ModuleList([
            LaplaceLatentSurrogate(
                latent_dim=latent_dim,
                theta_dim=theta_dim,
                freq_ratio=k / max(K - 1, 1),
                hidden_dim=hidden_dim,
            )
            for k in range(K)
        ])
        self.shared_decoder = LaplaceDecoder(N=N, latent_dim=latent_dim, freq_L=freq_L)
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

        self.K          = K
        self.Nt         = Nt
        self.N          = N
        self.theta_dim  = theta_dim
        self.latent_dim = latent_dim
        self.k_max      = k_max
        self.hidden_dim = hidden_dim
        self.register_buffer('target_mean', torch.zeros(K, 2, N, N))
        self.register_buffer('target_std',  torch.ones(K,  2, N, N))
        self.register_buffer('s_real', torch.zeros(K, dtype=torch.float64))
        self.register_buffer('s_imag', torch.zeros(K, dtype=torch.float64))

    def _inject_decoder(self):
        for s in self.surrogates:
            assert isinstance(s, LaplaceLatentSurrogate)
            s.set_decoder(self.shared_decoder)

    def set_ae_decoder(self, ae: LaplaceAE):
        """Charge les poids du décodeur AE dans shared_decoder."""
        self.shared_decoder.load_state_dict(ae.decoder.state_dict())
        self.shared_decoder.requires_grad_(False)
        self._inject_decoder()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._inject_decoder()
        return result

    def _forward_k_diff(self, theta_norm: torch.Tensor, n_active: int) -> torch.Tensor:
        """
        Override optimisé : toutes les projections θ→z en n_active appels MLP,
        puis UN SEUL appel au shared_decoder batché.
        Retourne M_active : (B, N*N, n_active) complex64.
        """
        B  = theta_norm.shape[0]
        NN = self.N * self.N

        z_all = torch.cat(
            [self.surrogates[k].proj(theta_norm) for k in range(n_active)],  # type: ignore[operator]
            dim=0,
        )  # (n_active*B, latent_dim)

        fr = torch.tensor(
            [self.surrogates[k].freq_ratio for k in range(n_active)],
            device=theta_norm.device, dtype=torch.float32,
        ).repeat_interleave(B)  # (n_active*B,)

        preds = self.shared_decoder(z_all.float(), fr)  # (n_active*B, 2, N, N)
        preds = preds.view(n_active, B, 2, self.N, self.N)
        return torch.complex(
            preds[:, :, 0].reshape(n_active, B, NN).permute(1, 2, 0).float(),
            preds[:, :, 1].reshape(n_active, B, NN).permute(1, 2, 0).float(),
        )  # (B, NN, n_active)

    def loss(self, *_, **__):
        raise NotImplementedError(
            "L'entraînement se fait par fréquence via LaplaceLatentSurrogate.loss()."
        )

    def __repr__(self) -> str:
        return (f"LaplaceLatentModel(K={self.K}, Nt={self.Nt}, "
                f"N={self.N}, theta_dim={self.theta_dim}, latent_dim={self.latent_dim})\n"
                f"Shared decoder : {self.shared_decoder.__class__.__name__}\n"
                f"Surrogate par fréquence : {self.surrogates[0].__repr__()}")



if __name__ == "__main__":
    ae = LaplaceAE(N=128, latent_dim=64, freq_L=8)
    total_params   = sum(p.numel() for p in ae.parameters())
    encoder_params = sum(p.numel() for p in ae.encoder.parameters())
    decoder_params = sum(p.numel() for p in ae.decoder.parameters())
    print(f"Total parameters : {total_params:,}")
    print(f"Encoder parameters : {encoder_params:,}")
    print(f"Decoder parameters : {decoder_params:,}")
    print(f"Decoder / Total ratio : {decoder_params / total_params:.2%}")
    print(ae)

    U = torch.randn(2, 2, 128, 128)

    # Vérification forward
    U_hat, z = ae(U, freq_ratio=0.75)
    assert U_hat.shape == U.shape,  f"U_hat : attendu {U.shape}, obtenu {U_hat.shape}"
    assert z.shape == (2, 64),      f"z : attendu (2, 64), obtenu {z.shape}"
    print(f"U {U.shape} → U_hat {U_hat.shape}, z {z.shape} ")

    # Vérification loss
    loss, metrics = ae.loss(U, U_hat, z)
    print(f"Loss : {loss.item():.4f}  |  metrics : {metrics}")

    # Vérification SinusoidalFreqEncoding
    enc = SinusoidalFreqEncoding(L=8, out_dim=64)
    f_test = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    emb = enc(f_test)
    assert emb.shape == (5, 64)
    print(f"SinusoidalFreqEncoding : {f_test.flatten().tolist()} → embeddings shape {emb.shape}")


    import time
    model = ae.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    # Avant la boucle train, après le premier batch
    for i in range(5):
        u, freq_ratio = torch.randn(256, 2, 128, 128), torch.rand(256)  # batch fictif
        u = u.to(device, non_blocking=True)
        freq_ratio = freq_ratio.to(device, non_blocking=True)
        torch.cuda.synchronize()

        # Forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            u_hat, z = model(u, freq_ratio)
            loss, metrics = model.loss(u, u_hat, z)
        torch.cuda.synchronize()
        print(f"forward : {(time.perf_counter()-t0)*1000:.1f}ms")

        # Backward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        print(f"backward : {(time.perf_counter()-t0)*1000:.1f}ms")