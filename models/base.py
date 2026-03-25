import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Union


class BaseAutoEncoder(nn.Module):
    """
    Classe de base pour les modèles encodeur-décodeur.

    Chaque sous-classe doit définir :
      - self.encoder
      - self.decoder
      - self.latent_dim
      - loss(...)
    """

    encoder    : nn.Module
    decoder    : nn.Module
    latent_dim : int

    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def _latent_picker(self, n_sample, device: torch.device, theta: Union[None, torch.Tensor] = None, ) -> torch.Tensor:
        """
        Retourne un element de l'espace latent à partir de theta, si nécessaire.
        Par défaut : retourne un tenseur aleatoire de dimension (B, latent_dim).
        Les sous-classes peuvent surcharger cette méthode si la logique de sampling
        est plus complexe (ex. : VAE avec reparametrization).
        """
        return torch.randn(n_sample, self.latent_dim, device=device)

    
    @torch.no_grad()
    def generate(self, n_samples: int = 1, theta: Union[None, torch.Tensor] = None, grad: bool = False) -> torch.Tensor:
        """
        Génère n_samples échantillons à partir du modèle.
        Par défaut : échantillonnage aléatoire dans l'espace latent, puis décodage.
        Les sous-classes peuvent surcharger cette méthode si la génération diffère
        (ex. : conditionnement sur des paramètres).
        """
        self.eval()
        z = self._latent_picker(n_samples, next(self.parameters()).device, theta)  # (n_samples, latent_dim)
        return self.decoder(z)

    def __repr__(self) -> str:
        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return (
            f"{type(self).__name__}(\n"
            f"  latent_dim={self.latent_dim}\n"
            f"  encoder [{n_enc:,} params]:\n"
            f"{self._indent(repr(self.encoder))}\n"
            f"  decoder [{n_dec:,} params]:\n"
            f"{self._indent(repr(self.decoder))}\n"
            f")"
        )

    @staticmethod
    def _indent(s: str, spaces: int = 4) -> str:
        pad = ' ' * spaces
        return '\n'.join(pad + line for line in s.splitlines())


class BaseDecoder(nn.Module):
    """
    Classe de base pour tous les modèles du projet.

    Chaque sous-classe doit implémenter :
      - forward(...)
      - loss(...)
      - _generate(theta, **kwargs)  — optionnel, si la génération diffère du forward
    """

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Calcule la loss du modèle. À implémenter par chaque sous-classe."""
        raise NotImplementedError

    def _generate(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Logique de génération. Les sous-classes peuvent surcharger cette méthode
        (ex. : sampling latent pour un VAE).
        Par défaut : appel direct au forward.
        """
        return self(theta, **kwargs)

    @torch.no_grad()
    def generate(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Interface publique de génération.
        Passe en mode eval, normalise theta en (B, D) si nécessaire,
        puis délègue à _generate.
        Ne pas surcharger dans les sous-classes — surcharger _generate à la place.
        """
        self.eval()
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        return self._generate(theta, **kwargs)
