import torch
import torch.nn as nn
from abc import abstractmethod


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
