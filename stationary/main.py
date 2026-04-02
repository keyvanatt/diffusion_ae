"""
stationary/main.py — Inférence DirectDecoder : theta → grille U prédite
=============================================================
Fonctions réutilisables :
  load_model(ckpt_path, device)            → model, ckpt
  run_inference(theta_raw, model, ckpt, device) → U_pred (N, N)
  predict(theta_raw, ckpt_path, device)    → U_pred (N, N)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from models.base import BaseDecoder

from models.direct_decoder import DirectDecoder, DirectDecoderDenseOut


def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state']

    model_type = ckpt.get('model_type', 'decoder')

    if model_type in ('decoder', 'DirectDecoder'):
        fc_out = state['fc.0.weight'].shape[0]
        base   = int((fc_out / 256) ** 0.5)
        model  = DirectDecoder(N=base * 16, theta_dim=4).to(device)
    elif model_type == 'DirectDecoderDenseOut':
        fc_out = state['fc.0.weight'].shape[0]
        base   = int((fc_out / 256) ** 0.5)
        model  = DirectDecoderDenseOut(N=base * 32, theta_dim=4).to(device)
    elif model_type == 'IndirectDecoder':
        from models.variationalAutoEncoder import VAE, IndirectDecoder
        latent_dim = state['theta_proj.2.weight'].shape[0]
        theta_dim  = state['theta_proj.0.weight'].shape[1]
        N          = int(state['decoder.out_fc.3.weight'].shape[0] ** 0.5)
        dummy_vae  = VAE(N=N, latent_dim=latent_dim)
        model      = IndirectDecoder(dummy_vae, N=N, theta_dim=theta_dim,
                                     latent_dim=latent_dim).to(device)
    elif model_type == 'IndirectDecoderSVD':
        from models.AE_SVD import AutoencoderSVD, IndirectDecoderSVD
        theta_dim  = state['theta_proj.0.weight'].shape[1]
        latent_dim = state['theta_proj.2.weight'].shape[0]
        N          = int(state['decoder.out_fc.3.weight'].shape[0] ** 0.5)
        kmax       = state['svd_proj.fixed_basis_buffer'].shape[1]
        dummy_ae   = AutoencoderSVD(N=N, latent_dim=latent_dim, kmax=kmax)
        model      = IndirectDecoderSVD(N=N, kmax=kmax, theta_dim=theta_dim,
                                        latent_dim=latent_dim,
                                        trained_autoencoder=dummy_ae).to(device)
        # Le buffer None n'apparaît pas dans les clés attendues — on l'initialise
        # pour que load_state_dict puisse le reconnaître et le remplacer.
        model.svd_proj.set_fixed_basis(
            torch.zeros(latent_dim, kmax, device=device)
        )
    else:
        raise ValueError(f"Checkpoint de type '{model_type}' non supporté.")

    model.load_state_dict(state)
    model.eval()

    return model, ckpt


def denorm_U(U_norm: torch.Tensor, ckpt: dict) -> torch.Tensor:
    """Inverse la normalisation avec les stats du checkpoint.
    Supporte les deux formats :
      - nouveau : U_mean (grille), U_std (scalaire) — Z-score
      - ancien  : U_mean (grille), U_min, U_max     — min-max centré
    """
    U_mean = ckpt['U_mean'].cpu() if torch.is_tensor(ckpt['U_mean']) else ckpt['U_mean']
    if 'U_std' in ckpt:
        return U_norm * float(ckpt['U_std']) + U_mean
    else:
        U_min = float(ckpt['U_min'])
        U_max = float(ckpt['U_max'])
        return (U_norm + 1.0) / 2.0 * (U_max - U_min) + U_min + U_mean


@torch.no_grad()
def run_inference(theta_raw: list[float], model: BaseDecoder,
                  ckpt: dict, device: torch.device) -> np.ndarray:
    """
    Inference avec un modèle déjà chargé.

    Retourne U_pred : np.ndarray shape (N, N) en valeurs physiques.
    """
    theta_mean = ckpt['theta_mean'].to(device)
    theta_std  = ckpt['theta_std'].to(device)

    theta_t    = torch.tensor(theta_raw, dtype=torch.float32, device=device)
    theta_norm = (theta_t - theta_mean) / theta_std
    U_hat_norm = model.generate(theta_norm)                           # (1, 1, N, N)
    return denorm_U(U_hat_norm.cpu(), ckpt)[0, 0].numpy()            # (N, N)


def predict(theta_raw: list[float], ckpt_path: str = 'checkpoints/decoder_best.pt',
            device_str: str = 'auto') -> np.ndarray:
    """
    Paramètres
    ----------
    theta_raw  : [D, bx, by, f] — 4 valeurs physiques (non normalisées)
    ckpt_path  : chemin vers le checkpoint
    device_str : 'auto', 'cpu' ou 'cuda'

    Retourne
    --------
    U_pred : np.ndarray  shape (N, N)  en valeurs physiques
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    model, ckpt = load_model(ckpt_path, device)
    return run_inference(theta_raw, model, ckpt, device)


def main(
    ckpt_path = 'checkpoints/DirectDecoderDenseOut_best.pt',
    theta     = [0.02, 0.5, 0.3, 10.0],   # [D, bx, by, f]
    out       = None,                       # chemin .npy pour sauvegarder, ou None
    plot      = True,
    device    = 'auto',
):
    import matplotlib.pyplot as plt

    U_pred = predict(theta, ckpt_path, device)
    print(f'Prédiction : shape={U_pred.shape}  min={U_pred.min():.4f}  max={U_pred.max():.4f}')

    if out:
        np.save(out, U_pred)
        print(f'Sauvegardé → {out}')

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(U_pred, origin='lower', cmap='viridis')
        ax.set_title(f'theta = {theta}')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    return U_pred


if __name__ == '__main__':
    main()
