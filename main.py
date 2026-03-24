"""
main.py — Inférence CVAE : theta → grille U prédite
=====================================================
Usage :
    python main.py --theta 1.0 0.5 0.3 0.8 0.2 0.6
    python main.py --theta 1.0 0.5 0.3 0.8 0.2 0.6 --n_samples 4 --out pred.npy
    python main.py --theta 1.0 0.5 0.3 0.8 0.2 0.6 --plot
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from models.CVAE import CVAE
from models.CVAELight import CVAELight


def load_model(ckpt_path: str, device: torch.device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    state      = ckpt['model_state']
    model_type = ckpt.get('model_type', 'cvae')

    # Déduire N depuis la couche fc du décodeur
    if model_type == 'cvae_light':
        fc_out = state['dec_fc.0.weight'].shape[0]
    else:
        fc_out = state['decoder.fc.0.weight'].shape[0]

    base = int((fc_out / 256) ** 0.5)
    N    = base * 16

    cfg = ckpt['config']
    if model_type == 'cvae_light':
        model = CVAELight(
            N          = N,
            theta_dim  = 6,
            latent_dim = cfg['latent_dim'],
            beta       = cfg['beta'],
            free_bits  = cfg.get('free_bits', 0.2),
        ).to(device)
    else:  # 'cvae' ou anciens checkpoints
        model = CVAE(
            N          = N,
            theta_dim  = 6,
            latent_dim = cfg['latent_dim'],
            beta       = cfg['beta'],
            free_bits  = cfg['free_bits'],
        ).to(device)

    model.load_state_dict(state)
    model.eval()

    return model, ckpt


def predict(theta_raw: list[float], ckpt_path: str = 'checkpoints/cvae_best.pt',
            n_samples: int = 1, device_str: str = 'auto') -> np.ndarray:
    """
    Paramètres
    ----------
    theta_raw  : 6 valeurs physiques (non normalisées)
    ckpt_path  : chemin vers le checkpoint
    n_samples  : nombre de tirages stochastiques (z ~ N(0,I))
    device_str : 'auto', 'cpu' ou 'cuda'

    Retourne
    --------
    U_pred : np.ndarray  shape (n_samples, N, N)  en valeurs physiques
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    model, ckpt = load_model(ckpt_path, device)

    theta_mean = ckpt['theta_mean'].to(device)   # (6,)
    theta_std  = ckpt['theta_std'].to(device)    # (6,)
    U_min      = float(ckpt['U_min'])
    U_max      = float(ckpt['U_max'])

    theta_raw_t = torch.tensor(theta_raw, dtype=torch.float32, device=device)
    theta_norm  = (theta_raw_t - theta_mean) / theta_std              # (6,)

    U_hat_norm = model.generate(theta_norm, n_samples=n_samples)      # (n, 1, N, N)

    U_pred = U_hat_norm * (U_max - U_min) + U_min
    return U_pred[:, 0, :, :].cpu().numpy()                           # (n, N, N)


def main():
    parser = argparse.ArgumentParser(description='CVAE inference: theta → U grid')
    parser.add_argument('--theta',     nargs=6, type=float, required=True,
                        metavar=('t1','t2','t3','t4','t5','t6'),
                        help='6 valeurs physiques de theta')
    parser.add_argument('--ckpt',      default='checkpoints/cvae_best.pt',
                        help='Chemin checkpoint (défaut: checkpoints/cvae_best.pt)')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Nombre de tirages stochastiques (défaut: 1)')
    parser.add_argument('--out',       default=None,
                        help='Sauvegarder la prédiction en .npy')
    parser.add_argument('--plot',      action='store_true',
                        help='Afficher la grille avec matplotlib')
    parser.add_argument('--device',    default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    print(f'theta = {args.theta}')
    print(f'Chargement du checkpoint : {args.ckpt}')

    U_pred = predict(args.theta, args.ckpt, args.n_samples, args.device)

    print(f'Prédiction : shape={U_pred.shape}  min={U_pred.min():.4f}  max={U_pred.max():.4f}')

    if args.out:
        np.save(args.out, U_pred)
        print(f'Sauvegardé → {args.out}')

    if args.plot:
        import matplotlib.pyplot as plt
        n = U_pred.shape[0]
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(U_pred[i], origin='lower', cmap='viridis')
            ax.set_title(f'Sample {i}')
            plt.colorbar(im, ax=ax)
        plt.suptitle(f'theta = {args.theta}')
        plt.tight_layout()
        plt.show()

    return U_pred


if __name__ == '__main__':
    main()
