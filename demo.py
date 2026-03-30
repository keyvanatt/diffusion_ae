"""
demo.py — Visualisation d'un sample : prédictions des meilleurs modèles vs ground truth
========================================================================================
Usage :
    .conda/bin/python demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from main import load_model, run_inference

DATASET_PATH = 'dataset/dataset.npz'


def random_theta(seed=None):
    """Tire un theta au hasard."""
    rng       = np.random.default_rng(seed)
    data      = np.load(DATASET_PATH, allow_pickle=True)
    theta_all = data['theta'].astype(np.float32)
    idx       = int(rng.integers(len(theta_all)))
    return theta_all[idx].tolist()


def find_nearest(theta_raw, theta_all):
    theta_raw = np.array(theta_raw, dtype=np.float32)
    std  = theta_all.std(axis=0) + 1e-8
    diff = (theta_all - theta_raw) / std
    return int(np.argmin((diff ** 2).sum(axis=1)))


def load_ground_truth(theta_raw):
    data      = np.load(DATASET_PATH, allow_pickle=True)
    U_all     = data['U'].astype(np.float32)
    theta_all = data['theta'].astype(np.float32)
    idx       = find_nearest(theta_raw, theta_all)
    return U_all[idx], idx, theta_all[idx]


def plot_demo(theta_raw, results, U_gt, gt_idx, out_path='plots/demo.png'):
    n_models = len(results)
    n_cols   = n_models + 1   # GT + un par modèle

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Échelle commune prédictions/GT
    all_vals = np.concatenate([U_gt.ravel()] + [u.ravel() for _, u in results])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    # Erreur max commune
    emax = float(max(np.abs(u - U_gt).max() for _, u in results))

    # ── Ligne 1 : GT + prédictions ─────────────────────────────────────
    im = axes[0, 0].imshow(U_gt, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Ground truth\n(idx {gt_idx})', fontsize=9)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for col, (name, U_pred) in enumerate(results, start=1):
        rmse = float(np.sqrt(np.mean((U_pred - U_gt) ** 2)))
        im   = axes[0, col].imshow(U_pred, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'{name}\nRMSE={rmse:.4f}', fontsize=9)
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # ── Ligne 2 : erreurs absolues ─────────────────────────────────────
    axes[1, 0].axis('off')

    for col, (name, U_pred) in enumerate(results, start=1):
        err = np.abs(U_pred - U_gt)
        im  = axes[1, col].imshow(err, origin='lower', cmap='Reds', vmin=0, vmax=emax)
        axes[1, col].set_title(f'|Erreur|\n{name}', fontsize=9)
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    theta_str = f'D={theta_raw[0]:.3f}  bx={theta_raw[1]:.3f}  by={theta_raw[2]:.3f}  f={theta_raw[3]:.3f}'
    fig.suptitle(theta_str, fontsize=10)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure sauvegardée → {out_path}')
    plt.show()


def demo(theta_raw: list, ckpt_paths: list, out_path: str = 'plots/demo.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    print('Chargement ground truth...')
    U_gt, gt_idx, gt_theta = load_ground_truth(theta_raw)
    print(f'  voisin idx={gt_idx}  theta={gt_theta.tolist()}')

    results = []
    for ckpt_path in ckpt_paths:
        label = Path(ckpt_path).stem
        print(f'── {label}')
        model, ckpt = load_model(ckpt_path, device)
        U_pred = run_inference(theta_raw, model, ckpt, device)
        results.append((label, U_pred))

    plot_demo(theta_raw, results, U_gt, gt_idx, out_path=out_path)


if __name__ == '__main__':
    demo(
        theta_raw  = random_theta(),
        ckpt_paths = [
            'checkpoints/DirectDecoderDenseOut_best.pt',
            'checkpoints/finetune_IndirectDecoder_best.pt',
            'checkpoints/finetune_IndirectDecoderSVD_best.pt',
            'checkpoints/finetune_smallLD_IndirectDecoder_best.pt',
        ],
        out_path   = 'plots/demo.png',
    )
