"""
benchmark.py — Compare plusieurs checkpoints BaseDecoder sur le dataset de test
================================================================================
Usage :
    python benchmark.py --ckpts checkpoints/DirectDecoder_best.pt checkpoints/DirectDecoderDenseOut_best.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from utils.dataset import ConvDiffDataset
from models.base import BaseDecoder
from main import load_model, denorm_U


# ---------------------------------------------------------------------------
# Métriques par sample
# ---------------------------------------------------------------------------

def _grad_loss_per_sample(U: torch.Tensor, U_hat: torch.Tensor) -> torch.Tensor:
    """MSE sur les gradients spatiaux, retourne (B,)."""
    dx_gt  = U[:, :, :, 1:] - U[:, :, :, :-1]
    dy_gt  = U[:, :, 1:, :] - U[:, :, :-1, :]
    dx_hat = U_hat[:, :, :, 1:] - U_hat[:, :, :, :-1]
    dy_hat = U_hat[:, :, 1:, :] - U_hat[:, :, :-1, :]
    return (
        (dx_hat - dx_gt).pow(2).mean(dim=(1, 2, 3)) +
        (dy_hat - dy_gt).pow(2).mean(dim=(1, 2, 3))
    ) * 0.5


@torch.no_grad()
def evaluate(model: BaseDecoder, ckpt: dict, loader,
             device: torch.device) -> dict:
    """Évalue le modèle sur loader. Retourne des arrays numpy par sample."""
    # Warmup
    theta_dummy, _ = next(iter(loader))
    model(theta_dummy[:1].to(device))

    mse_all, mae_all, grad_all, time_all = [], [], [], []

    for theta, U_norm in loader:
        theta  = theta.to(device)
        B      = theta.shape[0]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        U_hat_norm = model(theta)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms_per_sample = (time.perf_counter() - t0) / B * 1000

        U_phys     = denorm_U(U_norm,            ckpt)
        U_hat_phys = denorm_U(U_hat_norm.cpu(),  ckpt)

        mse_all.append( (U_phys - U_hat_phys).pow(2).mean(dim=(1, 2, 3)).numpy())
        mae_all.append( (U_phys - U_hat_phys).abs().mean(dim=(1, 2, 3)).numpy())
        grad_all.append(_grad_loss_per_sample(U_phys, U_hat_phys).numpy())
        time_all.extend([ms_per_sample] * B)

    return {
        'MSE'              : np.concatenate(mse_all),
        'MAE'              : np.concatenate(mae_all),
        'Grad loss'        : np.concatenate(grad_all),
        'Temps (ms/sample)': np.array(time_all),
    }


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def benchmark(ckpt_paths: list, dataset_path: str, batch_size: int, seed: int) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    dataset = ConvDiffDataset(dataset_path)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    results = {}

    for ckpt_path in ckpt_paths:
        label = Path(ckpt_path).stem
        print(f'── {label}')

        model, ckpt = load_model(ckpt_path, device)

        # Normalise le dataset avec les stats de ce checkpoint
        dataset.U_mean = ckpt['U_mean']
        dataset.U_std  = ckpt['U_std']
        dataset.U      = (dataset.U_raw - dataset.U_mean) / dataset.U_std

        loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'   Params : {n_params:,}  |  Samples test : {n_test}')

        metrics = evaluate(model, ckpt, loader, device)
        results[label] = metrics

        for key, vals in metrics.items():
            print(f'   {key:<22s}: {vals.mean():.4f} ± {vals.std():.4f}')
        print()

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_path: str = 'plots/benchmark.png'):
    labels      = list(results.keys())
    metric_keys = list(next(iter(results.values())).keys())
    colors      = [plt.cm.get_cmap('tab10')(i) for i in range(len(labels))]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, metric_keys):
        data  = [results[lbl][key] for lbl in labels]
        parts = ax.violinplot(data, positions=range(len(labels)),
                              showmedians=True, showextrema=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.75)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        for i, d in enumerate(data):
            ax.text(i, np.median(d), f'{np.median(d):.3g}',
                    ha='center', va='bottom', fontsize=7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_title(key, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if key != 'Temps (ms/sample)':
            ax.set_yscale('log')

    plt.suptitle('Benchmark — test set (espace physique)', fontsize=11, y=1.02)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure sauvegardée → {out_path}')
    plt.show()


if __name__ == '__main__':
    results = benchmark(
        ckpt_paths   = [
            'checkpoints/DirectDecoderDenseOut_best.pt',
        ],
        dataset_path = 'dataset/dataset.npz',
        batch_size   = 64,
        seed         = 42,
    )
    plot_results(results, out_path='plots/benchmark.png')
