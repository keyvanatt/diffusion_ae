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
from tqdm import tqdm
from utils.sim import ConvDiffSimulator, to_grid



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
             device: torch.device, n_timing_batches: int = 10) -> dict:
    """Évalue le modèle sur loader. Retourne des arrays numpy par sample + temps moyen sur n_timing_batches."""
    # Erreurs sur tous les batches
    mse_all, mae_all, grad_all = [], [], []
    timing_batches = []   # batches réservés pour le timing

    def _record(U_norm, U_hat_norm):
        U_phys     = denorm_U(U_norm,           ckpt)
        U_hat_phys = denorm_U(U_hat_norm.cpu(), ckpt)
        mse_all.append( (U_phys - U_hat_phys).pow(2).mean(dim=(1, 2, 3)).numpy())
        mae_all.append( (U_phys - U_hat_phys).abs().mean(dim=(1, 2, 3)).numpy())
        grad_all.append(_grad_loss_per_sample(U_phys, U_hat_phys).numpy())

    for i, (theta, U_norm) in enumerate(loader):
        theta_dev  = theta.to(device)
        U_hat_norm = model(theta_dev)
        _record(U_norm, U_hat_norm)
        if i < n_timing_batches:
            timing_batches.append(theta_dev)

    # Warmup
    model(timing_batches[0][:1])

    # Timing sur n_timing_batches batches
    times_us = []
    B = timing_batches[0].shape[0]
    for t_batch in timing_batches:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        model(t_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_us.append((time.perf_counter_ns() - t0) / 1e3 / B)

    batch_time_us = float(np.mean(times_us))

    return {
        'MSE'              : np.concatenate(mse_all),
        'MAE'              : np.concatenate(mae_all),
        'Grad loss'        : np.concatenate(grad_all),
        'Temps (µs/sample)': batch_time_us,
    }



def sim_timing(dataset: ConvDiffDataset, test_indices, n_samples: int, N_mesh: int = 64) -> dict:
    """
    Mesure le temps d'exécution du simulateur FEniCS sur n_samples du test set.
    Les métriques d'erreur sont NaN (sim = ground truth).
    """
    indices = list(test_indices)[:n_samples]
    times   = []

    # Compilation JIT une seule fois avant la boucle de timing
    sim = ConvDiffSimulator(n=N_mesh)

    # Dénormalise theta : theta_raw = theta_norm * std + mean
    for idx in tqdm(indices, desc='Sim baseline'):
        theta_norm = dataset.theta[idx].numpy()
        theta_raw  = theta_norm * dataset.theta_std.numpy() + dataset.theta_mean.numpy()
        D, bx, by, f = float(theta_raw[0]), float(theta_raw[1]), \
                       float(theta_raw[2]), float(theta_raw[3])

        t0    = time.perf_counter_ns()
        u_sol = sim.solve(D=D, b_val=np.array([bx, by]), f=f, x0=np.array([0.5, 0.5]))
        to_grid(u_sol, N_out=dataset.N)
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    nan = np.full(len(times), np.nan)
    print(f'   Temps (ms/sample)     : {np.mean(times):.1f} ± {np.std(times):.1f}')
    return {
        'MSE'              : nan,
        'MAE'              : nan,
        'Grad loss'        : nan,
        'Temps (ms/sample)': np.array(times),
    }


def benchmark(ckpt_paths: list, dataset_path: str, batch_size: int, seed: int | None,
              n_sim_samples: int = 0, N_mesh: int = 64) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    dataset = ConvDiffDataset(dataset_path)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=generator
    )

    results = {}

    for ckpt_path in ckpt_paths:
        label = Path(ckpt_path).stem
        print(f'── {label}')

        model, ckpt = load_model(ckpt_path, device)

        # Normalise le dataset avec les stats de ce checkpoint (deux formats supportés)
        dataset.U_mean = ckpt['U_mean'].cpu()
        if 'U_std' in ckpt:
            dataset.U_std = ckpt['U_std']
            dataset.U     = (dataset.U_raw - dataset.U_mean) / float(dataset.U_std)
        else:
            U_min = float(ckpt['U_min'])
            U_max = float(ckpt['U_max'])
            dataset.U = 2.0 * (dataset.U_raw - dataset.U_mean - U_min) / (U_max - U_min) - 1.0

        loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'   Params : {n_params:,}  |  Samples test : {n_test}')

        metrics = evaluate(model, ckpt, loader, device)
        results[label] = metrics

        for key, val in metrics.items():
            if np.isscalar(val):
                print(f'   {key:<22s}: {val:.1f} µs')
            else:
                print(f'   {key:<22s}: {val.mean():.4f} ± {val.std():.4f}')
        print()

    if n_sim_samples > 0:
        print('── Sim (FEniCS)')
        results['Sim (FEniCS)'] = sim_timing(dataset, test_set.indices,
                                             n_sim_samples, N_mesh)
        print()

    return results



def plot_results(results: dict, batch_size: int, n_timing_batches: int = 10,
                 out_path: str = 'plots/benchmark.png'):
    sim_time_ms = None
    if 'Sim (FEniCS)' in results:
        sim_time_ms = np.nanmean(results['Sim (FEniCS)']['Temps (ms/sample)'])

    labels      = [lbl for lbl in results if lbl != 'Sim (FEniCS)']
    error_keys  = ['MSE', 'MAE', 'Grad loss']
    colors      = [plt.colormaps['tab10'](i) for i in range(len(labels))]

    fig, axes = plt.subplots(1, len(error_keys) + 1,
                             figsize=(4 * (len(error_keys) + 1), 5))

    # Violins pour les métriques d'erreur
    for ax, key in zip(axes[:len(error_keys)], error_keys):
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
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Bar chart pour le temps
    ax_t = axes[-1]
    times = [results[lbl]['Temps (µs/sample)'] for lbl in labels]
    bars  = ax_t.bar(range(len(labels)), times, color=colors[:len(labels)], alpha=0.8)
    ax_t.bar_label(bars, fmt='%.1f µs', fontsize=8, padding=3)
    ax_t.set_xticks(range(len(labels)))
    ax_t.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    time_title = 'Temps / sample'
    if sim_time_ms is not None:
        time_title += f'\nSim (FEniCS) : {sim_time_ms:.1f} ms/sample'
    ax_t.set_title(time_title, fontsize=10)
    ax_t.set_ylabel('µs')
    ax_t.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(
        f'Benchmark — test set | batch B={batch_size}, timing moyenné sur {n_timing_batches} batches',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Figure sauvegardée → {out_path}')
    plt.show()


if __name__ == '__main__':
    results = benchmark(
        ckpt_paths   = [
            'checkpoints/DirectDecoderDenseOut_best.pt',
            'checkpoints/IndirectDecoder_best.pt',
            'checkpoints/finetune_IndirectDecoder_best.pt',
            "checkpoints/finetune_IndirectDecoderSVD_best.pt",
        ],
        dataset_path   = 'dataset/dataset.npz',
        batch_size     = 256,
        seed           = None,
        n_sim_samples  = 20,   # nb de simulations FEniCS pour la baseline temps
        N_mesh         = 64,
    )
    plot_results(results, batch_size=256, n_timing_batches=10, out_path='plots/benchmark.png')
