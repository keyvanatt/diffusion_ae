"""
transient/gamma_opti.py — Optimisation conjointe de gamma et k_max
===================================================================
Pour chaque paire (gamma, k_max), calcule l'erreur L2 relative de
reconstruction :

  - k <= k_max : coefficients Laplace vrais
  - k  > k_max : moyenne du dataset d'entraînement dans l'espace de Laplace

La moyenne d'entraînement est calculée une seule fois par gamma,
puis réutilisée pour tous les k_max.

Usage :
    .conda/bin/python transient/gamma_opti.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.laplace import laplace_forward, laplace_inverse


def compute_train_mean(
    U_raw,
    train_idx: np.ndarray,
    gamma: float,
    dt: float = 1.0,
    rule: str = 'trap',
) -> np.ndarray:
    """
    Calcule la moyenne Laplace par fréquence sur le train.
    Retourne mean_k_pos de shape (Nt_half, N) complexe.
    """
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N       = H * W

    mean_k_pos = np.zeros((Nt_half, N), dtype=np.complex64)
    for i in tqdm(train_idx, desc=f"  [gamma={gamma:.4f}] train mean", leave=False):
        U_i = U_raw[i].copy().astype(np.float32)
        C_i = U_i.reshape(Nt, N).T                      # (N, Nt)
        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)
        mean_k_pos += M[:, :Nt_half].T.astype(np.complex64)
    mean_k_pos /= len(train_idx)
    return mean_k_pos


def compute_mean_energy(
    U_raw,
    sample_idx: np.ndarray,
    gamma: float,
    dt: float = 1.0,
    rule: str = 'trap',
) -> np.ndarray:
    """
    Énergie moyenne du spectre de Laplace par fréquence k.
    Retourne energy (Nt_half,) : mean over samples of sum_pixels(|M_k|^2).
    """
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N       = H * W

    energy = np.zeros(Nt_half, dtype=np.float64)
    for i in tqdm(sample_idx, desc=f"  [gamma={gamma:.4f}] energy", leave=False):
        U_i = U_raw[i].copy().astype(np.float32)
        C_i = U_i.reshape(Nt, N).T
        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)
        energy += (np.abs(M[:, :Nt_half]) ** 2).sum(axis=0)
    energy /= len(sample_idx)
    return energy.astype(np.float32)


def compute_l2rel_for_kmax(
    U_raw,
    gamma: float,
    k_max: int,
    mean_k_pos: np.ndarray,
    test_idx: np.ndarray,
    dt: float = 1.0,
    rule: str = 'trap',
) -> np.ndarray:
    """
    Calcule les erreurs L2 relatives pour un (gamma, k_max) donné.
    mean_k_pos (Nt_half, N) est déjà calculé pour ce gamma.
    """
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N       = H * W
    n_tail  = Nt - Nt_half

    l2rel_list = []
    for i in tqdm(test_idx, desc=f"  k_max={k_max}", leave=False):
        U_true = U_raw[i].copy().astype(np.float32)
        C_i    = U_true.reshape(Nt, N).T

        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)

        M_pos = M[:, :Nt_half].copy().astype(np.complex64)
        if k_max + 1 < Nt_half:
            M_pos[:, k_max + 1:] = mean_k_pos[k_max + 1:, :].T

        M_full = np.zeros((N, Nt), dtype=np.complex64)
        M_full[:, :Nt_half] = M_pos
        if n_tail > 0:
            M_full[:, Nt_half:] = np.conj(M_pos[:, n_tail:0:-1])

        C_rec, _ = laplace_inverse(M_full, dt, Nt, rule=rule, gamma=gamma)
        U_pred   = C_rec.T.reshape(Nt, H, W)

        err = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
        l2rel_list.append(float(err))

    return np.array(l2rel_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str = '/Data/KAT/ch4_rotated.npy',
    k_max_list     = [5, 10, 15, 20, 30, 40, 50],
    gamma_list     = np.sort(np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, -0.001, -0.002, -0.003, -0.005, -0.01, -0.02])),
    dt: float      = 1.0,
    rule: str      = 'trap',
    n_train: int   = 200,
    n_test: int    = 100,
    seed: int      = 42,
    save_plot: str = 'plots/gamma_opti.png',
):
    rng   = np.random.default_rng(seed)
    U_raw = np.load(data_path, mmap_mode='r')
    ns    = U_raw.shape[0]

    idx       = rng.permutation(ns)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:n_train + n_test]

    print(f"Dataset  : {U_raw.shape}")
    print(f"n_train  : {n_train}    n_test : {n_test}")
    print(f"gammas   : {gamma_list}")
    print(f"k_max    : {k_max_list}")
    print()

    # ------------------------------------------------------------------
    # Plot énergie spectrale en premier
    # ------------------------------------------------------------------
    os.makedirs(Path(save_plot).parent, exist_ok=True)
    save_energy = str(Path(save_plot).with_stem(Path(save_plot).stem + '_energy'))

    print("Calcul de l'énergie spectrale...")
    energy_per_gamma = {}
    for gamma in gamma_list:
        energy_per_gamma[gamma] = compute_mean_energy(
            U_raw, test_idx, gamma, dt=dt, rule=rule,
        )

    fig_e, ax_e = plt.subplots(figsize=(10, 5))
    ks = np.arange(U_raw.shape[1] // 2 + 1)
    for gamma in gamma_list:
        ax_e.semilogy(ks, energy_per_gamma[gamma], lw=1.5, label=f'γ={gamma:.4f}')
    for k in k_max_list:
        ax_e.axvline(k, color='gray', linestyle=':', lw=0.8)
    ax_e.set_xlabel('k (fréquence Laplace)', fontsize=12)
    ax_e.set_ylabel('Énergie moyenne (log)', fontsize=12)
    ax_e.set_title(f'Énergie spectrale moyenne par fréquence (n={len(test_idx)} samples)', fontsize=12)
    ax_e.legend(fontsize=8, ncol=3)
    ax_e.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(save_energy, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure énergie sauvegardée → {save_energy}\n")

    # results[(gamma, k_max)] = l2rel array (n_test,)
    results = {}

    for gamma in gamma_list:
        print(f"gamma = {gamma}")
        mean_k_pos = compute_train_mean(U_raw, train_idx, gamma, dt=dt, rule=rule)

        for k_max in tqdm(k_max_list, desc=f"  k_max sweep", leave=False):
            l2rel = compute_l2rel_for_kmax(
                U_raw, gamma, k_max, mean_k_pos, test_idx, dt=dt, rule=rule,
            )
            results[(gamma, k_max)] = l2rel
            print(
                f"  k_max={k_max:<4d}  "
                f"mean={l2rel.mean()*100:.2f}%  "
                f"std={l2rel.std()*100:.2f}%  "
                f"median={np.median(l2rel)*100:.2f}%"
            )
        print()

    # ------------------------------------------------------------------
    # Tableau récapitulatif
    # ------------------------------------------------------------------
    mean_grid = np.array([
        [results[(g, k)].mean() * 100 for k in k_max_list]
        for g in gamma_list
    ])  # (n_gamma, n_kmax)

    best_idx   = np.unravel_index(np.argmin(mean_grid), mean_grid.shape)
    best_gamma = gamma_list[best_idx[0]]
    best_kmax  = k_max_list[best_idx[1]]
    print(f"Meilleure combinaison : gamma={best_gamma}, k_max={best_kmax}  "
          f"(L2rel moyen = {mean_grid[best_idx]:.2f}%)")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    os.makedirs(Path(save_plot).parent, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap (gamma × k_max)
    ax = axes[0]
    im = ax.imshow(
        mean_grid, aspect='auto', origin='upper',
        cmap='viridis_r',
    )
    ax.set_xticks(range(len(k_max_list)))
    ax.set_xticklabels([str(k) for k in k_max_list])
    ax.set_yticks(range(len(gamma_list)))
    ax.set_yticklabels([f"{g:.4f}" for g in gamma_list])
    ax.set_xlabel('k_max', fontsize=12)
    ax.set_ylabel('γ (gamma)', fontsize=12)
    ax.set_title('Mean L2 relative error (%) — grid γ × k_max', fontsize=12)
    ax.plot(best_idx[1], best_idx[0], 'r*', markersize=14, label=f'best ({best_gamma}, {best_kmax})')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='L2rel (%)')
    # Annotate cells
    for i in range(len(gamma_list)):
        for j in range(len(k_max_list)):
            ax.text(j, i, f"{mean_grid[i, j]:.1f}", ha='center', va='center',
                    fontsize=7, color='white' if mean_grid[i, j] > mean_grid.mean() else 'black')

    # Courbes mean L2rel vs k_max pour chaque gamma
    ax2 = axes[1]
    for i, g in enumerate(gamma_list):
        ax2.plot(k_max_list, mean_grid[i], 'o-', label=f'γ={g}', lw=1.5, markersize=5)
    ax2.axvline(best_kmax, color='red', linestyle='--', lw=1.2, label=f'best k_max={best_kmax}')
    ax2.set_xlabel('k_max', fontsize=12)
    ax2.set_ylabel('Mean L2 relative error (%)', fontsize=12)
    ax2.set_title('Error vs k_max per γ', fontsize=12)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure sauvegardée → {save_plot}")

    return results, mean_grid, gamma_list, k_max_list


if __name__ == '__main__':
    main()
