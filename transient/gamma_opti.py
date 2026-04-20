"""
transient/gamma_opti.py — Optimisation du paramètre gamma de la transformée de Laplace
=========================================================================================
Pour chaque valeur de gamma, calcule l'erreur L2 relative de reconstruction avec
truncature à k_max fréquences :

  - k <= k_max : coefficients Laplace vrais (calculés depuis le champ U)
  - k  > k_max : moyenne du dataset d'entraînement dans l'espace de Laplace
                 (c'est ce que prédisent les surrogates pour ces fréquences :
                  sortie 0 normalisée → target_mean après dénormalisation)

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


def compute_l2rel_for_gamma(
    U_raw,
    gamma: float,
    k_max: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dt: float = 1.0,
    rule: str = 'trap',
) -> np.ndarray:
    """
    Calcule les erreurs L2 relatives de reconstruction pour un gamma donné.

    Paramètres
    ----------
    U_raw     : mmap array (ns, Nt, H, W)
    gamma     : paramètre d'amortissement de la transformée de Laplace
    k_max     : fréquence de troncature
    train_idx : indices d'entraînement (pour le calcul de la moyenne)
    test_idx  : indices de test (évalués)
    dt, rule  : paramètres de la transformée de Laplace

    Retour
    ------
    l2rel : np.ndarray (n_test,) — erreurs L2 relatives
    """
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N       = H * W
    n_tail  = Nt - Nt_half

    # ------------------------------------------------------------------
    # Étape 1 : moyenne Laplace par fréquence sur le train
    # mean_k_pos[k] : (N,) complex — moyenne de FFT(C * weights)[k]
    # ------------------------------------------------------------------
    mean_k_pos = np.zeros((Nt_half, N), dtype=np.complex64)
    for i in tqdm(train_idx, desc=f"  [gamma={gamma:.4f}] train mean", leave=False):
        U_i = U_raw[i].copy().astype(np.float32)        # (Nt, H, W)
        C_i = U_i.reshape(Nt, N).T                      # (N, Nt)
        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)
        mean_k_pos += M[:, :Nt_half].T.astype(np.complex64)
    mean_k_pos /= len(train_idx)                         # (Nt_half, N)

    # ------------------------------------------------------------------
    # Étape 2 : reconstruction tronquée sur le test set
    # ------------------------------------------------------------------
    l2rel_list = []
    for i in tqdm(test_idx, desc=f"  [gamma={gamma:.4f}] test eval", leave=False):
        U_true = U_raw[i].copy().astype(np.float32)     # (Nt, H, W)
        C_i    = U_true.reshape(Nt, N).T                # (N, Nt)

        # Transformée de Laplace
        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)

        # Spectre tronqué (partie positive des fréquences, k = 0..Nt_half-1)
        M_pos = M[:, :Nt_half].copy().astype(np.complex64)   # (N, Nt_half)
        # k > k_max → remplacer par la moyenne du train
        if k_max + 1 < Nt_half:
            M_pos[:, k_max + 1:] = mean_k_pos[k_max + 1:, :].T

        # Reconstruction du spectre complet via symétrie conjuguée
        # M_full[:, Nt-k] = conj(M_full[:, k])  pour  k = 1..Nt_half-1
        M_full = np.zeros((N, Nt), dtype=np.complex64)
        M_full[:, :Nt_half] = M_pos
        if n_tail > 0:
            # indices positifs 1..n_tail (dans l'ordre décroissant) → fréquences négatives
            M_full[:, Nt_half:] = np.conj(M_pos[:, n_tail:0:-1])

        # Transformée inverse de Laplace
        C_rec, _ = laplace_inverse(M_full, dt, Nt, rule=rule, gamma=gamma)
        U_pred   = C_rec.T.reshape(Nt, H, W)            # (Nt, H, W)

        # Erreur L2 relative
        err = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
        l2rel_list.append(float(err))

    return np.array(l2rel_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str    = '/Data/KAT/ch4_rotated.npy',
    k_max: int        = 20,
    gamma_list        = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    dt: float         = 1.0,
    rule: str         = 'trap',
    n_train: int      = 200,
    n_test: int       = 100,
    seed: int         = 42,
    save_plot: str    = 'plots/gamma_opti.png',
):
    rng   = np.random.default_rng(seed)
    U_raw = np.load(data_path, mmap_mode='r')            # (ns, Nt, H, W)
    ns    = U_raw.shape[0]

    idx       = rng.permutation(ns)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:n_train + n_test]

    print(f"Dataset  : {U_raw.shape}")
    print(f"n_train  : {n_train}    n_test : {n_test}    k_max : {k_max}")
    print(f"gammas   : {gamma_list}")
    print()

    results = {}
    for gamma in gamma_list:
        l2rel = compute_l2rel_for_gamma(
            U_raw, gamma, k_max, train_idx, test_idx, dt=dt, rule=rule,
        )
        results[gamma] = l2rel
        print(
            f"  gamma={gamma:<8.4f}  "
            f"L2rel mean={l2rel.mean()*100:.2f}%  "
            f"std={l2rel.std()*100:.2f}%  "
            f"median={np.median(l2rel)*100:.2f}%"
        )

    # ------------------------------------------------------------------
    # Tableau récapitulatif
    # ------------------------------------------------------------------
    print()
    print(f"{'gamma':>10}  {'mean (%)':>10}  {'std (%)':>9}  {'median (%)':>11}  {'min (%)':>8}  {'max (%)':>8}")
    print("-" * 68)
    gammas = list(results.keys())
    means  = []
    for g in gammas:
        arr = results[g]
        m   = arr.mean() * 100
        means.append(m)
        print(
            f"{g:>10.4f}  {m:>10.2f}  {arr.std()*100:>9.2f}  "
            f"{np.median(arr)*100:>11.2f}  {arr.min()*100:>8.2f}  {arr.max()*100:>8.2f}"
        )

    best_gamma = gammas[int(np.argmin(means))]
    print(f"\nMeilleur gamma : {best_gamma}  (L2rel moyen = {min(means):.2f}%)")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    os.makedirs(Path(save_plot).parent, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    stds = [results[g].std() * 100 for g in gammas]

    # Courbe principale
    ax1.errorbar(
        gammas, means, yerr=stds,
        fmt='o-', capsize=5, lw=1.5, markersize=6,
    )
    ax1.axvline(best_gamma, color='red', linestyle='--', lw=1.2,
                label=f'best γ = {best_gamma}')
    ax1.set_xscale('symlog', linthresh=0.001)
    ax1.set_xlabel('γ (gamma)', fontsize=12)
    ax1.set_ylabel('L2 relative error (%)', fontsize=12)
    ax1.set_title(f'Reconstruction error vs. γ\n(k_max={k_max}, n_test={n_test})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Boxplot
    labels = [str(g) for g in gammas]
    data_box = [results[g] * 100 for g in gammas]
    ax2.boxplot(data_box, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_xlabel('γ (gamma)', fontsize=12)
    ax2.set_ylabel('L2 relative error (%)', fontsize=12)
    ax2.set_title('Distribution des erreurs par γ', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure sauvegardée → {save_plot}")

    return results


if __name__ == '__main__':
    main()
