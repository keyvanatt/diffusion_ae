"""
scripts/gamma_opti_hist.py — Histogramme L2rel (gamma=0, k_max fixé)
=====================================================================
Calcule les erreurs L2rel sur 100 samples de test : reconstruction Laplace
tronquée à k_max, hautes fréquences remplacées par la moyenne du train.

Usage :
    .conda/bin/python scripts/gamma_opti_hist.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.laplace import laplace_forward, laplace_inverse


def compute_train_mean(U_raw, train_idx, gamma, dt=1.0, rule='trap'):
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N = H * W
    mean_k_pos = np.zeros((Nt_half, N), dtype=np.complex64)
    for i in tqdm(train_idx, desc="train mean", leave=False):
        U_i = U_raw[i].copy().astype(np.float32)
        C_i = U_i.reshape(Nt, N).T
        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)
        mean_k_pos += M[:, :Nt_half].T.astype(np.complex64)
    mean_k_pos /= len(train_idx)
    return mean_k_pos


def compute_l2rel(U_raw, gamma, k_max, mean_k_pos, test_idx, dt=1.0, rule='trap'):
    _, Nt, H, W = U_raw.shape
    Nt_half = Nt // 2 + 1
    N = H * W
    n_tail = Nt - Nt_half

    l2rel_list = []
    for i in tqdm(test_idx, desc="test l2rel", leave=False):
        U_true = U_raw[i].copy().astype(np.float32)
        C_i = U_true.reshape(Nt, N).T

        M, _, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)

        M_pos = M[:, :Nt_half].copy().astype(np.complex64)
        if k_max + 1 < Nt_half:
            M_pos[:, k_max + 1:] = mean_k_pos[k_max + 1:, :].T

        M_full = np.zeros((N, Nt), dtype=np.complex64)
        M_full[:, :Nt_half] = M_pos
        if n_tail > 0:
            M_full[:, Nt_half:] = np.conj(M_pos[:, n_tail:0:-1])

        C_rec, _ = laplace_inverse(M_full, dt, Nt, rule=rule, gamma=gamma)
        U_pred = C_rec.T.reshape(Nt, H, W)

        err = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
        l2rel_list.append(float(err))

    return np.array(l2rel_list, dtype=np.float32)


if __name__ == '__main__':
    data_path = '/Data/KAT/ch4_rotated.npy'
    gamma     = 0.0
    k_max     = 19
    dt        = 1.0
    rule      = 'trap'
    n_train   = 200
    n_test    = 100
    seed      = 42
    save_plot = 'plots/gamma_opti_hist.png'

    rng   = np.random.default_rng(seed)
    U_raw = np.load(data_path, mmap_mode='r')
    ns    = U_raw.shape[0]

    idx       = rng.permutation(ns)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:n_train + n_test]

    print(f"Dataset : {U_raw.shape}  |  n_train={n_train}  n_test={n_test}")
    print(f"gamma={gamma}  k_max={k_max}")

    mean_k_pos = compute_train_mean(U_raw, train_idx, gamma, dt=dt, rule=rule)
    l2rel = compute_l2rel(U_raw, gamma, k_max, mean_k_pos, test_idx, dt=dt, rule=rule)

    print(f"mean={l2rel.mean()*100:.2f}%  median={np.median(l2rel)*100:.2f}%  std={l2rel.std()*100:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(l2rel * 100, bins=30, color='steelblue',
            label=f'γ=0   med={np.median(l2rel)*100:.1f}%')
    ax.set_xlabel('L2 relative error (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'L2rel distribution — γ=0, k_max={k_max}, n_test={n_test}', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(Path(save_plot).parent, exist_ok=True)
    fig.savefig(save_plot, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée → {save_plot}")
    plt.close(fig)
