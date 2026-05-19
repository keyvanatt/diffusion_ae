"""
train_ae.py — Entraîne un LatentLaplaceAE ou SpatialLaplaceAE.

Sélectionner AE_TYPE dans le bloc __main__.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from transient.spatial_and_latent_laplace.train import train
from transient.dataset import TransientDataset


if __name__ == "__main__":
    AE_TYPE = 'latent'   # 'latent'  |  'spatial'

    data_path = os.path.join("dataset", "ch4_rotated.npy")
    ckpt_dir  = "checkpoints"
    project   = "convdiff"
    seed      = 42
    dt        = 1.0
    N         = 128

    cfg = dict(
        N          = N,
        epochs     = 300,
        patience   = 40,
        latent_dim = 64,
        K          = 16,
        dt         = dt,
        beta       = 1e-2,
        gamma_init = 1e-2,   # gain initiale pour les fréquences dans Laplace
        alpha_t    = 7e-3,   # lissage temporel dans l'inversion de Laplace
        lam        = 3e-5,   # ridge dans l'inversion de Laplace
    )

    if AE_TYPE == 'latent':
        cfg.update(dict(
            beta_latent = 5.0,
            time_L      = 8,
            batch_size  = 4,
            lr          = 5e-4,
        ))
    else:
        cfg.update(dict(
            beta_freq  = 1.0,
            freq_L     = 8,
            batch_size = 4,
            lr         = 5e-4,
        ))

    torch.backends.cudnn.benchmark = True
    dataset    = TransientDataset(data_path, laplace=False, dt=dt, interp_size=N)
    cfg['Nt']  = dataset.Nt

    _split    = np.load('dataset/split.npz')
    test_idx  = _split['test_idx'].tolist()
    non_test  = [i for i in range(len(dataset)) if i not in set(test_idx)]
    torch.manual_seed(seed)
    perm      = torch.randperm(len(non_test))
    n_train   = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)

    train(
        ae_type=AE_TYPE, mode='ae',
        dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        cfg=cfg, ckpt_dir=ckpt_dir, project=project,
    )
