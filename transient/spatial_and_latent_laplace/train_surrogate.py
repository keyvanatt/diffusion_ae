"""
train_surrogate.py — Entraîne un LatentLaplaceSurrogate ou SpatialLaplaceSurrogate.

Requiert un AE pré-entraîné (checkpoints/<AE>_best.pt).
Sélectionner AE_TYPE dans le bloc __main__.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from transient.train import train
from transient.dataset import TransientDataset


if __name__ == "__main__":
    AE_TYPE = 'latent'   # 'latent'  |  'spatial'

    data_path = os.path.join("dataset", "ch4_rotated.npy")
    ckpt_dir  = "checkpoints"
    project   = "convdiff"
    seed      = 42
    dt        = 1.0
    N         = 128

    ae_ckpt_name = 'LatentLaplaceAE' if AE_TYPE == 'latent' else 'SpatialLaplaceAE'

    cfg = dict(
        N            = N,
        epochs       = 300,
        patience     = 40,
        batch_size   = 16 if AE_TYPE == 'latent' else 4,
        ae_ckpt_path = os.path.join(ckpt_dir, f"{ae_ckpt_name}_best.pt"),
        lr_surrogate = 3e-4,
        lr_decoder   = 5e-5,
        alpha_lat    = 1.0,
        alpha_spat   = 1.0,
        shared_dim   = 256,
        head_dim     = 128,
        n_trunk      = 4,
        n_head       = 2,
        freq_L       = 6,
    )

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
        ae_type=AE_TYPE, mode='surrogate',
        dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        cfg=cfg, ckpt_dir=ckpt_dir, project=project,
    )
