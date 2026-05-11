"""
Calcule et met en cache la transformée de Laplace (N=128) + les stats de
normalisation du train set, en reproduisant exactement le split de train_ae_laplace.py.

Lance : .conda/bin/python transient/compute_laplace_stats.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from transient.dataset import TransientDataset

data_path   = os.path.join("dataset", "ch4_rotated.npy")
interp_size = 128
dt          = 1.0
rule        = 'trap'
seed        = 42
k_max       = 20   # nombre de fréquences Laplace

Nt_data = np.load(data_path, mmap_mode='r').shape[1]
s_list  = (1j * 2 * np.pi * np.fft.rfftfreq(Nt_data, d=dt))[:k_max]

print("Chargement du dataset + transformée de Laplace (N=128)…")
print("(premier lancement : ~20 min pour le cache, ensuite instantané)")
dataset = TransientDataset(data_path, laplace=True, s_list=s_list, rule=rule,
                           interp_size=interp_size, dt=dt)

torch.manual_seed(seed)
idx     = torch.randperm(len(dataset))
n_train = int(0.8 * len(dataset))
train_idx = idx[:n_train].tolist()

print(f"Calcul des stats de normalisation sur {n_train} simulations…")
dataset.fit(train_idx)   # met en cache automatiquement dans /Data/KAT/

print("Stats calculées et mises en cache.")
print(f"  target_mean : {dataset.target_mean.shape}")
print(f"  target_std  : {dataset.target_std.shape}")
