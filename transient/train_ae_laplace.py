"""
train_ae_laplace.py — Entraîne un LaplaceVAE sur tous les champs Laplace (toutes fréquences).

=========================================================================

Checkpoint :
  checkpoints/LaplaceVAE_best.pt 

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
import wandb

from models.laplace_ae_surrogate import LaplaceVAE, LaplaceLatentSurrogate, LaplaceLatentModel
from transient.dataset import TransientDataset


def train_vae(
    dataset,
    train_idx,
    val_idx,
    latent_dim  = 64,
    epochs      = 300,
    batch_size  = 64,
    lr          = 1e-3,
    beta        = 1.0,
    patience    = 30,
    ckpt_dir    = 'checkpoints/laplace_vae',
    project     = 'convdiff',
):
    """
    Entraîne LaplaceVAE sur TOUS les champs Laplace (toutes fréquences confondues).
    Chaque paire (simulation n, fréquence k) est traitée comme un échantillon.
    """
    N         = dataset.N
    Nt_half   = dataset.Nt_half
    theta_dim = dataset.theta_dim
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Aplatir (ns, Nt_half, 2, N, N) → (ns*Nt_half, 2, N, N), normalisé par fréquence
    all_targets  = []
    all_freq     = []
    ns           = len(dataset)
    for k in range(Nt_half):
        tm = dataset.target_mean[k]
        ts = dataset.target_std[k]
        all_targets.append((dataset.U_laplace[:, k] - tm) / ts)          # (ns, 2, N, N)
        all_freq.append(torch.full((ns, 1), k / Nt_half))                 # (ns, 1)
    U_all    = torch.cat(all_targets, dim=0)   # (ns*Nt_half, 2, N, N)
    freq_all = torch.cat(all_freq,    dim=0).float()  # (ns*Nt_half, 1)

    train_idx_all = [i + k * ns for k in range(Nt_half) for i in train_idx]
    val_idx_all   = [i + k * ns for k in range(Nt_half) for i in val_idx]

    ds           = TensorDataset(U_all, freq_all)
    train_loader = DataLoader(Subset(ds, train_idx_all), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idx_all),   batch_size=batch_size)

    model     = LaplaceVAE(N=N, latent_dim=latent_dim, beta=beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LaplaceVAE : {n_params:,} paramètres  |  device={device}")

    wandb.init(project=project, name='LaplaceVAE', config=dict(
        N=N, latent_dim=latent_dim, beta=beta,
        epochs=epochs, batch_size=batch_size, lr=lr,
        n_samples_train=len(train_idx_all),
    ))

    best_val  = float('inf')
    patience_ = 0
    ckpt_path = os.path.join(ckpt_dir, 'LaplaceVAE_best.pt')

    for epoch in tqdm(range(1, epochs + 1), desc='LaplaceVAE'):
        model.train()
        train_loss = 0.
        for u, freq in train_loader:
            u, freq = u.to(device), freq.to(device)
            u_hat, mu, logvar = model(u, freq)
            loss, _ = model.loss(u, u_hat, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for u, freq in val_loader:
                u, freq = u.to(device), freq.to(device)
                u_hat, mu, logvar = model(u, freq)
                loss, _ = model.loss(u, u_hat, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        wandb.log({'vae/train': train_loss, 'vae/val': val_loss,
                   'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if val_loss < best_val:
            best_val  = val_loss
            patience_ = 0
            torch.save({'model_state': model.state_dict(),
                        'N': N, 'latent_dim': latent_dim, 'val_loss': best_val}, ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                print(f"Early stopping VAE à l'époque {epoch}")
                break

    wandb.finish()
    print(f"VAE entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    # Recharger le meilleur
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model


def main(
    data_path  = os.path.join("dataset", "dataset_transient.npz"),
    ckpt_dir   = os.path.join("checkpoints", "laplace_vae"),
    latent_dim = 64,
    seed       = 42,
    gamma = 0.0, 
    rule = 'trap',
    epochs      = 300,
    batch_size  = 64,
    lr          = 1e-3,
    beta        = 1.0,
    patience    = 30,
    project     = 'convdiff',
):
    
    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule)

    torch.manual_seed(seed)
    idx       = torch.randperm(len(dataset))
    n_train   = int(0.8 * len(dataset))
    n_val     = int(0.1 * len(dataset))
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].numpy()

    dataset.fit(train_idx)
    print(f"Dataset : {tuple(dataset.U_laplace.shape)}  (Nt_half={dataset.Nt_half}/{dataset.Nt})")

    os.makedirs(ckpt_dir, exist_ok=True)

    # Étape 1
    print("\n=== Training LaplaceVAE ===")
    vae = train_vae(dataset, train_idx, val_idx,
                    latent_dim=latent_dim,
                    epochs=epochs, batch_size=batch_size,
                    lr=lr, beta=beta, patience=patience,
                    ckpt_dir=ckpt_dir)
