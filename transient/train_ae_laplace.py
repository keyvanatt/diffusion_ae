"""
train_ae_laplace.py — Entraîne un LaplaceVAE sur tous les champs Laplace (toutes fréquences).

=========================================================================

Checkpoint :
  checkpoints/LaplaceVAE_best.pt 

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from torch.utils.data import Dataset as _Dataset

from models.laplace_ae_surrogate import LaplaceVAE, LaplaceLatentSurrogate, LaplaceLatentModel
from transient.dataset import TransientDataset


class _LaplaceFlatDataset(_Dataset):
    """
    Dataset lazy indexé sur (simulation, fréquence).
    Lit depuis le memmap disque — ~(2, N, N) par accès, pas d'OOM.

    Ordre sim-first : pour chaque sim, toutes les fréquences en séquence.
    Cela rend les accès memmap quasi-séquentiels (une sim = un bloc contigu).
    Appeler reshuffle() au début de chaque epoch pour mélanger l'ordre des sims.
    """
    def __init__(self, U_laplace, target_mean, target_std, indices, Nt_half):
        self.U_laplace   = U_laplace
        self.target_mean = target_mean   # (Nt_half, 2, 1, 1)
        self.target_std  = target_std
        self._indices    = [int(i) for i in indices]
        self._Nt_half    = Nt_half
        self.pairs       = self._make_pairs()

    def _make_pairs(self):
        """sim-first : (sim0, k0), (sim0, k1), …, (sim1, k0), …"""
        return [(i, k) for i in self._indices for k in range(self._Nt_half)]

    def reshuffle(self):
        """Mélange l'ordre des sims et reconstruit pairs. Appeler avant chaque epoch."""
        random.shuffle(self._indices)
        self.pairs = self._make_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sim_i, k = self.pairs[idx]
        if isinstance(self.U_laplace, np.ndarray):
            u = torch.from_numpy(self.U_laplace[sim_i, k].copy()).float()  # (2, N, N)
        else:
            u = self.U_laplace[sim_i, k].float()
        return ((u - self.target_mean[k]) / self.target_std[k],)


def train_vae(
    dataset,
    train_idx,
    val_idx,
    latent_dim,
    epochs,
    batch_size,
    lr,
    beta,
    patience,
    ckpt_dir,
    project,
    free_bits,
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

    # Dataset lazy : lit (sim, freq) depuis le memmap sans tout charger en RAM
    train_ds = _LaplaceFlatDataset(dataset.U_laplace, dataset.target_mean, dataset.target_std,
                                   train_idx, Nt_half)
    val_ds   = _LaplaceFlatDataset(dataset.U_laplace, dataset.target_mean, dataset.target_std,
                                   val_idx,   Nt_half)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    print("Laplace Dataset : train %d samples  |  val %d samples" % (len(train_ds), len(val_ds)))
    model     = LaplaceVAE(N=N, latent_dim=latent_dim, beta=beta, free_bits=free_bits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LaplaceVAE : {n_params:,} paramètres  |  device={device}")

    wandb.init(project=project, name='LaplaceVAE', config=dict(
        N=N, latent_dim=latent_dim, beta=beta, free_bits=free_bits,
        epochs=epochs, batch_size=batch_size, lr=lr,
        n_samples_train=len(train_ds),
        n_params=n_params, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=50)

    best_val  = float('inf')
    patience_ = 0
    ckpt_path = os.path.join(ckpt_dir, 'LaplaceVAE_best.pt')

    epoch_bar = tqdm(range(1, epochs + 1), desc='LaplaceVAE', position=0, leave=True)
    for epoch in epoch_bar:
        t0 = time.perf_counter()
        train_ds.reshuffle()   # mélange sims, garde accès memmap séquentiels

        # --- Train ---
        model.train()
        train_elbo = train_recon = train_kl = train_l2rel = 0.
        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>4}', position=1,
                         leave=False, unit='batch')
        for (u,) in train_bar:
            u = u.to(device)
            u_hat, mu, logvar = model(u)
            loss, metrics = model.loss(u, u_hat, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            elbo_  = loss.item()
            recon_ = metrics['recon'].item()
            kl_    = metrics['kl'].item()
            train_elbo  += elbo_
            train_recon += recon_
            train_kl    += kl_
            train_l2rel += ((u_hat - u).flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-8)).mean().item()
            wandb.log({'batch/elbo': elbo_, 'batch/recon': recon_, 'batch/kl': kl_})
            train_bar.set_postfix(elbo=f"{elbo_:.3e}")
        n = len(train_loader)
        train_elbo /= n; train_recon /= n; train_kl /= n; train_l2rel /= n

        # --- Val ---
        model.eval()
        val_elbo = val_recon = val_kl = val_l2rel = 0.
        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>4}', position=1,
                       leave=False, unit='batch')
        with torch.no_grad():
            for (u,) in val_bar:
                u = u.to(device)
                u_hat, mu, logvar = model(u)
                loss, metrics = model.loss(u, u_hat, mu, logvar)
                val_elbo   += loss.item()
                val_recon  += metrics['recon'].item()
                val_kl     += metrics['kl'].item()
                val_l2rel  += ((u_hat - u).flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-8)).mean().item()
                val_bar.set_postfix(elbo=f"{loss.item():.3e}")
        n = len(val_loader)
        val_elbo /= n; val_recon /= n; val_kl /= n; val_l2rel /= n

        scheduler.step(val_elbo)
        epoch_bar.set_postfix(
            tr_elbo=f"{train_elbo:.3e}", vl_elbo=f"{val_elbo:.3e}",
            l2=f"{val_l2rel:.2%}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best=f"{best_val:.3e}",
        )
        wandb.log({
            'train/elbo'   : train_elbo,
            'train/recon'  : train_recon,
            'train/kl'     : train_kl,
            'train/l2rel'  : train_l2rel,
            'val/elbo'     : val_elbo,
            'val/recon'    : val_recon,
            'val/kl'       : val_kl,
            'val/l2rel'    : val_l2rel,
            'lr'           : optimizer.param_groups[0]['lr'],
            'epoch_time_s' : time.perf_counter() - t0,
            'epoch'         : epoch,
        })

        if val_elbo < best_val:
            best_val  = val_elbo
            patience_ = 0
            torch.save({'model_state': model.state_dict(),
                        'N': N, 'latent_dim': latent_dim, 'val_loss': best_val}, ckpt_path)
            wandb.save(ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val={best_val:.4e})")
                break

    wandb.finish()
    print(f"VAE entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    # Recharger le meilleur
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model


def main(
    data_path   = os.path.join("dataset", "ch4_rotated.npy"),
    ckpt_dir    = os.path.join("checkpoints", "laplace_vae"),
    latent_dim  = 64,
    seed        = 42,
    gamma       = 0.0,
    rule        = 'trap',
    epochs      = 100,
    batch_size  = 76*32, # un multiple de 76 = Nt_half pour éviter les batches partiels
    lr          = 5e-4,
    beta        = 0.01,
    patience    = 30,
    free_bits   = 1.00,
    project     = 'convdiff',
    interp_size = 128,    # résolution spatiale cible (multiple de 32 requis par le VAE)
    dt          = 1.0,
):

    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule,
                               interp_size=interp_size, dt=dt)

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
                    lr=lr, beta=beta, patience=patience, free_bits=free_bits,
                    ckpt_dir=ckpt_dir,project=project)

if __name__ == "__main__":
    main()