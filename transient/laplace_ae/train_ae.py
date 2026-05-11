"""
train_ae_laplace.py — Entraîne un LaplaceAE sur tous les champs Laplace (toutes fréquences).

=========================================================================

Checkpoint :
  checkpoints/LaplaceAE_best.pt

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from torch.utils.data import Dataset as _Dataset

from models.laplace_ae_surrogate import LaplaceAE
from transient.dataset import TransientDataset


class _LaplaceFlatDataset(_Dataset):
    """
    Dataset lazy indexé sur (simulation, fréquence).
    Lit depuis le memmap disque — ~(2, N, N) par accès, pas d'OOM.

    Ordre sim-first : pour chaque sim, toutes les fréquences en séquence.
    Cela rend les accès memmap quasi-séquentiels (une sim = un bloc contigu).
    Appeler reshuffle() au début de chaque epoch pour mélanger l'ordre des sims.
    """
    def __init__(self, U_laplace, target_mean, target_std, indices, K, k_max=None):
        self.U_laplace   = U_laplace
        self.target_mean = target_mean   # (K, 2, N, N)
        self.target_std  = target_std
        self._indices    = [int(i) for i in indices]
        self._n_freqs    = (min(k_max, K - 1) + 1) if k_max is not None else K
        self._freq_ratio = [k / max(K - 1, 1) for k in range(self._n_freqs)]
        self.pairs       = self._make_pairs()

    def _make_pairs(self):
        """sim-first : (sim0, k0), (sim0, k1), …, (sim1, k0), …"""
        return [(i, k) for i in self._indices for k in range(self._n_freqs)]

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
        u_norm = (u - self.target_mean[k]) / self.target_std[k]
        return u_norm, torch.tensor(self._freq_ratio[k], dtype=torch.float32)


def train_ae(
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
    freq_L,
    k_max     = None,
):
    """
    Entraîne LaplaceAE sur les champs Laplace pour k <= k_max (toutes si k_max=None).
    Chaque paire (simulation n, fréquence k) est traitée comme un échantillon.
    """
    N         = dataset.N
    K         = dataset.K
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = _LaplaceFlatDataset(dataset.U_laplace, dataset.target_mean, dataset.target_std,
                                   train_idx, K, k_max=k_max)
    val_ds   = _LaplaceFlatDataset(dataset.U_laplace, dataset.target_mean, dataset.target_std,
                                   val_idx,   K, k_max=k_max)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    print("Laplace Dataset : train %d samples  |  val %d samples" % (len(train_ds), len(val_ds)))

    model     = LaplaceAE(N=N, latent_dim=latent_dim, beta=beta, freq_L=freq_L).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LaplaceAE : {n_params:,} paramètres  |  device={device}")

    wandb.init(project=project, name='LaplaceAE', config=dict(
        N=N, latent_dim=latent_dim, beta=beta, freq_L=freq_L,
        epochs=epochs, batch_size=batch_size, lr=lr,
        n_samples_train=len(train_ds), k_max=k_max,
        n_params=n_params, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=50)

    best_val    = float('inf')
    best_state  = None
    patience_   = 0
    ckpt_path   = os.path.join(ckpt_dir, 'LaplaceAE_best.pt')
    global_step = 0
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    epoch_bar = tqdm(range(1, epochs + 1), desc='LaplaceAE', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()
        train_ds.reshuffle()

        # --- Train ---
        model.train()
        train_loss  = torch.zeros(1, device=device)
        train_recon = torch.zeros(1, device=device)
        train_ridge = torch.zeros(1, device=device)
        train_l2rel = torch.zeros(1, device=device)
        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>4}', position=1,
                         leave=False, unit='batch')
        for batch_idx, (u, freq_ratio) in enumerate(train_bar):
            u          = u.to(device, non_blocking=True)
            freq_ratio = freq_ratio.to(device, non_blocking=True)

            # --- Forward pass ---
            t_forward = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                u_hat, z        = model(u, freq_ratio)
                loss, metrics   = model.loss(u, u_hat, z)
            t_forward = time.perf_counter() - t_forward

            # --- Backward pass ---
            t_backward = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(total_norm):
                epoch_bar.write(f"  [NaN/inf] epoch={epoch} batch={batch_idx} "
                                f"loss={loss.item():.3e} norm={total_norm.item():.3e} "
                                f"— rollback au dernier bon état")
                model.load_state_dict(last_good_state)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                t_backward = time.perf_counter() - t_backward
                continue
            scaler.step(optimizer)
            scaler.update()
            t_backward = time.perf_counter() - t_backward

            train_loss  += loss.detach()
            train_recon += metrics['recon_loss']
            train_ridge += metrics['ridge']
            train_l2rel += ((u_hat.detach() - u).flatten(1).norm(dim=1)
                    / (u.flatten(1).norm(dim=1) + 1e-8)).mean()

            if batch_idx % 50 == 0:
                loss_item = loss.item()
                recon_item = metrics['recon_loss'].item()
                ridge_item = metrics['ridge'].item()
                train_bar.set_postfix(loss=f"{loss_item:.3e}", recon=f"{recon_item:.3e}",
                            fwd_ms=f"{t_forward*1000:.1f}",
                            bwd_ms=f"{t_backward*1000:.1f}")
            wandb.log({
                'batch/loss' : loss_item,
                'batch/recon': recon_item,
                'batch/ridge': ridge_item,
                'batch/forward_time_ms': t_forward * 1000,
                'batch/backward_time_ms': t_backward * 1000,
            }, step=global_step)
            global_step += 1

        n = len(train_loader)
        train_loss  = train_loss.item()  / n
        train_recon = train_recon.item() / n
        train_ridge = train_ridge.item() / n
        train_l2rel = train_l2rel.item() / n

        # --- Val ---
        model.eval()
        val_loss   = torch.zeros(1, device=device)
        val_recon  = torch.zeros(1, device=device)
        val_ridge  = torch.zeros(1, device=device)
        val_l2rel  = torch.zeros(1, device=device)
        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>4}', position=1,
                       leave=False, unit='batch')
        with torch.no_grad():
            for u, freq_ratio in val_bar:
                u          = u.to(device)
                freq_ratio = freq_ratio.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    u_hat, z      = model(u, freq_ratio)
                    loss, metrics = model.loss(u, u_hat, z)
                val_loss  += loss
                val_recon += metrics['recon_loss'].item()
                val_ridge += metrics['ridge'].item()
                val_l2rel += ((u_hat - u).flatten(1).norm(dim=1)
                              / (u.flatten(1).norm(dim=1) + 1e-8)).mean()

        n = len(val_loader)
        val_loss  = val_loss.item()  / n
        val_recon = val_recon.item() / n
        val_ridge = val_ridge.item() / n
        val_l2rel = val_l2rel.item() / n

        scheduler.step(val_loss)
        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.3e}", vl_loss=f"{val_loss:.3e}",
            l2=f"{val_l2rel:.2%}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best=f"{best_val:.3e}",
        )
        wandb.log({
            'train/loss'   : train_loss,
            'train/recon'  : train_recon,
            'train/ridge'  : train_ridge,
            'train/l2rel'  : train_l2rel,
            'val/loss'     : val_loss,
            'val/recon'    : val_recon,
            'val/ridge'    : val_ridge,
            'val/l2rel'    : val_l2rel,
            'lr'           : optimizer.param_groups[0]['lr'],
            'epoch_time_s' : time.perf_counter() - t0,
            'epoch'        : epoch,
        }, step=global_step)

        # Mettre à jour last_good_state si la val loss est valide
        if math.isfinite(val_loss):
            last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_  = 0
            torch.save({'model_state': best_state,
                        'N': N, 'latent_dim': latent_dim, 'val_loss': best_val}, ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val={best_val:.4e})")
                break

    assert best_state is not None, "Aucune epoch n'a amélioré la val loss"
    wandb.save(ckpt_path)
    wandb.finish()
    print(f"AE entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    model.load_state_dict(best_state)
    return model


def main(
    data_path   = os.path.join("dataset", "ch4_rotated.npy"),
    ckpt_dir    = "checkpoints",
    latent_dim  = 64,
    seed        = 42,
    gamma       = 0.0,
    rule        = 'trap',
    epochs      = 100,
    batch_size  = 256,
    lr          = 5e-4,
    beta        = 1e-3,
    patience    = 30,
    freq_L      = 8,
    project     = 'convdiff',
    interp_size = 128,
    dt          = 1.0,
    k_max       = 20,
):

    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule,
                               interp_size=interp_size, dt=dt)

    torch.backends.cudnn.benchmark = True
    _split   = np.load('dataset/split.npz')
    test_idx = _split['test_idx']
    non_test = [i for i in range(len(dataset)) if i not in set(test_idx.tolist())]
    torch.manual_seed(seed)
    perm     = torch.randperm(len(non_test))
    n_train  = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)
    print("Chargement en RAM...", end=' ', flush=True)
    t0 = time.perf_counter()
    U = np.ascontiguousarray(dataset.U_laplace)  # force contigu
    dataset.U_laplace = U
    print(f"OK — {U.nbytes/1e9:.1f} Go RAM, {time.perf_counter()-t0:.1f}s")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n=== Training LaplaceAE ===")
    ae = train_ae(dataset, train_idx, val_idx,
                  latent_dim=latent_dim,
                  epochs=epochs, batch_size=batch_size,
                  lr=lr, beta=beta, patience=patience,
                  freq_L=freq_L, k_max=k_max,
                  ckpt_dir=ckpt_dir, project=project)


if __name__ == "__main__":
    main()