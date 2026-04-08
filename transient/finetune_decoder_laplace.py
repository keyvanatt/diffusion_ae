"""
finetune_decoder_laplace.py — Finetune end-to-end du LaplaceLatentModel assemblé.

Pipeline :
  1. train_ae_laplace.py    → LaplaceAE (autoencoder)
  2. train_laplace.py        → LaplaceLatentSurrogate par fréquence (θ→z, décodeur gelé)
  3. CE SCRIPT               → finetune end-to-end (surrogates + décodeur dégelé)

Le modèle assemblé (LaplaceLatentModel.pt) contient N_half surrogates (proj: θ→z)
et un shared_decoder gelé. On dégèle le décodeur et on finetune tout avec un faible LR,
en utilisant un LR différentiel (décodeur plus bas que les surrogates).

Astuce vitesse : les z sont calculés par fréquence (proj petit MLP), puis le décodeur
est appelé UNE SEULE FOIS sur le batch complet avec freq_ratio par sample.

Checkpoint : checkpoints/LaplaceLatentModel_finetuned.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as _Dataset
from tqdm import tqdm
import wandb

from models.laplace_ae_surrogate import LaplaceLatentModel
from transient.dataset import TransientDataset


class _FinetuneDataset(_Dataset):
    """
    Dataset aplati (simulation, fréquence) pour le finetuning end-to-end.
    Retourne (theta_norm, u_laplace_norm, freq_idx, freq_ratio).
    """
    def __init__(self, U_laplace, theta_norm, target_mean, target_std, indices, Nt_half):
        self.U_laplace   = U_laplace
        self.theta_norm  = theta_norm       # (ns, theta_dim)
        self.target_mean = target_mean      # (Nt_half, 2, 1, 1)
        self.target_std  = target_std
        self._indices    = [int(i) for i in indices]
        self._Nt_half    = Nt_half
        self._freq_ratio = [k / max(Nt_half - 1, 1) for k in range(Nt_half)]
        self.pairs       = self._make_pairs()

    def _make_pairs(self):
        """freq-first : toutes les sims pour k=0, puis k=1, etc.
        Chaque batch a (quasi) une seule fréquence → 1 proj + 1 decoder call."""
        return [(i, k) for k in range(self._Nt_half) for i in self._indices]

    def reshuffle(self):
        random.shuffle(self._indices)
        # Mélange l'ordre des fréquences aussi
        ks = list(range(self._Nt_half))
        random.shuffle(ks)
        self.pairs = [(i, k) for k in ks for i in self._indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sim_i, k = self.pairs[idx]
        th = self.theta_norm[sim_i]
        if isinstance(self.U_laplace, np.ndarray):
            u = torch.from_numpy(self.U_laplace[sim_i, k].copy()).float()
        else:
            u = self.U_laplace[sim_i, k].float()
        u_norm = (u - self.target_mean[k]) / self.target_std[k]
        return th, u_norm, torch.tensor(k, dtype=torch.long), torch.tensor(self._freq_ratio[k], dtype=torch.float32)


def _forward_batch(model, th, k_idx, freq_ratio, device):
    """
    Forward batché. Cas rapide (freq-first) : 1 seule fréquence par batch.
    Fallback : boucle sur les fréquences uniques, puis 1 appel décodeur.
    """
    B = th.shape[0]
    latent_dim = model.latent_dim

    unique_ks = k_idx.unique()
    if unique_ks.numel() == 1:
        # Fast path : tout le batch a la même fréquence
        z = model.surrogates[unique_ks.item()].proj(th)
        return model.shared_decoder(z, freq_ratio[0].item())

    # Slow path (batch à cheval sur 2 fréquences)
    amp_dtype = torch.float16 if (device.type == 'cuda' and torch.is_autocast_enabled()) else th.dtype
    z_all = torch.empty(B, latent_dim, device=device, dtype=amp_dtype)
    for k in unique_ks:
        mask = (k_idx == k)
        z_all[mask] = model.surrogates[k.item()].proj(th[mask])
    return model.shared_decoder(z_all, freq_ratio)


def finetune(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    ckpt_path,
    epochs,
    batch_size,
    lr_surrogate,
    lr_decoder,
    patience,
    save_dir,
    project,
):
    N         = dataset.N
    Nt        = dataset.Nt
    Nt_half   = dataset.Nt_half
    theta_dim = dataset.theta_dim
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # --- Charger le modèle assemblé ---
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = LaplaceLatentModel(
        N_freq=ckpt['N_freq'], N_half=ckpt['N_half'], N=ckpt['N'],
        theta_dim=ckpt['theta_dim'], latent_dim=ckpt['latent_dim'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Modèle chargé depuis {ckpt_path}")

    # Dégeler le shared_decoder pour le finetuning
    model.shared_decoder.requires_grad_(True)


    # --- Theta normalisé ---
    theta_norm = (dataset.theta - dataset.theta_mean) / dataset.theta_std

    # --- Datasets ---
    train_ds = _FinetuneDataset(dataset.U_laplace, theta_norm,
                                dataset.target_mean, dataset.target_std,
                                train_idx, Nt_half)
    val_ds   = _FinetuneDataset(dataset.U_laplace, theta_norm,
                                dataset.target_mean, dataset.target_std,
                                val_idx, Nt_half)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    print(f"Finetune Dataset : train {len(train_ds)} samples  |  val {len(val_ds)} samples")

    # --- Optimiseur avec LR différentiel ---
    surrogate_params = []
    for s in model.surrogates:
        surrogate_params.extend(s.proj.parameters())
    decoder_params = list(model.shared_decoder.parameters())

    optimizer = torch.optim.AdamW([
        {'params': surrogate_params, 'lr': lr_surrogate},
        {'params': decoder_params,   'lr': lr_decoder},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    n_surr   = sum(p.numel() for p in surrogate_params if p.requires_grad)
    n_dec    = sum(p.numel() for p in decoder_params if p.requires_grad)
    n_total  = n_surr + n_dec
    print(f"Params entraînés : {n_total:,} (surrogates {n_surr:,} + decoder {n_dec:,})")
    print(f"LR surrogate={lr_surrogate:.1e}  decoder={lr_decoder:.1e}  device={device}")

    # --- Wandb ---
    wandb.init(project=project, name='LaplaceLatentModel_finetune', config=dict(
        N=N, Nt=Nt, Nt_half=Nt_half, theta_dim=theta_dim,
        latent_dim=ckpt['latent_dim'],
        epochs=epochs, batch_size=batch_size,
        lr_surrogate=lr_surrogate, lr_decoder=lr_decoder,
        n_params_total=n_total, n_params_surrogates=n_surr, n_params_decoder=n_dec,
        n_samples_train=len(train_ds), n_samples_val=len(val_ds),
        source_ckpt=ckpt_path, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=100)

    best_val    = float('inf')
    best_state  = None
    patience_   = 0
    out_path    = os.path.join(save_dir, 'LaplaceLatentModel_finetuned.pt')
    global_step = 0
    log_every   = 20  # log wandb batch metrics toutes les N batches

    epoch_bar = tqdm(range(1, epochs + 1), desc='Finetune', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()
        train_ds.reshuffle()

        # --- Train ---
        model.train()
        train_loss  = 0.0
        train_l2rel = 0.0
        n_batches   = 0

        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>3}', position=1,
                         leave=False, unit='batch')
        for batch_idx, (th, u, k_idx, freq_ratio) in enumerate(train_bar):
            th         = th.to(device, non_blocking=True)
            u          = u.to(device, non_blocking=True)
            k_idx      = k_idx.to(device, non_blocking=True)
            freq_ratio = freq_ratio.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                u_hat = _forward_batch(model, th, k_idx, freq_ratio, device)
                loss  = F.mse_loss(u_hat.float(), u)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not torch.isfinite(total_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            with torch.no_grad():
                l2rel = ((u_hat.float() - u).flatten(1).norm(dim=1)
                         / (u.flatten(1).norm(dim=1) + 1e-8)).mean().item()

            train_loss  += loss_val
            train_l2rel += l2rel
            n_batches   += 1

            if batch_idx % 50 == 0:
                train_bar.set_postfix(loss=f"{loss_val:.3e}", l2=f"{l2rel:.2%}")

            if batch_idx % log_every == 0:
                wandb.log({
                    'batch/loss':      loss_val,
                    'batch/l2rel':     l2rel,
                    'batch/grad_norm': total_norm.item(),
                }, step=global_step)
            global_step += 1

        train_loss  /= max(n_batches, 1)
        train_l2rel /= max(n_batches, 1)

        # --- Val ---
        model.eval()
        val_loss  = 0.0
        val_l2rel = 0.0
        freq_losses = torch.zeros(Nt_half, device=device)
        freq_counts = torch.zeros(Nt_half, device=device)
        n_val_b = 0

        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>3}', position=1,
                       leave=False, unit='batch')
        with torch.no_grad():
            for th, u, k_idx, freq_ratio in val_bar:
                th         = th.to(device, non_blocking=True)
                u          = u.to(device, non_blocking=True)
                k_idx      = k_idx.to(device, non_blocking=True)
                freq_ratio = freq_ratio.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    u_hat = _forward_batch(model, th, k_idx, freq_ratio, device)
                    loss  = F.mse_loss(u_hat.float(), u)

                val_loss  += loss.item()
                val_l2rel += ((u_hat.float() - u).flatten(1).norm(dim=1)
                              / (u.flatten(1).norm(dim=1) + 1e-8)).mean().item()
                n_val_b += 1

                # Per-frequency loss
                for k in k_idx.unique():
                    mask = (k_idx == k)
                    freq_losses[k] += F.mse_loss(u_hat[mask].float(), u[mask]).item() * mask.sum()
                    freq_counts[k] += mask.sum()

        val_loss  /= max(n_val_b, 1)
        val_l2rel /= max(n_val_b, 1)

        scheduler.step(val_loss)
        epoch_time = time.perf_counter() - t0

        epoch_bar.set_postfix(
            tr=f"{train_loss:.3e}", vl=f"{val_loss:.3e}",
            l2=f"{val_l2rel:.2%}",
            lr_s=f"{optimizer.param_groups[0]['lr']:.1e}",
            lr_d=f"{optimizer.param_groups[1]['lr']:.1e}",
            best=f"{best_val:.3e}",
        )

        # --- Wandb epoch log ---
        log_dict = {
            'train/loss':    train_loss,
            'train/l2rel':   train_l2rel,
            'val/loss':      val_loss,
            'val/l2rel':     val_l2rel,
            'lr/surrogate':  optimizer.param_groups[0]['lr'],
            'lr/decoder':    optimizer.param_groups[1]['lr'],
            'epoch_time_s':  epoch_time,
            'epoch':         epoch,
        }
        # Per-frequency val loss (every 5 epochs)
        if epoch % 5 == 0 or epoch == 1:
            mask_nz = freq_counts > 0
            freq_avg = freq_losses.clone()
            freq_avg[mask_nz] /= freq_counts[mask_nz]
            for k in range(Nt_half):
                if freq_counts[k] > 0:
                    log_dict[f'val_freq/loss_{k:03d}'] = freq_avg[k].item()
        wandb.log(log_dict, step=global_step)

        # --- Early stopping ---
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            patience_  = 0
            torch.save({
                'model_state': best_state,
                'model_type':  'LaplaceLatentModel',
                'N_freq':      Nt,
                'N_half':      Nt_half,
                'N':           N,
                'theta_dim':   theta_dim,
                'latent_dim':  ckpt['latent_dim'],
                'dt':          dataset.dt,
                'gamma':       ckpt.get('gamma', 0.0),
                'theta_mean':  dataset.theta_mean,
                'theta_std':   dataset.theta_std,
                'test_idx':    np.asarray(test_idx),
                'finetuned':   True,
            }, out_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val={best_val:.4e})")
                break

    assert best_state is not None, "Aucune epoch n'a amélioré la val loss"
    wandb.save(out_path)
    wandb.finish()
    print(f"Finetune terminé — best val : {best_val:.4e}  → {out_path}")

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    data_path    = os.path.join("dataset", "ch4_rotated.npy")
    model_ckpt   = os.path.join("checkpoints", "LaplaceLatentModel.pt")
    save_dir     = "checkpoints"
    seed         = 42
    gamma        = 0.0
    rule         = 'trap'
    interp_size  = 128
    dt           = 1.0

    # Hyperparamètres finetuning
    epochs       = 50
    batch_size   = 512
    lr_surrogate = 5e-5
    lr_decoder   = 1e-5
    patience     = 20
    project      = 'convdiff'

    # --- Dataset ---
    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule,
                               interp_size=interp_size, dt=dt)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    idx       = torch.randperm(len(dataset))
    n_train   = int(0.8 * len(dataset))
    n_val     = int(0.1 * len(dataset))
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].numpy()

    dataset.fit(train_idx)
    print("Chargement en RAM...", end=' ', flush=True)
    t0 = time.perf_counter()
    U = np.ascontiguousarray(dataset.U_laplace)
    dataset.U_laplace = U
    print(f"OK — {U.nbytes/1e9:.1f} Go RAM, {time.perf_counter()-t0:.1f}s")

    # --- Finetune ---
    print("\n=== Finetune LaplaceLatentModel (end-to-end) ===")
    finetune(dataset, train_idx, val_idx, test_idx,
             ckpt_path=model_ckpt,
             epochs=epochs, batch_size=batch_size,
             lr_surrogate=lr_surrogate, lr_decoder=lr_decoder,
             patience=patience, save_dir=save_dir, project=project)
