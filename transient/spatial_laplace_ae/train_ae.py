"""
train_spatial_ae.py — Entraîne un SpatialLaplaceAE sur les séquences temporelles brutes.

Le modèle applique la transformée de Laplace directement sur les champs U(t)
(pixel par pixel), puis encode/décode chaque frame fréquentielle via un AE
conditionné sur freq_ratio. Les K points s_k sont apprenables.

Checkpoint : checkpoints/SpatialLaplaceAE_best.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as _Dataset, DataLoader
from tqdm import tqdm
import wandb

from models.transient.spatial_laplace_ae import SpatialLaplaceAE
from transient.dataset import TransientDataset
from transient.laplace_ae.laplace_opti import _log_s_scatter, _log_s_text


class _FrameDataset(_Dataset):
    def __init__(self, dataset: TransientDataset, indices: list[int], N: int,
                 U_mean: torch.Tensor, U_std: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
        self.N      = N
        self.U_mean = U_mean
        self.U_std  = U_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        theta_n, U = self.dataset[idx]
        if isinstance(U, np.ndarray):
            U = torch.from_numpy(U.copy()).float()
        if U.shape[-1] != self.N:
            U = F.interpolate(
                U.unsqueeze(0), size=(self.N, self.N),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
        U = (U - self.U_mean) / self.U_std
        return theta_n, U


def train_ae(
    dataset    : TransientDataset,
    train_idx  : list[int],
    val_idx    : list[int],
    test_idx   : list[int],
    N          : int,
    latent_dim : int,
    K          : int,
    epochs     : int,
    batch_size : int,
    lr         : float,
    beta       : float,
    beta_freq  : float,
    patience   : int,
    ckpt_dir   : str,
    project    : str,
    gamma_init : float = 0.0,
    freq_L     : int   = 8,
):
    Nt     = dataset.Nt
    dt     = dataset.dt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)

    U_mean, U_std = dataset.U_mean, dataset.U_std
    print(f"U_mean : [{U_mean.min():.3f}, {U_mean.max():.3f}]  "
          f"U_std  : [{U_std.min():.3f}, {U_std.max():.3f}]")

    train_ds = _FrameDataset(dataset, train_idx, N, U_mean, U_std)
    val_ds   = _FrameDataset(dataset, val_idx,   N, U_mean, U_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Dataset : train {len(train_ds)}  |  val {len(val_ds)}")

    model = SpatialLaplaceAE(
        N=N, Nt=Nt, K=K, latent_dim=latent_dim, dt=dt,
        beta=beta, beta_freq=beta_freq, gamma_init=gamma_init, freq_L=freq_L,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpatialLaplaceAE : {n_params:,} params  |  N={N}  Nt={Nt}  K={K}  latent={latent_dim}  device={device}")

    wandb.init(project=project, name='SpatialLaplaceAE', config=dict(
        N=N, Nt=Nt, K=K, latent_dim=latent_dim, dt=dt,
        beta=beta, beta_freq=beta_freq, gamma_init=gamma_init, freq_L=freq_L,
        epochs=epochs, batch_size=batch_size, lr=lr,
        n_train=len(train_ds), n_params=n_params, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=50)

    s_init = model.laplace.s_list.detach().cpu().clone()

    best_val    = float('inf')
    best_state  = None
    patience_   = 0
    ckpt_path   = os.path.join(ckpt_dir, 'SpatialLaplaceAE_best.pt')
    global_step = 0
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    epoch_bar = tqdm(range(1, epochs + 1), desc='SpatialLaplaceAE', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        tr_loss = tr_recon = tr_freq = tr_ridge = tr_l2 = 0.0
        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>4}', position=1, leave=False, unit='batch')
        for batch_idx, (_, U) in enumerate(train_bar):
            U = U.to(device, non_blocking=True)

            t_fwd = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                U_rec, U_hat, U_rec_hat, z = model(U)
                loss, metrics = model.loss(U, U_rec, U_hat, U_rec_hat, z)
            t_fwd = time.perf_counter() - t_fwd

            t_bwd = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(total_norm):
                epoch_bar.write(f"  [NaN/inf] epoch={epoch} batch={batch_idx} — rollback")
                model.load_state_dict(last_good_state)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            t_bwd = time.perf_counter() - t_bwd

            loss_v  = loss.item()
            recon_v = metrics['recon'].item()
            freq_v  = metrics['freq_rec'].item()
            ridge_v = metrics['ridge'].item()
            l2rel   = ((U_rec.detach() - U).flatten(1).norm(dim=1)
                       / (U.flatten(1).norm(dim=1) + 1e-8)).mean().item()

            tr_loss  += loss_v;  tr_recon += recon_v
            tr_freq  += freq_v;  tr_ridge += ridge_v;  tr_l2 += l2rel

            if batch_idx % 20 == 0:
                train_bar.set_postfix(loss=f"{loss_v:.3e}", recon=f"{recon_v:.3e}",
                                      freq=f"{freq_v:.3e}",
                                      fwd=f"{t_fwd*1e3:.0f}ms", bwd=f"{t_bwd*1e3:.0f}ms")
            wandb.log({
                'batch/loss': loss_v, 'batch/recon': recon_v,
                'batch/freq_rec': freq_v, 'batch/ridge': ridge_v,
            }, step=global_step)
            global_step += 1

        n = len(train_loader)
        tr_loss /= n;  tr_recon /= n;  tr_freq /= n;  tr_ridge /= n;  tr_l2 /= n

        # --- Val ---
        model.eval()
        vl_loss = vl_recon = vl_freq = vl_ridge = vl_l2 = 0.0
        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>4}', position=1, leave=False, unit='batch')
        with torch.no_grad():
            for _, U in val_bar:
                U = U.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    U_rec, U_hat, U_rec_hat, z = model(U)
                    loss, metrics = model.loss(U, U_rec, U_hat, U_rec_hat, z)
                vl_loss  += loss.item()
                vl_recon += metrics['recon'].item()
                vl_freq  += metrics['freq_rec'].item()
                vl_ridge += metrics['ridge'].item()
                vl_l2    += ((U_rec - U).flatten(1).norm(dim=1)
                             / (U.flatten(1).norm(dim=1) + 1e-8)).mean().item()

        n = len(val_loader)
        vl_loss /= n;  vl_recon /= n;  vl_freq /= n;  vl_ridge /= n;  vl_l2 /= n

        scheduler.step(vl_loss)
        epoch_bar.set_postfix(
            tr=f"{tr_loss:.3e}", vl=f"{vl_loss:.3e}",
            l2=f"{vl_l2:.2%}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best=f"{best_val:.3e}",
        )

        s_cur = model.laplace.s_list.detach().cpu()
        log = {
            'train/loss': tr_loss, 'train/recon': tr_recon,
            'train/freq_rec': tr_freq, 'train/ridge': tr_ridge, 'train/l2rel': tr_l2,
            'val/loss': vl_loss, 'val/recon': vl_recon,
            'val/freq_rec': vl_freq, 'val/ridge': vl_ridge, 'val/l2rel': vl_l2,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_s': time.perf_counter() - t0,
            'epoch': epoch,
            'params/alpha_t': model.laplace.log_alpha_t.exp().item(),
            'params/lam':     model.laplace.log_lam.exp().item(),
            's_points/text':  _log_s_text(s_cur),
        }
        if epoch % 5 == 0:
            log['s_points/scatter'] = _log_s_scatter(s_cur, s_init, epoch)
        wandb.log(log, step=global_step)

        if math.isfinite(vl_loss):
            last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_  = 0
            torch.save({
                'model_state' : best_state,
                'model_type'  : 'SpatialLaplaceAE',
                'N'           : N,
                'Nt'          : Nt,
                'K'           : K,
                'latent_dim'  : latent_dim,
                'dt'          : dt,
                'beta'        : beta,
                'beta_freq'   : beta_freq,
                'gamma_init'  : gamma_init,
                'freq_L'      : freq_L,
                'theta_mean'  : dataset.theta_mean,
                'theta_std'   : dataset.theta_std,
                'U_mean'      : U_mean,
                'U_std'       : U_std,
                'test_idx'    : test_idx,
                'val_loss'    : best_val,
            }, ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val={best_val:.4e})")
                break

    assert best_state is not None
    wandb.save(ckpt_path)
    wandb.finish()
    print(f"SpatialLaplaceAE entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    model.load_state_dict(best_state)
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    data_path  = os.path.join("dataset", "ch4_rotated.npy")
    ckpt_dir   = "checkpoints"
    N          = 128
    latent_dim = 64
    K          = 32
    epochs     = 200
    batch_size = 4   # réduit car la transformée inverse (B, Nt, N²) est mémoire-intensive
    lr         = 5e-4
    beta       = 1e-3
    beta_freq  = 1.0
    patience   = 30
    gamma_init = 0.0
    freq_L     = 8
    seed       = 42
    project    = 'convdiff'
    dt         = 1.0

    dataset = TransientDataset(data_path, laplace=False, dt=dt, interp_size=N)

    torch.backends.cudnn.benchmark = True
    _split    = np.load('dataset/split.npz')
    test_idx  = _split['test_idx'].tolist()
    non_test  = [i for i in range(len(dataset)) if i not in set(test_idx)]
    torch.manual_seed(seed)
    perm      = torch.randperm(len(non_test))
    n_train   = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)

    train_ae(
        dataset=dataset, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        N=N, latent_dim=latent_dim, K=K,
        epochs=epochs, batch_size=batch_size, lr=lr,
        beta=beta, beta_freq=beta_freq, patience=patience,
        gamma_init=gamma_init, freq_L=freq_L, ckpt_dir=ckpt_dir, project=project,
    )
