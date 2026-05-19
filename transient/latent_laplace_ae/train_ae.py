"""
train_ae.py — Entraîne un LatentLaplaceAE sur les séquences temporelles brutes.

Le modèle encode chaque frame individuellement, applique la transformée de
Laplace dans l'espace latent (fréquences s_k apprenables), reconstruit via
inversion régularisée, et décode.

Checkpoint : checkpoints/LatentLaplaceAE_best.pt
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

from models.transient.latent_laplace_ae import LatentLaplaceAE
from transient.dataset import TransientDataset
from transient.laplace_ae.laplace_opti import _log_s_scatter, _log_s_text


class _FrameDataset(_Dataset):
    """
    Wrapper sur TransientDataset (laplace=False).
    Charge U depuis le mmap, interpole à N×N, normalise avec les stats du train set.
    Retourne (theta_norm, U_norm) avec U_norm : (Nt, N, N) float32.
    """
    def __init__(self, dataset: TransientDataset, indices: list[int], N: int,
                 U_mean: torch.Tensor, U_std: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
        self.N      = N
        self.U_mean = U_mean   # (N, N)
        self.U_std  = U_std    # (N, N)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        theta_n, U = self.dataset[idx]         # U : (Nt, H, W)
        if isinstance(U, np.ndarray):
            U = torch.from_numpy(U.copy()).float()
        if U.shape[-1] != self.N:
            U = F.interpolate(
                U.unsqueeze(0), size=(self.N, self.N),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
        U = (U - self.U_mean) / self.U_std     # normalisation z-score pixel-par-pixel
        return theta_n, U                      # (Nt, N, N)


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
    beta_latent: float,
    patience   : int,
    ckpt_dir   : str,
    project    : str,
    gamma_init : float = 0.0,
    time_L     : int   = 8,
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

    model = LatentLaplaceAE(
        N=N, Nt=Nt, latent_dim=latent_dim, K=K, dt=dt,
        beta=beta, beta_latent=beta_latent, gamma_init=gamma_init, time_L=time_L,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LatentLaplaceAE : {n_params:,} params  |  N={N}  Nt={Nt}  latent={latent_dim}  K={K}  device={device}")

    wandb.init(project=project, name='LatentLaplaceAE', config=dict(
        N=N, Nt=Nt, latent_dim=latent_dim, K=K, dt=dt,
        beta=beta, beta_latent=beta_latent, gamma_init=gamma_init, time_L=time_L,
        epochs=epochs, batch_size=batch_size, lr=lr,
        n_train=len(train_ds), n_params=n_params, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=50)

    s_init = model.laplace.s_list.detach().cpu().clone()

    best_val    = float('inf')
    best_state  = None
    patience_   = 0
    ckpt_path   = os.path.join(ckpt_dir, 'LatentLaplaceAE_best.pt')
    global_step = 0
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    epoch_bar = tqdm(range(1, epochs + 1), desc='LatentLaplaceAE', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        tr_loss = 0.0
        tr_recon = 0.0
        tr_lat = 0.0
        tr_ridge = 0.0
        tr_l2 = 0.0
        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>4}', position=1, leave=False, unit='batch')
        for batch_idx, (_, U) in enumerate(train_bar):
            U = U.to(device, non_blocking=True)

            t_fwd = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                U_rec, z_hat, z, z_rec   = model(U)
                loss, metrics     = model.loss(U, U_rec, z_hat, z, z_rec)
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
            lat_v   = metrics['lat_rec'].item()
            ridge_v = metrics['ridge'].item()
            l2rel   = ((U_rec.detach() - U).flatten(1).norm(dim=1)
                       / (U.flatten(1).norm(dim=1) + 1e-8)).mean().item()

            tr_loss  += loss_v;  tr_recon += recon_v
            tr_lat   += lat_v;   tr_ridge += ridge_v;  tr_l2 += l2rel

            if batch_idx % 20 == 0:
                train_bar.set_postfix(loss=f"{loss_v:.3e}", recon=f"{recon_v:.3e}",
                                      lat=f"{lat_v:.3e}",
                                      fwd=f"{t_fwd*1e3:.0f}ms", bwd=f"{t_bwd*1e3:.0f}ms")
            wandb.log({
                'batch/loss': loss_v, 'batch/recon': recon_v,
                'batch/lat_rec': lat_v, 'batch/ridge': ridge_v,
            }, step=global_step)
            global_step += 1

        n = len(train_loader)
        tr_loss /= n;  tr_recon /= n;  tr_lat /= n;  tr_ridge /= n;  tr_l2 /= n

        # --- Val ---
        model.eval()
        vl_loss = vl_recon = vl_lat = vl_ridge = vl_l2 = 0.0
        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>4}', position=1, leave=False, unit='batch')
        with torch.no_grad():
            for _, U in val_bar:
                U = U.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    U_rec, z_hat, z, z_rec = model(U)
                    loss, metrics   = model.loss(U, U_rec, z_hat, z, z_rec)
                vl_loss  += loss.item()
                vl_recon += metrics['recon'].item()
                vl_lat   += metrics['lat_rec'].item()
                vl_ridge += metrics['ridge'].item()
                vl_l2    += ((U_rec - U).flatten(1).norm(dim=1)
                             / (U.flatten(1).norm(dim=1) + 1e-8)).mean().item()

        n = len(val_loader)
        vl_loss /= n;  vl_recon /= n;  vl_lat /= n;  vl_ridge /= n;  vl_l2 /= n

        scheduler.step(vl_loss)
        epoch_bar.set_postfix(
            tr=f"{tr_loss:.3e}", vl=f"{vl_loss:.3e}",
            l2=f"{vl_l2:.2%}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best=f"{best_val:.3e}",
        )
        s_cur = model.laplace.s_list.detach().cpu()
        log = {
            'train/loss': tr_loss, 'train/recon': tr_recon,
            'train/lat_rec': tr_lat, 'train/ridge': tr_ridge, 'train/l2rel': tr_l2,
            'val/loss': vl_loss, 'val/recon': vl_recon,
            'val/lat_rec': vl_lat, 'val/ridge': vl_ridge, 'val/l2rel': vl_l2,
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
                'model_type'  : 'LatentLaplaceAE',
                'N'           : N,
                'Nt'          : Nt,
                'latent_dim'  : latent_dim,
                'K'           : K,
                'dt'          : dt,
                'beta'        : beta,
                'beta_latent' : beta_latent,
                'gamma_init'  : gamma_init,
                'time_L'      : time_L,
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
    print(f"LatentLaplaceAE entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    model.load_state_dict(best_state)
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    data_path  = os.path.join("dataset", "ch4_rotated.npy")
    ckpt_dir   = "checkpoints"
    N          = 128
    latent_dim = 64
    K          = 20
    epochs     = 200
    batch_size = 16
    lr         = 5e-4
    beta       = 1e-2
    beta_latent = 5
    patience   = 30
    gamma_init = 0.0
    time_L     = 8
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
        beta=beta, beta_latent=beta_latent, patience=patience,
        gamma_init=gamma_init, time_L=time_L, ckpt_dir=ckpt_dir, project=project,
    )
