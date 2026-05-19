"""
train.py — Script d'entraînement unifié.

Couvre les quatre combinaisons :
  AE_TYPE ∈ {'latent', 'spatial'}
  MODE    ∈ {'ae', 'surrogate'}

Configurer AE_TYPE, MODE et les hyperparamètres dans le bloc __main__.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as _Dataset, DataLoader
from tqdm import tqdm
import wandb

from models.transient.latent_laplace_ae import LatentLaplaceAE
from models.transient.spatial_laplace_ae import SpatialLaplaceAE
from models.transient.freq_latent_surrogate import (
    LatentLaplaceSurrogateModel, SpatialLaplaceSurrogateModel,
)
from transient.dataset import TransientDataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _FrameDataset(_Dataset):
    def __init__(self, dataset: TransientDataset, indices: list[int], N: int,
                 U_mean: torch.Tensor, U_std: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
        self.N       = N
        self.U_mean  = U_mean
        self.U_std   = U_std

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
        return theta_n, (U - self.U_mean) / self.U_std


# ---------------------------------------------------------------------------
# Construction du modèle
# ---------------------------------------------------------------------------

def _build_ae(ae_type: str, cfg: dict, device: torch.device):
    """Construit et retourne (model, ckpt_keys) pour un AE vierge."""
    if ae_type == 'latent':
        model = LatentLaplaceAE(
            N=cfg['N'], Nt=cfg['Nt'], latent_dim=cfg['latent_dim'], K=cfg['K'],
            dt=cfg['dt'], beta=cfg['beta'], beta_latent=cfg['beta_latent'],
            gamma_init=cfg['gamma_init'], time_L=cfg['time_L'],
            alpha_t=cfg.get('alpha_t', math.exp(-2.0)), lam=cfg.get('lam', math.exp(-2.0)),
        ).to(device)
        ckpt_keys = {
            'model_type': 'LatentLaplaceAE',
            'N': cfg['N'], 'Nt': cfg['Nt'], 'latent_dim': cfg['latent_dim'],
            'K': cfg['K'], 'dt': cfg['dt'], 'beta': cfg['beta'],
            'beta_latent': cfg['beta_latent'], 'gamma_init': cfg['gamma_init'],
            'time_L': cfg['time_L'],
            'alpha_t': cfg.get('alpha_t', math.exp(-2.0)), 'lam': cfg.get('lam', math.exp(-2.0)),
        }
    else:
        model = SpatialLaplaceAE(
            N=cfg['N'], Nt=cfg['Nt'], K=cfg['K'], latent_dim=cfg['latent_dim'],
            dt=cfg['dt'], beta=cfg['beta'], beta_freq=cfg['beta_freq'],
            gamma_init=cfg['gamma_init'], freq_L=cfg['freq_L'],
            alpha_t=cfg.get('alpha_t', 1e-2), lam=cfg.get('lam', 1e-2),
        ).to(device)
        ckpt_keys = {
            'model_type': 'SpatialLaplaceAE',
            'N': cfg['N'], 'Nt': cfg['Nt'], 'K': cfg['K'], 'latent_dim': cfg['latent_dim'],
            'dt': cfg['dt'], 'beta': cfg['beta'], 'beta_freq': cfg['beta_freq'],
            'gamma_init': cfg['gamma_init'], 'freq_L': cfg['freq_L'],
            'alpha_t': cfg.get('alpha_t', 1e-2), 'lam': cfg.get('lam', 1e-2),
        }
    return model, ckpt_keys


def _load_ae(ae_type: str, ckpt_path: str, device: torch.device):
    """Charge un AE depuis un checkpoint, retourne (ae, ae_ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ae_type == 'latent':
        ae = LatentLaplaceAE(
            N=ckpt['N'], Nt=ckpt['Nt'], latent_dim=ckpt['latent_dim'], K=ckpt['K'],
            dt=ckpt['dt'], beta=ckpt.get('beta', 1e-2), beta_latent=ckpt.get('beta_latent', 5.0),
            gamma_init=ckpt.get('gamma_init', 0.0), time_L=ckpt.get('time_L', 8),
            alpha_t=ckpt.get('alpha_t', math.exp(-2.0)), lam=ckpt.get('lam', math.exp(-2.0)),
        ).to(device)
    else:
        ae = SpatialLaplaceAE(
            N=ckpt['N'], Nt=ckpt['Nt'], K=ckpt['K'], latent_dim=ckpt['latent_dim'],
            dt=ckpt['dt'], beta=ckpt.get('beta', 1e-3), beta_freq=ckpt.get('beta_freq', 1.0),
            gamma_init=ckpt.get('gamma_init', 0.0), freq_L=ckpt.get('freq_L', 8),
            alpha_t=ckpt.get('alpha_t', 1e-2), lam=ckpt.get('lam', 1e-2),
        ).to(device)
    ae.load_state_dict(ckpt['model_state'])
    ae.eval()
    return ae, ckpt


def _build_surrogate(ae_type: str, ae, ae_ckpt: dict, theta_dim: int,
                     cfg: dict, device: torch.device):
    """Construit et retourne (model, ckpt_keys) pour un surrogate."""
    sur_kwargs = dict(
        theta_dim=theta_dim, shared_dim=cfg['shared_dim'],
        head_dim=cfg['head_dim'], n_trunk=cfg['n_trunk'],
        n_head=cfg['n_head'], freq_L=cfg['freq_L'],
    )
    sur_ckpt = {
        'theta_dim': theta_dim,
        'shared_dim': cfg['shared_dim'], 'head_dim': cfg['head_dim'],
        'n_trunk': cfg['n_trunk'], 'n_head': cfg['n_head'], 'freq_L': cfg['freq_L'],
        'N': ae_ckpt['N'], 'Nt': ae_ckpt['Nt'], 'K': ae_ckpt['K'],
        'latent_dim': ae_ckpt['latent_dim'], 'dt': ae_ckpt['dt'],
    }
    if ae_type == 'latent':
        model = LatentLaplaceSurrogateModel(ae=ae, **sur_kwargs).to(device)
        sur_ckpt.update({'model_type': 'LatentLaplaceSurrogate',
                         'time_L': ae_ckpt.get('time_L', 8)})
    else:
        model = SpatialLaplaceSurrogateModel(ae=ae, latent_dim=ae_ckpt['latent_dim'],
                                              **sur_kwargs).to(device)
        sur_ckpt.update({'model_type': 'SpatialLaplaceSurrogate',
                         'freq_L_ae': ae_ckpt.get('freq_L', 8)})
    return model, sur_ckpt


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def _run_epoch(
    model, loader, is_train: bool, device: torch.device,
    mode: str, optimizer=None, scaler=None,
    alpha_lat: float = 1.0, alpha_spat: float = 1.0,
    epoch: int = 0, epoch_bar=None,
) -> tuple[float, dict, float]:
    """Retourne (loss_mean, metrics_mean, l2rel_mean)."""
    use_theta = mode == 'surrogate'
    total_loss = 0.0;  total_metrics = {};  total_l2 = 0.0

    desc = f"  {'Train' if is_train else 'Val  '} {epoch:>4}"
    bar  = tqdm(loader, desc=desc, position=1, leave=False, unit='batch')

    for batch_idx, (theta_norm, U) in enumerate(bar):
        theta_norm = theta_norm.to(device, non_blocking=True)
        U          = U.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            if use_theta:
                U_rec, *latents = model(theta_norm, U)
                loss, metrics   = model.loss(U, U_rec, *latents, alpha_lat, alpha_spat)
            else:
                outputs       = model(U)
                loss, metrics = model.loss(U, *outputs)
                U_rec         = outputs[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            trainable = [p for g in optimizer.param_groups for p in g['params']]
            norm      = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                if epoch_bar is not None:
                    epoch_bar.write(f"  [NaN/inf] epoch={epoch} batch={batch_idx} — skip")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

        loss_v = loss.item()
        l2rel  = ((U_rec.detach() - U).flatten(1).norm(dim=1)
                  / (U.flatten(1).norm(dim=1) + 1e-8)).mean().item()
        total_loss += loss_v;  total_l2 += l2rel
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

        if is_train and batch_idx % 20 == 0:
            bar.set_postfix(loss=f"{loss_v:.3e}",
                            **{k: f"{v.item():.3e}" for k, v in metrics.items()})

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}, total_l2 / n


# ---------------------------------------------------------------------------
# Entraînement principal
# ---------------------------------------------------------------------------

def train(
    ae_type    : str,
    mode       : str,
    dataset    : TransientDataset,
    train_idx  : list[int],
    val_idx    : list[int],
    test_idx   : list[int],
    cfg        : dict,
    ckpt_dir   : str,
    project    : str,
):
    """
    ae_type : 'latent' | 'spatial'
    mode    : 'ae' | 'surrogate'
    cfg     : dict de tous les hyperparamètres (varie selon ae_type × mode)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(ckpt_dir, exist_ok=True)

    N     = cfg['N']
    U_mean, U_std = dataset.U_mean, dataset.U_std
    theta_dim     = len(dataset.theta_mean)

    train_ds = _FrameDataset(dataset, train_idx, N, U_mean, U_std)
    val_ds   = _FrameDataset(dataset, val_idx,   N, U_mean, U_std)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Dataset : train {len(train_ds)}  |  val {len(val_ds)}")

    # --- Construction du modèle ---
    if mode == 'ae':
        model, ckpt_base = _build_ae(ae_type, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    else:
        ae, ae_ckpt = _load_ae(ae_type, cfg['ae_ckpt_path'], device)
        model, ckpt_base = _build_surrogate(ae_type, ae, ae_ckpt, theta_dim, cfg, device)
        del ae
        optimizer = torch.optim.AdamW([
            {'params': model.surrogate.parameters(), 'lr': cfg['lr_surrogate']},
            {'params': model.decoder.parameters(),   'lr': cfg['lr_decoder']},
        ], weight_decay=1e-4)
        s_init = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    run_name = f"{ae_type.capitalize()}Laplace{'AE' if mode == 'ae' else 'Surrogate'}"
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_name} — {n_params:,} params trainables  |  device: {device}")

    wandb.init(project=project, name=run_name, config={**cfg, 'ae_type': ae_type, 'mode': mode,
               'n_train': len(train_ds), 'n_params': n_params, 'device': str(device)})

    ckpt_name = f"{'LatentLaplace' if ae_type == 'latent' else 'SpatialLaplace'}"
    ckpt_name += 'AE' if mode == 'ae' else 'Surrogate'
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_best.pt")

    best_val        = float('inf')
    best_state      = None
    patience_left   = cfg['patience']
    global_step     = 0
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    epoch_bar = tqdm(range(1, cfg['epochs'] + 1), desc=run_name, position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        model.train()
        tr_loss, tr_metrics, tr_l2 = _run_epoch(
            model, train_loader, is_train=True, device=device, mode=mode,
            optimizer=optimizer, scaler=scaler,
            alpha_lat=cfg.get('alpha_lat', 1.0), alpha_spat=cfg.get('alpha_spat', 1.0),
            epoch=epoch, epoch_bar=epoch_bar,
        )

        # Rollback check : si le modèle a divergé pendant l'époque, restaurer
        with torch.no_grad():
            params_ok = all(torch.isfinite(p).all() for p in model.parameters())
        if not params_ok:
            epoch_bar.write(f"  [divergence] epoch={epoch} — rollback global")
            model.load_state_dict(last_good_state)
            continue

        model.eval()
        with torch.no_grad():
            vl_loss, vl_metrics, vl_l2 = _run_epoch(
                model, val_loader, is_train=False, device=device, mode=mode,
                alpha_lat=cfg.get('alpha_lat', 1.0), alpha_spat=cfg.get('alpha_spat', 1.0),
                epoch=epoch,
            )

        scheduler.step(vl_loss)

        # Postfix tqdm
        lr_info = (f"lr={optimizer.param_groups[0]['lr']:.1e}" if mode == 'ae'
                   else f"lr_s={optimizer.param_groups[0]['lr']:.1e} "
                        f"lr_d={optimizer.param_groups[1]['lr']:.1e}")
        epoch_bar.set_postfix(tr=f"{tr_loss:.3e}", vl=f"{vl_loss:.3e}",
                              l2=f"{vl_l2:.2%}", best=f"{best_val:.3e}")

        # Wandb log
        log = {
            'train/loss': tr_loss, 'train/l2rel': tr_l2,
            'val/loss':   vl_loss, 'val/l2rel':   vl_l2,
            **{f'train/{k}': v for k, v in tr_metrics.items()},
            **{f'val/{k}':   v for k, v in vl_metrics.items()},
            'epoch_time_s': time.perf_counter() - t0,
            'epoch': epoch,
        }
        if mode == 'ae':
            log['lr'] = optimizer.param_groups[0]['lr']
            log['s_points/text']    = model.laplace.log_text()
            log['s_points/scatter'] = model.laplace.log_scatter(epoch)
        else:
            log['lr/surrogate'] = optimizer.param_groups[0]['lr']
            log['lr/decoder']   = optimizer.param_groups[1]['lr']
        wandb.log(log, step=global_step)
        global_step += len(train_loader)

        if math.isfinite(vl_loss):
            last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if vl_loss < best_val:
            best_val     = vl_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg['patience']
            torch.save({
                **ckpt_base,
                'model_state': best_state,
                'theta_mean' : dataset.theta_mean,
                'theta_std'  : dataset.theta_std,
                'U_mean'     : U_mean,
                'U_std'      : U_std,
                'test_idx'   : test_idx,
                'val_loss'   : best_val,
            }, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val={best_val:.4e})")
                break

    assert best_state is not None
    wandb.save(ckpt_path)
    wandb.finish()
    print(f"{run_name} entraîné — best val : {best_val:.4e}  → {ckpt_path}")

    model.load_state_dict(best_state)
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return model

