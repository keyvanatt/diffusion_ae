"""
train_correction_ae.py — Entraîne un CorrectionAE frame-par-frame.

Pré-calcul (une seule fois) :
  precompute() génère deux memmaps (ns, kt, N, N) :
    - U_pred : prédictions du surrogate sur kt instants aléatoires par simulation
    - U_true : vérité terrain correspondante (interpolée à N×N)
  Ces fichiers sont mis en cache dans cache_dir.

Entraînement :
  Le dataset lit directement depuis les memmaps → pas de generate() pendant le training.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as _Dataset
from tqdm import tqdm
import wandb

from transient.dataset import TransientDataset
from transient.main import load_model
from models.correction_ae import CorrectionAE


# Pré-calcul : (ns, kt, N, N) × 2 memmaps
def precompute(dataset, surrogate_ckpt, kt, cache_dir, batch_size=32,
               dt=1.0, gamma=0.0, rule='trap', seed=0):
    """
    Génère ou charge les memmaps U_pred et U_true de shape (ns, kt, N, N).

    kt instants t tirés aléatoirement par simulation (reproductible via seed).
    Retourne (U_pred_mmap, U_true_mmap) en lecture seule.
    """
    ns  = dataset.ns
    Nt  = dataset.Nt
    N   = dataset.N
    os.makedirs(cache_dir, exist_ok=True)

    surr_name = os.path.splitext(os.path.basename(surrogate_ckpt))[0]
    pred_path = os.path.join(cache_dir, f"correction_upred_{surr_name}_kt{kt}_N{N}_s{seed}.npy")
    true_path = os.path.join(cache_dir, f"correction_utrue_kt{kt}_N{N}_s{seed}.npy")

    if os.path.exists(pred_path) and os.path.exists(true_path):
        print(f"Cache trouvé :\n  {pred_path}\n  {true_path}")
        return (np.load(pred_path, mmap_mode='r'),
                np.load(true_path, mmap_mode='r'))

    print(f"Pré-calcul U_pred/U_true → {cache_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surrogate, surr_ckpt_data = load_model(surrogate_ckpt, device)
    surrogate.eval()
    theta_mean = surr_ckpt_data['theta_mean'].to(device)
    theta_std  = surr_ckpt_data['theta_std'].to(device)

    rng   = np.random.default_rng(seed)
    t_idx = rng.integers(0, Nt, size=(ns, kt))   # (ns, kt) — instants fixes par sim

    pred_mmap = np.lib.format.open_memmap(pred_path, mode='w+', dtype=np.float32, shape=(ns, kt, N, N))
    true_mmap = np.lib.format.open_memmap(true_path, mode='w+', dtype=np.float32, shape=(ns, kt, N, N))

    for start in tqdm(range(0, ns, batch_size), desc='Pré-calcul'):
        end = min(start + batch_size, ns)
        B   = end - start

        u_raw = torch.from_numpy(dataset._U_raw[start:end].copy()).float()  # (B, Nt, H, W)
        if u_raw.shape[-1] != N:
            u_raw = F.interpolate(
                u_raw.reshape(B * Nt, 1, u_raw.shape[-2], u_raw.shape[-1]),
                size=(N, N), mode='bilinear', align_corners=False,
            ).reshape(B, Nt, N, N)

        theta_n = (dataset.theta[start:end].to(device) - theta_mean) / theta_std
        with torch.no_grad():
            u_pred = surrogate.generate(theta_n, dt=dt, gamma=gamma, rule=rule).cpu()  # (B, Nt, N, N)

        for i in range(B):
            ti = t_idx[start + i]
            pred_mmap[start + i] = u_pred[i, ti].numpy()
            true_mmap[start + i] = u_raw[i, ti].numpy()

    pred_mmap.flush(); true_mmap.flush()
    print(f"Sauvegardé — {pred_mmap.nbytes / 1e9:.1f} Go × 2")

    # Libérer le surrogate GPU avant l'entraînement
    del surrogate
    torch.cuda.empty_cache()

    return (np.load(pred_path, mmap_mode='r'),
            np.load(true_path, mmap_mode='r'))


# Dataset
class _FrameDataset(_Dataset):
    """(U_pred_frames, U_true_frames) de shape (kt, N, N) par simulation."""

    def __init__(self, U_pred, U_true, indices):
        self.U_pred  = U_pred
        self.U_true  = U_true
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.U_pred[i].copy()),
                torch.from_numpy(self.U_true[i].copy()))


# Entraînement
def train(
    U_pred, U_true,
    train_idx, val_idx, test_idx,
    surrogate_ckpt,
    epochs, batch_size, kt,
    base_ch, lr, patience,
    save_dir, project, N, Nt,
    lambda_grad=10.0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    ae = CorrectionAE(N=N, base_ch=base_ch).to(device)
    print(ae)

    train_ds = _FrameDataset(U_pred, U_true, train_idx)
    val_ds   = _FrameDataset(U_pred, U_true, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train : {len(train_idx)} sims  |  Val : {len(val_idx)} sims  |  {batch_size * kt} frames/step")

    optimizer = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-7)

    wandb.init(project=project, name='CorrectionAE', config=dict(
        N=N, Nt=Nt, base_ch=base_ch, lr=lr,
        batch_size=batch_size, kt=kt, lambda_grad=lambda_grad,
        n_params=sum(p.numel() for p in ae.parameters()),
        n_train=len(train_idx), n_val=len(val_idx),
        surrogate=os.path.basename(surrogate_ckpt),
    ))

    best_val   = float('inf')
    best_state = None
    patience_  = 0
    out_path   = os.path.join(save_dir, 'CorrectionAE_best.pt')
    step       = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc='CorrectionAE', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:

        # ---- Train ----
        ae.train()
        tr_loss = tr_l2rel = 0.0
        n_tr = 0

        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>3}', position=1, leave=False, unit='batch')
        for pred_frames, true_frames in train_bar:
            B    = pred_frames.shape[0]
            pred = pred_frames.reshape(B * kt, N, N).to(device)
            true = true_frames.reshape(B * kt, N, N).to(device)

            U_corr         = ae(pred)
            loss, metrics  = ae.loss(U_corr, true, pred, lambda_grad=lambda_grad)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                denom      = true.norm() + 1e-8
                l2rel      = (U_corr - true).norm().item() / denom.item()
                l2rel_surr = (pred   - true).norm().item() / denom.item()

            tr_loss  += loss.item(); tr_l2rel += l2rel; n_tr += 1
            train_bar.set_postfix(loss=f"{loss.item():.3e}", l2=f"{l2rel:.2%}", surr=f"{l2rel_surr:.2%}")
            wandb.log({'batch/loss': loss.item(), 'batch/mse': metrics['mse'].item(),
                       'batch/grad_loss': metrics['grad'].item(),
                       'batch/l2rel': l2rel, 'batch/l2rel_surrogate': l2rel_surr}, step=step)
            step += 1

        tr_loss /= n_tr; tr_l2rel /= n_tr

        # ---- Val ----
        ae.eval()
        vl_loss = vl_mse = vl_grad = vl_l2rel = 0.0
        n_vl = 0

        log_images = (epoch % 5 == 0)
        img_logged = False

        with torch.no_grad():
            for pred_frames, true_frames in tqdm(val_loader, desc=f'  Val   {epoch:>3}', position=1, leave=False, unit='batch'):
                B    = pred_frames.shape[0]
                pred = pred_frames.reshape(B * kt, N, N).to(device)
                true = true_frames.reshape(B * kt, N, N).to(device)

                U_corr          = ae(pred)
                loss, metrics   = ae.loss(U_corr, true, pred, lambda_grad=lambda_grad)
                denom           = true.norm() + 1e-8
                l2rel           = (U_corr - true).norm().item() / denom.item()

                vl_loss += loss.item(); vl_mse += metrics['mse'].item()
                vl_grad += metrics['grad'].item(); vl_l2rel += l2rel; n_vl += 1

                # Log quelques frames du premier batch val
                if log_images and not img_logged:
                    def _to_img(t):   # (N, N) → wandb.Image normalisé
                        t = t.float().cpu()
                        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
                        return wandb.Image(t.numpy())
                    wandb.log({
                        'images/u_pred':      _to_img(pred[0]),
                        'images/u_corrected': _to_img(U_corr[0]),
                        'images/u_true':      _to_img(true[0]),
                        'images/residual':    _to_img((U_corr[0] - pred[0]).abs()),
                    }, step=step)
                    img_logged = True

        vl_loss /= n_vl; vl_mse /= n_vl; vl_grad /= n_vl; vl_l2rel /= n_vl
        scheduler.step(vl_l2rel)

        epoch_bar.set_postfix(
            tr_l2=f"{tr_l2rel:.2%}", vl_l2=f"{vl_l2rel:.2%}",
            loss=f"{vl_loss:.3e}", lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            best=f"{best_val:.2%}",
        )
        wandb.log({
            'train/loss': tr_loss, 'train/l2rel': tr_l2rel,
            'val/loss':   vl_loss, 'val/mse':     vl_mse,
            'val/grad_loss': vl_grad, 'val/l2rel': vl_l2rel,
            'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch,
        }, step=step)

        if vl_l2rel < best_val:
            best_val   = vl_l2rel
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
            patience_  = 0
            torch.save({
                'model_state':   best_state,
                'model_type':    'CorrectionAE',
                'N':             N,
                'surrogate_ckpt': surrogate_ckpt,
                'val_l2rel':     best_val,
                'test_idx':      np.asarray(test_idx),
                'base_ch':       base_ch,
            }, out_path)
            epoch_bar.write(f"  → best sauvegardé (val L2={best_val:.2%})")
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best={best_val:.2%})")
                break

    wandb.finish()
    print(f"Entraînement terminé — best val L2 : {best_val:.2%}  → {out_path}")
    assert best_state is not None
    ae.load_state_dict(best_state)
    return ae


if __name__ == '__main__':
    data_path      = os.path.join('dataset', 'ch4_rotated.npy')
    surrogate_ckpt = os.path.join('checkpoints', 'LaplaceLatentModel_finetuned.pt')
    save_dir       = 'checkpoints'
    cache_dir      = '/Data/KAT'
    seed           = 42
    dt             = 1.0
    gamma          = 0.0
    rule           = 'trap'
    interp_size    = 128

    epochs      = 100
    batch_size  = 64
    kt          = 20
    base_ch     = 16
    lr          = 1e-3
    patience    = 15
    lambda_grad = 10.0
    project     = 'convdiff'

    dataset = TransientDataset(data_path, laplace=False, interp_size=interp_size, dt=dt)

    _surr_ckpt = torch.load(surrogate_ckpt, map_location='cpu', weights_only=False)
    test_idx   = _surr_ckpt['test_idx']
    test_set   = set(test_idx.tolist())
    non_test   = [i for i in range(len(dataset)) if i not in test_set]

    torch.manual_seed(seed)
    perm      = torch.randperm(len(non_test))
    n_train   = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)

    U_pred, U_true = precompute(
        dataset, surrogate_ckpt, kt=kt, cache_dir=cache_dir,
        batch_size=32, dt=dt, gamma=gamma, rule=rule, seed=seed,
    )

    train(
        U_pred, U_true,
        train_idx, val_idx, test_idx,
        surrogate_ckpt=surrogate_ckpt,
        epochs=epochs, batch_size=batch_size, kt=kt,
        base_ch=base_ch, lr=lr, lambda_grad=lambda_grad,
        patience=patience, save_dir=save_dir, project=project,
        N=dataset.N, Nt=dataset.Nt,
    )
