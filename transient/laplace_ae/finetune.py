"""
finetune_decoder_laplace.py — Finetune end-to-end du LaplaceLatentModel assemblé.

Pipeline :
  1. train_ae_laplace.py    → LaplaceAE
  2. train_laplace.py        → LaplaceLatentSurrogate par fréquence (θ→z, décodeur gelé)
  3. CE SCRIPT               → finetune end-to-end, loss L2 directement sur U(t) physique

La transformée inverse de Laplace est implémentée en PyTorch (torch.fft.ifft),
ce qui rend le chemin θ → spectre → U(t) entièrement différentiable.
La loss et le critère d'arrêt anticipé opèrent sur U(t), pas sur les spectres normalisés.

Optimisations vitesse :
  - Toutes les projections θ→z sont calculées, puis UN SEUL appel batché au
    shared_decoder (K*B samples) au lieu de K appels séquentiels.
  - La cible U(t) est reconstruite une seule fois par batch (via IFFT, no_grad).
  - Le denom de l'inverse Laplace est pré-calculé une fois pour tout le training.

Checkpoint : checkpoints/LaplaceLatentModel_finetuned.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as _Dataset
from tqdm import tqdm
import wandb

from models.transient.laplace_latent_surrogate import LaplaceLatentModel
from transient.dataset import TransientDataset


# ---------------------------------------------------------------------------
# Dataset — retourne toutes les fréquences d'une simulation en un seul item
# Toutes les données restent CPU (les workers DataLoader n'ont pas accès au GPU).
# ---------------------------------------------------------------------------

class _FinetuneDataset(_Dataset):
    """
    Retourne (theta_norm, U_laplace_norm) où U_laplace_norm est de shape
    (K, 2, N, N) — toutes les fréquences d'une simulation.

    Chaque item = 1 simulation complète.
    """
    def __init__(self, U_laplace, theta_norm, target_mean, target_std, indices):
        self.U_laplace   = U_laplace             # ndarray ou tensor CPU
        self.theta_norm  = theta_norm.cpu()       # (ns, theta_dim) CPU
        self.target_mean = target_mean.cpu()      # (K, 2, 1, 1) CPU
        self.target_std  = target_std.cpu()       # (K, 2, 1, 1) CPU
        self._indices    = [int(i) for i in indices]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sim_i = self._indices[idx]
        th = self.theta_norm[sim_i]
        if isinstance(self.U_laplace, np.ndarray):
            u = torch.from_numpy(self.U_laplace[sim_i].copy()).float()
        else:
            u = self.U_laplace[sim_i].float()
        u_norm = (u - self.target_mean) / self.target_std
        return th, u_norm


# ---------------------------------------------------------------------------
# Reconstruction U(t) depuis les spectres normalisés (cible, sans gradient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _laplace_to_u(U_laplace_norm, target_mean, target_std, s_list, Nt, dt,
                  alpha_t=0.0, lam=1e-6, rule='trap'):
    """
    Reconstruit U(t) physique depuis les spectres normalisés via laplace_inverse_tik.
    La symétrie conjuguée est gérée dans laplace_inverse_tik.

    U_laplace_norm : (B, K, 2, N, N) normalisé
    s_list         : (K,) complex128 CPU
    → U_true (B, Nt, N, N) float32
    """
    from utils.laplace import laplace_inverse_tik
    B, K, _, N, _ = U_laplace_norm.shape
    NN = N * N

    # Dénorm
    U_phys = U_laplace_norm * target_std + target_mean  # (B, K, 2, N, N)

    # → (B, NN, K) complexe
    re = U_phys[:, :, 0].reshape(B, K, NN).permute(0, 2, 1)
    im = U_phys[:, :, 1].reshape(B, K, NN).permute(0, 2, 1)
    M  = torch.complex(re, im)  # (B, NN, K) complex64

    # CPU float64 inverse (pas de gradient ici)
    M_flat = M.cpu().cdouble().reshape(B * NN, K)
    U_flat = laplace_inverse_tik(M_flat, s_list, dt, Nt, alpha_t, lam, rule)
    # (B*NN, Nt) float64

    return U_flat.float().reshape(B, NN, Nt).permute(0, 2, 1).reshape(B, Nt, N, N)


# ---------------------------------------------------------------------------
# Finetune
# ---------------------------------------------------------------------------

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
    dt,
    rule,
    alpha_t=0.0,
    lam=1e-6,
):
    N         = dataset.N
    Nt        = dataset.Nt
    K         = dataset.K
    theta_dim = dataset.theta_dim
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # --- Charger le modèle assemblé ---
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = LaplaceLatentModel(
        K=ckpt['K'], Nt=ckpt['Nt'], N=ckpt['N'],
        theta_dim=ckpt['theta_dim'], latent_dim=ckpt['latent_dim'],
        hidden_dim=ckpt['hidden_dim'], k_max=ckpt.get('k_max'),
        freq_L=ckpt.get('freq_L', 8),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Modèle chargé depuis {ckpt_path}")

    # Dégeler le shared_decoder
    model.shared_decoder.requires_grad_(True)

    # s_list pour la reconstruction ground-truth (CPU, float64)
    s_list_cpu = model.s_list  # (K,) complex128 CPU

    # --- Theta normalisé (CPU pour le dataset) ---
    theta_norm = (dataset.theta - dataset.theta_mean) / dataset.theta_std   # CPU

    # --- Datasets (tout CPU) ---
    train_ds = _FinetuneDataset(dataset.U_laplace, theta_norm,
                                dataset.target_mean, dataset.target_std,
                                train_idx)
    val_ds   = _FinetuneDataset(dataset.U_laplace, theta_norm,
                                dataset.target_mean, dataset.target_std,
                                val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Finetune Dataset : train {len(train_ds)} sims  |  val {len(val_ds)} sims")
    print(f"Batch U(t) shape : ({batch_size}, {Nt}, {N}, {N})")

    # --- Optimiseur avec LR différentiel ---
    surrogate_params = [p for s in model.surrogates for p in s.proj.parameters()]
    decoder_params   = list(model.shared_decoder.parameters())

    optimizer = torch.optim.AdamW([
        {'params': surrogate_params, 'lr': lr_surrogate},
        {'params': decoder_params,   'lr': lr_decoder},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    n_surr  = sum(p.numel() for p in surrogate_params)
    n_dec   = sum(p.numel() for p in decoder_params)
    print(f"Params : {n_surr + n_dec:,} total  (surrogates {n_surr:,} + decoder {n_dec:,})")
    print(f"LR surrogate={lr_surrogate:.1e}  decoder={lr_decoder:.1e}  device={device}")

    # --- Wandb ---
    wandb.init(project=project, name='LaplaceLatentModel_finetune_e2e', config=dict(
        N=N, Nt=Nt, K=K, theta_dim=theta_dim,
        latent_dim=ckpt['latent_dim'],
        epochs=epochs, batch_size=batch_size,
        lr_surrogate=lr_surrogate, lr_decoder=lr_decoder,
        loss='L2_on_U_t',
        n_params_total=n_surr + n_dec,
        n_samples_train=len(train_ds), n_samples_val=len(val_ds),
        source_ckpt=ckpt_path, device=str(device),
    ))
    wandb.watch(model, log='gradients', log_freq=100)

    best_val_l2  = float('inf')
    best_state   = None
    patience_    = 0
    out_path     = os.path.join(save_dir, 'LaplaceLatentModel_finetuned.pt')
    global_step  = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc='Finetune e2e', position=0, leave=True, unit='epoch')
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss  = 0.0
        train_l2rel = 0.0
        n_batches   = 0

        train_bar = tqdm(train_loader, desc=f'  Train {epoch:>3}', position=1,
                         leave=False, unit='batch')
        for batch_idx, (th, u_laplace_norm) in enumerate(train_bar):
            th             = th.to(device, non_blocking=True)
            u_laplace_norm = u_laplace_norm.to(device, non_blocking=True)

            # Cible U(t) physique
            u_true = _laplace_to_u(u_laplace_norm, model.target_mean, model.target_std,
                                   s_list_cpu, Nt, dt, alpha_t=alpha_t, lam=lam, rule=rule).to(device)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                u_pred = model._generate_diff(th, dt=dt, alpha_t=alpha_t, lam=lam, rule=rule)
                loss   = F.mse_loss(u_pred.float(), u_true)

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
                l2rel = ((u_pred.float() - u_true).flatten(1).norm(dim=1)
                         / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean().item()

            train_loss  += loss_val
            train_l2rel += l2rel
            n_batches   += 1

            if batch_idx % 20 == 0:
                train_bar.set_postfix(loss=f"{loss_val:.3e}", l2=f"{l2rel:.2%}")
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
        n_val_b   = 0

        val_bar = tqdm(val_loader, desc=f'  Val   {epoch:>3}', position=1,
                       leave=False, unit='batch')
        with torch.no_grad():
            for th, u_laplace_norm in val_bar:
                th             = th.to(device, non_blocking=True)
                u_laplace_norm = u_laplace_norm.to(device, non_blocking=True)

                u_true = _laplace_to_u(u_laplace_norm, model.target_mean, model.target_std,
                                       s_list_cpu, Nt, dt, alpha_t=alpha_t, lam=lam, rule=rule).to(device)

                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    u_pred = model._generate_diff(th, dt=dt, alpha_t=alpha_t, lam=lam, rule=rule)

                val_loss  += F.mse_loss(u_pred.float(), u_true).item()
                val_l2rel += ((u_pred.float() - u_true).flatten(1).norm(dim=1)
                              / (u_true.flatten(1).norm(dim=1) + 1e-8)).mean().item()
                n_val_b   += 1

        val_loss  /= max(n_val_b, 1)
        val_l2rel /= max(n_val_b, 1)

        scheduler.step(val_l2rel)
        epoch_time = time.perf_counter() - t0

        epoch_bar.set_postfix(
            tr_l2=f"{train_l2rel:.2%}", vl_l2=f"{val_l2rel:.2%}",
            lr_s=f"{optimizer.param_groups[0]['lr']:.1e}",
            lr_d=f"{optimizer.param_groups[1]['lr']:.1e}",
            best=f"{best_val_l2:.2%}",
        )
        wandb.log({
            'train/loss':   train_loss,
            'train/l2rel':  train_l2rel,
            'val/loss':     val_loss,
            'val/l2rel':    val_l2rel,
            'lr/surrogate': optimizer.param_groups[0]['lr'],
            'lr/decoder':   optimizer.param_groups[1]['lr'],
            'epoch_time_s': epoch_time,
            'epoch':        epoch,
        }, step=global_step)

        if val_l2rel < best_val_l2:
            best_val_l2 = val_l2rel
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_   = 0
            torch.save({
                'model_state': best_state,
                'model_type':  'LaplaceLatentModel',
                'K':           K,
                'Nt':          Nt,
                'N':           N,
                'theta_dim':   theta_dim,
                'latent_dim':  ckpt['latent_dim'],
                'hidden_dim':  ckpt['hidden_dim'],
                'k_max':       ckpt.get('k_max'),
                'freq_L':      ckpt.get('freq_L', 8),
                'dt':          dt,
                'alpha_t':     alpha_t,
                'lam':         lam,
                'theta_mean':  dataset.theta_mean,
                'theta_std':   dataset.theta_std,
                'test_idx':    np.asarray(test_idx),
                'finetuned':   True,
            }, out_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.write(f"Early stopping à l'époque {epoch}  (best val L2={best_val_l2:.2%})")
                break

    assert best_state is not None
    wandb.save(out_path)
    wandb.finish()
    print(f"Finetune terminé — best val L2 : {best_val_l2:.2%}  → {out_path}")

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    data_path    = os.path.join("dataset", "ch4_rotated.npy")
    model_ckpt   = os.path.join("checkpoints", "LaplaceLatentModel.pt")
    save_dir     = "checkpoints"
    seed         = 42
    rule         = 'trap'
    interp_size  = 128
    dt           = 1.0
    alpha_t      = 0.0
    lam          = 1e-6

    epochs       = 50
    batch_size   = 16       # chaque item = simulation complète (K freqs)
    lr_surrogate = 5e-5
    lr_decoder   = 1e-5
    patience     = 15
    project      = 'convdiff'

    # Charger s_list depuis le checkpoint du modèle assemblé
    _ckpt_s = torch.load(model_ckpt, map_location='cpu', weights_only=False)
    _s_real = _ckpt_s['model_state']['s_real'].numpy()
    _s_imag = _ckpt_s['model_state']['s_imag'].numpy()
    s_list  = (_s_real + 1j * _s_imag).astype(np.complex128)

    dataset = TransientDataset(data_path, laplace=True, s_list=s_list, rule=rule,
                               interp_size=interp_size, dt=dt)

    _split   = np.load('dataset/split.npz')
    test_idx = _split['test_idx']
    non_test = [i for i in range(len(dataset)) if i not in set(test_idx.tolist())]
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    perm     = torch.randperm(len(non_test))
    n_train  = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)
    print("Chargement en RAM...", end=' ', flush=True)
    t0 = time.perf_counter()
    U = np.ascontiguousarray(dataset.U_laplace)
    dataset.U_laplace = U
    print(f"OK — {U.nbytes/1e9:.1f} Go RAM, {time.perf_counter()-t0:.1f}s")

    print("\n=== Finetune LaplaceLatentModel end-to-end (loss sur U(t)) ===")
    finetune(dataset, train_idx, val_idx, test_idx,
             ckpt_path=model_ckpt,
             epochs=epochs, batch_size=batch_size,
             lr_surrogate=lr_surrogate, lr_decoder=lr_decoder,
             patience=patience, save_dir=save_dir, project=project,
             dt=dt, rule=rule, alpha_t=alpha_t, lam=lam)
