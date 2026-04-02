"""
train_laplace.py — Entraînement des surrogates dans l'espace de Laplace
=======================================================================
Un LaplaceSurrogate indépendant est entraîné par fréquence de Laplace.
La transformation des champs en espace de Laplace est assurée par TransientDataset.

Usage :
    python transient/train_laplace.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
import wandb

from models.laplace_surrogate import LaplaceSurrogate, LaplaceModel, LaplaceModel


def train_one(
    k,
    s_k,
    train_loader,
    val_loader,
    theta_mean,
    theta_std,
    target_mean,
    target_std,
    N,
    Nt,
    epochs       = 300,
    lr           = 1e-3,
    patience     = 30,
    theta_dim    = 4,
    device       = None,
    ckpt_dir     = 'checkpoints',
    global_step  = 0,
):
    """
    Entraîne un LaplaceSurrogate pour la fréquence k.

    Retour
    ------
    best_val : float – meilleure val loss atteinte
    n_epochs : int   – nombre d'époques réalisées
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    best_val  = float('inf')
    patience_ = 0
    ckpt_path = os.path.join(ckpt_dir, f'LaplaceSurrogate_freq{k:03d}.pt')

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"  freq {k:03d}  s={s_k.imag:.3g}j",
        leave=False,
        position=1,
    )
    for epoch in epoch_bar:
        # --- Train ---
        model.train()
        train_loss = 0.
        for th, u in train_loader:
            th, u = th.to(device), u.to(device)
            loss = model.loss(model(th), u)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Val ---
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for th, u in val_loader:
                th, u = th.to(device), u.to(device)
                val_loss += model.loss(model(th), u).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        epoch_bar.set_postfix(train=f"{train_loss:.3e}", val=f"{val_loss:.3e}")

        wandb.log({
            'freq': k,
            'train/loss': train_loss,
            'val/loss':   val_loss,
            'lr':         optimizer.param_groups[0]['lr'],
        }, step=global_step + epoch)

        if val_loss < best_val:
            best_val  = val_loss
            patience_ = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'theta_mean':  theta_mean.numpy(),
                'theta_std':   theta_std.numpy(),
                'target_mean': target_mean.numpy(),
                'target_std':  target_std.numpy(),
                'N': N, 'Nt': Nt, 'theta_dim': theta_dim,
                'freq_idx': k, 's_k_real': s_k.real, 's_k_imag': s_k.imag,
            }, ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.set_postfix(
                    train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", stop=f"ep{epoch}"
                )
                break

    wandb.log({f'best_val/freq_{k:03d}': best_val})
    return best_val, epoch


def train_all(
    dataset,
    train_idx,
    val_idx,
    epochs     = 300,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 30,
    ckpt_dir   = 'checkpoints/laplace',
    project    = 'convdiff',
):
    """
    Entraîne un LaplaceSurrogate par fréquence de Laplace.

    Paramètres
    ----------
    dataset   : TransientDataset avec laplace=True et fit() déjà appelé
    train_idx : liste d'indices train
    val_idx   : liste d'indices val
    """
    Nt_half   = dataset.Nt_half
    N         = dataset.N
    Nt        = dataset.Nt
    theta_dim = dataset.theta_dim
    ns        = len(dataset)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device : {device}   ns={ns}  Nt={Nt} (Nt_half={Nt_half})  N={N}  theta_dim={theta_dim}")
    os.makedirs(ckpt_dir, exist_ok=True)

    theta_n = (dataset.theta - dataset.theta_mean) / dataset.theta_std  # (ns, theta_dim)

    n_params = sum(p.numel() for p in LaplaceSurrogate(s=0j, N=N, theta_dim=theta_dim).parameters())
    wandb.init(project=project, name=f'LaplaceSurrogate_Nt{Nt}', config=dict(
        Nt=Nt, Nt_half=Nt_half, N=N, ns=ns, theta_dim=theta_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, n_params=n_params,
    ))

    best_vals   = []
    global_step = 0

    freq_bar = tqdm(range(Nt_half), desc="Fréquences", position=0, leave=True)
    for k in freq_bar:
        s_k           = complex(dataset.s[k])
        target_mean_k = dataset.target_mean[k]                                      # (2, 1, 1)
        target_std_k  = dataset.target_std[k]
        target_k_n    = (dataset.U_laplace[:, k] - target_mean_k) / target_std_k   # (ns, 2, N, N)

        ds_k         = TensorDataset(theta_n, target_k_n)
        train_loader = DataLoader(Subset(ds_k, train_idx), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(ds_k, val_idx),   batch_size=batch_size)

        best_val, n_epochs = train_one(
            k, s_k, train_loader, val_loader,
            dataset.theta_mean, dataset.theta_std,
            target_mean=target_mean_k, target_std=target_std_k,
            N=N, Nt=Nt, epochs=epochs, lr=lr, patience=patience,
            theta_dim=theta_dim, device=device, ckpt_dir=ckpt_dir,
            global_step=global_step,
        )

        best_vals.append(best_val)
        global_step += n_epochs
        freq_bar.set_postfix(freq=k, best_val=f"{best_val:.3e}", s=f"{s_k:.3g}")

    wandb.log({'best_val/mean': np.mean(best_vals)})
    wandb.finish()
    print(f"\nEntraînement terminé. Val loss moyenne : {np.mean(best_vals):.3e}")
    assemble_model(dataset, ckpt_dir)
    return best_vals


def assemble_model(dataset, ckpt_dir: str):
    """
    Charge les checkpoints individuels et assemble un LaplaceModel unique.
    Sauvegarde dans <ckpt_dir>/LaplaceModel.pt avec toutes les stats de normalisation.
    """
    Nt_half   = dataset.Nt_half
    N         = dataset.N
    Nt        = dataset.Nt
    theta_dim = dataset.theta_dim
    gamma     = float(dataset.s[0].real) if hasattr(dataset, 's') else 0.0

    model = LaplaceModel(N_freq=Nt, N_half=Nt_half, N=N, theta_dim=theta_dim)
    for k in range(Nt_half):
        path_k = os.path.join(ckpt_dir, f'LaplaceSurrogate_freq{k:03d}.pt')
        ckpt_k = torch.load(path_k, map_location='cpu', weights_only=False)
        model.surrogates[k].load_state_dict(ckpt_k['model_state'])

    model.set_normalization(dataset.target_mean, dataset.target_std)

    out_path = os.path.join(ckpt_dir, 'LaplaceModel.pt')
    torch.save({
        'model_state': model.state_dict(),
        'model_type':  'LaplaceModel',
        'N_freq':      Nt,
        'N_half':      Nt_half,
        'N':           N,
        'theta_dim':   theta_dim,
        'dt':          dataset.dt,
        'gamma':       gamma,
        'theta_mean':  dataset.theta_mean,
        'theta_std':   dataset.theta_std,
    }, out_path)
    print(f"LaplaceModel assemblé → {out_path}")


if __name__ == '__main__':
    from transient.dataset import TransientDataset

    data_path  = os.path.join("dataset", "dataset_transient.npz")
    ckpt_dir   = os.path.join("checkpoints", "laplace")
    epochs, batch_size, lr, patience = 300, 64, 1e-3, 30
    gamma, rule, seed = 0.0, 'trap', 42

    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule)

    torch.manual_seed(seed)
    idx       = torch.randperm(len(dataset))
    n_train   = int(0.8 * len(dataset))
    n_val     = int(0.1 * len(dataset))
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].numpy()

    dataset.fit(train_idx)

    os.makedirs(ckpt_dir, exist_ok=True)
    np.save(os.path.join(ckpt_dir, 'test_idx.npy'), test_idx)
    print(f"Dataset : {tuple(dataset.U_laplace.shape)}  (Nt_half={dataset.Nt_half}/{dataset.Nt})")

    train_all(dataset, train_idx, val_idx,
              epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, ckpt_dir=ckpt_dir)
