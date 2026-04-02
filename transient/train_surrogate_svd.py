import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm

from models.svd_surrogate import SVDSurrogate
from transient.dataset import TransientDataset


def train(
    svd_path,
    data_path,
    epochs     = 500,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 50,
    seed       = 42,
    project    = 'convdiff',
    ckpt_dir   = 'checkpoints',
):

    # --- Données ---
    svd   = np.load(svd_path)
    G     = svd['G'].astype(np.float32)       # (ns, nf_eff)
    alph  = svd['alph'].astype(np.float32)    # (nf_eff,)
    F     = svd['F'].astype(np.float32)       # (nr, nf_eff)
    P     = svd['P'].astype(np.float32)       # (Nt, nf_eff)

    # Pondérer G par alph pour que la loss MSE soit alignée avec l'erreur L2 champ
    G = G * alph[None, :]

    dataset   = TransientDataset(data_path)
    ns        = len(dataset)
    theta_dim = dataset.theta_dim

    assert ns == G.shape[0], "theta et G doivent avoir le même nombre de simulations"
    nf_eff = G.shape[1]
    print(f"ns={ns}  nf_eff={nf_eff}  theta_dim={theta_dim}")

    # Splits
    torch.manual_seed(seed)
    n_train = int(0.8 * ns)
    n_val   = int(0.1 * ns)
    idx = torch.randperm(ns, generator=torch.Generator().manual_seed(seed))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    dataset.fit(train_idx.tolist())

    theta_mean = dataset.theta_mean.numpy()
    theta_std  = dataset.theta_std.numpy()

    theta   = dataset.theta.numpy()            # (ns, theta_dim)
    theta_n = (theta - theta_mean) / theta_std

    G_mean = G[train_idx.numpy()].mean(axis=0)
    G_std  = G[train_idx.numpy()].std(axis=0) + 1e-8
    G_n    = (G - G_mean) / G_std

    theta_t = torch.tensor(theta_n)
    G_t     = torch.tensor(G_n)

    dataset = TensorDataset(theta_t, G_t)
    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_set   = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_set  = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    # --- Modèle ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model = SVDSurrogate(nf_eff=nf_eff, theta_dim=theta_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,}")

    # --- W&B ---
    run_name = f'SVDSurrogate_{time.strftime("%Y%m%d-%H%M%S")}'
    wandb.init(project=project, name=run_name, config=dict(
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        ns=ns, nf_eff=nf_eff, theta_dim=theta_dim, n_params=n_params,
    ))
    wandb.watch(model, log='gradients', log_freq=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-6)
    criterion = nn.MSELoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, 'SVDSurrogate_best.pt')
    best_val  = float('inf')
    patience_ = 0

    for epoch in tqdm(range(1, epochs + 1)):
        # Train
        model.train()
        train_loss = 0.
        for th, g in train_loader:
            th, g = th.to(device), g.to(device)
            loss = criterion(model(th), g)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for th, g in val_loader:
                th, g = th.to(device), g.to(device)
                val_loss += criterion(model(th), g).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        wandb.log({'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss,
                   'lr': optimizer.param_groups[0]['lr']})

        if val_loss < best_val:
            best_val  = val_loss
            patience_ = 0
            torch.save({
                'model_type': 'SVDSurrogate',
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'nf_eff': nf_eff,
                'theta_dim': theta_dim,
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'G_mean': G_mean,
                'G_std': G_std,
                'alph': alph,
                'F': F,
                'P': P,
                'test_idx': test_idx.numpy(),
            }, best_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                print(f'Early stopping à l\'époque {epoch}')
                break

    # Test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for th, g in test_loader:
            th, g = th.to(device), g.to(device)
            test_loss += criterion(model(th), g).item()
    test_loss /= len(test_loader)
    print(f'Test loss : {test_loss:.6e}')
    wandb.log({'test/loss': test_loss})
    wandb.finish()


if __name__ == '__main__':
    train(
        svd_path   = os.path.join("dataset", "svd_train_diff.npz"),
        data_path  = os.path.join("dataset", "dataset_transient.npz"),
        epochs     = 600,
        batch_size = 256,
        lr         = 1e-3,
        patience   = 50,
    )
