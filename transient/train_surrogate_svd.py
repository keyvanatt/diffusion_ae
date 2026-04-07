import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import wandb
from tqdm import tqdm

from models.svd_surrogate import SVDSurrogate
from transient.dataset import TransientDataset


def train(
    svd_path,
    data_path,
    dt         = 1.0,
    epochs     = 500,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 50,
    num_workers = 4,
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

    dataset   = TransientDataset(data_path, dt=dt)
    ns        = len(dataset)
    nr        = F.shape[0]
    Nt        = P.shape[0]
    theta_dim = dataset.theta_dim

    assert ns == G.shape[0], "theta et G doivent avoir le même nombre de simulations"
    nf_eff = G.shape[1]
    print(f"ns={ns}  nf_eff={nf_eff}  theta_dim={theta_dim}")

    # Splits
    torch.manual_seed(seed)
    idx     = torch.randperm(ns, generator=torch.Generator().manual_seed(seed))
    n_train = int(0.8 * ns)
    n_val   = int(0.1 * ns)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    dataset.fit(train_idx.tolist())

    theta_mean = dataset.theta_mean
    theta_std  = dataset.theta_std

    theta_n = (dataset.theta - theta_mean) / theta_std  # (ns, theta_dim)

    G_mean = G[train_idx.numpy()].mean(axis=0)
    G_std  = G[train_idx.numpy()].std(axis=0) + 1e-8
    G_n    = (G - G_mean) / G_std

    theta_t = theta_n                           # already a tensor
    G_t     = torch.tensor(G_n)

    full_ds   = TensorDataset(theta_t, G_t)
    train_loader = DataLoader(
        Subset(full_ds, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx.tolist()),
        batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        Subset(full_ds, test_idx.tolist()),
        batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # --- Modèle ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model = SVDSurrogate(nr, Nt, nf_eff=nf_eff, theta_dim=theta_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,}")

    # --- W&B ---
    run_name = f'SVDSurrogate_{time.strftime("%Y%m%d-%H%M%S")}'
    wandb.init(project=project, name=run_name, config=dict(
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        ns=ns, nf_eff=nf_eff, theta_dim=theta_dim, n_params=n_params,
    ))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-6)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    os.makedirs(ckpt_dir, exist_ok=True)
    best_path  = os.path.join(ckpt_dir, 'SVDSurrogate_best.pt')
    best_val   = float('inf')
    best_state = None
    best_epoch = 0
    patience_  = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc='SVDSurrogate')
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss  = torch.zeros(1, device=device)
        train_l2rel = torch.zeros(1, device=device)
        for th, g in train_loader:
            th, g = th.to(device), g.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                g_hat = model(th)
                loss  = criterion(g_hat, g)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss  += loss.detach()
            train_l2rel += (
                (g_hat.detach() - g).norm(dim=1) / (g.norm(dim=1) + 1e-8)
            ).mean()
        train_loss  = train_loss.item()  / len(train_loader)
        train_l2rel = train_l2rel.item() / len(train_loader)

        # Val
        model.eval()
        val_loss  = torch.zeros(1, device=device)
        val_l2rel = torch.zeros(1, device=device)
        with torch.no_grad():
            for th, g in val_loader:
                th, g = th.to(device), g.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    g_hat = model(th)
                    val_loss += criterion(g_hat, g)
                val_l2rel += (
                    (g_hat - g).norm(dim=1) / (g.norm(dim=1) + 1e-8)
                ).mean()
        val_loss  = val_loss.item()  / len(val_loader)
        val_l2rel = val_l2rel.item() / len(val_loader)

        epoch_time = time.perf_counter() - t0
        scheduler.step(val_loss)
        epoch_bar.set_postfix(train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", l2=f"{val_l2rel:.2%}")
        wandb.log({
            'train/loss':    train_loss,
            'train/l2rel':   train_l2rel,
            'val/loss':      val_loss,
            'val/l2rel':     val_l2rel,
            'lr':            optimizer.param_groups[0]['lr'],
            'epoch_time_s':  epoch_time,
        }, step=epoch)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_  = 0
        else:
            patience_ += 1
            if patience_ >= patience:
                print(f"Early stopping à l'époque {epoch}")
                break

    # Sauvegarde unique en fin de training
    assert best_state is not None
    model.load_state_dict(best_state)
    model.set_bases(F, P, alph, G_mean, G_std)
    torch.save({
        'model_type':  'SVDSurrogate',
        'model_state': model.state_dict(),
        'epoch':       best_epoch,
        'val_loss':    best_val,
        'nf_eff':      nf_eff,
        'nr':          nr,
        'Nt':          Nt,
        'theta_dim':   theta_dim,
        'theta_mean':  theta_mean.numpy(),
        'theta_std':   theta_std.numpy(),
        'test_idx':    test_idx.numpy(),
    }, best_path)
    print(f"Checkpoint sauvegardé : {best_path}")

    # Test
    model.eval()
    test_loss  = torch.zeros(1, device=device)
    test_l2rel = torch.zeros(1, device=device)
    with torch.no_grad():
        for th, g in test_loader:
            th, g = th.to(device), g.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                g_hat = model(th)
                test_loss += criterion(g_hat, g)
            test_l2rel += (
                (g_hat - g).norm(dim=1) / (g.norm(dim=1) + 1e-8)
            ).mean()
    test_loss  = test_loss.item()  / len(test_loader)
    test_l2rel = test_l2rel.item() / len(test_loader)
    print(f"Test loss={test_loss:.4e}  l2rel={test_l2rel:.2%}")
    wandb.log({'test/loss': test_loss, 'test/l2rel': test_l2rel})
    wandb.finish()


if __name__ == '__main__':
    train(
        svd_path    = '/Data/KAT/svd_train_diff.npz',
        data_path   = '/Data/KAT/ch4_rotated.npy',
        dt          = 1.0,
        epochs      = 600,
        batch_size  = 256,
        lr          = 1e-3,
        patience    = 50,
        num_workers = 4,
    )
