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


def train(
    svd_path,
    doe_path,
    epochs     = 500,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 50,
    seed       = 42,
    project    = 'convdiff',
    ckpt_dir   = 'checkpoints',
):
    # --- Données ---
    svd  = np.load(svd_path)
    G    = svd['G'].astype(np.float32)       # (ns, nf_eff)
    doe  = np.load(doe_path)
    if doe.dtype.names:                       # structured array
        theta = np.column_stack([doe[k] for k in doe.dtype.names]).astype(np.float32)
    else:
        theta = doe.astype(np.float32)        # (ns, theta_dim)

    assert theta.shape[0] == G.shape[0], "doe et G doivent avoir le même nombre de simulations"
    ns, nf_eff   = G.shape
    theta_dim    = theta.shape[1]
    print(f"ns={ns}  nf_eff={nf_eff}  theta_dim={theta_dim}")

    # Normalisation theta (z-score sur train)
    torch.manual_seed(seed)
    n_train = int(0.8 * ns)
    n_val   = int(0.1 * ns)
    idx = torch.randperm(ns, generator=torch.Generator().manual_seed(seed))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    theta_mean = theta[train_idx.numpy()].mean(axis=0)
    theta_std  = theta[train_idx.numpy()].std(axis=0) + 1e-8
    G_mean     = G[train_idx.numpy()].mean(axis=0)
    G_std      = G[train_idx.numpy()].std(axis=0) + 1e-8

    theta_n = (theta - theta_mean) / theta_std
    G_n     = (G     - G_mean)     / G_std

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)
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
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'nf_eff': nf_eff,
                'theta_dim': theta_dim,
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'G_mean': G_mean,
                'G_std': G_std,
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


def evaluate(svd_path, doe_path, ckpt_path, n_animate=3):
    """
    Évalue le surrogate sur le test set (jamais vu à l'entraînement).
    - Histogrammes MAE et MSE par simulation (espace champ reconstruit)
    - Animations de n_animate exemples avec animate_comparaison
    """
    import matplotlib.pyplot as plt
    from utils.SVD_Amine_3D import svd_inverse_3d
    from utils.animate import animate_comparaison

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    theta_mean = ckpt['theta_mean']
    theta_std  = ckpt['theta_std']
    G_mean     = ckpt['G_mean']
    G_std      = ckpt['G_std']
    test_idx   = ckpt['test_idx']               # indices jamais vus

    svd  = np.load(svd_path)
    F, P, alph = svd['F'], svd['P'], svd['alph']
    G_full = svd['G']                           # (ns, nf_eff)
    nr, nf_eff = F.shape
    Nt = P.shape[0]
    Hsub = Wsub = int(np.round(np.sqrt(nr)))
    assert Hsub * Wsub == nr, "Grille non carrée — ajuste Hsub/Wsub manuellement"

    doe = np.load(doe_path)
    if doe.dtype.names:
        theta = np.column_stack([doe[k] for k in doe.dtype.names]).astype(np.float32)
    else:
        theta = doe.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVDSurrogate(nf_eff=nf_eff, theta_dim=ckpt['theta_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Prédiction G sur tout le test set
    theta_test = theta[test_idx]
    theta_n    = (theta_test - theta_mean) / theta_std
    with torch.no_grad():
        G_pred_n = model(torch.tensor(theta_n).to(device)).cpu().numpy()
    G_pred = G_pred_n * G_std + G_mean          # (n_test, nf_eff)
    G_true = G_full[test_idx]                   # (n_test, nf_eff)

    # Reconstruction et métriques par simulation
    mae_list, mse_list = [], []
    for i in range(len(test_idx)):
        orig = svd_inverse_3d(F, G_true[i][None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)
        rec  = svd_inverse_3d(F, G_pred[i][None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)
        mae_list.append(np.abs(orig - rec).mean())
        mse_list.append(((orig - rec) ** 2).mean())

    mae = np.array(mae_list)
    mse = np.array(mse_list)
    print(f"Test set ({len(test_idx)} samples) — MAE: {mae.mean():.4e} ± {mae.std():.4e}  |  MSE: {mse.mean():.4e} ± {mse.std():.4e}")

    # Histogrammes
    os.makedirs('plots', exist_ok=True)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mae, bins=np.logspace(np.log10(mae.min()), np.log10(mae.max()), 30), color='steelblue', edgecolor='white')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('MAE')
    axes[0].set_title(f'MAE par simulation (test set)\nμ={mae.mean():.3e}  σ={mae.std():.3e}')
    axes[1].hist(mse, bins=np.logspace(np.log10(mse.min()), np.log10(mse.max()), 30), color='salmon', edgecolor='white')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('MSE')
    axes[1].set_title(f'MSE par simulation (test set)\nμ={mse.mean():.3e}  σ={mse.std():.3e}')
    plt.tight_layout()
    hist_path = os.path.join('plots', 'surrogate_test_hist.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Histogrammes sauvegardés : {hist_path}")

    # Animations : pires, médiane, meilleur
    sorted_idx = np.argsort(mse)
    picks = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    labels = ['best', 'median', 'worst']

    for pick, label in zip(picks[:n_animate], labels):
        sim_idx = test_idx[pick]
        orig = svd_inverse_3d(F, G_true[pick][None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)
        rec  = svd_inverse_3d(F, G_pred[pick][None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)
        animate_comparaison(
            orig, rec,
            output_path=os.path.join('plots', f'surrogate_{label}_{sim_idx}.gif'),
            title_fn=lambda t, s=sim_idx, l=label: f"#{s} ({l}) — t = {t}",
        )


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Results')

    svd_path  = os.path.join(results_dir, 'svd_train.npz')
    doe_path  = os.path.join(results_dir, 'doe.npy')
    ckpt_path = os.path.join('checkpoints', 'SVDSurrogate_best.pt')

    train(
        svd_path   = svd_path,
        doe_path   = doe_path,
        epochs     = 100,
        batch_size = 32,
        lr         = 1e-3,
        patience   = 50,
    )

    evaluate(svd_path, doe_path, ckpt_path, n_animate=3)
