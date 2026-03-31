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


def evaluate(svd_path, doe_path, concentration_path, ckpt_path, step=5, n_animate=3):
    """
    Évalue le surrogate sur le test set contre deux références :
      1. La reconstruction SVD (G_true → champ reconstruit)
      2. Le champ original (concentration sous-échantillonnée)
    Produit des histogrammes MAE/MSE et des animations pour les 3 cas best/median/worst.
    """
    import matplotlib.pyplot as plt
    from utils.SVD_Amine_3D import svd_inverse_3d
    from utils.animate import animate_comparaison

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    theta_mean = ckpt['theta_mean']
    theta_std  = ckpt['theta_std']
    G_mean     = ckpt['G_mean']
    G_std      = ckpt['G_std']
    test_idx   = ckpt['test_idx']

    svd  = np.load(svd_path)
    F, P, alph = svd['F'], svd['P'], svd['alph']
    G_full = svd['G']
    nr, nf_eff = F.shape
    Nt = P.shape[0]
    Hsub = Wsub = int(np.round(np.sqrt(nr)))
    assert Hsub * Wsub == nr, "Grille non carrée — ajuste Hsub/Wsub manuellement"

    doe = np.load(doe_path)
    if doe.dtype.names:
        theta = np.column_stack([doe[k] for k in doe.dtype.names]).astype(np.float32)
    else:
        theta = doe.astype(np.float32)

    # Champ original sous-échantillonné : (ns, Nt, Hsub, Wsub)
    concentration = np.load(concentration_path)
    conc_sub = concentration[:, :, ::step, ::step]  # (ns, Nt, Hsub, Wsub)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVDSurrogate(nf_eff=nf_eff, theta_dim=ckpt['theta_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    theta_test = theta[test_idx]
    theta_n    = (theta_test - theta_mean) / theta_std
    with torch.no_grad():
        G_pred_n = model(torch.tensor(theta_n).to(device)).cpu().numpy()
    G_pred = G_pred_n * G_std + G_mean
    G_true = G_full[test_idx]

    def to_field(G_row):
        return svd_inverse_3d(F, G_row[None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)

    mae_svd, mse_svd, l2rel_svd = [], [], []
    mae_orig, mse_orig, l2rel_orig, l2rel_svd_vs_orig = [], [], [], []
    for i, si in enumerate(test_idx):
        rec      = to_field(G_pred[i])
        ref_svd  = to_field(G_true[i])
        ref_orig = conc_sub[si]                 # (Nt, Hsub, Wsub)

        norm_orig = np.linalg.norm(ref_orig) + 1e-12

        mae_svd.append(np.abs(rec - ref_svd).mean())
        mse_svd.append(((rec - ref_svd) ** 2).mean())
        l2rel_svd.append(np.linalg.norm(rec - ref_svd) / (np.linalg.norm(ref_svd) + 1e-12))

        mae_orig.append(np.abs(rec - ref_orig).mean())
        mse_orig.append(((rec - ref_orig) ** 2).mean())
        l2rel_orig.append(np.linalg.norm(rec - ref_orig) / norm_orig)
        l2rel_svd_vs_orig.append(np.linalg.norm(ref_svd - ref_orig) / norm_orig)

    mae_svd,  mse_svd,  l2rel_svd  = np.array(mae_svd),  np.array(mse_svd),  np.array(l2rel_svd)
    mae_orig, mse_orig, l2rel_orig  = np.array(mae_orig), np.array(mse_orig), np.array(l2rel_orig)
    l2rel_svd_vs_orig = np.array(l2rel_svd_vs_orig)

    print(f"vs SVD  — MAE: {mae_svd.mean():.4e} ± {mae_svd.std():.4e}  |  MSE: {mse_svd.mean():.4e}  |  L2rel: {l2rel_svd.mean():.4e} ± {l2rel_svd.std():.4e}")
    print(f"vs Orig — MAE: {mae_orig.mean():.4e} ± {mae_orig.std():.4e}  |  MSE: {mse_orig.mean():.4e}  |  L2rel surrogate: {l2rel_orig.mean():.4e} ± {l2rel_orig.std():.4e}")
    print(f"SVD seul vs Orig — L2rel: {l2rel_svd_vs_orig.mean():.4e} ± {l2rel_svd_vs_orig.std():.4e}")

    # Histogrammes : 2 lignes (vs SVD, vs Original) × 2 colonnes (MAE, MSE)
    os.makedirs('plots', exist_ok=True)
    _, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (mae_arr, mse_arr, ref_label) in enumerate([
        (mae_svd,  mse_svd,  'vs SVD reconstruction'),
        (mae_orig, mse_orig, 'vs Original'),
    ]):
        for col, (arr, metric) in enumerate([(mae_arr, 'MAE'), (mse_arr, 'MSE')]):
            ax = axes[row, col]
            ax.hist(arr, bins=np.logspace(np.log10(arr.min()), np.log10(arr.max()), 30),
                    color='steelblue' if col == 0 else 'salmon', edgecolor='white')
            ax.set_xscale('log')
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} {ref_label}\nμ={arr.mean():.3e}  σ={arr.std():.3e}')
    plt.tight_layout()
    hist_path = os.path.join('plots', 'surrogate_test_hist.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Histogrammes sauvegardés : {hist_path}")

    # Animations best/median/worst selon MSE vs original (la vraie référence)
    sorted_idx = np.argsort(mse_orig)
    picks  = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    labels = ['best', 'median', 'worst']

    for pick, label in zip(picks[:n_animate], labels):
        si = test_idx[pick]
        rec      = to_field(G_pred[pick])
        ref_svd  = to_field(G_true[pick])
        ref_orig = conc_sub[si]

        animate_comparaison(
            ref_svd, rec,
            output_path=os.path.join('plots', f'surrogate_{label}_vs_svd.gif'),
            title_fn=lambda t, s=si, l=label: f"#{s} ({l}) SVD vs Surrogate — t={t}",
        )
        animate_comparaison(
            ref_orig, rec,
            output_path=os.path.join('plots', f'surrogate_{label}_vs_orig.gif'),
            title_fn=lambda t, s=si, l=label: f"#{s} ({l}) Original vs Surrogate — t={t}",
        )


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Results')

    svd_path  = os.path.join(results_dir, 'svd_train.npz')
    doe_path  = os.path.join(results_dir, 'doe_rotated.npy')
    ckpt_path = os.path.join('checkpoints', 'SVDSurrogate_best.pt')

    train(
        svd_path   = svd_path,
        doe_path   = doe_path,
        epochs     = 150,
        batch_size = 64,
        lr         = 1e-3,
        patience   = 50,
    )

    concentration_path = os.path.join(results_dir, 'ch4_rotated.npy')
    step = 5

    evaluate(svd_path, doe_path, concentration_path, ckpt_path, step=step, n_animate=3)
