"""
stationary/eval_vae.py — Évaluation quantitative et qualitative du VAE
=======================================================================
Produit :
  plots/vae_reconstruction.png   — grille GT / reconstruction / erreur
  plots/vae_l2rel_hist.png       — histogramme des erreurs L2 relatives (test set)
  plots/vae_latent_pca.png       — PCA de l'espace latent colorié par theta
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.variationalAutoEncoder import VAE
from stationary.dataset import ConvDiffDataset


# ─────────────────────────────────────────────────────────────────────────────
CKPT_PATH    = 'checkpoints/VAE_best.pt'
DATASET_PATH = '/Data/KAT/dataset.npz'
SEED         = 42
N_SHOW       = 4       # lignes dans la grille de reconstruction
OUT_DIR      = Path('plots')
# ─────────────────────────────────────────────────────────────────────────────


def load_vae(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state']
    latent_dim = state['encoder.fc_mu.2.weight'].shape[0]
    N          = int(state['decoder.out_fc.3.weight'].shape[0] ** 0.5)
    model = VAE(N=N, latent_dim=latent_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt, N, latent_dim



def get_test_loader(ckpt, dataset_path, seed):
    from stationary.dataset import ConvDiffDataset
    dataset = ConvDiffDataset(dataset_path)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    dataset.fit(train_set.indices)
    loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    return loader, dataset


@torch.no_grad()
def compute_errors(model, loader, dataset, device):
    """Retourne erreurs L2 relatives (%) et les arrays pour visualisation."""
    l2rel_list = []
    U_list, Uhat_list, theta_list, z_list = [], [], [], []
    for theta_norm, U_norm in loader:
        theta_norm = theta_norm.to(device)
        U_norm     = U_norm.to(device)
        U_hat_norm, mu, _ = model(U_norm)

        U_phys     = dataset.denorm_U(U_norm.cpu())
        Uhat_phys  = dataset.denorm_U(U_hat_norm.cpu())
        theta_raw  = theta_norm.cpu() * dataset.theta_std.cpu() + dataset.theta_mean.cpu()

        num  = ((U_phys - Uhat_phys) ** 2).sum(dim=(1, 2, 3)).sqrt()
        den  = (U_phys ** 2).sum(dim=(1, 2, 3)).sqrt().clamp(min=1e-8)
        l2rel_list.append((num / den * 100).numpy())

        U_list.append(U_phys.numpy())
        Uhat_list.append(Uhat_phys.numpy())
        theta_list.append(theta_raw.numpy())
        z_list.append(mu.cpu().numpy())

    l2rel  = np.concatenate(l2rel_list)
    U_all  = np.concatenate(U_list)
    Uh_all = np.concatenate(Uhat_list)
    th_all = np.concatenate(theta_list)
    z_all  = np.concatenate(z_list)
    return l2rel, U_all, Uh_all, th_all, z_all


# ── Figure 1 : grille reconstruction ─────────────────────────────────────────
def plot_reconstruction(U_all, Uh_all, l2rel, n_show, out_path):
    # Trie par erreur croissante pour montrer des cas typiques
    order = np.argsort(l2rel)
    # Prend des quantiles réguliers pour couvrir le spectre
    idx = order[np.linspace(0, len(order) - 1, n_show, dtype=int)]

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for row, i in enumerate(idx):
        U   = U_all[i, 0]
        Uh  = Uh_all[i, 0]
        err = np.abs(U - Uh)
        vmin, vmax = U.min(), U.max()

        im0 = axes[row, 0].imshow(U,   origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        im1 = axes[row, 1].imshow(Uh,  origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axes[row, 2].imshow(err, origin='lower', cmap='Reds',    vmin=0)

        for ax in axes[row]: ax.axis('off')
        axes[row, 0].set_title('Ground truth',    fontsize=8)
        axes[row, 1].set_title(f'VAE recon  (L2rel={l2rel[i]:.1f}%)', fontsize=8)
        axes[row, 2].set_title('|Error|',         fontsize=8)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.suptitle('VAE reconstruction — test samples at selected error quantiles', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()


# ── Figure 2 : histogramme L2 relatif ────────────────────────────────────────
def plot_hist(l2rel, out_path):
    med = np.median(l2rel)
    mu  = np.mean(l2rel)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(l2rel, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(med, color='red',    linestyle='--', linewidth=1.5, label=f'Median {med:.1f}%')
    ax.axvline(mu,  color='orange', linestyle='--', linewidth=1.5, label=f'Mean {mu:.1f}%')
    ax.set_xlabel('Relative $L^2$ error (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('VAE — reconstruction error on test set', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()
    return med, mu


# ── Figure 3 : PCA espace latent ─────────────────────────────────────────────
def plot_latent_pca(z_all, th_all, out_path):
    from sklearn.decomposition import PCA
    pca  = PCA(n_components=2)
    Z2   = pca.fit_transform(z_all)
    var  = pca.explained_variance_ratio_ * 100

    param_names = ['D', '$b_x$', '$b_y$', '$f$']
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for k, (ax, name) in enumerate(zip(axes, param_names)):
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=th_all[:, k],
                        cmap='plasma', s=4, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=8)
    plt.suptitle('PCA of VAE latent space — coloured by physical parameters', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model, ckpt, N, latent_dim = load_vae(CKPT_PATH, device)
    print(f'VAE  N={N}  latent_dim={latent_dim}')

    loader, dataset = get_test_loader(ckpt, DATASET_PATH, SEED)
    print(f'Samples: {len(loader.dataset)}')

    l2rel, U_all, Uh_all, th_all, z_all = compute_errors(model, loader, dataset, device)

    print(f'\nL2rel — mean={l2rel.mean():.2f}%  median={np.median(l2rel):.2f}%  '
          f'std={l2rel.std():.2f}%  max={l2rel.max():.2f}%')

    plot_reconstruction(U_all, Uh_all, l2rel, N_SHOW, OUT_DIR / 'vae_reconstruction.png')
    med, mu = plot_hist(l2rel, OUT_DIR / 'vae_l2rel_hist.png')
    plot_latent_pca(z_all, th_all, OUT_DIR / 'vae_latent_pca.png')

    print('\nDone.')
