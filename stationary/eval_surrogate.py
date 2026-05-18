"""
stationary/eval_surrogate.py — Évaluation des surrogates theta → U
===================================================================
Produit :
  plots/surrogate_l2rel_hist.png   — histogrammes comparatifs (plusieurs modèles)
  plots/surrogate_reconstruction.png — grille GT / prédictions / erreurs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from stationary.dataset import ConvDiffDataset
from stationary.main import load_model

# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = '/Data/KAT/dataset.npz'
SEED         = 42
BATCH_SIZE   = 128
OUT_DIR      = Path('plots')

MODELS = [
    ('Surrogate  $d_z=32$, frozen',    'checkpoints/IndirectDecoder_best.pt'),
    ('Surrogate  $d_z=32$, end-to-end','checkpoints/finetune_IndirectDecoder_best.pt'),
    ('Surrogate  $d_z=3$,  frozen',    'checkpoints/smallLD_IndirectDecoder_best.pt'),
    ('Surrogate  $d_z=3$,  end-to-end','checkpoints/finetune_smallLD_IndirectDecoder_best.pt'),
]
# ─────────────────────────────────────────────────────────────────────────────


def get_test_set(ckpt, seed):
    dataset = ConvDiffDataset(DATASET_PATH)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    dataset.fit(train_set.indices)
    return dataset, test_set


@torch.no_grad()
def evaluate(model, ckpt, dataset, test_set, device):
    """Retourne l2rel (%), U_phys, Uhat_phys pour tous les échantillons test."""
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    theta_mean = ckpt['theta_mean'].to(device)
    theta_std  = ckpt['theta_std'].to(device)

    l2rel_list, U_list, Uh_list = [], [], []

    for theta_norm, U_norm in loader:
        theta_norm = theta_norm.to(device)
        U_norm     = U_norm.to(device)

        # Dénorm theta pour renorm avec les stats du surrogate
        theta_raw  = theta_norm.cpu() * dataset.theta_std + dataset.theta_mean
        theta_s    = ((theta_raw.to(device) - theta_mean) / theta_std)

        Uh_norm = model.generate(theta_s)                         # (B, 1, N, N)

        U_phys  = dataset.denorm_U(U_norm.cpu())
        Uh_phys = dataset.denorm_U(Uh_norm.cpu())

        num   = ((U_phys - Uh_phys) ** 2).sum(dim=(1,2,3)).sqrt()
        den   = (U_phys ** 2).sum(dim=(1,2,3)).sqrt().clamp(min=1e-8)
        l2rel_list.append((num / den * 100).numpy())
        U_list.append(U_phys.numpy())
        Uh_list.append(Uh_phys.numpy())

    return (np.concatenate(l2rel_list),
            np.concatenate(U_list),
            np.concatenate(Uh_list))


# ── Figure 1 : histogrammes comparatifs ──────────────────────────────────────
def plot_comparison_hist(results, vae_l2rel, out_path):
    """results : list of (label, l2rel_array)"""
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4), sharey=False)
    if len(results) == 1:
        axes = [axes]

    colors = ['steelblue', 'tomato', 'mediumseagreen', 'mediumpurple']

    for ax, (label, l2rel), color in zip(axes, results, colors):
        med = np.median(l2rel)
        mu  = np.mean(l2rel)
        ax.hist(l2rel, bins=40, range=(0, 100), color=color,
                edgecolor='white', alpha=0.85, label='Surrogate')
        ax.set_xlim(0, 100)
        ax.axvline(med, color='black',  linestyle='--', lw=1.5, label=f'Median {med:.1f}%')
        ax.axvline(mu,  color='grey',   linestyle=':',  lw=1.5, label=f'Mean {mu:.1f}%')
        # VAE floor
        vae_med = np.median(vae_l2rel)
        ax.axvline(vae_med, color='orange', linestyle='-', lw=1.5, label=f'VAE floor {vae_med:.1f}%')
        ax.set_xlabel('Relative $L^2$ error (%)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle('Surrogate $\\vartheta \\to U$ — test-set error distributions', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()


# ── Figure 2 : grille reconstruction (meilleur modèle) ───────────────────────
def plot_reconstruction(U_all, Uh_all, l2rel, label, n_show, out_path, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(l2rel), size=n_show, replace=False)

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for row, i in enumerate(idx):
        U   = U_all[i, 0]
        Uh  = Uh_all[i, 0]
        err = np.abs(U - Uh)
        vmin, vmax = U.min(), U.max()

        axes[row, 0].imshow(U,   origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(Uh,  origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axes[row, 2].imshow(err, origin='lower', cmap='Reds', vmin=0)
        for ax in axes[row]:
            ax.axis('off')
        axes[row, 0].set_title('Ground truth', fontsize=8)
        axes[row, 1].set_title(f'{label}  (L2rel={l2rel[i]:.1f}%)', fontsize=8)
        axes[row, 2].set_title('|Error|', fontsize=8)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Surrogate reconstruction — {label}', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Charge VAE floor depuis l'évaluation précédente si dispo
    vae_errors_path = OUT_DIR / 'vae_l2rel.npy'
    if vae_errors_path.exists():
        vae_l2rel = np.load(vae_errors_path)
        print(f'VAE floor chargé depuis cache ({len(vae_l2rel)} samples)')
    else:
        # Recalcul VAE floor
        from models.stationary.vae import VAE
        vae_ckpt  = torch.load('checkpoints/VAE_best.pt', map_location=device)
        state     = vae_ckpt['model_state']
        latent_dim = state['encoder.fc_mu.2.weight'].shape[0]
        N          = int(state['decoder.out_fc.3.weight'].shape[0] ** 0.5)
        vae_model  = VAE(N=N, latent_dim=latent_dim).to(device)
        vae_model.load_state_dict(state)
        vae_model.eval()

        tmp_ckpt = torch.load('checkpoints/IndirectDecoder_best.pt', map_location=device)
        tmp_dataset, tmp_test = get_test_set(tmp_ckpt, SEED)
        loader = DataLoader(tmp_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        l2_v = []
        with torch.no_grad():
            for _, U_norm in loader:
                U_norm = U_norm.to(device)
                Uh_norm, _, _ = vae_model(U_norm)
                U_phys  = tmp_dataset.denorm_U(U_norm.cpu())
                Uh_phys = tmp_dataset.denorm_U(Uh_norm.cpu())
                num = ((U_phys - Uh_phys)**2).sum(dim=(1,2,3)).sqrt()
                den = (U_phys**2).sum(dim=(1,2,3)).sqrt().clamp(min=1e-8)
                l2_v.append((num / den * 100).numpy())
        vae_l2rel = np.concatenate(l2_v)
        np.save(vae_errors_path, vae_l2rel)
        print(f'VAE floor  median={np.median(vae_l2rel):.2f}%  mean={vae_l2rel.mean():.2f}%')

    results = []
    best_label, best_l2rel, best_U, best_Uh = None, None, None, None
    best_median = float('inf')

    for label, ckpt_path in MODELS:
        print(f'\n── {label}')
        model, ckpt = load_model(ckpt_path, device)
        dataset, test_set = get_test_set(ckpt, SEED)

        l2rel, U_all, Uh_all = evaluate(model, ckpt, dataset, test_set, device)
        med = float(np.median(l2rel))
        print(f'   median={med:.2f}%  mean={l2rel.mean():.2f}%  std={l2rel.std():.2f}%  max={l2rel.max():.2f}%')
        results.append((label, l2rel))

        if med < best_median:
            best_median = med
            best_label, best_l2rel = label, l2rel
            best_U, best_Uh = U_all, Uh_all

    plot_comparison_hist(results, vae_l2rel, OUT_DIR / 'surrogate_l2rel_hist.png')
    plot_reconstruction(best_U, best_Uh, best_l2rel, best_label, n_show=3,
                        out_path=OUT_DIR / 'surrogate_reconstruction.png')
    print('\nDone.')
