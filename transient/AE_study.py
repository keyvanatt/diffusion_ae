"""
AE_study.py — Analyse de l'erreur fréquence par fréquence de l'autoencoder Laplace
====================================================================================
Reprend le code de model_study.ipynb et ajoute :
- Calcul de l'erreur L2rel pour chaque fréquence
- Visualisations de quelques fréquences
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from transient.dataset import TransientDataset
from models.laplace_ae_surrogate import LaplaceAE

# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = "/Data/KAT/ch4_rotated.npy"
ae_path = "checkpoints/LaplaceAE_best.pt"

N = 128
latent_dim = 64
beta = 1e-3
n_samples = 64  # Nombre de samples pour l'analyse

# ============================================================
# Charger le dataset
# ============================================================

print("Chargement du dataset...")
dataset = TransientDataset(data_path, laplace=True, gamma=0, rule="trap",
                          interp_size=N, dt=1)

# Créer les indices
idx = torch.randperm(len(dataset))
n_train = int(0.8 * len(dataset))
n_val = int(0.1 * len(dataset))
train_idx = idx[:n_train].tolist()
test_idx = idx[n_train + n_val:].numpy()

dataset.fit(train_idx)

Nt_half = 76
k_max = 20  # L'AE a été entraîné avec k_max=20
print(f"Dataset: ns={len(dataset)}, Nt_half={Nt_half}, N={N}, k_max={k_max}")

# ============================================================
# Charger l'autoencoder
# ============================================================

print("\nChargement de l'autoencoder...")
ae_model = LaplaceAE(N=N, latent_dim=latent_dim, beta=beta).to(device)
ae_checkpoint = torch.load(ae_path, map_location=device)
ae_model.load_state_dict(ae_checkpoint['model_state'])
ae_model.eval()
print(f"AE chargé avec succès")

# ============================================================
# Sélectionner les samples de test
# ============================================================

batch_idx = np.random.choice(test_idx, size=n_samples, replace=False)
print(f"\nUtilisation de {n_samples} samples de test")

# ============================================================
# Calcul de l'erreur AE fréquence par fréquence
# ============================================================

print(f"\nCalcul de l'erreur AE fréquence par fréquence (k ≤ {k_max})...")
U_all = dataset[batch_idx][1].float().to(device)  # (B, Nt_half, 2, N, N)

ae_l2rel_losses = []
for k in tqdm(range(k_max + 1)):
    U = U_all[:, k]  # (B, 2, N, N)
    freq_ratio = torch.full((U.shape[0],), k / (Nt_half - 1), device=device)

    with torch.no_grad():
        U_hat, z = ae_model(U, freq_ratio)
        # Erreur L2 relative
        diff = torch.norm((U - U_hat).flatten(1), dim=1)
        norm_u = torch.norm(U.flatten(1), dim=1)
        l2rel = (diff / (norm_u + 1e-12)).mean()

    ae_l2rel_losses.append(l2rel.item())

ae_l2rel_losses = np.array(ae_l2rel_losses)
print(f"AE — Mean L2rel: {ae_l2rel_losses.mean():.2%}  Std: {ae_l2rel_losses.std():.2%}")

# ============================================================
# Figure 1 : Erreur fréquence par fréquence
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

k_freq = np.arange(k_max + 1)
ax.plot(k_freq, ae_l2rel_losses * 100, 'o-', label='AE reconstruction error', linewidth=2, markersize=6)

ax.set_xlabel('Frequency index k', fontsize=12)
ax.set_ylabel('L2 Relative Error (%)', fontsize=12)
ax.set_title('Autoencoder reconstruction error per frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('plots/ae_study_frequency_error.png', dpi=150, bbox_inches='tight')
print("\n✓ Sauvegardé : plots/ae_study_frequency_error.png")
plt.close()

# ============================================================
# Figure 2-6 : Visualisations de quelques fréquences
# ============================================================

selected_freqs = [0, 5, 10, 15, 20]  # Fréquences uniformément réparties dans [0, k_max]

for k in selected_freqs:
    sample_idx = 0  # Premier sample

    U = U_all[sample_idx, k].cpu()  # (2, N, N)

    # Prédiction AE
    with torch.no_grad():
        U_ae_input = U.unsqueeze(0).to(device)
        freq_ratio_batch = torch.full((1,), k / (Nt_half - 1), device=device)
        U_ae_pred, _ = ae_model(U_ae_input, freq_ratio_batch)
        U_ae_pred = U_ae_pred.squeeze(0).cpu()

    # Calculer les erreurs
    l2_ae = (torch.norm(U - U_ae_pred) / (torch.norm(U) + 1e-12)).item()

    # Créer la figure : Vrai Re, Vrai Im, Erreur Re, Erreur Im
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Frequency k={k}  |  AE L2rel={l2_ae:.2%}', fontsize=14, fontweight='bold')

    # Vrai champ
    im0 = axes[0, 0].imshow(U[0], cmap='RdBu_r')
    axes[0, 0].set_title('True Re(Û_k)', fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(U[1], cmap='RdBu_r')
    axes[0, 1].set_title('True Im(Û_k)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1])

    # Erreur AE
    im2 = axes[1, 0].imshow(U[0] - U_ae_pred[0], cmap='RdBu_r')
    axes[1, 0].set_title('Error Re(Û_k)', fontsize=11)
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(U[1] - U_ae_pred[1], cmap='RdBu_r')
    axes[1, 1].set_title('Error Im(Û_k)', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'plots/ae_study_freq_{k:02d}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Sauvegardé : plots/ae_study_freq_{k:02d}.png")
    plt.close()

# ============================================================
# Résumé et statistiques
# ============================================================

print("\n" + "="*70)
print("RÉSUMÉ — AUTOENCODER LAPLACE (k ≤ 20)")
print("="*70)

print(f"\nErroreurs par fréquence (k_max={k_max}):")
print(f"  Mean L2rel    : {ae_l2rel_losses.mean():.2%}")
print(f"  Median L2rel  : {np.median(ae_l2rel_losses):.2%}")
print(f"  Std L2rel     : {ae_l2rel_losses.std():.2%}")
print(f"  Min L2rel     : {ae_l2rel_losses.min():.2%}  (freq k={ae_l2rel_losses.argmin()})")
print(f"  Max L2rel     : {ae_l2rel_losses.max():.2%}  (freq k={ae_l2rel_losses.argmax()})")

# Regrouper par bandes de fréquence
low_freq = ae_l2rel_losses[:7]    # k=0-6
mid_freq = ae_l2rel_losses[7:14]  # k=7-13
high_freq = ae_l2rel_losses[14:]  # k=14-20

print(f"\nPar bande de fréquence:")
print(f"  Basses (k<7)    : mean={low_freq.mean():.2%}")
print(f"  Moyennes (7≤k<14) : mean={mid_freq.mean():.2%}")
print(f"  Hautes (k≥14)   : mean={high_freq.mean():.2%}")

print("\n" + "="*70)
