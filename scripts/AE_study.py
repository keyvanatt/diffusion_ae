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
from models.transient.laplace_ae import LaplaceAE
from utils.laplace import laplace_inverse_tik
from utils.animate import animate_comparaison

# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = "/Data/KAT/ch4_rotated.npy"
ae_path   = "checkpoints/LaplaceAE_best.pt"

N         = 128
latent_dim = 64
beta       = 1e-3
n_samples  = 64   # Nombre de samples pour l'analyse

# ============================================================
# Paramètres de la transformée de Laplace
# Copier depuis les logs wandb du Stage 1 du pipeline.
# ============================================================

alpha_t = 0.0068868
lam     = 0.000029221
rule    = 'trap'
dt      = 1.0

# s_list : array 1D complex128 issu du Stage 1.
# Si None → rfftfreq tronqué à k_max (cas simple, non optimisé).
k_max  = 20
s_list = np.array([
    0.0100+0.0000j, 0.0085+0.0414j, 0.0085+0.0827j, 0.0086+0.1236j,
    0.0088+0.1650j, 0.0087+0.2062j, 0.0088+0.2476j, 0.0089+0.2894j,
    0.0092+0.3325j, 0.0095+0.3780j, 0.0097+0.4269j, 0.0096+0.4785j,
    0.0099+0.5317j, 0.0101+0.5858j, 0.0102+0.6423j, 0.0102+0.7009j,
    0.0103+0.7606j, 0.0103+0.8212j, 0.0104+0.8815j, 0.0099+0.9353j,
], dtype=np.complex128)
# Pour utiliser rfftfreq simple à la place, décommenter :
# _u_tmp = np.load(data_path, mmap_mode='r'); Nt_data = _u_tmp.shape[1]; del _u_tmp
# s_list = (1j * 2 * np.pi * np.fft.rfftfreq(Nt_data, d=dt))[:k_max + 1]

K = len(s_list)
print(f"K={K}  alpha_t={alpha_t:.6f}  lam={lam:.5f}  rule={rule}")

# ============================================================
# Charger le dataset
# ============================================================

print("Chargement du dataset...")
dataset = TransientDataset(data_path, laplace=True, s_list=s_list, rule=rule,
                           interp_size=N, dt=dt)

# Créer les indices (alignés sur le split du pipeline si dispo)
_split_path = "dataset/split.npz"
import os as _os
if _os.path.exists(_split_path):
    _split   = np.load(_split_path)
    test_idx = _split['test_idx']
    non_test = [i for i in range(len(dataset)) if i not in set(test_idx.tolist())]
    perm     = torch.randperm(len(non_test))
    n_train  = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
else:
    idx       = torch.randperm(len(dataset))
    n_train   = int(0.8 * len(dataset))
    n_val     = int(0.1 * len(dataset))
    train_idx = idx[:n_train].tolist()
    test_idx  = idx[n_train + n_val:].numpy()

dataset.fit(train_idx)

print(f"Dataset: ns={len(dataset)}, K={K}, N={N}")

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

print(f"\nCalcul de l'erreur AE fréquence par fréquence (K={K})...")
U_all = dataset[batch_idx][1].float().to(device)  # (B, K, 2, N, N)

ae_l2rel_losses_re = []
ae_l2rel_losses_im = []
for k in tqdm(range(K)):
    U = U_all[:, k]  # (B, 2, N, N)
    freq_ratio = torch.full((U.shape[0],), k / (K - 1), device=device)

    with torch.no_grad():
        U_hat, z = ae_model(U, freq_ratio)

        # Erreur L2 relative — Real part
        diff_re = torch.norm((U[:, 0] - U_hat[:, 0]).flatten(1), dim=1)
        norm_u_re = torch.norm(U[:, 0].flatten(1), dim=1)
        # Masquer les samples avec très petite norme pour éviter division par zéro
        valid_re = norm_u_re > 1e-6
        if valid_re.sum() > 0:
            l2rel_re = (diff_re[valid_re] / (norm_u_re[valid_re] + 1e-12)).mean()
        else:
            l2rel_re = torch.tensor(0.0, device=device)

        # Erreur L2 relative — Imaginary part
        diff_im = torch.norm((U[:, 1] - U_hat[:, 1]).flatten(1), dim=1)
        norm_u_im = torch.norm(U[:, 1].flatten(1), dim=1)
        # Masquer les samples avec très petite norme pour éviter division par zéro
        valid_im = norm_u_im > 1e-6
        if valid_im.sum() > 0:
            l2rel_im = (diff_im[valid_im] / (norm_u_im[valid_im] + 1e-12)).mean()
        else:
            l2rel_im = torch.tensor(0.0, device=device)

    ae_l2rel_losses_re.append(l2rel_re.item())
    ae_l2rel_losses_im.append(l2rel_im.item())

ae_l2rel_losses_re = np.array(ae_l2rel_losses_re)
ae_l2rel_losses_im = np.array(ae_l2rel_losses_im)
print(f"AE (Real)      — Mean L2rel: {ae_l2rel_losses_re.mean():.2%}  Std: {ae_l2rel_losses_re.std():.2%}")
print(f"AE (Imaginary) — Mean L2rel: {ae_l2rel_losses_im.mean():.2%}  Std: {ae_l2rel_losses_im.std():.2%}")

# ============================================================
# Calcul de l'erreur L2rel dans le domaine temporel (reconstruction complète)
# ============================================================
# Prédiction AE sur toutes les K fréquences (dénormalisée).
# Référence : spectre vrai → laplace_inverse_tik avec la s_list optimale.
# ============================================================

print(f"\nReconstruction temporelle via laplace_inverse_tik (K={K}, alpha_t={alpha_t:.4f}, lam={lam:.4f})...")

Nt = dataset.Nt   # 150
B  = n_samples
NN = N * N

target_std_cpu  = dataset.target_std.cpu()   # (K, 2, N, N)
target_mean_cpu = dataset.target_mean.cpu()  # (K, 2, N, N)

M_pred = np.zeros((B, NN, K), dtype=np.complex64)
M_true = np.zeros((B, NN, K), dtype=np.complex64)

# Prédiction AE sur toutes les K fréquences + dénormalisation
for k in tqdm(range(K), desc="AE forward"):
    U_k = U_all[:, k]   # (B, 2, N, N) normalisé
    freq_ratio = torch.full((B,), k / (K - 1), device=device)
    with torch.no_grad():
        U_k_pred, _ = ae_model(U_k, freq_ratio)   # (B, 2, N, N) normalisé
    tm_k = target_mean_cpu[k]   # (2, N, N)
    ts_k = target_std_cpu[k]
    U_k_phys = (U_k_pred.cpu() * ts_k[None] + tm_k[None]).numpy()   # (B, 2, N, N)
    M_pred[:, :, k] = (U_k_phys[:, 0] + 1j * U_k_phys[:, 1]).reshape(B, NN)

# Spectre de référence dénormalisé
U_all_np = U_all.cpu().numpy()   # (B, K, 2, N, N)
for k in range(K):
    tm_k = target_mean_cpu[k].numpy()   # (2, N, N)
    ts_k = target_std_cpu[k].numpy()
    U_k_true = U_all_np[:, k]   # (B, 2, N, N) normalisé
    U_k_phys = U_k_true * ts_k[None] + tm_k[None]   # (B, 2, N, N)
    M_true[:, :, k] = (U_k_phys[:, 0] + 1j * U_k_phys[:, 1]).reshape(B, NN)

# Inversion Laplace via laplace_inverse_tik (gère la symétrie conjuguée)
s_torch = torch.tensor(s_list, dtype=torch.complex128)

anim_indices = set(np.random.choice(B, size=3, replace=False).tolist())
saved_pred_t = {}
saved_true_t = {}

l2rel_temporal_list = []
for b in tqdm(range(B), desc="Inversion Laplace"):
    U_pred_t = laplace_inverse_tik(
        torch.tensor(M_pred[b], dtype=torch.complex128),
        s_torch, dt=dt, Nt=Nt, alpha_t=alpha_t, lam=lam, rule=rule,
    ).numpy()   # (N², Nt)
    U_true_t = laplace_inverse_tik(
        torch.tensor(M_true[b], dtype=torch.complex128),
        s_torch, dt=dt, Nt=Nt, alpha_t=alpha_t, lam=lam, rule=rule,
    ).numpy()
    # (N², Nt) → (Nt, N, N)
    U_pred_t = U_pred_t.reshape(N, N, Nt).transpose(2, 0, 1)
    U_true_t = U_true_t.reshape(N, N, Nt).transpose(2, 0, 1)
    diff = np.linalg.norm(U_pred_t - U_true_t)
    norm_true = np.linalg.norm(U_true_t) + 1e-12
    l2rel_temporal_list.append(diff / norm_true)
    if b in anim_indices:
        saved_pred_t[b] = U_pred_t.copy()
        saved_true_t[b] = U_true_t.copy()

l2rel_temporal_array = np.array(l2rel_temporal_list)
print(f"Reconstruction temporelle (AE k≤{k_max}, laplace_inverse_tik):")
print(f"  Mean L2rel    : {l2rel_temporal_array.mean():.2%}")
print(f"  Median L2rel  : {np.median(l2rel_temporal_array):.2%}")
print(f"  Std L2rel     : {l2rel_temporal_array.std():.2%}")
print(f"  Min / Max     : {l2rel_temporal_array.min():.2%} / {l2rel_temporal_array.max():.2%}")

# ============================================================
# Figure 1 : Erreur fréquence par fréquence
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

k_freq = np.arange(K)
ax.plot(k_freq, ae_l2rel_losses_re * 100, 'o-', label='Re(Û_k) reconstruction error', linewidth=2, markersize=6, color='steelblue')
ax.plot(k_freq, ae_l2rel_losses_im * 100, 's-', label='Im(Û_k) reconstruction error', linewidth=2, markersize=6, color='coral')

ax.set_xlabel('Frequency index k', fontsize=12)
ax.set_ylabel('L2 Relative Error (%)', fontsize=12)
ax.set_title('Autoencoder reconstruction error per frequency (Re and Im components)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('plots/ae_study_frequency_error.png', dpi=150, bbox_inches='tight')
print("✓ Sauvegardé : plots/ae_study_frequency_error.png")
plt.close()

# ============================================================
# Figure 1b : Histogramme L2rel domaine temporel
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(l2rel_temporal_array * 100, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlim(0, 100)
ax.set_xlabel('L2 Relative Error (%)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(
    f'AE reconstruction error — temporal domain\n'
    f'(K={K}, alpha_t={alpha_t:.4f}, lam={lam:.4f},  '
    f'mean={l2rel_temporal_array.mean()*100:.2f}%)',
    fontsize=13, fontweight='bold',
)
ax.axvline(l2rel_temporal_array.mean() * 100, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {l2rel_temporal_array.mean()*100:.2f}%')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plots/ae_study_temporal_error.png', dpi=150, bbox_inches='tight')
print("✓ Sauvegardé : plots/ae_study_temporal_error.png")
plt.close()

# ============================================================
# Figure 2-6 : Visualisations de quelques fréquences
# ============================================================

selected_freqs = np.linspace(0, K - 1, min(5, K), dtype=int).tolist()

for k in selected_freqs:
    sample_idx = 0  # Premier sample

    U = U_all[sample_idx, k].cpu()  # (2, N, N)

    # Prédiction AE
    with torch.no_grad():
        U_ae_input = U.unsqueeze(0).to(device)
        freq_ratio_batch = torch.full((1,), k / (K - 1), device=device)
        U_ae_pred, _ = ae_model(U_ae_input, freq_ratio_batch)
        U_ae_pred = U_ae_pred.squeeze(0).cpu()

    # Calculer les erreurs
    l2_ae = (torch.norm(U - U_ae_pred) / (torch.norm(U) + 1e-12)).item()

    # Créer la figure : 3 colonnes (Vrai, Prédiction, Erreur) x 2 lignes (Re, Im)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Frequency k={k}  |  AE L2rel={l2_ae:.2%}', fontsize=14, fontweight='bold')

    # Row 0: Real part
    im0 = axes[0, 0].imshow(U[0], cmap='RdBu_r')
    axes[0, 0].set_title('True Re(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(U_ae_pred[0], cmap='RdBu_r')
    axes[0, 1].set_title('Pred Re(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(U[0] - U_ae_pred[0], cmap='RdBu_r')
    axes[0, 2].set_title('Error Re(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 1: Imaginary part
    im3 = axes[1, 0].imshow(U[1], cmap='RdBu_r')
    axes[1, 0].set_title('True Im(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(U_ae_pred[1], cmap='RdBu_r')
    axes[1, 1].set_title('Pred Im(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(U[1] - U_ae_pred[1], cmap='RdBu_r')
    axes[1, 2].set_title('Error Im(Û_k)', fontsize=11, fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 2])

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
print(f"RÉSUMÉ — AUTOENCODER LAPLACE  K={K}  alpha_t={alpha_t:.4f}  lam={lam:.4f}")
print("="*70)

print(f"\nErreurs par fréquence — REAL PART:")
print(f"  Mean L2rel    : {ae_l2rel_losses_re.mean():.2%}")
print(f"  Median L2rel  : {np.median(ae_l2rel_losses_re):.2%}")
print(f"  Std L2rel     : {ae_l2rel_losses_re.std():.2%}")
print(f"  Min L2rel     : {ae_l2rel_losses_re.min():.2%}  (freq k={ae_l2rel_losses_re.argmin()})")
print(f"  Max L2rel     : {ae_l2rel_losses_re.max():.2%}  (freq k={ae_l2rel_losses_re.argmax()})")

print(f"\nErreurs par fréquence — IMAGINARY PART:")
print(f"  Mean L2rel    : {ae_l2rel_losses_im.mean():.2%}")
print(f"  Median L2rel  : {np.median(ae_l2rel_losses_im):.2%}")
print(f"  Std L2rel     : {ae_l2rel_losses_im.std():.2%}")
print(f"  Min L2rel     : {ae_l2rel_losses_im.min():.2%}  (freq k={ae_l2rel_losses_im.argmin()})")
print(f"  Max L2rel     : {ae_l2rel_losses_im.max():.2%}  (freq k={ae_l2rel_losses_im.argmax()})")

# Regrouper par tiers de fréquence
t1, t2 = K // 3, 2 * K // 3
print(f"\nPar bande de fréquence (Real):")
print(f"  Basses  (k<{t1})      : mean={ae_l2rel_losses_re[:t1].mean():.2%}")
print(f"  Moyennes ({t1}≤k<{t2}) : mean={ae_l2rel_losses_re[t1:t2].mean():.2%}")
print(f"  Hautes  (k≥{t2})      : mean={ae_l2rel_losses_re[t2:].mean():.2%}")

print(f"\nPar bande de fréquence (Imaginary):")
print(f"  Basses  (k<{t1})      : mean={ae_l2rel_losses_im[:t1].mean():.2%}")
print(f"  Moyennes ({t1}≤k<{t2}) : mean={ae_l2rel_losses_im[t1:t2].mean():.2%}")
print(f"  Hautes  (k≥{t2})      : mean={ae_l2rel_losses_im[t2:].mean():.2%}")

print(f"\nERREUR DE RECONSTRUCTION TEMPORELLE (K={K}, laplace_inverse_tik):")
print(f"  Mean L2rel      : {l2rel_temporal_array.mean():.2%}")
print(f"  Median L2rel    : {np.median(l2rel_temporal_array):.2%}")
print(f"  Std L2rel       : {l2rel_temporal_array.std():.2%}")
print(f"  Min L2rel       : {l2rel_temporal_array.min():.2%}")
print(f"  Max L2rel       : {l2rel_temporal_array.max():.2%}")
print(f"  (Référence : spectre vrai → laplace_inverse_tik, grille {N}×{N})")

# ============================================================
# Animations de comparaison (3 samples)
# ============================================================

import os
os.makedirs('plots', exist_ok=True)

for i, b in enumerate(sorted(saved_pred_t)):
    l2 = l2rel_temporal_list[b]
    sim_id = int(batch_idx[b])
    anim_path = f'plots/ae_study_anim_{i}.gif'
    print(f"Animation {i+1}/3 — sample #{sim_id} (L2rel={l2*100:.1f}%)...")
    animate_comparaison(
        saved_true_t[b], saved_pred_t[b],
        output_path=anim_path,
        title_fn=lambda t, s=sim_id, e=l2: f"#{s}  L2rel={e*100:.1f}%  t={t}",
        title_a="Ground Truth",
        title_b="AE Reconstruction",
        title_err="|Error|",
        label="CH4 Concentration",
    )

print("\n" + "="*70)
