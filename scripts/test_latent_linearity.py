"""
Test linéarité de l'espace latent sur ~100 couples aléatoires.

Pour chaque couple (idx1, idx2) :
  - θ_mid = 0.5·θ1 + 0.5·θ2
  - Voisin le plus proche de θ_mid dans le dataset (référence physique)
  - Interpolation latente : decode(0.5·z1 + 0.5·z2) + UNet
  - Erreur L2 rel vs voisin

Sorties :
  - plots/latent_linearity_hist.png  : histogramme des erreurs
  - plots/latent_linearity_XXX.png   : 3 exemples aléatoires (snapshot médian)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models.laplace_ae_surrogate import LaplaceAE, LaplaceLatentModel
from models.correction_ae import CorrectionAE
from utils.laplace import laplace_inverse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

alpha     = 0.5
n_couples = 100

# ── AE avec décodeur finetuné ─────────────────────────────────────────────────
ckpt = torch.load('checkpoints/LaplaceAE_best.pt', map_location=device)
N_ae, lat = ckpt['N'], ckpt['latent_dim']
ae = LaplaceAE(N=N_ae, latent_dim=lat, beta=ckpt.get('beta', 1.0),
               freq_L=ckpt.get('freq_L', 8))
ae.load_state_dict(ckpt['model_state'])

ft_ckpt = torch.load('checkpoints/LaplaceLatentModel_finetuned.pt',
                     map_location=device, weights_only=False)
ft_cfg  = {k: ft_ckpt[k] for k in ('N_freq','N_half','N','theta_dim','latent_dim','hidden_dim')}
llm = LaplaceLatentModel(**ft_cfg, freq_L=ckpt.get('freq_L', 8))
llm.load_state_dict(ft_ckpt['model_state'])
ae.decoder = llm.shared_decoder
ae.to(device).eval()

# ── UNet de correction ────────────────────────────────────────────────────────
corr_ckpt = torch.load('checkpoints/CorrectionAE_best.pt', map_location=device, weights_only=False)
unet = CorrectionAE(N=corr_ckpt['N'], base_ch=corr_ckpt['base_ch'])
unet.load_state_dict(corr_ckpt['model_state'])
unet.to(device).eval()
print(f"Modèles chargés : AE (N={N_ae}, lat={lat}) + décodeur finetuné + UNet")

# ── Cache Laplace N=128 + stats ───────────────────────────────────────────────
cache_lap = np.load('/Data/KAT/ch4_rotated_laplace_N128_g0.000_trap.npy', mmap_mode='r')
ns, K, _, _, _ = cache_lap.shape
Nt = 2 * (K - 1)
dt = 1.0

stats_path = sorted(Path('/Data/KAT').glob('ch4_rotated_stats_N128_*.pt'))[0]
saved = torch.load(str(stats_path), weights_only=True)
freq_mean = saved['target_mean'].to(device)
freq_std  = saved['target_std'].to(device)
print(f"Stats : {stats_path.name}")

# ── Paramètres θ ──────────────────────────────────────────────────────────────
doe = np.load('dataset/doe_rotated.npy')
theta_all = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)
theta_mean_s = theta_all.mean(0)
theta_std_s  = theta_all.std(0) + 1e-8
theta_n_all  = (theta_all - theta_mean_s) / theta_std_s

def norm(x_k, k):   return (x_k - freq_mean[k]) / freq_std[k]
def denorm(x_k, k): return x_k * freq_std[k].cpu() + freq_mean[k].cpu()

def load_sim(idx):
    return torch.from_numpy(cache_lap[idx].copy()).float()

def encode_decode_interp(ul1, ul2):
    """Interpolation latente fréquence par fréquence. Retourne (K,2,N,N)."""
    out = torch.zeros_like(ul1)
    with torch.no_grad():
        for k in range(K):
            fr = k / max(K - 1, 1)
            z1 = ae.encoder(norm(ul1[k].to(device), k).unsqueeze(0), fr)
            z2 = ae.encoder(norm(ul2[k].to(device), k).unsqueeze(0), fr)
            z  = alpha * z1 + (1 - alpha) * z2
            out[k] = denorm(ae.decoder(z, fr).squeeze(0).cpu(), k)
    return out

def to_phys(spec):
    K, _, N, _ = spec.shape
    Nt_r = 2 * (K - 1)
    s = spec.numpy()
    flat = np.zeros((N * N, Nt_r), dtype=complex)
    for k in range(K):
        flat[:, k] = s[k, 0].reshape(-1) + 1j * s[k, 1].reshape(-1)
    for k in range(1, K - 1):
        flat[:, Nt_r - k] = np.conj(flat[:, k])
    rec, _ = laplace_inverse(flat, dt=dt, Nt=Nt_r, rule='trap', gamma=0.0)
    return rec.T.reshape(Nt_r, N, N)

def apply_unet(U_np):
    frames = torch.from_numpy(U_np).float().to(device)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(frames), 64):
            chunks.append(unet(frames[i:i+64]))
    return torch.cat(chunks).cpu().numpy()

def l2rel(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-8)

# ── Calibrer le seuil de proximité ────────────────────────────────────────────
# Distance médiane entre voisins directs dans le dataset (baseline)
rng = np.random.default_rng(0)
sample_idx = rng.choice(ns, 500, replace=False)
nn_dists = []
for si in sample_idx:
    d = np.linalg.norm(theta_n_all - theta_n_all[si], axis=1)
    d[si] = np.inf
    nn_dists.append(d.min())
dist_threshold = np.percentile(nn_dists, 25)
print(f"Seuil de proximité (percentile 25% des distances NN) : {dist_threshold:.4f}")

# ── Boucle sur les couples — on pioche jusqu'à avoir n_couples valides ─────────
errors   = []
meta     = []
n_tried  = 0

print(f"\nRecherche de {n_couples} couples avec voisin proche (dist < {dist_threshold:.4f})…")
pbar = tqdm(total=n_couples)
while len(errors) < n_couples:
    idx1, idx2 = rng.choice(ns, 2, replace=False)
    n_tried += 1

    theta1 = theta_all[idx1]
    theta2 = theta_all[idx2]
    theta_mid_n = ((alpha*theta1 + (1-alpha)*theta2) - theta_mean_s) / theta_std_s

    # Voisin le plus proche de θ_mid
    dists = np.linalg.norm(theta_n_all - theta_mid_n, axis=1)
    dists[[idx1, idx2]] = np.inf
    idx_ref = int(np.argmin(dists))

    if dists[idx_ref] > dist_threshold:
        continue

    ul1  = load_sim(int(idx1))
    ul2  = load_sim(int(idx2))
    ulr  = load_sim(idx_ref)

    ul_interp = encode_decode_interp(ul1, ul2)

    U_interp = apply_unet(to_phys(ul_interp))
    U_ref    = apply_unet(to_phys(ulr))

    err = l2rel(U_interp, U_ref)
    errors.append(err)
    pbar.update(1)
    meta.append({'idx1': idx1, 'idx2': idx2, 'idx_ref': idx_ref,
                 'dist_ref': dists[idx_ref],
                 'theta1': theta1, 'theta2': theta2, 'theta_ref': theta_all[idx_ref],
                 'U1': apply_unet(to_phys(ul1)),
                 'U2': apply_unet(to_phys(ul2)),
                 'U_ref': U_ref, 'U_interp': U_interp, 'err': err})

pbar.close()
errors = np.array(errors)
print(f"\n{n_couples} couples valides trouvés en {n_tried} tentatives.")
print(f"Erreur L2 relative (interpolation latente vs voisin physique) :")
print(f"  Médiane : {np.median(errors):.4f}")
print(f"  Moyenne : {errors.mean():.4f}")
print(f"  Std     : {errors.std():.4f}")
print(f"  Min/Max : {errors.min():.4f} / {errors.max():.4f}")

# ── Histogramme ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(errors * 100, bins=20, edgecolor='white', color='steelblue')
ax.axvline(np.median(errors) * 100, color='red', linestyle='--',
           label=f'Médiane : {np.median(errors)*100:.1f}%')
ax.set_xlabel('Erreur L2 relative (%)', fontsize=12)
ax.set_ylabel('Nombre de couples', fontsize=12)
ax.set_title('Linéarité espace latent — interpolation latente vs référence physique\n'
             f'({n_couples} couples avec dist(θ_mid, voisin) < {dist_threshold:.3f}, α={alpha})', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('plots/latent_linearity_hist.png', dpi=150)
plt.close()
print("Sauvegardé : plots/latent_linearity_hist.png")

# ── 3 exemples aléatoires ─────────────────────────────────────────────────────
example_idx = rng.choice(n_couples, 3, replace=False)
t_snap = Nt // 2

for ex_i, ei in enumerate(example_idx):
    m = meta[ei]
    fields = [m['U1'], m['U2'], m['U_ref'], m['U_interp']]
    labels = [
        f"U1 (sim {m['idx1']})\nk={m['theta1'][0]:.2f} A={m['theta1'][1]:.0f} C={m['theta1'][2]:.3f}",
        f"U2 (sim {m['idx2']})\nk={m['theta2'][0]:.2f} A={m['theta2'][1]:.0f} C={m['theta2'][2]:.3f}",
        f"Référence (sim {m['idx_ref']})\nk={m['theta_ref'][0]:.2f} A={m['theta_ref'][1]:.0f} C={m['theta_ref'][2]:.3f}\n(voisin, dist={m['dist_ref']:.3f})",
        f"Interpolation latente\ndecode(α·z1+(1-α)·z2) + UNet\nL2 rel = {m['err']*100:.1f}%",
    ]

    vmin = min(f[t_snap].min() for f in fields)
    vmax = max(f[t_snap].max() for f in fields)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f'Exemple {ex_i+1} — t = {t_snap}', fontsize=12, fontweight='bold')
    for ax, label, arr in zip(axes, labels, fields):
        im = ax.imshow(arr[t_snap], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(label, fontsize=8, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout()
    out = f'plots/latent_linearity_ex{ex_i+1}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Sauvegardé : {out}")
