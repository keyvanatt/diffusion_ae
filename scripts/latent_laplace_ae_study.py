"""
latent_laplace_ae_study.py — Analyse du LatentLaplaceAE
========================================================
Évalue la qualité de reconstruction du LatentLaplaceAE sur le test set.

Métriques :
  - L2rel temporelle par simulation (espace normalisé)
  - Roundtrip latent : ||z_rec - z|| / ||z|| (qualité Laplace dans le latent)
  - Profil d'erreur temporelle : erreur moyenne par pas de temps

Figures :
  - Histogramme L2rel temporal
  - Profil d'erreur temporelle (mean ± std)
  - Positions des s_k appris dans le plan complexe
  - Comparaisons frame-par-frame (5 instants clés)
  - Animations GIF (3 samples)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from models.transient.latent_laplace_ae import LatentLaplaceAE
from transient.dataset import TransientDataset
from utils.animate import animate_comparaison


# ============================================================
# Chargement checkpoint + modèle
# ============================================================

def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = LatentLaplaceAE(
        N          = ckpt['N'],
        Nt         = ckpt['Nt'],
        latent_dim = ckpt['latent_dim'],
        K          = ckpt['K'],
        dt         = ckpt['dt'],
        beta       = ckpt['beta'],
        beta_latent= ckpt['beta_latent'],
        gamma_init = ckpt['gamma_init'],
        time_L     = ckpt.get('time_L', 8),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt


# ============================================================
# Forward batché sur le test set
# ============================================================

@torch.no_grad()
def run_test(model, dataset, test_idx, N, U_mean, U_std, device, batch_size=8):
    """
    Retourne :
      U_true_norm : (ns, Nt, N, N)  séquences normalisées
      U_rec_norm  : (ns, Nt, N, N)  reconstructions normalisées
      z_arr       : (ns, Nt, D)     latents encodés
      z_rec_arr   : (ns, Nt, D)     latents reconstruits
    """
    ns = len(test_idx)
    Nt = model.Nt
    D  = model.latent_dim

    U_true_norm = np.zeros((ns, Nt, N, N), dtype=np.float32)
    U_rec_norm  = np.zeros((ns, Nt, N, N), dtype=np.float32)
    z_arr       = np.zeros((ns, Nt, D),    dtype=np.float32)
    z_rec_arr   = np.zeros((ns, Nt, D),    dtype=np.float32)

    for start in tqdm(range(0, ns, batch_size), desc='Inférence test'):
        end   = min(start + batch_size, ns)
        batch_idx = test_idx[start:end]
        B     = end - start

        # Charger + normaliser
        seqs = []
        for idx in batch_idx:
            _, U = dataset[idx]
            if isinstance(U, np.ndarray):
                U = torch.from_numpy(U.copy()).float()
            if U.shape[-1] != N:
                U = F.interpolate(U.unsqueeze(0), size=(N, N),
                                  mode='bilinear', align_corners=False).squeeze(0)
            seqs.append(U)
        U_batch = torch.stack(seqs).to(device)               # (B, Nt, N, N) raw
        U_norm  = (U_batch - U_mean.to(device)) / U_std.to(device)

        U_rec, z_hat, z, z_rec = model(U_norm)

        U_true_norm[start:end] = U_norm.cpu().numpy()
        U_rec_norm [start:end] = U_rec.cpu().numpy()
        z_arr      [start:end] = z.cpu().numpy()
        z_rec_arr  [start:end] = z_rec.cpu().numpy()

    return U_true_norm, U_rec_norm, z_arr, z_rec_arr


# ============================================================
# Métriques
# ============================================================

def l2rel_per_sample(U_true, U_rec):
    """(ns, Nt, N, N) → (ns,)"""
    diff = (U_rec - U_true).reshape(len(U_true), -1)
    norm = U_true.reshape(len(U_true), -1)
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(norm, axis=1) + 1e-8)


def l2rel_latent(z, z_rec):
    """(ns, Nt, D) → (ns,)"""
    diff = (z_rec - z).reshape(len(z), -1)
    norm = z.reshape(len(z), -1)
    return np.linalg.norm(diff, axis=1) / (np.linalg.norm(norm, axis=1) + 1e-8)


def temporal_error_profile(U_true, U_rec):
    """
    Erreur L2rel frame-par-frame moyennée sur les samples.
    → (Nt,) : mean L2rel à chaque t, (Nt,) : std
    """
    # ||U_rec_t - U_true_t||_F / ||U_true_t||_F  pour chaque (n, t)
    diff   = (U_rec - U_true).reshape(U_true.shape[0], U_true.shape[1], -1)
    norm   = U_true.reshape(U_true.shape[0], U_true.shape[1], -1)
    l2_nt  = np.linalg.norm(diff, axis=2) / (np.linalg.norm(norm, axis=2) + 1e-8)
    return l2_nt.mean(axis=0), l2_nt.std(axis=0)


# ============================================================
# Figures
# ============================================================

def fig_histogram(l2rel_arr, ckpt_path, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(l2rel_arr * 100, bins=25, edgecolor='black', alpha=0.75, color='steelblue')
    ax.axvline(l2rel_arr.mean() * 100, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {l2rel_arr.mean()*100:.2f}%')
    ax.axvline(np.median(l2rel_arr) * 100, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(l2rel_arr)*100:.2f}%')
    ax.set_xlabel('L2 Relative Error (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        f'LatentLaplaceAE — Reconstruction temporelle\n'
        f'({os.path.basename(ckpt_path)}  |  n={len(l2rel_arr)})',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def fig_temporal_profile(mean_t, std_t, Nt, dt, save_path):
    t = np.arange(Nt) * dt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_t * 100, color='steelblue', linewidth=2, label='Mean L2rel')
    ax.fill_between(t, (mean_t - std_t) * 100, (mean_t + std_t) * 100,
                    alpha=0.25, color='steelblue', label='±1 std')
    ax.set_xlabel('Time step t', fontsize=12)
    ax.set_ylabel('L2 Relative Error (%)', fontsize=12)
    ax.set_title('Profil d\'erreur temporelle — LatentLaplaceAE', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def fig_s_scatter(model, save_path):
    s = model.laplace.s_list.detach().cpu()
    re = s.real.numpy()
    im = s.im.numpy() if hasattr(s, 'im') else s.imag.numpy()
    K  = len(re)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(re, im, c=np.arange(K), cmap='viridis', s=80, zorder=3)
    for k in range(K):
        ax.annotate(str(k), (re[k], im[k]), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    plt.colorbar(sc, ax=ax, label='Indice k')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Re(s_k)  [γ appris]', fontsize=12)
    ax.set_ylabel('Im(s_k)  [ω_k appris]', fontsize=12)
    ax.set_title(f'Fréquences de Laplace apprises (K={K})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    alpha_t = model.laplace.log_alpha_t.exp().item()
    lam     = model.laplace.log_lam.exp().item()
    ax.set_title(
        f'Fréquences de Laplace apprises (K={K})\n'
        f'α_t={alpha_t:.4f}  λ={lam:.6f}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


def fig_frames(U_true_norm, U_rec_norm, U_mean, U_std, sample_idx, Nt, save_path):
    """Comparaison vrai / pred / erreur à 5 instants clés — valeurs physiques."""
    U_m = U_mean.numpy() if isinstance(U_mean, torch.Tensor) else U_mean
    U_s = U_std.numpy()  if isinstance(U_std,  torch.Tensor) else U_std

    U_true = U_true_norm[sample_idx] * U_s + U_m    # (Nt, N, N) physique
    U_rec  = U_rec_norm [sample_idx] * U_s + U_m

    t_steps = np.linspace(0, Nt - 1, 5, dtype=int).tolist()
    fig = plt.figure(figsize=(5 * len(t_steps), 10))
    gs  = gridspec.GridSpec(3, len(t_steps), hspace=0.35, wspace=0.25)

    vmin = min(U_true[t].min() for t in t_steps)
    vmax = max(U_true[t].max() for t in t_steps)

    for col, t in enumerate(t_steps):
        err = U_rec[t] - U_true[t]
        ax0 = fig.add_subplot(gs[0, col])
        ax1 = fig.add_subplot(gs[1, col])
        ax2 = fig.add_subplot(gs[2, col])

        im0 = ax0.imshow(U_true[t], cmap='viridis', vmin=vmin, vmax=vmax)
        im1 = ax1.imshow(U_rec[t],  cmap='viridis', vmin=vmin, vmax=vmax)
        emax = np.abs(err).max()
        im2 = ax2.imshow(err, cmap='RdBu_r', vmin=-emax, vmax=emax)

        plt.colorbar(im0, ax=ax0, fraction=0.046)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax0.set_title(f't={t}  True',  fontsize=10, fontweight='bold')
        ax1.set_title(f't={t}  Pred',  fontsize=10)
        ax2.set_title(f't={t}  Error', fontsize=10)
        for ax in (ax0, ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])

    l2 = np.linalg.norm(U_rec - U_true) / (np.linalg.norm(U_true) + 1e-8)
    fig.suptitle(f'Sample #{sample_idx}  —  L2rel={l2*100:.2f}%  (physique)',
                 fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'✓ {save_path}')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    data_path  = os.path.join('dataset', 'ch4_rotated.npy')
    ckpt_path  = os.path.join('checkpoints', 'LatentLaplaceAE_best.pt')
    plots_dir  = 'plots'
    n_samples  = 64     # samples de test à évaluer (None = tous)
    batch_size = 8
    n_anim     = 3      # animations GIF

    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # --- Modèle ---
    model, ckpt = load_model(ckpt_path, device)
    N          = ckpt['N']
    Nt         = ckpt['Nt']
    dt         = ckpt['dt']
    K          = ckpt['K']
    U_mean     = ckpt['U_mean']   # (N, N)
    U_std      = ckpt['U_std']    # (N, N)
    test_idx   = np.array(ckpt['test_idx'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f'LatentLaplaceAE  N={N}  Nt={Nt}  K={K}  latent={ckpt["latent_dim"]}')
    print(f'Paramètres : {n_params:,}  |  val_loss={ckpt["val_loss"]:.4e}')

    alpha_t = model.laplace.log_alpha_t.exp().item()
    lam     = model.laplace.log_lam.exp().item()
    s_cur   = model.laplace.s_list.detach().cpu()
    print(f'α_t={alpha_t:.5f}  λ={lam:.6f}')
    print(f's_k Re : [{s_cur.real.min():.4f}, {s_cur.real.max():.4f}]')
    print(f's_k Im : [{s_cur.imag.min():.4f}, {s_cur.imag.max():.4f}]')

    # --- Dataset ---
    dataset = TransientDataset(data_path, laplace=False, dt=dt, interp_size=N)
    dataset.fit(test_idx.tolist())

    if n_samples is not None and n_samples < len(test_idx):
        rng = np.random.default_rng(42)
        eval_idx = rng.choice(test_idx, size=n_samples, replace=False)
    else:
        eval_idx = test_idx
    print(f'Évaluation sur {len(eval_idx)} samples de test')

    # --- Inférence ---
    U_true_norm, U_rec_norm, z_arr, z_rec_arr = run_test(
        model, dataset, eval_idx, N, U_mean, U_std, device, batch_size=batch_size,
    )

    # --- Métriques ---
    l2_temp  = l2rel_per_sample(U_true_norm, U_rec_norm)
    l2_lat   = l2rel_latent(z_arr, z_rec_arr)
    mean_t, std_t = temporal_error_profile(U_true_norm, U_rec_norm)

    print('\n' + '=' * 65)
    print(f'RECONSTRUCTION TEMPORELLE (espace normalisé)')
    print(f'  Mean   : {l2_temp.mean()*100:.2f}%')
    print(f'  Median : {np.median(l2_temp)*100:.2f}%')
    print(f'  Std    : {l2_temp.std()*100:.2f}%')
    print(f'  Min    : {l2_temp.min()*100:.2f}%')
    print(f'  Max    : {l2_temp.max()*100:.2f}%')

    print(f'\nROUNDTRIP LATENT  z → ẑ → z̃')
    print(f'  Mean   : {l2_lat.mean()*100:.2f}%')
    print(f'  Median : {np.median(l2_lat)*100:.2f}%')
    print(f'  Std    : {l2_lat.std()*100:.2f}%')
    print('=' * 65)

    # --- Figures ---
    fig_histogram(
        l2_temp, ckpt_path,
        os.path.join(plots_dir, 'latent_laplace_ae_l2rel_hist.png'),
    )
    fig_temporal_profile(
        mean_t, std_t, Nt, dt,
        os.path.join(plots_dir, 'latent_laplace_ae_temporal_profile.png'),
    )
    fig_s_scatter(
        model,
        os.path.join(plots_dir, 'latent_laplace_ae_s_scatter.png'),
    )

    # Comparaisons frames : meilleur, médian, pire sample
    sorted_idx = np.argsort(l2_temp)
    for label, si in [('best', sorted_idx[0]),
                       ('median', sorted_idx[len(sorted_idx) // 2]),
                       ('worst', sorted_idx[-1])]:
        fig_frames(
            U_true_norm, U_rec_norm, U_mean, U_std,
            sample_idx=si, Nt=Nt,
            save_path=os.path.join(plots_dir, f'latent_laplace_ae_frames_{label}.png'),
        )

    # --- Animations ---
    anim_idx = list(np.random.default_rng(0).choice(len(eval_idx), size=n_anim, replace=False))
    U_m = U_mean.numpy() if isinstance(U_mean, torch.Tensor) else U_mean
    U_s = U_std.numpy()  if isinstance(U_std,  torch.Tensor) else U_std

    for i, si in enumerate(anim_idx):
        U_true_phys = U_true_norm[si] * U_s + U_m   # (Nt, N, N)
        U_rec_phys  = U_rec_norm [si] * U_s + U_m
        l2 = l2_temp[si]
        out = os.path.join(plots_dir, f'latent_laplace_ae_anim_{i}.gif')
        print(f'Animation {i+1}/{n_anim} — sample #{eval_idx[si]}  (L2rel={l2*100:.1f}%)...')
        animate_comparaison(
            U_true_phys, U_rec_phys,
            output_path=out,
            title_fn=lambda t, s=eval_idx[si], e=l2: f'#{s}  L2rel={e*100:.1f}%  t={t}',
            title_a='Ground Truth',
            title_b='AE Reconstruction',
            title_err='|Error|',
            label='CH4 Concentration',
        )
