"""
transient/main.py — Inférence transiente : theta → U(t)
========================================================
Supporte deux backends, détectés automatiquement depuis model_type dans le checkpoint :
  - LaplaceModel : ex. checkpoints/LaplaceModel.pt
  - SVDSurrogate : ex. checkpoints/SVDSurrogate_best.pt

Fonctions réutilisables :
  load_model(ckpt_path, device)                    → model, ckpt
  run_inference(theta_raw, model, ckpt, device)    → U_pred (B, Nt, H, W)
  predict(theta_raw, ckpt_path, device)            → U_pred (B, Nt, H, W)
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    """Charge LaplaceModel ou SVDSurrogate depuis un fichier checkpoint .pt."""
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'SVDSurrogate')

    if model_type == 'LaplaceModel':
        from models.laplace_surrogate import LaplaceModel
        model = LaplaceModel(
            N_freq    = ckpt['N_freq'],
            N_half    = ckpt['N_half'],
            N         = ckpt['N'],
            theta_dim = ckpt['theta_dim'],
            k_max     = ckpt.get('k_max'),
        ).to(device)

    elif model_type == 'LaplaceLatentModel':
        from models.laplace_ae_surrogate import LaplaceLatentModel
        model = LaplaceLatentModel(
            N_freq     = ckpt['N_freq'],
            N_half     = ckpt['N_half'],
            N          = ckpt['N'],
            theta_dim  = ckpt['theta_dim'],
            latent_dim = ckpt['latent_dim'],
            hidden_dim = ckpt['hidden_dim'],
            k_max      = ckpt.get('k_max'),
        ).to(device)

    elif model_type == 'LaplaceSVDModel':
        from models.laplace_svd_surrogate import LaplaceSVDModel
        model = LaplaceSVDModel(
            k_freq    = ckpt['k_freq'],
            N_freq    = ckpt['N_freq'],
            N_half    = ckpt['N_half'],
            N         = ckpt['N'],
            theta_dim = ckpt['theta_dim'],
            k_svd     = ckpt['k_svd'],
        ).to(device)

    elif model_type == 'SVDSurrogate':
        from models.svd_surrogate import SVDSurrogate

        model = SVDSurrogate(nr=ckpt['nr'], nt=ckpt['Nt'], nf_eff=ckpt['nf_eff'], theta_dim=ckpt['theta_dim']).to(device)

    elif model_type == 'CorrectionAE':
        from models.correction_ae import CorrectionAE, CorrectedPipeline
        ae = CorrectionAE(N=ckpt['N'], base_ch=ckpt['base_ch']).to(device)
        ae.load_state_dict(ckpt['model_state'])
        ae.eval()

        surr_model, surr_ckpt = load_model(ckpt['surrogate_ckpt'], device)
        model = CorrectedPipeline(surr_model, ae).to(device)
        # Expose les infos du surrogate pour run_inference (theta_mean/std, dt, gamma)
        ckpt['theta_mean'] = surr_ckpt['theta_mean']
        ckpt['theta_std']  = surr_ckpt['theta_std']
        ckpt['dt']         = surr_ckpt.get('dt', 1.0)
        ckpt['gamma']      = surr_ckpt.get('gamma', 0.0)
        model.eval()
        return model, ckpt

    else:
        raise ValueError(f"Checkpoint de type '{model_type}' non supporté.")

    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(theta_raw, model, ckpt: dict, device: torch.device,
                  dt: float | None = None,
                  gamma: float = 0.0,
                  rule: str = 'trap',
                  k_max: int | None = None) -> np.ndarray:
    """
    Inférence avec un modèle déjà chargé.

    Retourne U_pred : np.ndarray (B, Nt, N, N) en valeurs physiques.
    """
    theta_mean = torch.tensor(ckpt['theta_mean'], dtype=torch.float32, device=device)
    theta_std  = torch.tensor(ckpt['theta_std'],  dtype=torch.float32, device=device)

    theta_t    = torch.tensor(np.asarray(theta_raw, dtype=np.float32), device=device)
    if theta_t.dim() == 1:
        theta_t = theta_t.unsqueeze(0)
    theta_norm = (theta_t - theta_mean) / theta_std

    model_type = ckpt.get('model_type', 'SVDSurrogate')

    if model_type in ('LaplaceModel', 'LaplaceLatentModel', 'LaplaceSVDModel', 'CorrectionAE'):
        dt_eff = dt if dt is not None else float(ckpt.get('dt', 1.0))
        gamma  = float(ckpt.get('gamma', gamma))
        U_pred = model.generate(theta_norm, dt=dt_eff, gamma=gamma, rule=rule, k_max=k_max)
        return U_pred.cpu().numpy()

    else:  # SVDSurrogate
        U_pred = model.generate(theta_norm)
        return U_pred.cpu().numpy()


def predict(theta_raw, ckpt_path: str = 'checkpoints/LaplaceModel.pt',
            device_str: str = 'auto', dt: float | None = None,
            gamma: float = 0.0, rule: str = 'trap',
            k_max: int | None = None) -> np.ndarray:
    """
    Prédit U(t) pour un batch de theta.

    Paramètres
    ----------
    theta_raw  : array-like (B, theta_dim) ou (theta_dim,) — valeurs physiques
    ckpt_path  : fichier .pt (LaplaceModel.pt ou SVDSurrogate_best.pt)
    device_str : 'auto', 'cpu' ou 'cuda'

    Retourne U_pred : np.ndarray (B, Nt, N, N)
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    model, ckpt = load_model(ckpt_path, device)
    return run_inference(theta_raw, model, ckpt, device, dt=dt, gamma=gamma, rule=rule, k_max=k_max)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(U, theta, ckpt_path: str, test_idx=None,
             dt: float | None = None, gamma: float = 0.0,
             rule: str = 'trap', step: int = 1,
             n_animate: int = 3, batch_size: int = 32,
             device_str: str = 'auto', k_max: int | None = None):
    """
    Évalue le surrogate sur le test set et produit des GIFs + histogramme.

    Paramètres
    ----------
    U        : ndarray (ns, Nt, H, W) — peut être un mmap (jamais matérialisé en entier)
    theta    : ndarray (ns, theta_dim)
    ckpt_path: fichier .pt
    test_idx : indices du test set ; si None, chargé depuis le checkpoint
    step     : sous-échantillonnage spatial pour SVD
    batch_size: nombre de simulations traitées à la fois
    """
    from utils.animate import animate_comparaison

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    model, ckpt = load_model(ckpt_path, device)
    model_type  = ckpt.get('model_type', 'SVDSurrogate')
    os.makedirs('plots', exist_ok=True)

    if test_idx is None:
        assert 'test_idx' in ckpt, "test_idx absent du checkpoint — relance l'entraînement."
        test_idx = ckpt['test_idx']

    theta_arr = np.asarray(theta, dtype=np.float32)   # theta est petit, OK
    n_test    = len(test_idx)
    print(f"Test set : {n_test} simulations  |  backend : {model_type}")

    anim_pos       = set(np.random.choice(n_test, size=min(n_animate, n_test), replace=False))
    l2rel_list     = []
    mse_list       = []
    saved_pred     = {}   # pos → U_pred[pos]
    saved_true     = {}   # pos → U_true[pos]

    for start in tqdm(range(0, n_test, batch_size), desc='Évaluation'):
        pos_batch  = list(range(start, min(start + batch_size, n_test)))
        idx_batch  = [int(test_idx[p]) for p in pos_batch]

        # Prédiction
        U_pred_b = run_inference(
            theta_arr[idx_batch], model, ckpt, device, dt=dt, gamma=gamma, rule=rule, k_max=k_max
        )  # (B, Nt, H_pred, W_pred)

        # Vérité terrain — chargée sample par sample depuis le mmap
        H_pred, W_pred = U_pred_b.shape[-2], U_pred_b.shape[-1]
        slices = []
        for i in idx_batch:
            u = np.array(U[i], dtype=np.float32)   # (Nt, H, W) — copie locale
            if u.shape[-2] != H_pred or u.shape[-1] != W_pred:
                # Redimensionnement par interpolation bilinéaire
                u_t = torch.from_numpy(u).unsqueeze(0)  # (1, Nt, H, W)
                u_t = torch.nn.functional.interpolate(
                    u_t, size=(H_pred, W_pred), mode='bilinear', align_corners=False
                )
                u = u_t.squeeze(0).numpy()
            slices.append(u)
        U_true_b = np.stack(slices)                 # (B, Nt, H_pred, W_pred)

        diff       = U_pred_b - U_true_b
        norms_err  = np.linalg.norm(diff.reshape(len(pos_batch), -1), axis=1)
        norms_true = np.linalg.norm(U_true_b.reshape(len(pos_batch), -1), axis=1) + 1e-12
        l2rel_list.append(norms_err / norms_true)
        mse_list.append(np.mean(diff ** 2, axis=(1, 2, 3)))

        for local_p, global_p in enumerate(pos_batch):
            if global_p in anim_pos:
                saved_pred[global_p] = U_pred_b[local_p]
                saved_true[global_p] = U_true_b[local_p]

    l2rel   = np.concatenate(l2rel_list)
    mse_arr = np.concatenate(mse_list)

    print(f"MSE global    : {np.mean(mse_arr):.4e}")
    print(f"L2rel — mean  : {l2rel.mean()*100:.2f}%  std : {l2rel.std()*100:.2f}%")
    print(f"        min   : {l2rel.min()*100:.2f}%   max : {l2rel.max()*100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.hist(l2rel * 100, bins=30, edgecolor='black', alpha=0.7)
    plt.xlim((0, 100))
    plt.xlabel('L2 Relative Error (%)')
    plt.ylabel('Fréquence')
    plt.title(f'L2 Relative Error — {model_type}  (n={n_test})\n'
              f'mean={l2rel.mean()*100:.2f}%  std={l2rel.std()*100:.2f}%')
    plt.grid(True, alpha=0.3)
    hist_path = os.path.join('plots', f'{model_type}_l2rel_hist.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogramme -> {hist_path}")

    for i, global_p in enumerate(sorted(saved_pred)):
        si        = int(test_idx[global_p])
        l2        = l2rel[global_p]
        anim_path = os.path.join('plots', f'{model_type}_anim_{i}.gif')
        print(f"Animation {i+1}/{len(saved_pred)} — simulation #{si} (L2rel={l2*100:.1f}%)...")
        animate_comparaison(
            saved_true[global_p], saved_pred[global_p],
            output_path = anim_path,
            title_fn    = lambda t, s=si, e=l2: f"#{s}  L2rel={e*100:.1f}%  t={t}",
        )
        print(f"  -> {anim_path}")

    return l2rel


def main(
    ckpt_path = 'checkpoints/CorrectionAE_best.pt',
    data_path = '/Data/KAT/ch4_rotated.npy',
    theta     = [[1.0, 0.5, 0.3, 2.0]],
    dt        = None,
    gamma     = 0.0,
    rule      = 'trap',
    step      = 2,
    out       = None,
    plot      = True,
    do_evaluation  = True,
    k_max     = 20,
):

    if do_evaluation:
        if data_path.endswith('.npy'):
            U_data     = np.load(data_path, mmap_mode='r')
            doe_path   = str(Path(data_path).parent / 'doe_rotated.npy')
            doe        = np.load(doe_path)
            theta_data = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)
        else:
            data       = np.load(data_path)
            U_data     = data['U']
            theta_data = data['theta'].astype(np.float32)
        evaluate(U_data, theta_data, ckpt_path,
                 dt=dt, gamma=gamma, rule=rule, step=step, k_max=k_max)
        return

    U_pred = predict(theta, ckpt_path, dt=dt, gamma=gamma, rule=rule, k_max=k_max)[0]  # (Nt, N, N)
    print(f"Prédiction : shape={U_pred.shape}  min={U_pred.min():.4f}  max={U_pred.max():.4f}")

    if out:
        np.save(out, U_pred)
        print(f"Sauvegardé → {out}")


    return U_pred


if __name__ == '__main__':
    main()
