"""
transient/main.py — Inférence transiente : theta → U(t)
========================================================
Supporte deux backends, détectés automatiquement depuis le chemin checkpoint :
  - Laplace  : ckpt = répertoire  (ex. checkpoints/laplace/)
  - SVD      : ckpt = fichier .pt (ex. checkpoints/SVDSurrogate_best.pt)

Fonctions réutilisables :
  predict(theta, ckpt, ...)          → U_pred  (B, Nt, H, W)
  evaluate(U, theta, ckpt, ...)      → l2rel   (n_test,)
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Détection du backend
# ---------------------------------------------------------------------------

def _detect_backend(ckpt: str) -> str:
    """'laplace' si ckpt est un répertoire, 'svd' si c'est un fichier .pt."""
    if os.path.isdir(ckpt):
        return 'laplace'
    if ckpt.endswith('.pt'):
        return 'svd'
    raise ValueError(f"Impossible de détecter le backend depuis '{ckpt}' "
                     "(attendu : répertoire pour Laplace, fichier .pt pour SVD)")


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Backend Laplace
# ---------------------------------------------------------------------------

def _predict_laplace(theta, ckpt_dir: str, dt: float, gamma: float,
                     rule: str, device: torch.device) -> np.ndarray:
    from utils.laplace import laplace_inverse
    from models.laplace_surrogate import LaplaceSurrogate

    ckpt_files = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith('LaplaceSurrogate_freq') and f.endswith('.pt')
    ])
    assert len(ckpt_files) > 0, f"Aucun checkpoint trouvé dans {ckpt_dir}"

    theta_t = torch.tensor(np.asarray(theta, dtype=np.float32))
    B = theta_t.shape[0]

    ckpt0     = torch.load(os.path.join(ckpt_dir, ckpt_files[0]),
                           map_location='cpu', weights_only=False)
    N         = ckpt0['N']
    Nt_half   = len(ckpt_files)
    Nt        = ckpt0.get('Nt', 100)
    theta_dim = ckpt0['theta_dim']

    M_half = np.zeros((B, N * N, Nt_half), dtype=np.complex64)

    for k, fname in tqdm(enumerate(ckpt_files), total=Nt_half,
                         desc="Prédiction Laplace", leave=True):
        ckpt = torch.load(os.path.join(ckpt_dir, fname),
                          map_location=device, weights_only=False)

        theta_mean = torch.tensor(ckpt['theta_mean'], device=device)
        theta_std  = torch.tensor(ckpt['theta_std'],  device=device)
        theta_n    = (theta_t.to(device) - theta_mean) / theta_std

        s_k   = complex(ckpt['s_k_real'], ckpt['s_k_imag'])
        model = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        with torch.no_grad():
            pred = model(theta_n)                              # (B, 2, N, N)

        target_mean = torch.tensor(ckpt['target_mean'], device=device)
        target_std  = torch.tensor(ckpt['target_std'],  device=device)
        pred = pred * target_std + target_mean

        pred_np = pred.cpu().numpy()
        M_half[:, :, k] = (pred_np[:, 0] + 1j * pred_np[:, 1]).reshape(B, N * N)

    # Symétrie conjuguée → spectre complet
    M_full = np.zeros((B, N * N, Nt), dtype=np.complex64)
    M_full[:, :, :Nt_half] = M_half
    M_full[:, :, Nt_half:] = np.conj(M_half[:, :, Nt - Nt_half:0:-1])

    U_pred = np.zeros((B, Nt, N, N), dtype=np.float32)
    for b in tqdm(range(B), desc="Inverse Laplace", leave=True):
        C_b, _ = laplace_inverse(M_full[b], dt, Nt, rule=rule, gamma=gamma)
        U_pred[b] = C_b.reshape(N, N, Nt).transpose(2, 0, 1).astype(np.float32)

    return U_pred


# ---------------------------------------------------------------------------
# Backend SVD
# ---------------------------------------------------------------------------

def _predict_svd(theta, ckpt_path: str, device: torch.device) -> np.ndarray:
    from models.svd_surrogate import SVDSurrogate
    from utils.SVD_Amine_3D import svd_inverse_3d

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    theta_mean = ckpt['theta_mean']
    theta_std  = ckpt['theta_std']
    G_mean     = ckpt['G_mean']
    G_std      = ckpt['G_std']
    alph       = ckpt['alph']
    F          = ckpt['F']
    P          = ckpt['P']
    nf_eff     = ckpt['nf_eff']

    model = SVDSurrogate(nf_eff=nf_eff, theta_dim=ckpt['theta_dim']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    theta = np.asarray(theta, dtype=np.float32)
    theta_n = (theta - theta_mean) / theta_std
    with torch.no_grad():
        G_pred_n = model(torch.tensor(theta_n).to(device)).cpu().numpy()
    G_pred = (G_pred_n * G_std + G_mean) / alph[None, :]   # (B, nf_eff)

    nr   = F.shape[0]
    Nt   = P.shape[0]
    Hsub = Wsub = int(np.round(np.sqrt(nr)))
    assert Hsub * Wsub == nr, "Grille spatiale non carrée — ajuste Hsub/Wsub"
    B = G_pred.shape[0]

    # (nr, B, Nt) → (B, Nt, Hsub, Wsub)
    fields = svd_inverse_3d(F, G_pred, P, alph)             # (nr, B, Nt)
    U_pred = fields.transpose(1, 2, 0).reshape(B, Nt, Hsub, Wsub).astype(np.float32)

    return U_pred


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def predict(theta, ckpt: str, dt: float = 1.0, gamma: float = 0.0,
            rule: str = 'trap', device: str = 'auto') -> np.ndarray:
    """
    Prédit U(t) pour un batch de theta.

    Paramètres
    ----------
    theta  : array-like (B, theta_dim)
    ckpt   : répertoire Laplace ou fichier .pt SVD
    dt, gamma, rule : paramètres Laplace (ignorés pour SVD)
    device : 'auto', 'cpu' ou 'cuda'

    Retour
    ------
    U_pred : ndarray (B, Nt, H, W)
    """
    dev     = _resolve_device(device)
    backend = _detect_backend(ckpt)
    if backend == 'laplace':
        return _predict_laplace(theta, ckpt, dt, gamma, rule, dev)
    else:
        return _predict_svd(theta, ckpt, dev)


def evaluate(U, theta, ckpt: str, test_idx=None, dt: float = 1.0,
             gamma: float = 0.0, rule: str = 'trap', step: int = 1,
             n_animate: int = 3, device: str = 'auto'):
    """
    Évalue le surrogate sur le test set et produit des GIFs + histogramme.

    Paramètres
    ----------
    U        : ndarray (ns, Nt, H, W) — champs originaux (pleine résolution)
    theta    : ndarray (ns, theta_dim)
    ckpt     : répertoire Laplace ou fichier .pt SVD
    test_idx : indices du test set ; si None, chargé depuis le checkpoint
    step     : sous-échantillonnage spatial pour SVD (doit correspondre à learn_svd)
    """
    import matplotlib.pyplot as plt
    from utils.animate import animate_comparaison

    backend = _detect_backend(ckpt)
    os.makedirs('plots', exist_ok=True)

    # Récupération de test_idx
    if test_idx is None:
        if backend == 'laplace':
            idx_path = os.path.join(ckpt, 'test_idx.npy')
            assert os.path.exists(idx_path), \
                f"test_idx.npy introuvable dans {ckpt} — relance l'entraînement ou passe test_idx manuellement"
            test_idx = np.load(idx_path)
        else:
            ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
            test_idx  = ckpt_data['test_idx']

    theta = np.asarray(theta, dtype=np.float32)
    U     = np.asarray(U, dtype=np.float32)

    # Prédiction
    U_pred = predict(theta[test_idx], ckpt, dt=dt, gamma=gamma, rule=rule, device=device)

    # Référence : champ original (sous-échantillonné si SVD)
    if backend == 'svd':
        U_true = U[test_idx][:, :, ::step, ::step]
    else:
        U_true = U[test_idx]

    n_test = len(test_idx)

    # Métriques L2 relative
    diff       = U_pred - U_true
    norms_err  = np.linalg.norm(diff.reshape(n_test, -1), axis=1)
    norms_true = np.linalg.norm(U_true.reshape(n_test, -1), axis=1) + 1e-12
    l2rel      = norms_err / norms_true
    mse_arr    = np.mean(diff ** 2, axis=(1, 2, 3))

    print(f"Test set : {n_test} simulations  |  backend : {backend}")
    print(f"MSE global    : {np.mean(mse_arr):.4e}")
    print(f"L2rel — mean  : {l2rel.mean():.4e}  std : {l2rel.std():.4e}")
    print(f"        min   : {l2rel.min():.4e}  max : {l2rel.max():.4e}")

    # Histogramme L2rel
    plt.figure(figsize=(10, 5))
    plt.hist(l2rel, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('L2 Relative Error')
    plt.ylabel('Fréquence')
    plt.title(f'L2 Relative Error — {backend}  (n={n_test})\n'
              f'mean={l2rel.mean():.3e}  std={l2rel.std():.3e}')
    plt.grid(True, alpha=0.3)
    hist_path = os.path.join('plots', f'{backend}_l2rel_hist.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogramme → {hist_path}")

    # Animations best / median / worst
    sorted_idx = np.argsort(l2rel)
    picks  = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    labels = ['best', 'median', 'worst']

    for pick, label in zip(picks[:n_animate], labels):
        si = test_idx[pick]
        gif_path = os.path.join('plots', f'{backend}_{label}.gif')
        animate_comparaison(
            U_true[pick], U_pred[pick],
            output_path=gif_path,
            title_fn=lambda t, s=si, l=label: f"#{s} ({l}) — L2rel={l2rel[pick]:.3e}  t={t}",
        )
        print(f"Animation {label} → {gif_path}")

    return l2rel


def main(
    ckpt      = 'checkpoints/laplace',          # ou 'checkpoints/SVDSurrogate_best.pt'
    data_path = 'dataset/dataset_transient.npz',
    theta     = [[1.0, 0.5, 0.3, 2.0]],        # (B, theta_dim)
    dt        = 1.0,
    gamma     = 0.0,
    rule      = 'trap',
    step      = 2,    # sous-échantillonnage spatial (SVD uniquement)
    out       = None, # chemin .npy pour sauvegarder, ou None
    plot      = True,
    EVALUATE  = False,
):
    import matplotlib.pyplot as plt

    if EVALUATE:
        data  = np.load(data_path)
        dt_   = float(data['dt'][0]) if 'dt' in data else dt
        evaluate(data['U'], data['theta'], ckpt,
                 dt=dt_, gamma=gamma, rule=rule, step=step)
        return

    theta_arr = np.array(theta, dtype=np.float32)
    U_pred = predict(theta_arr, ckpt, dt=dt, gamma=gamma, rule=rule)[0]  # (Nt, H, W)
    print(f"Prédiction : shape={U_pred.shape}  min={U_pred.min():.4f}  max={U_pred.max():.4f}")

    if out:
        np.save(out, U_pred)
        print(f"Sauvegardé → {out}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, t in zip(axes, [0, len(U_pred) // 2, len(U_pred) - 1]):
            im = ax.imshow(U_pred[t], origin='lower', cmap='viridis')
            ax.set_title(f't={t}')
            plt.colorbar(im, ax=ax)
        plt.suptitle(f'theta = {theta}')
        plt.tight_layout()
        plt.show()

    return U_pred


if __name__ == '__main__':
    main()
