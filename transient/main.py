"""
transient/main.py — Inférence transiente : theta → U(t)
========================================================
Supporte deux backends, détectés automatiquement depuis model_type dans le checkpoint :
  - LaplaceModel : ex. checkpoints/laplace/LaplaceModel.pt
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
        ).to(device)

    elif model_type == 'SVDSurrogate':
        from models.svd_surrogate import SVDSurrogate
        model = SVDSurrogate(nf_eff=ckpt['nf_eff'], theta_dim=ckpt['theta_dim']).to(device)

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
                  rule: str = 'trap') -> np.ndarray:
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

    if model_type == 'LaplaceModel':
        dt_eff = dt if dt is not None else float(ckpt.get('dt', 1.0))
        gamma  = float(ckpt.get('gamma', gamma))
        U_pred = model.generate(theta_norm, dt=dt_eff, gamma=gamma, rule=rule)
        return U_pred.cpu().numpy()

    else:  # SVDSurrogate
        U_pred = model.generate(theta_norm)
        return U_pred.cpu().numpy()


def predict(theta_raw, ckpt_path: str = 'checkpoints/laplace/LaplaceModel.pt',
            device_str: str = 'auto', dt: float | None = None,
            gamma: float = 0.0, rule: str = 'trap') -> np.ndarray:
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
    return run_inference(theta_raw, model, ckpt, device, dt=dt, gamma=gamma, rule=rule)


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(U, theta, ckpt_path: str, test_idx=None,
             dt: float | None = None, gamma: float = 0.0,
             rule: str = 'trap', step: int = 1,
             n_animate: int = 3, device_str: str = 'auto'):
    """
    Évalue le surrogate sur le test set et produit des GIFs + histogramme.

    Paramètres
    ----------
    U        : ndarray (ns, Nt, H, W) — champs originaux
    theta    : ndarray (ns, theta_dim)
    ckpt_path: fichier .pt ou répertoire contenant LaplaceModel.pt
    test_idx : indices du test set ; si None, chargé depuis le checkpoint
    step     : sous-échantillonnage spatial pour SVD
    """
    import matplotlib.pyplot as plt
    from utils.animate import animate_comparaison

    ckpt_data  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_type = ckpt_data.get('model_type', 'SVDSurrogate')
    os.makedirs('plots', exist_ok=True)

    if test_idx is None:
        idx_path = os.path.join(os.path.dirname(ckpt_path), 'test_idx.npy')
        assert os.path.exists(idx_path), \
            f"test_idx.npy introuvable dans {os.path.dirname(ckpt_path)}"
        test_idx = np.load(idx_path)

    theta = np.asarray(theta, dtype=np.float32)
    U     = np.asarray(U,     dtype=np.float32)

    U_pred = predict(theta[test_idx], ckpt_path, device_str, dt=dt, gamma=gamma, rule=rule)

    U_true = U[test_idx]
    if model_type == 'SVDSurrogate':
        U_true = U_true[:, :, ::step, ::step]

    n_test = len(test_idx)
    diff        = U_pred - U_true
    norms_err   = np.linalg.norm(diff.reshape(n_test, -1), axis=1)
    norms_true  = np.linalg.norm(U_true.reshape(n_test, -1), axis=1) + 1e-12
    l2rel       = norms_err / norms_true
    mse_arr     = np.mean(diff ** 2, axis=(1, 2, 3))

    print(f"Test set : {n_test} simulations  |  backend : {model_type}")
    print(f"MSE global    : {np.mean(mse_arr):.4e}")
    print(f"L2rel — mean  : {l2rel.mean():.4e}  std : {l2rel.std():.4e}")
    print(f"        min   : {l2rel.min():.4e}  max : {l2rel.max():.4e}")

    plt.figure(figsize=(10, 5))
    plt.hist(l2rel, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('L2 Relative Error')
    plt.ylabel('Fréquence')
    plt.title(f'L2 Relative Error — {model_type}  (n={n_test})\n'
              f'mean={l2rel.mean():.3e}  std={l2rel.std():.3e}')
    plt.grid(True, alpha=0.3)
    hist_path = os.path.join('plots', f'{model_type}_l2rel_hist.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogramme -> {hist_path}")

    # Animation pour quelques simulations
    n_animate = min(n_animate, n_test)
    positions = np.random.choice(n_test, size=n_animate, replace=False)
    for pos in positions:
        si        = test_idx[pos]
        anim_path = os.path.join('plots', f'{model_type}_anim_{si}.gif')
        animate_comparaison(U_true[pos], U_pred[pos], output_path=anim_path)
        print(f"Animation simulation {si} -> {anim_path}")

    return l2rel


def main(
    ckpt_path = 'checkpoints/laplace/LaplaceModel.pt',
    data_path = 'dataset/dataset_transient.npz',
    theta     = [[1.0, 0.5, 0.3, 2.0]],
    dt        = None,
    gamma     = 0.0,
    rule      = 'trap',
    step      = 2,
    out       = None,
    plot      = True,
    EVALUATE  = False,
):
    import matplotlib.pyplot as plt

    if EVALUATE:
        data = np.load(data_path)
        evaluate(data['U'], data['theta'], ckpt_path,
                 dt=dt, gamma=gamma, rule=rule, step=step)
        return

    U_pred = predict(theta, ckpt_path, dt=dt, gamma=gamma, rule=rule)[0]  # (Nt, N, N)
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
