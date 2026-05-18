"""
Pipeline Laplace-SVD (transient)
=================================
Pour chaque fréquence k = 0 .. k_freq-1 :
  1. Charger Re_k et Im_k du spectre de Laplace depuis le cache mmap
  2. SVD tronquée (torch.svd_lowrank) sur Re_k et Im_k séparément (train set)
  3. Entraîner un MLP par composante : theta_norm → k_svd coefficients SVD normalisés

Résultat : LaplaceSVDModel sauvegardé dans checkpoints/LaplaceSVDModel.pt,
           compatible avec transient/main.py (model_type='LaplaceSVDModel').
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from transient.dataset import TransientDataset
from models.transient.laplace_svd_surrogate import LaplaceSVDSurrogate, LaplaceSVDModel


# ---------------------------------------------------------------------------
# SVD tronquée pour une fréquence
# ---------------------------------------------------------------------------

def compute_svd_for_freq(U_laplace, train_idx, val_idx, k, k_svd, N):
    """
    SVD tronquée (torch.svd_lowrank) sur Re_k et Im_k du train set.

    Paramètres
    ----------
    U_laplace : ndarray mmap (ns, K, 2, N, N)
    train_idx : liste d'indices train
    val_idx   : liste d'indices val
    k         : indice de fréquence
    k_svd     : nombre de composantes SVD à garder
    N         : taille spatiale

    Retour
    ------
    Vt_re    : (k_svd, N*N)  — base Re (lignes orthonormées)
    Vt_im    : (k_svd, N*N)  — base Im
    coeff_re_train : (ns_train, k_svd)
    coeff_im_train : (ns_train, k_svd)
    coeff_re_val   : (ns_val,   k_svd)
    coeff_im_val   : (ns_val,   k_svd)
    """
    train_np = np.array(train_idx)
    val_np   = np.array(val_idx)
    NN       = N * N

    # Charger Re et Im train (matérialise ~1 Go chacun pour N=200)
    Re_train = torch.from_numpy(
        np.array(U_laplace[train_np, k, 0], dtype=np.float32).reshape(-1, NN)
    )  # (ns_train, N*N)
    Im_train = torch.from_numpy(
        np.array(U_laplace[train_np, k, 1], dtype=np.float32).reshape(-1, NN)
    )

    # SVD tronquée sur CPU (matrices potentiellement très larges)
    U_re, S_re, Vh_re = torch.svd_lowrank(Re_train, q=k_svd)
    U_im, S_im, Vh_im = torch.svd_lowrank(Im_train, q=k_svd)
    # Vh_re : (N*N, k_svd) → Vt_re : (k_svd, N*N)
    Vt_re = Vh_re.T.contiguous()
    Vt_im = Vh_im.T.contiguous()

    # Coefficients train = U * S   (reconstruction X ≈ coeff @ Vt)
    coeff_re_train = (U_re * S_re.unsqueeze(0)).contiguous()
    coeff_im_train = (U_im * S_im.unsqueeze(0)).contiguous()

    # Coefficients val = projection sur la base train
    Re_val = torch.from_numpy(
        np.array(U_laplace[val_np, k, 0], dtype=np.float32).reshape(-1, NN)
    )
    Im_val = torch.from_numpy(
        np.array(U_laplace[val_np, k, 1], dtype=np.float32).reshape(-1, NN)
    )
    coeff_re_val = Re_val @ Vt_re.T   # (ns_val, k_svd)
    coeff_im_val = Im_val @ Vt_im.T

    # Erreur d'approximation SVD sur le val set
    norm_re = torch.norm(Re_val)
    norm_im = torch.norm(Im_val)
    err_svd_re = (torch.norm(Re_val - coeff_re_val @ Vt_re) / norm_re).item() if norm_re > 0 else float('nan')
    err_svd_im = (torch.norm(Im_val - coeff_im_val @ Vt_im) / norm_im).item() if norm_im > 0 else float('nan')

    return Vt_re, Vt_im, coeff_re_train, coeff_im_train, coeff_re_val, coeff_im_val, err_svd_re, err_svd_im


# ---------------------------------------------------------------------------
# Entraînement d'une paire de surrogates (Re + Im) pour la fréquence k
# ---------------------------------------------------------------------------

def train_one_freq(k, k_svd, theta_dim, N,
                   theta_norm_train, theta_norm_val,
                   coeff_re_train, coeff_re_val,
                   coeff_im_train, coeff_im_val,
                   n_epochs, lr, batch_size, device,
                   step_offset: int = 0, use_wandb=True):
    """
    Entraîne LaplaceSVDSurrogate Re et Im pour la fréquence k.

    Retour
    ------
    sur_re, sur_im      : surrogates entraînés (CPU, meilleur val)
    cr_mean, cr_std     : stats normalisation coefficients Re  (k_svd,)
    ci_mean, ci_std     : stats normalisation coefficients Im  (k_svd,)
    """
    # Normalisation des coefficients (stats train)
    cr_mean = coeff_re_train.mean(0); cr_std = coeff_re_train.std(0) + 1e-8
    ci_mean = coeff_im_train.mean(0); ci_std = coeff_im_train.std(0) + 1e-8

    cre_tr_n = (coeff_re_train - cr_mean) / cr_std
    cre_va_n = (coeff_re_val   - cr_mean) / cr_std
    cim_tr_n = (coeff_im_train - ci_mean) / ci_std
    cim_va_n = (coeff_im_val   - ci_mean) / ci_std

    # DataLoaders
    ds_tr = TensorDataset(theta_norm_train.to(device),
                           cre_tr_n.to(device), cim_tr_n.to(device))
    ds_va = TensorDataset(theta_norm_val.to(device),
                           cre_va_n.to(device), cim_va_n.to(device))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size)

    sur_re = LaplaceSVDSurrogate(k_svd, theta_dim, N).to(device)
    sur_im = LaplaceSVDSurrogate(k_svd, theta_dim, N).to(device)
    params = list(sur_re.parameters()) + list(sur_im.parameters())
    opt    = torch.optim.Adam(params, lr=lr)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    best_val = float('inf')
    best_re  = None
    best_im  = None

    pbar = tqdm(range(n_epochs), desc=f'freq {k:03d}', leave=False)
    for epoch in pbar:
        sur_re.train(); sur_im.train()
        for theta_b, cre_b, cim_b in loader_tr:
            opt.zero_grad()
            loss = (nn.functional.mse_loss(sur_re(theta_b), cre_b) +
                    nn.functional.mse_loss(sur_im(theta_b), cim_b))
            loss.backward()
            opt.step()
        sched.step()

        if epoch % 5 == 0:
            sur_re.eval(); sur_im.eval()
            with torch.no_grad():
                val_re = sum(
                    nn.functional.mse_loss(sur_re(th), cre).item()
                    for th, cre, _ in loader_va
                ) / len(loader_va)
                val_im = sum(
                    nn.functional.mse_loss(sur_im(th), cim).item()
                    for th, _, cim in loader_va
                ) / len(loader_va)
            val_loss = val_re + val_im
            pbar.set_postfix(val=f'{val_loss:.4f}', best=f'{best_val:.4f}')

            if use_wandb:
                wandb.log({
                    f'val_freq/loss_re_{k:03d}': val_re,
                    f'val_freq/loss_im_{k:03d}': val_im,
                    f'val_freq/loss_{k:03d}':    val_loss,
                }, step=step_offset + epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_re  = {kk: v.clone() for kk, v in sur_re.state_dict().items()}
                best_im  = {kk: v.clone() for kk, v in sur_im.state_dict().items()}

    sur_re.load_state_dict(best_re)
    sur_im.load_state_dict(best_im)
    return sur_re.cpu(), sur_im.cpu(), cr_mean, cr_std, ci_mean, ci_std


# ---------------------------------------------------------------------------
# Assemblage du modèle
# ---------------------------------------------------------------------------

def assemble_and_save(model, train_idx, val_idx, test_idx,
                      dataset, N, Nt, K, theta_dim,
                      k_freq, k_svd, dt, alpha_t, lam, rule, save_path):
    """Sauvegarde le LaplaceSVDModel avec toutes les métadonnées."""
    model.set_s_list(dataset.s)
    ckpt = {
        'model_type':  'LaplaceSVDModel',
        'K':           K,
        'Nt':          Nt,
        'N':           N,
        'theta_dim':   theta_dim,
        'k_freq':      k_freq,
        'k_svd':       k_svd,
        'model_state': model.state_dict(),
        'theta_mean':  dataset.theta_mean.numpy(),
        'theta_std':   dataset.theta_std.numpy(),
        'dt':          dt,
        'alpha_t':     alpha_t,
        'lam':         lam,
        'rule':        rule,
        'test_idx':    test_idx,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"Modèle sauvegardé → {save_path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Hyperparamètres ──────────────────────────────────────────────────────
    data_path  = '/Data/KAT/ch4_rotated.npy'
    doe_path   = '/Data/KAT/doe_rotated.npy'
    Nt_data    = 150          # nombre de pas de temps du dataset ch4_rotated
    dt         = 1.0          # pas de temps
    rule       = 'trap'       # règle de quadrature (forward et inverse)
    k_svd      = 100          # composantes SVD par (fréquence, composante)
    alpha_t        = 0.092214
    lam            = 0.32193
    n_epochs   = 200
    lr         = 1e-3
    batch_size = 128
    save_path  = 'checkpoints/LaplaceSVDModel.pt'
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s_list = [0.0233+0.0000j, 0.0233+0.0435j, 0.0234+0.0913j, 0.0247+0.1426j, 0.0245+0.1971j, 0.0254+0.2533j, 0.0260+0.3107j, 0.0261+0.3697j, 0.0265+0.4298j, 0.0270+0.4911j, 0.0273+0.5534j, 0.0276+0.6163j, 0.0280+0.6800j, 0.0284+0.7446j, 0.0287+0.8095j, 0.0291+0.8744j, 0.0295+0.9397j, 0.0298+1.0054j, 0.0296+1.0689j, 0.0309+1.1200j]
    s_list = np.array(s_list)
    # ─────────────────────────────────────────────────────────────────────────

    k_freq = len(s_list)
    wandb.init(project='convdiff', name=f'LaplaceSVD_kf{k_freq}_ks{k_svd}')

    # Dataset avec transformée de Laplace (calcule/charge le cache)
    dataset = TransientDataset(data_path, doe_path=doe_path,
                               laplace=True, s_list=s_list, rule=rule, dt=dt)
    ns        = dataset.ns
    N         = dataset.N
    Nt        = dataset.Nt
    K    = dataset.K
    theta_dim = dataset.theta_dim

    # Split test fixe, val aléatoire sur le reste
    _split    = np.load('dataset/split.npz')
    test_idx  = _split['test_idx'].tolist()
    non_test  = np.array([i for i in range(ns) if i not in set(test_idx)])
    rng       = np.random.default_rng(42)
    perm      = rng.permutation(len(non_test))
    n_train   = int(0.8 * len(non_test))
    train_idx = non_test[perm[:n_train]].tolist()
    val_idx   = non_test[perm[n_train:]].tolist()
    dataset.fit(train_idx)

    # Theta normalisé pour tous les samples
    theta_all      = dataset.theta                              # (ns, theta_dim)
    theta_mean     = dataset.theta_mean                         # (theta_dim,)
    theta_std      = dataset.theta_std                          # (theta_dim,)
    theta_norm_all = (theta_all - theta_mean) / theta_std      # (ns, theta_dim)

    theta_norm_train = theta_norm_all[train_idx]               # (ns_train, theta_dim)
    theta_norm_val   = theta_norm_all[val_idx]                 # (ns_val,   theta_dim)

    # Modèle global
    model = LaplaceSVDModel(k_freq, K, Nt, N, theta_dim, k_svd)
    svd_errors_re = []
    svd_errors_im = []

    # ── Boucle fréquences ────────────────────────────────────────────────────
    for k in tqdm(range(k_freq), desc='Fréquences'):
        # 1. SVD
        tqdm.write(f"\n[freq {k:03d}] Calcul SVD tronquée...")
        (Vt_re, Vt_im,
         coeff_re_train, coeff_im_train,
         coeff_re_val,   coeff_im_val,
         err_svd_re,     err_svd_im) = compute_svd_for_freq(
            dataset.U_laplace, train_idx, val_idx, k, k_svd, N
        )
        tqdm.write(f"  → Erreur SVD val (k_svd={k_svd}) : Re {err_svd_re:.4f}, Im {err_svd_im:.4f}")
        svd_errors_re.append(err_svd_re)
        svd_errors_im.append(err_svd_im)


        # 2. Entraînement
        #tqdm.write(f"[freq {k:03d}] Entraînement ({n_epochs} epochs, device={device})...")
        # (sur_re, sur_im,
        #  cr_mean, cr_std,
        #  ci_mean, ci_std) = train_one_freq(
        #     k, k_svd, theta_dim, N,
        #     theta_norm_train, theta_norm_val,
        #     coeff_re_train, coeff_re_val,
        #     coeff_im_train, coeff_im_val,
        #     n_epochs, lr, batch_size, device,
        #     step_offset=k * n_epochs,
        # )

        # # 3. Injecter bases SVD dans les surrogates
        # sur_re.set_svd(Vt_re, cr_mean, cr_std)
        # sur_im.set_svd(Vt_im, ci_mean, ci_std)

        # # Copier dans le modèle global (CPU)
        # model.surrogates_re[k].load_state_dict(sur_re.state_dict())
        # model.surrogates_im[k].load_state_dict(sur_im.state_dict())

        # Libérer la mémoire avant la prochaine fréquence
        del Vt_re, Vt_im
        del coeff_re_train, coeff_im_train
        del coeff_re_val,   coeff_im_val
        #del sur_re, sur_im

    # ── Plot erreur SVD par fréquence ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = list(range(k_freq))
    ax.plot(freqs, svd_errors_re, 'o-', label='Re')
    ax.plot(freqs, svd_errors_im, 's--', label='Im')
    ax.set_xlabel('Indice de fréquence k')
    ax.set_ylabel('Erreur relative L2')
    ax.set_title(f'Erreur approximation SVD tronquée (k_svd={k_svd}) sur val set')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_path = f'plots/LaplaceSVDModel_svd_error_ksvd{k_svd}.png'
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot SVD error → {plot_path}")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    assemble_and_save(model, train_idx, val_idx, test_idx,
                      dataset, N, Nt, K, theta_dim,
                      k_freq, k_svd, dt, alpha_t, lam, rule, save_path)

    wandb.finish()
