import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from utils.SVD_Amine_3D import svd_inverse_3d
from utils.animate import animate_comparaison


def load_data(svd_path, doe_path):
    svd = np.load(svd_path)
    G   = svd['G'].astype(np.float64)          # (ns, nf_eff)

    doe = np.load(doe_path)
    if doe.dtype.names:
        theta = np.column_stack([doe[k] for k in doe.dtype.names]).astype(np.float64)
    else:
        theta = doe.astype(np.float64)          # (ns, theta_dim)

    assert theta.shape[0] == G.shape[0]
    return theta, G, svd


def train_and_evaluate(svd_path, doe_path, concentration_path, step=5, degree=2, alpha=1.0, seed=42, ckpt_path='checkpoints/SVDSurrogate_sklearn.pkl'):
    theta, G, svd = load_data(svd_path, doe_path)
    ns = theta.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(ns)
    n_train = int(0.8 * ns)
    n_val   = int(0.1 * ns)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    theta_train, G_train = theta[train_idx], G[train_idx]
    theta_val,   G_val   = theta[val_idx],   G[val_idx]
    theta_test,  G_test  = theta[test_idx],  G[test_idx]

    # Pipeline : features polynomiales + régression Ridge
    model = Pipeline([
        ('poly',  PolynomialFeatures(degree=degree, include_bias=True)),
        ('scale', StandardScaler()),
        ('ridge', Ridge(alpha=alpha)),
    ])

    model.fit(theta_train, G_train)

    val_pred  = model.predict(theta_val)
    val_mse   = np.mean((val_pred - G_val) ** 2)
    print(f"Degré={degree}  alpha={alpha}  |  Val MSE (espace G) : {val_mse:.4e}")

    # Sauvegarde
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump({'model': model, 'test_idx': test_idx}, ckpt_path)
    print(f"Modèle sauvegardé : {ckpt_path}")

    # Évaluation test
    F, P, alph = svd['F'], svd['P'], svd['alph']
    nr, nf_eff = F.shape
    Nt         = P.shape[0]
    Hsub = Wsub = int(np.round(np.sqrt(nr)))
    assert Hsub * Wsub == nr, "Grille non carrée — ajuste Hsub/Wsub manuellement"

    concentration = np.load(concentration_path)
    conc_sub = concentration[:, :, ::step, ::step]  # (ns, Nt, Hsub, Wsub)

    test_pred = model.predict(theta_test)   # (n_test, nf_eff)

    def to_field(G_row):
        return svd_inverse_3d(F, G_row[None, :], P, alph)[:, 0, :].reshape(Hsub, Wsub, Nt).transpose(2, 0, 1)

    mae_svd, mse_svd, l2rel_svd = [], [], []
    mae_orig, mse_orig, l2rel_orig, l2rel_svd_vs_orig = [], [], [], []
    for i, si in enumerate(test_idx):
        rec      = to_field(test_pred[i])
        ref_svd  = to_field(G_test[i])
        ref_orig = conc_sub[si]
        norm_orig = np.linalg.norm(ref_orig) + 1e-12

        mae_svd.append(np.abs(rec - ref_svd).mean())
        mse_svd.append(((rec - ref_svd) ** 2).mean())
        l2rel_svd.append(np.linalg.norm(rec - ref_svd) / (np.linalg.norm(ref_svd) + 1e-12))

        mae_orig.append(np.abs(rec - ref_orig).mean())
        mse_orig.append(((rec - ref_orig) ** 2).mean())
        l2rel_orig.append(np.linalg.norm(rec - ref_orig) / norm_orig)
        l2rel_svd_vs_orig.append(np.linalg.norm(ref_svd - ref_orig) / norm_orig)

    mae_svd,  mse_svd,  l2rel_svd  = np.array(mae_svd),  np.array(mse_svd),  np.array(l2rel_svd)
    mae_orig, mse_orig, l2rel_orig  = np.array(mae_orig), np.array(mse_orig), np.array(l2rel_orig)
    l2rel_svd_vs_orig = np.array(l2rel_svd_vs_orig)

    print(f"vs SVD  — MAE: {mae_svd.mean():.4e} ± {mae_svd.std():.4e}  |  MSE: {mse_svd.mean():.4e}  |  L2rel: {l2rel_svd.mean():.4e} ± {l2rel_svd.std():.4e}")
    print(f"vs Orig — MAE: {mae_orig.mean():.4e} ± {mae_orig.std():.4e}  |  MSE: {mse_orig.mean():.4e}  |  L2rel surrogate: {l2rel_orig.mean():.4e} ± {l2rel_orig.std():.4e}")
    print(f"SVD seul vs Orig — L2rel: {l2rel_svd_vs_orig.mean():.4e} ± {l2rel_svd_vs_orig.std():.4e}")

    # Histogrammes : 2 lignes (vs SVD, vs Original) × 2 colonnes (MAE, MSE)
    os.makedirs('plots', exist_ok=True)
    _, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (mae_arr, mse_arr, ref_label) in enumerate([
        (mae_svd,  mse_svd,  'vs SVD reconstruction'),
        (mae_orig, mse_orig, 'vs Original'),
    ]):
        for col, (arr, metric) in enumerate([(mae_arr, 'MAE'), (mse_arr, 'MSE')]):
            ax = axes[row, col]
            ax.hist(arr, bins=np.logspace(np.log10(arr.min()), np.log10(arr.max()), 30),
                    color='steelblue' if col == 0 else 'salmon', edgecolor='white')
            ax.set_xscale('log')
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} {ref_label}\nμ={arr.mean():.3e}  σ={arr.std():.3e}')
    plt.tight_layout()
    hist_path = os.path.join('plots', 'surrogate_sklearn_hist.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Histogrammes sauvegardés : {hist_path}")

    # Animations best/median/worst selon L2rel vs original
    sorted_idx = np.argsort(l2rel_orig)
    picks  = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    labels = ['best', 'median', 'worst']
    for pick, label in zip(picks, labels):
        si = test_idx[pick]
        rec      = to_field(test_pred[pick])
        ref_svd  = to_field(G_test[pick])
        ref_orig = conc_sub[si]
        animate_comparaison(
            ref_svd, rec,
            output_path=os.path.join('plots', f'surrogate_sklearn_{label}_vs_svd.gif'),
            title_fn=lambda t, s=si, l=label: f"#{s} ({l}) SVD vs Surrogate — t={t}",
        )
        animate_comparaison(
            ref_orig, rec,
            output_path=os.path.join('plots', f'surrogate_sklearn_{label}_vs_orig.gif'),
            title_fn=lambda t, s=si, l=label: f"#{s} ({l}) Original vs Surrogate — t={t}",
        )


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Results')

    svd_path  = os.path.join(results_dir, 'svd_train.npz')
    doe_path  = os.path.join(results_dir, 'doe.npy')
    ckpt_path = os.path.join('checkpoints', 'SVDSurrogate_sklearn.pkl')

    degree = 3
    alpha  = 1.0   # régularisation Ridge

    concentration_path = os.path.join(results_dir, 'CH4.npy')
    step = 5

    train_and_evaluate(svd_path, doe_path, concentration_path, step=step, degree=degree, alpha=alpha, ckpt_path=ckpt_path)
