"""
Génération du dataset
====================================
Chaque sample est une paire :
  θ = (D, bx, by, f)   :vecteur de paramètres physiques
  U = grille N×N        :solution de l'EDP

La position source est fixée à x0 = (0.5, 0.5).

Stockage : fichier .npz  (numpy compressé)
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path

from sim import simulate, to_grid



PARAM_RANGES = {
    #          min      max
    'D'  : (1e-3,    1e-1 ),   # diffusivité
    'bx' : (-2.0,    2.0  ),   # convection x
    'by' : (-2.0,    2.0  ),   # convection y
    'f'  : (1.0,     20.0 ),   # intensité source
}

X0_FIXED = np.array([0.5, 0.5])   # position source fixée


def sample_params(rng):
    """Tire un vecteur θ aléatoire dans PARAM_RANGES."""
    p = {}
    for name, (lo, hi) in PARAM_RANGES.items():
        if name == 'D':
            # Log-uniforme pour D (couvre mieux les ordres de grandeur)
            p[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            p[name] = float(rng.uniform(lo, hi))
    return p


def params_to_vector(p):
    """Convertit le dict de paramètres en vecteur numpy (4,)."""
    return np.array([p['D'], p['bx'], p['by'], p['f']], dtype=np.float32)



def generate_dataset(
    n_samples  = 5000,
    N_grid     = 64,
    N_mesh     = 32,
    seed       = 42,
    output_path= 'dataset.npz',
    use_supg   = True,
):
    """
    Génère n_samples paires (θ, U) et les sauvegarde dans output_path.

    Paramètres
    ----------
    n_samples   : nombre de simulations
    N_grid      : résolution de la grille de sortie (N×N)
    N_mesh      : résolution du maillage FEM interne
    seed        : graine aléatoire
    output_path : chemin du fichier .npz de sortie
    use_supg    : activer la stabilisation SUPG
    """
    rng = np.random.default_rng(seed)

    # Pré-allouer les tableaux
    U_all     = np.zeros((n_samples, N_grid, N_grid), dtype=np.float32)
    theta_all = np.zeros((n_samples, 4),              dtype=np.float32)

    n_ok   = 0   # simulations réussies
    n_fail = 0   # simulations échouées

    pbar = tqdm(total=n_samples, desc='Génération dataset')

    while n_ok < n_samples:

        p = sample_params(rng)

        try:
            # Résoudre l'EDP
            u_sol = simulate(
                D      = p['D'],
                b_val  = np.array([p['bx'], p['by']]),
                f      = p['f'],
                x0     = X0_FIXED,
                n      = N_mesh,
                use_supg = use_supg,
            )

            # Interpoler sur grille N×N
            U = to_grid(u_sol, N_out=N_grid)

            # Vérification basique : la solution ne doit pas diverger
            if not np.isfinite(U).all():
                raise ValueError("Solution non finie")
            if np.abs(U).max() > 1e6:
                raise ValueError(f"Solution divergente : max={np.abs(U).max():.2e}")

            # Stocker
            U_all[n_ok]     = U
            theta_all[n_ok] = params_to_vector(p)
            n_ok += 1
            pbar.update(1)

        except Exception as e:
            n_fail += 1
            if n_fail % 50 == 0:
                print(f"\n[warn] {n_fail} échecs cumulés. Dernier : {e}")

    pbar.close()

    # Normalisation des paramètres
    theta_mean = theta_all.mean(axis=0)
    theta_std  = theta_all.std(axis=0) + 1e-8   # éviter division par zéro

    theta_norm = (theta_all - theta_mean) / theta_std

    np.savez_compressed(
        output_path,
        U          = U_all,          # (n_samples, N, N)   float32
        theta      = theta_all,      # (n_samples, 4)      float32  brut
        theta_norm = theta_norm,      # (n_samples, 4)      float32  normalisé
        theta_mean = theta_mean,      # (4,)  pour dénormaliser plus tard
        theta_std  = theta_std,       # (4,)
        param_names= np.array(['D', 'bx', 'by', 'f']),
        N_grid     = np.array([N_grid]),
        N_mesh     = np.array([N_mesh]),
    )

    print(f"\nDataset sauvegardé → {output_path}")
    print(f"  Samples OK    : {n_ok}")
    print(f"  Échecs        : {n_fail}")
    print(f"  Shape U       : {U_all.shape}")
    print(f"  Shape theta   : {theta_all.shape}")
    print(f"  Taille disque : {Path(output_path).stat().st_size / 1e6:.1f} MB")

    return U_all, theta_all, theta_norm



def check_dataset(path='dataset.npz'):
    """Charge et inspecte le dataset."""
    import matplotlib.pyplot as plt

    data        = np.load(path, allow_pickle=True)
    U           = data['U']
    theta       = data['theta']
    param_names = data['param_names']

    print(f"U       : {U.shape}    min={U.min():.3f}  max={U.max():.3f}")
    print(f"theta   : {theta.shape}")
    print()

    # Stats par paramètre
    print("Statistiques des paramètres :")
    for i, name in enumerate(param_names):
        print(f"  {name:5s}  min={theta[:,i].min():.3e}  "
              f"max={theta[:,i].max():.3e}  "
              f"mean={theta[:,i].mean():.3e}")

    # Afficher 6 solutions aléatoires
    rng  = np.random.default_rng(0)
    idxs = rng.choice(len(U), size=6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, idx in zip(axes.flat, idxs):
        im = ax.imshow(U[idx], origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, shrink=0.8)

        p = theta[idx]
        ax.set_title(
            f"D={p[0]:.3f}  b=({p[1]:.1f},{p[2]:.1f})  f={p[3]:.1f}",
            fontsize=8
        )
        # Marquer la source (position fixe)
        ax.plot(0.5, 0.5, 'co', markersize=6)

    plt.suptitle(f'Dataset — {len(U)} samples', fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/dataset_preview.png', dpi=150)
    plt.show()



if __name__ == '__main__':
    import argparse
    DEFAULT_PATH = 'dataset/dataset_huge.npz'

    parser = argparse.ArgumentParser()
    parser.add_argument('--check', nargs='?', const=DEFAULT_PATH, default=None,
                        metavar='PATH',
                        help=f'Inspecter un dataset existant (défaut : {DEFAULT_PATH})')
    args = parser.parse_args()

    if args.check is not None:
        check_dataset(args.check)
    else:
        generate_dataset(
            n_samples   = 50_000,
            N_grid      = 64,
            N_mesh      = 64,
            output_path = DEFAULT_PATH,
        )
