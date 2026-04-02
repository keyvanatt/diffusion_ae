"""
Génération du dataset
====================================
Chaque sample est une paire :
  θ = (D, bx, by, f)   :vecteur de paramètres physiques
  U = grille N×N        :solution de l'EDP

La position source est fixée à x0 = (0.5, 0.5).

Avec obstacles (with_obstacles=True) :
  rect = (rx0, ry0, rx1, ry1) stocké séparément, zeros si pas de mur.
  has_wall = booléen par sample.
  theta reste (D, bx, by, f) — inchangé.

Stockage : fichier .npz  (numpy compressé)
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path

from sim import ConvDiffSimulator, to_grid, ConvDiffTransientSimulator, to_grid_sequence



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


# Contraintes sur l'obstacle
RECT_MIN_SIDE = 0.08    # côté minimum du rectangle
RECT_MAX_SIDE = 0.30    # côté maximum
RECT_MIN_AREA = 0.02    # aire minimale (fraction du domaine)
RECT_MAX_AREA = 0.10    # aire maximale
RECT_MARGIN   = 0.05    # distance minimale au bord du domaine


def sample_rect(rng, x0=X0_FIXED, b=None, max_tries=100):
    """
    Échantillonne un rectangle obstacle garanti sur la trajectoire des particules.

    Stratégie selon le régime :
    - Convection dominante (|b| > 0.3) : centre sur la ligne de courant x0 + t·b̂,
      avec bruit gaussien perpendiculaire faible. Le mur coupe le flux.
    - Diffusion dominante (|b| ≤ 0.3) : centre à distance t de x0 dans une
      direction aléatoire (le champ se propage radialement).

    Contraintes géométriques :
    - Aire dans [RECT_MIN_AREA, RECT_MAX_AREA]
    - Côtés dans [RECT_MIN_SIDE, RECT_MAX_SIDE]
    - Distance RECT_MARGIN aux bords
    - Ne contient pas la source x0

    Retourne (rx0, ry0, rx1, ry1) ou None si échec.
    """
    b_norm_val = np.linalg.norm(b) if b is not None else 0.0

    for _ in range(max_tries):
        if b_norm_val > 0.3:
            # Régime convectif : placer sur la ligne de courant
            b_hat = b / b_norm_val                        # direction du flux
            b_perp = np.array([-b_hat[1], b_hat[0]])     # direction perpendiculaire

            t     = rng.uniform(0.08, 0.35)               # distance le long du flux
            noise = rng.normal(0.0, 0.04)                 # bruit latéral faible
            cx = x0[0] + t * b_hat[0] + noise * b_perp[0]
            cy = x0[1] + t * b_hat[1] + noise * b_perp[1]
        else:
            # Régime diffusif : direction aléatoire à distance t
            angle = rng.uniform(0, 2 * np.pi)
            t     = rng.uniform(0.08, 0.30)
            cx    = x0[0] + t * np.cos(angle)
            cy    = x0[1] + t * np.sin(angle)

        cx = float(np.clip(cx, RECT_MARGIN + 0.05, 1 - RECT_MARGIN - 0.05))
        cy = float(np.clip(cy, RECT_MARGIN + 0.05, 1 - RECT_MARGIN - 0.05))

        hw = rng.uniform(RECT_MIN_SIDE / 2, RECT_MAX_SIDE / 2)
        hh = rng.uniform(RECT_MIN_SIDE / 2, RECT_MAX_SIDE / 2)

        rx0 = max(cx - hw, RECT_MARGIN)
        ry0 = max(cy - hh, RECT_MARGIN)
        rx1 = min(cx + hw, 1 - RECT_MARGIN)
        ry1 = min(cy + hh, 1 - RECT_MARGIN)

        if rx1 - rx0 < RECT_MIN_SIDE or ry1 - ry0 < RECT_MIN_SIDE:
            continue
        area = (rx1 - rx0) * (ry1 - ry0)
        if not (RECT_MIN_AREA <= area <= RECT_MAX_AREA):
            continue
        if rx0 <= x0[0] <= rx1 and ry0 <= x0[1] <= ry1:
            continue

        return (rx0, ry0, rx1, ry1)

    return None   # échec — appelant doit gérer



def generate_dataset(
    n_samples    = 5000,
    N_grid       = 64,
    N_mesh       = 32,
    seed         = 42,
    output_path  = 'dataset.npz',
    use_supg     = True,
    with_obstacles = False,
    p_wall       = 0.5,
):
    """
    Génère n_samples paires (θ, U) et les sauvegarde dans output_path.

    Paramètres
    ----------
    n_samples      : nombre de simulations
    N_grid         : résolution de la grille de sortie (N×N)
    N_mesh         : résolution du maillage FEM interne
    seed           : graine aléatoire
    output_path    : chemin du fichier .npz de sortie
    use_supg       : activer la stabilisation SUPG
    with_obstacles : activer le tirage d'obstacles rectangulaires
    p_wall         : probabilité d'avoir un mur par sample
    """
    rng = np.random.default_rng(seed)

    U_all      = np.zeros((n_samples, N_grid, N_grid), dtype=np.float32)
    theta_all  = np.zeros((n_samples, 4),              dtype=np.float32)
    rect_all   = np.zeros((n_samples, 4),              dtype=np.float32)  # zeros = pas de mur
    wall_all   = np.zeros(n_samples,                   dtype=bool)

    n_ok   = 0
    n_fail = 0

    sim = ConvDiffSimulator(n=N_mesh, use_supg=use_supg)
    pbar = tqdm(total=n_samples, desc='Génération dataset')

    while n_ok < n_samples:
        p = sample_params(rng)
        b = np.array([p['bx'], p['by']])

        rect = None
        if with_obstacles and rng.random() < p_wall:
            rect = sample_rect(rng, x0=X0_FIXED, b=b)

        try:
            u_sol = sim.solve(
                D     = p['D'],
                b_val = b,
                f     = p['f'],
                x0    = X0_FIXED,
                rect  = rect,
            )

            U = to_grid(u_sol, N_out=N_grid)

            if not np.isfinite(U).all():
                raise ValueError("Solution non finie")
            if np.abs(U).max() > 1e6:
                raise ValueError(f"Solution divergente : max={np.abs(U).max():.2e}")

            U_all[n_ok]     = U
            theta_all[n_ok] = params_to_vector(p)
            if rect is not None:
                rect_all[n_ok] = rect
                wall_all[n_ok] = True
            n_ok += 1
            pbar.update(1)

        except Exception as e:
            n_fail += 1
            if n_fail % 50 == 0:
                print(f"\n[warn] {n_fail} échecs cumulés. Dernier : {e}")

    pbar.close()

    theta_mean = theta_all.mean(axis=0)
    theta_std  = theta_all.std(axis=0) + 1e-8
    theta_norm = (theta_all - theta_mean) / theta_std

    np.savez_compressed(
        output_path,
        U          = U_all,
        theta      = theta_all,
        theta_norm = theta_norm,
        theta_mean = theta_mean,
        theta_std  = theta_std,
        rect       = rect_all,   # (n_samples, 4)  zeros si pas de mur
        has_wall   = wall_all,   # (n_samples,)    bool
        param_names= np.array(['D', 'bx', 'by', 'f']),
        N_grid     = np.array([N_grid]),
        N_mesh     = np.array([N_mesh]),
    )

    print(f"\nDataset sauvegardé → {output_path}")
    print(f"  Samples OK    : {n_ok}  (dont {wall_all.sum()} avec mur)")
    print(f"  Échecs        : {n_fail}")
    print(f"  Shape U       : {U_all.shape}")
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



def generate_dataset_transient(
    n_samples      = 5000,
    N_grid         = 64,
    N_mesh         = 32,
    dt             = 0.02,
    n_steps        = 50,
    seed           = 42,
    output_path    = 'dataset.npz',
    use_supg       = True,
    with_obstacles = False,
    p_wall         = 0.5,
):
    """
    Génère n_samples séquences temporelles (θ, U) et les sauvegarde dans output_path.

    Paramètres
    ----------
    n_samples      : nombre de simulations
    N_grid         : résolution de la grille de sortie (N×N)
    N_mesh         : résolution du maillage FEM interne
    dt             : pas de temps
    n_steps        : nombre de pas de temps par simulation
    seed           : graine aléatoire
    output_path    : chemin du fichier .npz de sortie
    use_supg       : activer la stabilisation SUPG
    with_obstacles : activer le tirage d'obstacles rectangulaires
    p_wall         : probabilité d'avoir un mur par sample
    """
    rng = np.random.default_rng(seed)

    U_all     = np.zeros((n_samples, n_steps, N_grid, N_grid), dtype=np.float32)
    theta_all = np.zeros((n_samples, 4),                       dtype=np.float32)
    rect_all  = np.zeros((n_samples, 4),                       dtype=np.float32)
    wall_all  = np.zeros(n_samples,                            dtype=bool)

    n_ok   = 0
    n_fail = 0

    sim = ConvDiffTransientSimulator(n=N_mesh, dt=dt, use_supg=use_supg)
    pbar = tqdm(total=n_samples, desc='Génération dataset transitoire')

    while n_ok < n_samples:
        p = sample_params(rng)
        b = np.array([p['bx'], p['by']])

        rect = None
        if with_obstacles and rng.random() < p_wall:
            rect = sample_rect(rng, x0=X0_FIXED, b=b)

        try:
            u_list = sim.solve(
                D       = p['D'],
                b_val   = b,
                f       = p['f'],
                x0      = X0_FIXED,
                n_steps = n_steps,
                rect    = rect,
            )

            U = to_grid_sequence(u_list, N_out=N_grid)  # (n_steps, N, N)

            if not np.isfinite(U).all():
                raise ValueError("Solution non finie")
            if np.abs(U).max() > 1e6:
                raise ValueError(f"Solution divergente : max={np.abs(U).max():.2e}")

            U_all[n_ok]     = U
            theta_all[n_ok] = params_to_vector(p)
            if rect is not None:
                rect_all[n_ok] = rect
                wall_all[n_ok] = True
            n_ok += 1
            pbar.update(1)

        except Exception as e:
            n_fail += 1
            if n_fail % 50 == 0:
                print(f"\n[warn] {n_fail} échecs cumulés. Dernier : {e}")

    pbar.close()

    theta_mean = theta_all.mean(axis=0)
    theta_std  = theta_all.std(axis=0) + 1e-8
    theta_norm = (theta_all - theta_mean) / theta_std

    np.savez_compressed(
        output_path,
        U           = U_all,
        theta       = theta_all,
        theta_norm  = theta_norm,
        theta_mean  = theta_mean,
        theta_std   = theta_std,
        rect        = rect_all,
        has_wall    = wall_all,
        param_names = np.array(['D', 'bx', 'by', 'f']),
        N_grid      = np.array([N_grid]),
        N_mesh      = np.array([N_mesh]),
        dt          = np.array([dt]),
        n_steps     = np.array([n_steps]),
    )

    print(f"\nDataset sauvegardé → {output_path}")
    print(f"  Samples OK    : {n_ok}  (dont {wall_all.sum()} avec mur)")
    print(f"  Échecs        : {n_fail}")
    print(f"  Shape U       : {U_all.shape}")
    print(f"  Taille disque : {Path(output_path).stat().st_size / 1e6:.1f} MB")

    return U_all, theta_all, theta_norm


def check_dataset_transient(path='dataset.npz'):
    """Charge et inspecte un dataset transitoire."""
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.animate import animate

    data        = np.load(path, allow_pickle=True)
    U           = data['U']           # (n_samples, n_steps, N, N)
    theta       = data['theta']
    param_names = data['param_names']
    dt          = float(data['dt'][0])
    n_steps     = int(data['n_steps'][0])

    print(f"U       : {U.shape}    min={U.min():.3f}  max={U.max():.3f}")
    print(f"theta   : {theta.shape}")
    print(f"dt={dt}  n_steps={n_steps}  T_end={dt * n_steps:.3f}")
    print()

    print("Statistiques des paramètres :")
    for i, name in enumerate(param_names):
        print(f"  {name:5s}  min={theta[:,i].min():.3e}  "
              f"max={theta[:,i].max():.3e}  "
              f"mean={theta[:,i].mean():.3e}")

    # Aperçu statique : dernier pas de temps de 6 samples aléatoires
    rng  = np.random.default_rng(0)
    idxs = rng.choice(len(U), size=6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, idx in zip(axes.flat, idxs):
        im = ax.imshow(U[idx, -1], origin='lower', cmap='hot', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, shrink=0.8)
        p = theta[idx]
        ax.set_title(f"D={p[0]:.3f}  b=({p[1]:.1f},{p[2]:.1f})  f={p[3]:.1f}", fontsize=8)
        ax.plot(0.5, 0.5, 'co', markersize=6)

    plt.suptitle(f'Dataset transitoire — {len(U)} samples — T_end={dt * n_steps:.2f}', fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/dataset_transient_preview.png', dpi=150)
    plt.show()

    # GIF pour un sample aléatoire
    animate(
        U[idxs[0]],
        output_path='plots/dataset_transient_check.gif',
        fps=10,
        cmap='hot',
        label='u',
        title_fn=lambda t: f"t = {(t + 1) * dt:.3f}",
    )


if __name__ == '__main__':
    CHECK          = False
    TRANSIENT      = True
    WITH_OBSTACLES = True  

    if CHECK:
        if TRANSIENT:
            check_dataset_transient('dataset/dataset_transient.npz')
        else:
            check_dataset('dataset/dataset_huge.npz')
    else:
        if TRANSIENT:
            generate_dataset_transient(
                n_samples      = 5_000,
                N_grid         = 64,
                N_mesh         = 64,
                dt             = 0.05,
                n_steps        = 100,
                output_path    = 'dataset/dataset_transient_obstacles.npz',
                with_obstacles = WITH_OBSTACLES,
                p_wall         = 0.8,
            )
        else:
            generate_dataset(
                n_samples      = 50_000,
                N_grid         = 64,
                N_mesh         = 64,
                output_path    = 'dataset/dataset_huge.npz',
                with_obstacles = WITH_OBSTACLES,
                p_wall         = 0.8,
            )
