"""
laplace_complex_plane.py — Champs spatiaux hatU(s) dans le plan complexe.

Pour une simulation choisie, calcule hatU(gamma + i*omega) pour chaque point
(gamma, omega) d'une grille et affiche le champ spatial (H×W) réel en chaque point.

Grille :
  - Re(s) = gamma : quelques valeurs négatives ∈ [-0.1, -0.01],
                    zéro, puis positives jusqu'à la limite underflow float64
                    (exp(-gamma*dt) >= float64_eps, soit ~36 pour dt=1)
  - Im(s) = omega : log-espacé de 1/T à 5*pi/dt (bien au-delà du Nyquist pi/dt)

Calcul direct (quadrature trapézoïdale vectorisée sur tous les omega simultanément).

Sortie — trois fichiers distincts :
  plots/laplace_fields_re.png   — Re(hatU(s))  par point de grille
  plots/laplace_fields_im.png   — Im(hatU(s))  par point de grille
  plots/laplace_fields_abs.png  — |hatU(s)|    par point de grille

Run:
    .conda/bin/python transient/laplace_complex_plane.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


# ── Calcul du noyau vectorisé ─────────────────────────────────────────────────

def compute_laplace_multi_omega(U_flat: np.ndarray, omegas: np.ndarray,
                                gamma: float, dt: float) -> np.ndarray:
    """
    Transformée de Laplace trapézoïdale pour un gamma et plusieurs omegas.

        hatU(gamma + i*omega_k) = dt * sum_n w_n * U[n] * exp(-(gamma+i*omega_k)*t_n)

    Paramètres
    ----------
    U_flat : (HW, Nt)      float64
    omegas : (N_omega,)    float64
    gamma  : float         Re(s)
    dt     : float

    Retour
    ------
    M : (HW, N_omega)  complexe  (complex128)
    """
    Nt    = U_flat.shape[1]
    t     = np.arange(Nt, dtype=np.float64) * dt
    w     = np.ones(Nt, dtype=np.float64); w[0] = 0.5; w[-1] = 0.5

    amp   = dt * w * np.exp(-gamma * t)              # (Nt,)
    phase = omegas[:, None] * t[None, :]             # (N_omega, Nt)
    k_re  =  amp[None, :] * np.cos(phase)            # (N_omega, Nt)
    k_im  = -amp[None, :] * np.sin(phase)            # (N_omega, Nt)

    # (HW, Nt) @ (Nt, N_omega) → (HW, N_omega) par composante
    M_re = U_flat @ k_re.T
    M_im = U_flat @ k_im.T
    return M_re + 1j * M_im                          # (HW, N_omega) complex128


# ── Grilles ───────────────────────────────────────────────────────────────────

def build_gamma_grid(dt: float, N_neg: int, N_pos: int) -> np.ndarray:
    """
    Grille Re(s) :
      - négatif : log-espacé entre -0.1 et -0.01
      - zéro
      - positif : log-espacé de 0.01 à la limite underflow (≈36 pour dt=1)
    """
    re_max_pos = float(-np.log(np.finfo(np.float64).eps) / dt)
    gamma_neg  = -np.logspace(np.log10(0.01), np.log10(0.1), N_neg)[::-1]
    gamma_zero = np.array([0.0])
    gamma_pos  = np.logspace(-2, np.log10(re_max_pos), N_pos)
    return np.concatenate([gamma_neg, gamma_zero, gamma_pos])


def build_omega_grid(Nt: int, dt: float, N: int) -> np.ndarray:
    """Log-espacé de 1/(Nt*dt) à 5*pi/dt (5× le Nyquist)."""
    T = Nt * dt
    return np.logspace(np.log10(1.0 / T), np.log10(5.0 * np.pi / dt), N)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_field_grid(fields: np.ndarray, gammas: np.ndarray, omegas: np.ndarray,
                    title: str, cmap: str, diverging: bool, path: str,
                    nyquist: float):
    """
    Grille de champs spatiaux (H×W) orientée comme la heatmap :
      colonnes → Re(s) = gamma  (de gauche à droite : négatif → positif)
      lignes   → Im(s) = omega  (de bas en haut : basse fréq → haute fréq)

    fields : (N_gamma, N_omega, H, W)
    """
    Ng, No, _, _ = fields.shape
    # No lignes (omegas, ordre inversé : plus haut en haut) × Ng colonnes (gammas)
    fig, axes = plt.subplots(No, Ng,
                             figsize=(Ng * 2.0, No * 2.0),
                             squeeze=False)

    for gi in range(Ng):
        for oi in range(No):
            row = No - 1 - oi          # omega croissant vers le haut
            ax  = axes[row, gi]
            img = fields[gi, oi]       # (H, W)

            if diverging:
                vmax = float(np.abs(img).max()) or 1.0
                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax,
                          aspect='auto', origin='lower')
            else:
                vmax = float(img.max()) or 1.0
                ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
                          aspect='auto', origin='lower')
            ax.axis('off')

    # Étiquettes gamma en haut (première ligne)
    for gi, gm in enumerate(gammas):
        axes[0, gi].set_title(f'γ={gm:.2g}', fontsize=5.5, pad=2)

    # Étiquettes omega à gauche (première colonne), ordre inversé
    for oi in range(No):
        row = No - 1 - oi
        nyq = ' *' if omegas[oi] > nyquist else ''
        axes[row, 0].text(-0.05, 0.5, f'ω={omegas[oi]:.2g}{nyq}',
                          transform=axes[row, 0].transAxes,
                          ha='right', va='center', fontsize=6, clip_on=False)

    fig.suptitle(title + f'\n(* au-delà du Nyquist ω={nyquist:.2f})',
                 fontsize=10, y=1.005)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Config ────────────────────────────────────────────────────────────────
    DATA_PATH = 'dataset/ch4_rotated.npy'
    dt        = 1.0
    Nt        = 150
    SIM_IDX   = 0        # simulation à visualiser
    N_NEG     = 3        # points Re(s) < 0  dans [-0.1, -0.01]
    N_POS     = 7        # points Re(s) > 0  dans [0.01, ~36]
    N_OMEGA   = 10       # points Im(s)
    OUT_RE    = 'plots/laplace_fields_re.png'
    OUT_IM    = 'plots/laplace_fields_im.png'
    OUT_ABS   = 'plots/laplace_fields_abs.png'

    # ── Données ───────────────────────────────────────────────────────────────
    U_raw = np.load(DATA_PATH, mmap_mode='r')              # (ns, Nt, H, W)
    _, _, H, W = U_raw.shape

    U      = U_raw[SIM_IDX].astype(np.float64)             # (Nt, H, W)
    U_flat = U.reshape(Nt, H * W).T                        # (H*W, Nt)

    # ── Grilles ───────────────────────────────────────────────────────────────
    gammas  = build_gamma_grid(dt, N_NEG, N_POS)           # (N_NEG+1+N_POS,)
    omegas  = build_omega_grid(Nt, dt, N_OMEGA)            # (N_OMEGA,)
    Ng      = len(gammas)
    nyquist = np.pi / dt

    print(f"Simulation #{SIM_IDX}  —  U: {U.shape}  |  H×W={H}×{W}")
    print(f"Grille : {Ng} γ × {N_OMEGA} ω  →  {Ng * N_OMEGA} champs spatiaux")
    print(f"  Re(s) : {gammas}")
    print(f"  Im(s) ∈ [{omegas[0]:.4f}, {omegas[-1]:.4f}]"
          f"  (Nyquist={nyquist:.4f})")

    # ── Calcul des champs (N_gamma, N_omega, H, W) ───────────────────────────
    fields_re  = np.zeros((Ng, N_OMEGA, H, W), dtype=np.float64)
    fields_im  = np.zeros_like(fields_re)
    grid_abs = np.zeros((Ng, N_OMEGA))   # intensité moyenne par point de grille

    for gi, gamma in enumerate(tqdm(gammas, desc='γ (Re(s))')):
        M = compute_laplace_multi_omega(U_flat, omegas, gamma, dt)
        # M : (H*W, N_omega) complex128
        fields_re[gi] = np.real(M).T.reshape(N_OMEGA, H, W)
        fields_im[gi] = np.imag(M).T.reshape(N_OMEGA, H, W)
        grid_abs[gi]  = np.abs(M).mean(axis=0)              # (N_omega,) scalaire

    # ── Trois figures séparées ────────────────────────────────────────────────
    print("\nGénération des figures...")
    plot_field_grid(fields_re, gammas, omegas,
                    f'Re$[\\hat{{U}}(s)]$  —  simulation #{SIM_IDX}',
                    'RdBu_r', True, OUT_RE, nyquist)

    plot_field_grid(fields_im, gammas, omegas,
                    f'Im$[\\hat{{U}}(s)]$  —  simulation #{SIM_IDX}',
                    'RdBu_r', True, OUT_IM, nyquist)

    # Heatmap scalaire de l'intensité dans le plan complexe
    pos  = grid_abs[grid_abs > 0]
    norm = mcolors.LogNorm(vmin=float(pos.min()) if len(pos) else 1e-10,
                           vmax=float(grid_abs.max()))
    fig, ax = plt.subplots(figsize=(8, 5))
    pm = ax.pcolormesh(gammas, omegas, grid_abs.T,
                       cmap='plasma', norm=norm, shading='nearest')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('log')
    ax.axvline(0,       color='white', lw=0.8, ls='--', alpha=0.6)
    ax.axhline(nyquist, color='white', lw=0.8, ls=':',  alpha=0.6,
               label=f'Nyquist ω={nyquist:.2f}')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Re(s) = γ')
    ax.set_ylabel('Im(s) = ω')
    ax.set_title(f'Intensité moyenne $\\langle|\\hat{{U}}(s)|\\rangle$'
                 f'  —  simulation #{SIM_IDX}')
    plt.colorbar(pm, ax=ax, label='|hatU|  [moy. spatiale]')
    plt.tight_layout()
    plt.savefig(OUT_ABS, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {OUT_ABS}")

    print("Done.")
