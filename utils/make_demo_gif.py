"""
make_demo_gif.py — GIF de démonstration du dataset CH4 transitoire.
Montre 4 simulations côte à côte animées sur les 150 pas de temps.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter

DATA_PATH = "dataset/ch4_rotated.npy"
DOE_PATH  = "dataset/doe_rotated.npy"
OUT_PATH  = "plots/ch4_demo.gif"
FPS       = 15
STEP      = 2   # affiche 1 frame sur STEP (150/2 = 75 frames)
CMAP      = "RdBu_r"
LEVELS    = 50


def pick_indices(doe, n=4):
    """Choisit n simulations variées via quantiles sur k et A."""
    k_vals = np.array([d[0] for d in doe])
    A_vals = np.array([d[1] for d in doe])
    # quartiles croisés
    k_lo, k_hi = np.percentile(k_vals, [20, 80])
    A_lo, A_hi = np.percentile(A_vals, [20, 80])
    masks = [
        (k_vals <= k_lo) & (A_vals <= A_lo),
        (k_vals <= k_lo) & (A_vals >= A_hi),
        (k_vals >= k_hi) & (A_vals <= A_lo),
        (k_vals >= k_hi) & (A_vals >= A_hi),
    ]
    idxs = []
    for m in masks:
        candidates = np.where(m)[0]
        idxs.append(int(candidates[len(candidates) // 2]))
    return idxs


def make_gif():
    print("Chargement du dataset...")
    ch4 = np.load(DATA_PATH, mmap_mode='r')   # (8100, 150, 200, 200)
    doe = np.load(DOE_PATH)                     # structured array

    idxs = pick_indices(doe)
    print(f"Simulations sélectionnées : {idxs}")

    # Pré-charger les 4 simulations choisies
    sims = [ch4[i, ::STEP] for i in idxs]      # (Nt_sub, 200, 200)
    params = [doe[i] for i in idxs]
    Nt_sub = sims[0].shape[0]

    # Colormap et normalisation communes (style animate.py)
    vmin = min(s.min() for s in sims)
    vmax = max(s.max() for s in sims)
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[CMAP]

    x = np.linspace(-100, 100, sims[0].shape[2])
    y = np.linspace(-100, 100, sims[0].shape[1])
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.subplots_adjust(left=0.04, right=0.88, top=0.85, bottom=0.05, wspace=0.12)

    cfs = []
    for ax, sim, p in zip(axes, sims, params):
        ax.set_aspect("equal")
        ax.set_title(f"k={p[0]:.2f}  A={p[1]:.0f}  C={p[2]:.3f}", fontsize=9, pad=4)
        cf = ax.contourf(X, Y, sim[0], levels=LEVELS, cmap=cmap_obj, norm=norm)
        cfs.append(cf)

    cbar_ax = fig.add_axes([0.90, 0.05, 0.015, 0.80])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                 cax=cbar_ax, label="CH₄ concentration")

    title = fig.suptitle("Dataset CH₄ transitoire — t = 0", fontsize=12)

    def update(frame):
        for ax, sim in zip(axes, sims):
            for c in ax.collections:
                c.remove()
            ax.contourf(X, Y, sim[frame], levels=LEVELS, cmap=cmap_obj, norm=norm)
        t_label = frame * STEP + 1
        title.set_text(f"Dataset CH₄ transitoire — pas de temps {t_label:3d} / 150")

    anim = FuncAnimation(fig, update, frames=Nt_sub, interval=1000 // FPS, blit=False)

    print(f"Écriture du GIF dans {OUT_PATH} ...")
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS), dpi=110)
    plt.close(fig)
    print("Terminé :", OUT_PATH)


if __name__ == '__main__':
    make_gif()
