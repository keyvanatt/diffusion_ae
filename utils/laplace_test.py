import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.laplace_forward import laplace_forward
from utils.laplace_inverse import laplace_inverse
from utils.animate import animate

results_dir = os.path.join(os.path.dirname(__file__), '..', 'Results')
concentration = np.load(os.path.join(results_dir, 'CH4.npy'))  # (cases, T, H, W)
doe = np.load(os.path.join(results_dir, 'doe.npy'))             # (cases, 3)

case_idx = 2
frames = concentration[case_idx]  # (T, H, W) = (150, 200, 200)
D, angle, inj = doe[case_idx]
print(f"Case {case_idx}: D={D}, angle={angle}°, injection={inj} kg/m³/s")
print(f"Frames shape: {frames.shape}")

Nt, H, W = frames.shape

# Sous-échantillonnage spatial pour accélérer l'inverse (step=5 → 40×40=1600 nœuds)
step = 5
frames_sub = frames[:, ::step, ::step]  # (Nt, 40, 40)
Hsub, Wsub = frames_sub.shape[1], frames_sub.shape[2]
print(f"Grille sous-échantillonnée : {Hsub}×{Wsub} = {Hsub*Wsub} nœuds")

# Reshape en (Nnodes, Nt) pour les fonctions Laplace
C = frames_sub.reshape(Hsub * Wsub, Nt)  # (1600, 150)

dt = 1.0  # pas de temps normalisé

# --- Transformée de Laplace forward ---
print("Forward Laplace...")
M, s_vals, meta = laplace_forward(C, dt, rule='trap', Ns=60)
print(f"  M shape: {M.shape}, s range: [{s_vals[0]:.3e}, {s_vals[-1]:.3e}]")

# --- Transformée de Laplace inverse ---
print("Inverse Laplace...")
Crec, info = laplace_inverse(
    M, dt, s_vals, Nt,
    E=meta['E'],
    rule='trap',
    Lambda=1e-3,
    LambdaSmooth=1e-3,
)
print(f"  Erreur relative : {info['data_relmisfit']:.4f}")
print(f"  Erreur L2 reconstruction : {np.linalg.norm(Crec - C) / (np.linalg.norm(C) + 1e-12):.4f}")

# Reshape retour en (Nt, Hsub, Wsub)
C_orig = C.reshape(Nt, Hsub, Wsub)
C_rec  = Crec.reshape(Nt, Hsub, Wsub)

x = np.linspace(-100, 100, Wsub)
y = np.linspace(-100, 100, Hsub)
X, Y = np.meshgrid(x, y)

# --- Visualisation ---
t_ids = [0, Nt // 2, Nt - 1]
vmin = C_orig.min()
vmax = C_orig.max()
err_abs = np.abs(C_rec - C_orig)

fig, axes = plt.subplots(3, len(t_ids), figsize=(14, 10))
fig.suptitle(
    f"Laplace forward → inverse  |  Case {case_idx}  D={D}, angle={angle}°",
    fontsize=12
)

for col, t in enumerate(t_ids):
    # Ligne 0 : original
    ax = axes[0, col]
    cf = ax.contourf(X, Y, C_orig[t], levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"Original  t={t}")
    ax.set_aspect('equal')

    # Ligne 1 : reconstruit
    ax = axes[1, col]
    cf = ax.contourf(X, Y, C_rec[t], levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"Reconstruit  t={t}")
    ax.set_aspect('equal')

    # Ligne 2 : erreur absolue
    ax = axes[2, col]
    cf = ax.contourf(X, Y, err_abs[t], levels=40, cmap='hot_r')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"|Erreur|  t={t}")
    ax.set_aspect('equal')

plt.tight_layout()
out_path = os.path.join(results_dir, 'laplace_test.png')
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Figure sauvegardée : {out_path}")

# --- Animations : original, reconstruit, erreur ---
print("Génération des animations...")

animate(
    C_orig, os.path.join(results_dir, 'laplace_original.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_fn=lambda t: f"Original  |  Case {case_idx} D={D}, angle={angle}°  t={t}",
)

animate(
    C_rec, os.path.join(results_dir, 'laplace_reconstruit.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_fn=lambda t: f"Reconstruit  |  Case {case_idx} D={D}, angle={angle}°  t={t}",
)

animate(
    err_abs, os.path.join(results_dir, 'laplace_erreur.gif'),
    fps=10, cmap='hot_r', label='|err|', X=X, Y=Y,
    title_fn=lambda t: f"|Erreur|  |  Case {case_idx} D={D}, angle={angle}°  t={t}",
)
