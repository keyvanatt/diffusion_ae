import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.SVD_Amine_3D import svd_amine_3d, svd_inverse_3d
from utils.animate import animate

results_dir = os.path.join(os.path.dirname(__file__), '..', 'Results')
concentration = np.load(os.path.join(results_dir, 'CH4.npy'))  # (cases, T, H, W)
doe = np.load(os.path.join(results_dir, 'doe.npy'))             # structured array (k, A, C)

case_idx = 2
frames = concentration[case_idx]  # (T, H, W) = (150, 200, 200)
k, A, C_inj = doe[case_idx]['k'], doe[case_idx]['A'], doe[case_idx]['C']
print(f"Case {case_idx}: k={k}, A={A}, C={C_inj}")
print(f"Frames shape: {frames.shape}")

Nt, H, W = frames.shape

# Sous-échantillonnage spatial (step=5 → 40×40 = 1600 nœuds)
step = 5
frames_sub = frames[:, ::step, ::step]  # (Nt, 40, 40)
Hsub, Wsub = frames_sub.shape[1], frames_sub.shape[2]
print(f"Grille sous-échantillonnée : {Hsub}×{Wsub} = {Hsub*Wsub} nœuds")

# Reshape en tenseur (nr, ns, nt) avec ns=1 (un seul cas)
nr = Hsub * Wsub          # 1600
ns = 1
nt = Nt                   # 150
nf = Hsub**2            # 1600 = côté de la grille²

HH = frames_sub.reshape(nr, ns, nt)  # (nr, 1, nt)

print(f"Tenseur HH : {HH.shape}  |  nf={nf}")

# --- Décomposition Tucker rang-1 (SVD Amine 3D) ---
erreur = 1e-6
print(f"SVD Amine 3D  (nf={nf}, erreur={erreur})...")
F, G, P, alph, Hist_ErrL2 = svd_amine_3d(HH, nf=nf, erreur=erreur)
nf_eff = len(alph)
print(f"  Modes retenus : {nf_eff}")
print(f"  Erreur L2 finale : {Hist_ErrL2[-1]:.6e}")

# --- Reconstruction ---
HH_rec = svd_inverse_3d(F, G, P, alph)

# Reshape retour en (Nt, Hsub, Wsub)
C_orig = HH[:, 0, :].reshape(Nt, Hsub, Wsub)
C_rec  = HH_rec[:, 0, :].reshape(Nt, Hsub, Wsub)

rel_err = np.linalg.norm(C_rec - C_orig) / (np.linalg.norm(C_orig) + 1e-12)
print(f"  Erreur L2 reconstruction : {rel_err:.4f}")

x = np.linspace(-100, 100, Wsub)
y = np.linspace(-100, 100, Hsub)
X, Y = np.meshgrid(x, y)

# --- Visualisation statique ---
t_ids = [0, Nt // 2, Nt - 1]
vmin = C_orig.min()
vmax = C_orig.max()
err_abs = np.abs(C_rec - C_orig)

fig, axes = plt.subplots(3, len(t_ids), figsize=(14, 10))
fig.suptitle(
    f"SVD Amine 3D  |  Case {case_idx}  k={k}, A={A}, C={C_inj}  "
    f"|  {nf_eff} modes  |  erreur L2={rel_err:.4f}",
    fontsize=11
)

for col, t in enumerate(t_ids):
    ax = axes[0, col]
    cf = ax.contourf(X, Y, C_orig[t], levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"Original  t={t}")
    ax.set_aspect('equal')

    ax = axes[1, col]
    cf = ax.contourf(X, Y, C_rec[t], levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"Reconstruit  t={t}")
    ax.set_aspect('equal')

    ax = axes[2, col]
    cf = ax.contourf(X, Y, err_abs[t], levels=40, cmap='hot_r')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"|Erreur|  t={t}")
    ax.set_aspect('equal')

plt.tight_layout()
out_path = os.path.join(results_dir, 'svd_test.png')
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Figure sauvegardée : {out_path}")

# --- Courbe de convergence ---
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.semilogy(Hist_ErrL2, marker='o', markersize=3)
ax2.set_xlabel("Mode")
ax2.set_ylabel("Erreur L2 relative")
ax2.set_title(f"Convergence SVD Amine 3D  |  Case {case_idx}")
ax2.grid(True, which='both', ls='--', alpha=0.5)
conv_path = os.path.join(results_dir, 'svd_convergence.png')
plt.savefig(conv_path, dpi=150)
plt.show()
print(f"Courbe de convergence : {conv_path}")

# --- Animations ---
print("Génération des animations...")

animate(
    C_orig, os.path.join(results_dir, 'svd_original.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_fn=lambda t: f"Original  |  Case {case_idx} k={k}, A={A}  t={t}",
)

animate(
    C_rec, os.path.join(results_dir, 'svd_reconstruit.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_fn=lambda t: f"Reconstruit ({nf_eff} modes)  |  Case {case_idx}  t={t}",
)

animate(
    err_abs, os.path.join(results_dir, 'svd_erreur.gif'),
    fps=10, cmap='hot_r', label='|err|', X=X, Y=Y,
    title_fn=lambda t: f"|Erreur|  |  Case {case_idx}  t={t}",
)
