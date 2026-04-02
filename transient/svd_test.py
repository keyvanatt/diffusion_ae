import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.SVD_Amine_3D import svd_amine_3d, svd_inverse_3d
from utils.animate import animate_comparaison

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
nf = min(nr, nt)          # rang maximal de la matrice (nr, nt)

# frames_sub : (Nt, Hsub, Wsub) → HH : (nr, 1, nt)
# On regroupe d'abord les dimensions spatiales, puis on transpose
# pour que l'axe 0 = espace et l'axe 2 = temps.
HH = frames_sub.reshape(Nt, nr).T[:, np.newaxis, :]  # (nr, 1, nt)

print(f"Tenseur HH : {HH.shape}  |  nf={nf}")

# --- Décomposition Tucker rang-1 (SVD Amine 3D) ---
erreur = 1e-8
print(f"SVD Amine 3D  (nf={nf}, erreur={erreur})...")
F, G, P, alph, Hist_ErrL2 = svd_amine_3d(HH, nf=nf, erreur=erreur)
nf_eff = len(alph)
print(f"  Modes retenus : {nf_eff}")
print(f"  Erreur L2 finale : {Hist_ErrL2[-1]:.6e}")

# --- Reconstruction ---
HH_rec = svd_inverse_3d(F, G, P, alph)

# Reshape retour en (Nt, Hsub, Wsub)
C_orig = HH[:, 0, :].T.reshape(Nt, Hsub, Wsub)
C_rec  = HH_rec[:, 0, :].T.reshape(Nt, Hsub, Wsub)

rel_err = np.linalg.norm(C_rec - C_orig) / (np.linalg.norm(C_orig) + 1e-12)
print(f"  Erreur L2 reconstruction : {rel_err:.6e}")

x = np.linspace(-100, 100, Wsub)
y = np.linspace(-100, 100, Hsub)
X, Y = np.meshgrid(x, y)

# --- Courbe de convergence ---
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.semilogy(Hist_ErrL2, marker='o', markersize=3)
ax2.set_xlabel("Mode")
ax2.set_ylabel("Erreur L2 relative")
ax2.set_title(f"Convergence SVD Amine 3D  |  Case {case_idx}")
ax2.grid(True, which='both', ls='--', alpha=0.5)
conv_path = os.path.join("plots", 'svd_convergence.png')
plt.savefig(conv_path, dpi=150)
plt.show()
print(f"Courbe de convergence : {conv_path}")

# --- Animation comparaison ---
print("Génération de l'animation de comparaison...")

animate_comparaison(
    C_orig, C_rec,
    os.path.join("plots", 'svd_comparaison.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_a='Original',
    title_b=f'Reconstruit ({nf_eff} modes)',
    title_err='|Erreur|',
    title_fn=lambda t: f"Case {case_idx}  k={k}, A={A}, C={C_inj}  |  t={t}",
)
