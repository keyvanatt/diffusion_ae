import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils.laplace_forward import laplace_forward
from utils.laplace_inverse import laplace_inverse
from utils.animate import animate_comparaison

results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Results')
concentration = np.load(os.path.join(results_dir, 'CH4.npy'))  # (cases, T, H, W)
doe = np.load(os.path.join(results_dir, 'doe.npy'))             # (cases, 3)

case_idx = 2
frames = concentration[case_idx]
D, angle, inj = doe[case_idx]
print(f"Case {case_idx}: D={D}, angle={angle}°, injection={inj} kg/m³/s")

Nt, H, W = frames.shape
step = 5
frames_sub = frames[:, ::step, ::step]
Hsub, Wsub = frames_sub.shape[1], frames_sub.shape[2]
print(f"Grille sous-échantillonnée : {Hsub}×{Wsub} = {Hsub*Wsub} nœuds")

C = frames_sub.reshape(Hsub * Wsub, Nt)
dt = 1.0

print("Forward Laplace...")
M, s, meta = laplace_forward(C, dt, rule='trap', gamma=0.0)
print(f"  M shape: {M.shape} (complex)")

print("Inverse Laplace...")
Crec, info = laplace_inverse(M, dt, Nt, rule='trap', gamma=0.0)
err = np.linalg.norm(Crec - C) / (np.linalg.norm(C) + 1e-12)
print(f"  Partie imaginaire résiduelle : {info['residual_imag_ratio']:.2e}")
print(f"  Erreur L2 reconstruction    : {err:.2e}")

C_orig = C.reshape(Nt, Hsub, Wsub)
C_rec  = Crec.reshape(Nt, Hsub, Wsub)
x = np.linspace(-100, 100, Wsub)
y = np.linspace(-100, 100, Hsub)
X, Y = np.meshgrid(x, y)

print("Génération des animations...")
animate_comparaison(
    C_orig, C_rec,
    os.path.join("plots", 'laplace_comparaison.gif'),
    fps=10, cmap='RdBu_r', label='CH4', X=X, Y=Y,
    title_a="Original", title_b="Reconstruit", title_err="|Erreur|",
    title_fn=lambda t: f"Case {case_idx} D={D}, angle={angle}°  t={t}",
)
