"""
Regularised Laplace inversion test on real CH4 data.

Given K complex frequencies s_1, ..., s_K:
  1. Forward Laplace on all pixels simultaneously.
  2. Conjugate extension  (Im(s) > 0 only — skips DC duplicate).
  3. Regularised inversion:  A v* = F* û,  A = F*F + α_t Dt*Dt + λI.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.animate import animate_comparaison

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Nt      = 150
dt      = 1.0
K       = 20        # s points before conjugate extension
gamma   = 0.0       # Bromwich damping

case_idx = 0
step     = 1        # spatial subsampling
t_frame  = 50       # time step shown in spatial panel

# ---------------------------------------------------------------------------
# Path generators  —  choose one by setting `path`
# ---------------------------------------------------------------------------
omega_max = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[K - 1]   # highest freq used

def path_bromwich(K, gamma=0.0):
    """Bromwich: horizontal line at Re(s)=gamma, linearly spaced omega."""
    omega = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[:K]
    return gamma + 1j * omega

def path_bromwich_log(K, gamma=0.0):
    """Bromwich log-spaced: denser at low frequencies."""
    omega = np.logspace(np.log10(2 * np.pi / (Nt * dt)),
                        np.log10(omega_max), K)
    return gamma + 1j * omega

def path_quadratic(K, gamma=0.0, a=0.5):
    """Parabolic contour: Re(s) = gamma + a * (omega/omega_max)^2."""
    omega = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[:K]
    return (gamma + a * (omega / (omega_max + 1e-12)) ** 2) + 1j * omega

def path_log_shifted(K, gamma=0.0, c=1.0):
    """Log-horizontal shift: Re(s) = gamma + c * log(1 + omega/omega_1)."""
    omega  = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[:K]
    omega1 = 2 * np.pi / (Nt * dt)
    return (gamma + c * np.log1p(omega / omega1)) + 1j * omega

def path_horizontal_log_re(K, gamma=0.0, omega_0=None, re_min=1e-3, re_max=2.0):
    """Horizontal line at Im(s)=omega_0, Re(s) log-spaced in [re_min, re_max].
    omega_0 defaults to the fundamental frequency 2π/(Nt*dt)."""
    if omega_0 is None:
        omega_0 = 2 * np.pi / (Nt * dt)
    re = np.logspace(np.log10(re_min), np.log10(re_max), K)
    return re + 1j * omega_0


def path_talbot(K, r=1.0, sigma=0.0):
    theta = np.linspace(0.1, np.pi - 0.1, K) # Éviter 0 et pi pour la stabilité initiale
    s = sigma + r * theta * (1/np.tan(theta) + 1j)
    return s

def path_c_shape(K, R=5.0, center=1.0):
    """
    C-shaped contour: Half-circle in the complex plane.
    Focuses density around the most relevant frequencies.
    """
    phi = np.linspace(-np.pi/2, np.pi/2, K)
    s = center + R * (np.cos(phi) + 1j * np.sin(phi))
    return s


s_list = [0.0233+0.0000j, 0.0233+0.0435j, 0.0234+0.0913j, 0.0247+0.1426j, 0.0245+0.1971j, 0.0254+0.2533j, 0.0260+0.3107j, 0.0261+0.3697j, 0.0265+0.4298j, 0.0270+0.4911j, 0.0273+0.5534j, 0.0276+0.6163j, 0.0280+0.6800j, 0.0284+0.7446j, 0.0287+0.8095j, 0.0291+0.8744j, 0.0295+0.9397j, 0.0298+1.0054j, 0.0296+1.0689j, 0.0309+1.1200j]
s_list = np.array(s_list)

alpha_t = 0.042193
lam     = 0.092214


# ---------------------------------------------------------------------------
# Load CH4 case
# ---------------------------------------------------------------------------
data_path = os.path.join("dataset", "ch4_rotated.npy")
C_full = np.load(data_path, mmap_mode='r')[case_idx]          # (Nt, H, W)
C_sub  = C_full[:, ::step, ::step].copy().astype(float)       # (Nt, Hsub, Wsub)
Nt_ch4, Hsub, Wsub = C_sub.shape
assert Nt_ch4 == Nt
V = C_sub.reshape(Nt, -1).T                                   # (Nnodes, Nt)
print(f"Loaded case {case_idx}: {Nt}×{Hsub}×{Wsub}, {V.shape[0]} nodes")

# ---------------------------------------------------------------------------
# Forward Laplace at K s points  (all nodes at once)
# ---------------------------------------------------------------------------
t = np.arange(Nt) * dt
w = np.ones(Nt); w[0] = 0.5; w[-1] = 0.5

F_base = dt * w[None, :] * np.exp(-s_list[:, None] * t[None, :])  # (K, Nt)
U_hat  = V @ F_base.T                                              # (Nnodes, K)

# ---------------------------------------------------------------------------
# Conjugate extension  —  Im(s) > 0 only
# ---------------------------------------------------------------------------
cmask      = s_list.imag > 0
s_full     = np.concatenate([s_list,                  np.conj(s_list[cmask])])
U_hat_full = np.concatenate([U_hat, np.conj(U_hat[:, cmask])], axis=1)
F_full     = dt * w[None, :] * np.exp(-s_full[:, None] * t[None, :])
print(f"Frequencies: {K} + {cmask.sum()} conjugates = {len(s_full)} total")

# ---------------------------------------------------------------------------
# Normal equations:  A = F*F + α_t Dt*Dt + λI
# ---------------------------------------------------------------------------
FH    = np.conj(F_full).T
FtF   = np.real(FH @ F_full)
Dt    = np.eye(Nt - 1, Nt, k=1) - np.eye(Nt - 1, Nt, k=0)
DtTDt = Dt.T @ Dt

A   = FtF + alpha_t * DtTDt + lam * np.eye(Nt)
RHS = np.real(U_hat_full @ FH.T)             # (Nnodes, Nt)

V_rec = np.linalg.solve(A, RHS.T)            # (Nt, Nnodes)
C_rec = V_rec.reshape(Nt, Hsub, Wsub)

err_field = np.linalg.norm(C_rec - C_sub) / (np.linalg.norm(C_sub) + 1e-12)
norm_F    = np.linalg.norm(F_full, ord=2)
sig_min   = np.linalg.eigvalsh(A).min()
print(f"||F||={norm_F:.3e}  σ_min(A)={sig_min:.3e}  "
      f"cond={norm_F/sig_min:.3e}  L2 err={err_field:.3e}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

vmin = min(C_sub[t_frame].min(), C_rec[t_frame].min())
vmax = max(C_sub[t_frame].max(), C_rec[t_frame].max())

axes[0].imshow(C_sub[t_frame], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0].set_title(f'Original  t={t_frame}')

im = axes[1].imshow(C_rec[t_frame], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].set_title(f'Reconstructed  K={K}  err={err_field:.2e}')
plt.colorbar(im, ax=axes[1])

im_e = axes[2].imshow(np.abs(C_rec[t_frame] - C_sub[t_frame]), origin='lower', cmap='hot_r')
axes[2].set_title(f'|Error|  t={t_frame}')
plt.colorbar(im_e, ax=axes[2])

ax = axes[3]
# draw all contours in background for reference
ax.scatter(s_list.real, s_list.imag,
           c='steelblue', s=60, zorder=4, label=f'used')
ax.scatter(np.conj(s_list[cmask]).real, np.conj(s_list[cmask]).imag,
           c='tomato', s=30, marker='x', zorder=4, label=f'conj ({cmask.sum()})')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('Re(s)'); ax.set_ylabel('Im(s)')
ax.set_xscale('symlog', linthresh=1e-3)
ax.set_title(f's points  ({len(s_full)} total)')
ax.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join("plots", "laplace_inversion_test.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure saved → {out_path}")

# ---------------------------------------------------------------------------
# Laplace spectrum figure — Re and Im side by side for each s_k
# ---------------------------------------------------------------------------
U_re = U_hat.real.reshape(Hsub, Wsub, K)   # (Hsub, Wsub, K)
U_im = U_hat.imag.reshape(Hsub, Wsub, K)

ncols_k = min(K, 5)
nrows_k  = (K + ncols_k - 1) // ncols_k
# 2 sub-columns per k (Re | Im)
fig2, axes2 = plt.subplots(nrows_k, ncols_k * 2,
                            figsize=(3.0 * ncols_k * 2, 2.8 * nrows_k))
axes2 = np.array(axes2).reshape(nrows_k, ncols_k, 2)

for k in range(K):
    row, col = divmod(k, ncols_k)
    sk = s_list[k]

    re_map = U_re[:, :, k]
    im_map = U_im[:, :, k]

    for part, data, cmap, label in [
        (0, re_map, 'RdBu_r', 'Re'),
        (1, im_map, 'PiYG',   'Im'),
    ]:
        ax = axes2[row, col, part]
        lim = np.abs(data).max() or 1.0
        im2 = ax.imshow(data, origin='lower', cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(f'k={k} {label}\ns={sk.real:.3f}{sk.imag:+.3f}j', fontsize=6)
        ax.axis('off')
        plt.colorbar(im2, ax=ax, shrink=0.8, pad=0.02)

# hide leftover axes
for k in range(K, nrows_k * ncols_k):
    row, col = divmod(k, ncols_k)
    axes2[row, col, 0].axis('off')
    axes2[row, col, 1].axis('off')

fig2.suptitle(f'Laplace transform Re/Im at K={K} frequencies', fontsize=11)
plt.tight_layout()
out_path2 = os.path.join("plots", "laplace_spectrum_all_k.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure saved → {out_path2}")

# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------
animate_comparaison(
    C_sub, C_rec,
    os.path.join("plots", "laplace_inversion_ch4.gif"),
    fps=10, cmap='RdBu_r', label='CH4',
    title_a='Original',
    title_b=f'Regularised inverse (K={K}, λ={lam:.0e}, α={alpha_t:.0e})',
    title_err='|Error|',
    title_fn=lambda t_: f"Case {case_idx}  t={t_}  (L2={err_field:.2e})",
)
