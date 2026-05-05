"""
Optimization of Laplace inversion paths for transient diffusion problems.
This script uses AdamW to optimize the K points in C, minimizing the MSE between the reconstructed and true transient responses.


"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm
from utils.animate import animate_comparaison




def path_bromwich(K, gamma=0.0):
    """Bromwich: horizontal line at Re(s)=gamma, linearly spaced omega."""
    omega = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[:K]
    return gamma + 1j * omega

def forward_laplace(s_list, V, w):
    # V    : (n_cases, N_spatial, Nt)
    # return: (n_cases, N_spatial, K)
    s_exp  = torch.exp(-s_list[:, None] * t)     # (K, Nt)
    F_base = dt * w[None, :] * s_exp             # (K, Nt)
    return V.to(F_base.dtype) @ F_base.T         # (..., Nt) @ (Nt, K) → (n_cases, N_spatial, K)

def conj_extension(s_list, U_hat, w):
    # U_hat : (n_cases, N_spatial, K)
    c_mask     = (s_list.imag > 0)
    s_full     = torch.cat([s_list, torch.conj(s_list[c_mask])])               # (K_full,)
    U_hat_full = torch.cat([U_hat, torch.conj(U_hat[..., c_mask])], dim=-1)   # (n_cases, N_spatial, K_full)
    F_full     = dt * w[None, :] * torch.exp(-s_full[:, None] * t[None, :])   # (K_full, Nt)
    return s_full, U_hat_full, F_full

def reg_inv_laplace(s_list, alpha_t, lam, U_hat, w):
    # U_hat : (n_cases, N_spatial, K)
    s_full, U_hat_full, F_full = conj_extension(s_list, U_hat, w)
    n_cases, N_spatial, K_full = U_hat_full.shape
    FH  = torch.conj(F_full).T                                                  # (Nt, K_full)
    FtF = torch.real(FH @ F_full)                                               # (Nt, Nt)
    A = FtF + alpha_t * _DtTDt + lam * _eye_Nt                                 # (Nt, Nt)
    LU, pivots = torch.linalg.lu_factor(A)                                      # factorisé une seule fois
    U_flat = U_hat_full.reshape(n_cases * N_spatial, K_full)                    # (N_nodes, K_full)
    RHS    = torch.real(FH @ U_flat.T)                                          # (Nt, N_nodes)
    V_recon = torch.linalg.lu_solve(LU, pivots, RHS).T                         # (N_nodes, Nt)
    V_recon = V_recon.reshape(n_cases, N_spatial, Nt)                           # (n_cases, N_spatial, Nt)
    return V_recon, LU, pivots, F_full

def ae_error(U_full, n_latent):
    # U_full: (n_cases, N_spatial, K) complex
    # Gram matrix approach: G = M @ M.T = real(U_k @ U_k†) — même spectre que svdvals(M)
    # mais sans allouer M = (K, n_cases, 2*N_spatial) (~1.3 GB avec step=1).
    # eigvalsh stable ici : avec n_cases=100, n_latent=64, seulement 36 valeurs dégénérées.
    U_k = U_full.permute(2, 0, 1)                               # (K, n_cases, N_spatial)
    G   = torch.real(torch.bmm(U_k, U_k.conj().mT))            # (K, n_cases, n_cases) Gram
    ev  = torch.linalg.eigvalsh(G)                              # (K, n_cases) σ² croissants
    n   = ev.shape[-1]
    r   = min(n_latent, n)
    tail_sq  = ev[:, : n - r].sum(dim=-1)                      # n_cases-r plus petits σ²
    total_sq = ev.sum(dim=-1)
    return (tail_sq / total_sq.clamp(min=1e-30)).sqrt().mean()

def amplyfication_factor(LU, pivots, F_full):
    """||A^{-1} F^H||_2 spectrale exacte, sans tensor complexe sur GPU.
    Identité : sv(A+iB) = sv([A;B]) car (A+iB)†(A+iB) = AᵀA + BᵀB = [A;B]ᵀ[A;B].
    Tout en float64 réel → pas de kernel NVRTC complexe."""
    FH     = torch.conj(F_full).T                                      # (Nt, K_full) cdouble
    amp_re = torch.linalg.lu_solve(LU, pivots, FH.real)               # (Nt, K_full) float64
    amp_im = torch.linalg.lu_solve(LU, pivots, FH.imag)               # (Nt, K_full) float64
    return torch.linalg.matrix_norm(torch.cat([amp_re, amp_im], dim=0), ord='fro')

def _log_s_scatter(s_cur, s_ref, epoch):
    """Scatter plot du plan complexe : position courante vs initiale."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = s_cur.detach().cpu().numpy()
    si = s_ref.cpu().numpy()
    cmap = plt.cm.plasma
    for k, (a, b) in enumerate(zip(si, sc)):
        col = cmap(k / max(len(si) - 1, 1))
        ax.plot([a.real, b.real], [a.imag, b.imag], color=col, lw=0.6, alpha=0.5)
    ax.scatter(si.real, si.imag, c=np.arange(len(si)), cmap='plasma',
               s=50, marker='o', zorder=3, alpha=0.4, label='initial')
    ax.scatter(sc.real, sc.imag, c=np.arange(len(sc)), cmap='plasma',
               s=80, marker='*', zorder=4, label='current')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Re(s)'); ax.set_ylabel('Im(s)')
    ax.set_title(f's-points  epoch {epoch}'); ax.legend(fontsize=7)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Nt      = 150
dt      = 1.0
K       = 20        # s points before conjugate extension
gamma   = 0.0       # Bromwich damping initial
lambda_diff = 2.0  # weight of finite-diff temporal smoothness in the loss (relative to L2)
step     = 1        # spatial subsampling
t_frame  = 50       # time step shown in spatial panel
seed = 42
n_cases = 100
n_latent = 64
lambda_ae = 1.25 # amelioration factor of the AE in comparison to a linear SVD trunc of the same rank
gamma_min = -0.05
lr = 5e-3
n_epochs = 100

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t       = torch.arange(Nt, dtype=torch.float64, device=device) * dt
_Dt     = (torch.diag(torch.ones(Nt - 1), 1) - torch.eye(Nt, dtype=torch.float64))[:Nt - 1, :]
_DtTDt  = (_Dt.T @ _Dt).to(device)                # (Nt, Nt) constante, précalculée une fois
_eye_Nt = torch.eye(Nt, dtype=torch.float64, device=device)

wandb.init(
    project="convdiff",
    name="laplace_opti_s",
    config=dict(K=K, Nt=Nt, dt=dt, gamma=gamma, lambda_diff=lambda_diff,
                step=step, n_cases=n_cases, n_latent=n_latent, lambda_ae=lambda_ae,
                gamma_min=gamma_min, lr_init=lr, n_epochs=n_epochs, seed=seed),
)

initial_s = path_bromwich(K, gamma=gamma)

s_list  = torch.tensor(initial_s, dtype=torch.cdouble, device=device).requires_grad_(True)
lam     = torch.tensor(1e-8, dtype=torch.float64,  device=device).requires_grad_(True)
alpha_t = torch.tensor(1e-8, dtype=torch.float64,  device=device).requires_grad_(True)

data_path = os.path.join("dataset", "CH4.npy")
C_full = np.load(data_path, mmap_mode='r')
n_total = C_full.shape[0]
np.random.seed(seed)
cases_idx = np.random.choice(n_total, size=n_cases, replace=False)
C_cases = C_full[cases_idx, :, ::step, ::step].copy().astype(np.float64)  # (n_cases, Nt, H_sub, W_sub)
n_cases_, Nt_, H_sub, W_sub = C_cases.shape
N_spatial = H_sub * W_sub
V = C_cases.transpose(0, 2, 3, 1).reshape(n_cases, N_spatial, Nt)        # (n_cases, N_spatial, Nt)
V_tensor = torch.tensor(V, dtype=torch.float64, device=device)
print(f"Loaded {n_cases} cases: V {V_tensor.shape}  device={device}  (N_spatial={N_spatial})")

torch.manual_seed(seed)
optimizer = torch.optim.AdamW([s_list, lam, alpha_t], lr=lr, weight_decay=1e-4)
w = torch.ones(Nt, dtype=torch.float64, device=device); w[0] = 0.5; w[-1] = 0.5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ── Baseline avant optimisation ───────────────────────────────────────────────
s_init = s_list.detach().clone()
with torch.no_grad():
    U0 = forward_laplace(s_list, V_tensor, w)
    V_recon0, _, _, _ = reg_inv_laplace(s_list, alpha_t, lam, U0, w)
    l2rel_init = (
        torch.norm(V_recon0 - V_tensor, dim=(1, 2))
        / torch.norm(V_tensor, dim=(1, 2))
    ).cpu().numpy()



# ── Boucle d'optimisation ─────────────────────────────────────────────────────
pbar = tqdm(range(n_epochs), desc="Optim s-points", unit="epoch")
for epoch in pbar:
    optimizer.zero_grad()
    U_hat                        = forward_laplace(s_list, V_tensor, w)
    V_recon, LU, pivots, F_full = reg_inv_laplace(s_list, alpha_t, lam, U_hat, w)

    error = V_recon - V_tensor
    biais_loss_l2 = torch.norm(error) / torch.norm(V_tensor)
    diff_error = error[:, :, 1:] - error[:, :, :-1]
    biais_loss_diff = torch.norm(diff_error) / torch.norm(V_tensor[:, :, 1:] - V_tensor[:, :, :-1])

    amp        = amplyfication_factor(LU, pivots, F_full)
    ae         = ae_error(U_hat, n_latent)
    var_los    = amp * ae / lambda_ae
    loss       = (biais_loss_l2 + lambda_diff * biais_loss_diff)/ (lambda_diff + 1.0) + var_los
    loss.backward()
    torch.nn.utils.clip_grad_norm_([s_list, lam, alpha_t], max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.detach())
    with torch.no_grad():
        s_list.real.clamp_(min=gamma_min)  # projection : Re(s) ≥ gamma_min
        lam.clamp_(min=1e-10, max=1e-2)
        alpha_t.clamp_(min=1e-10, max=1e-2)
        s_list.imag.clamp_(min=0)  # éviter les points négatif dans la partie imaginaire (déjà gérés par la symétrie de conjugaison)
    with torch.no_grad():
        l2rel = (
            torch.norm(V_recon - V_tensor, dim=(1, 2))
            / torch.norm(V_tensor, dim=(1, 2))
        ).mean().item()
    cur_lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix(
        loss=f"{loss.item():.3e}",
        biais_l2=f"{biais_loss_l2.item():.3e}",
        biais_diff=f"{biais_loss_diff.item():.3e}",
        var=f"{var_los.item():.3e}",
        l2rel=f"{l2rel:.4f}",
        lr=f"{cur_lr:.1e}",
    )

    # ── Log scalaires ──────────────────────────────────────────────────────────
    log = {
        "loss/total":    loss.item(),
        "loss/biais_l2":    biais_loss_l2.item(),
        "loss/biais_diff": biais_loss_diff.item(),
        "loss/var":      var_los.item(),
        "metrics/l2rel": l2rel,
        "metrics/amp":   amp.item(),
        "metrics/ae":    ae.item(),
        "optim/lr":      cur_lr,
        "params/lam":     lam.item(),
        "params/alpha_t": alpha_t.item(),
    }

    # ── Scatter plan complexe ──────────────────────────────────────────
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        log["s_points/scatter"] = _log_s_scatter(s_list, s_init, epoch)

    wandb.log(log, step=epoch)

# ── Résultats finaux ──────────────────────────────────────────────────────────
s_opt = s_list.detach().clone()
with torch.no_grad():
    U_opt = forward_laplace(s_list, V_tensor, w)
    V_recon_opt, _, _, _ = reg_inv_laplace(s_list, alpha_t, lam, U_opt, w)
    l2rel_opt = (
        torch.norm(V_recon_opt - V_tensor, dim=(1, 2))
        / torch.norm(V_tensor, dim=(1, 2))
    ).cpu().numpy()

print("Optimized s points:", ", ".join(
    f"{z.real:.4f}+{z.imag:.4f}j" for z in s_opt.numpy().tolist()))
print("Optimized lam: ", lam.item())
print("Optimized alpha_t: ", alpha_t.item())

# ── Figures finales loggées sur wandb ─────────────────────────────────────────

# Figure A — s-points initial → optimisé avec flèches
fig, ax = plt.subplots(figsize=(7, 5))
si, so = s_init.numpy(), s_opt.numpy()
cmap = plt.cm.plasma
for k, (a, b) in enumerate(zip(si, so)):
    col = cmap(k / max(K - 1, 1))
    ax.annotate('', xy=(b.real, b.imag), xytext=(a.real, a.imag),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
sc0 = ax.scatter(si.real, si.imag, c=np.arange(K), cmap='plasma',
                 s=60, marker='o', zorder=3, label='Initial')
sc1 = ax.scatter(so.real, so.imag, c=np.arange(K), cmap='plasma',
                 s=100, marker='*', zorder=4, label='Optimisé')
plt.colorbar(sc1, ax=ax, label='k')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('Re(s)'); ax.set_ylabel('Im(s)')
ax.set_title('s-points : initial → optimisé'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
wandb.log({"final/s_points_trajectory": wandb.Image(fig)}, step=n_epochs)
plt.close(fig)

# Figure B — Histogramme L2rel initial vs optimisé
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(l2rel_init, bins=30, alpha=0.6, color='steelblue',
        label=f'Initial   méd={np.median(l2rel_init):.2%}')
ax.hist(l2rel_opt,  bins=30, alpha=0.6, color='tomato',
        label=f'Optimisé  méd={np.median(l2rel_opt):.2%}')
ax.set_xlabel('L2 relative error'); ax.set_ylabel('Count')
ax.set_title('Distribution L2rel : initial vs optimisé')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
wandb.log({"final/l2rel_histogram": wandb.Image(fig)}, step=n_epochs)
plt.close(fig)

# Figure C — Série temporelle (cas 0, nœud central)
node  = N_spatial // 2
t_ax  = np.arange(Nt) * dt
v_true  = V_tensor[0, node].cpu().numpy()
v_init  = V_recon0[0, node].detach().cpu().numpy()
v_final = V_recon_opt[0, node].detach().cpu().numpy()
e_i = np.linalg.norm(v_init  - v_true) / (np.linalg.norm(v_true) + 1e-12)
e_f = np.linalg.norm(v_final - v_true) / (np.linalg.norm(v_true) + 1e-12)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_ax, v_true,  'k',         lw=1.5, label='Vrai')
axes[0].plot(t_ax, v_init,  'steelblue', lw=1.2, ls='--', label=f'Initial  L2={e_i:.2%}')
axes[0].plot(t_ax, v_final, 'tomato',    lw=1.2, ls='--', label=f'Optimisé L2={e_f:.2%}')
axes[0].set_xlabel('t'); axes[0].set_ylabel('CH4')
axes[0].set_title('Série temporelle'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].plot(t_ax, v_true - v_init,  'steelblue', lw=1.2,
             label=f'Résidu initial  rms={np.std(v_true - v_init):.2e}')
axes[1].plot(t_ax, v_true - v_final, 'tomato',    lw=1.2,
             label=f'Résidu optimisé rms={np.std(v_true - v_final):.2e}')
axes[1].axhline(0, color='gray', lw=0.5)
axes[1].set_xlabel('t'); axes[1].set_ylabel('Résidu')
axes[1].set_title('Résidus temporels'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
fig.tight_layout()
wandb.log({"final/reconstruction": wandb.Image(fig)}, step=n_epochs)
plt.close(fig)

wandb.finish()
print("Run wandb terminé.")
