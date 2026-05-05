"""
Optimization of Laplace inversion paths for transient diffusion problems.
This script uses AdamW to optimize the K points in C, minimizing the MSE between the reconstructed and true transient responses.

Memory strategy: forward + inverse Laplace are fused inside a single checkpoint per chunk
of cases so U_hat (n_cases, N_spatial, K) is never fully materialized. The Gram matrix
for ae_error is accumulated over spatial chunks to avoid storing full U_hat too.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.checkpoint as checkpoint
import wandb
from tqdm import tqdm


def path_bromwich(K, gamma=0.0):
    """Bromwich: horizontal line at Re(s)=gamma, linearly spaced omega."""
    omega = 2 * np.pi * np.fft.rfftfreq(Nt, d=dt)[:K]
    return gamma + 1j * omega


# ---------------------------------------------------------------------------
# Low-memory Laplace helpers
# ---------------------------------------------------------------------------

def _compute_laplace_matrices(s_list, alpha_t, lam, w):
    """
    Compute frequency-independent (Nt x Nt) matrices from s_list.
    Returns small tensors only — no N_spatial dimension.
    """
    c_mask = (s_list.imag > 0)
    s_full = torch.cat([s_list, torch.conj(s_list[c_mask])])
    F_full = dt * w[None, :] * torch.exp(-s_full[:, None] * t)   # (K_full, Nt)
    FH     = torch.conj(F_full).T                                  # (Nt, K_full)
    FtF    = torch.real(FH @ F_full)                               # (Nt, Nt)
    A      = FtF + alpha_t * _DtTDt + lam * _eye_Nt
    LU, pivots = torch.linalg.lu_factor(A)
    return c_mask, F_full, FH, LU, pivots


def _fused_chunk(V_c, F_re, F_im, FH_re, FH_im, LU, pivots, c_mask):
    """
    Fused forward Laplace + inverse Laplace + loss for one chunk of cases.
    Designed to be wrapped in torch.utils.checkpoint.checkpoint so activations
    are recomputed during backward instead of stored.

    V_c : (cc, N_spatial, Nt)  float64, no grad
    Returns: (sq_err, sq_diff, sum_l2rel)  three scalar float64 tensors.
    """
    cc, N_spatial, _ = V_c.shape

    # Forward Laplace: V_c → U_hat_c  (cc, N_spatial, K)
    U_re = V_c @ F_re.T   # (cc, N_spatial, K)
    U_im = V_c @ F_im.T

    # Conjugate extension → (cc, N_spatial, K_full)
    U_re_full = torch.cat([U_re,  U_re[..., c_mask]],  dim=-1)
    U_im_full = torch.cat([U_im, -U_im[..., c_mask]], dim=-1)

    K_full = U_re_full.shape[-1]
    U_flat_re = U_re_full.reshape(cc * N_spatial, K_full)
    U_flat_im = U_im_full.reshape(cc * N_spatial, K_full)

    # Regularised inverse: solve A @ V_recon = Re(FH @ U_flat^T)
    RHS       = FH_re @ U_flat_re.T - FH_im @ U_flat_im.T   # (Nt, cc*N_spatial)
    V_recon_c = torch.linalg.lu_solve(LU, pivots, RHS).T.reshape(cc, N_spatial, -1)

    # Loss terms
    err_c  = V_recon_c - V_c
    diff_c = err_c[:, :, 1:] - err_c[:, :, :-1]

    norm_err = torch.norm(err_c, dim=(1, 2))
    norm_v   = torch.norm(V_c,   dim=(1, 2))

    return torch.sum(err_c ** 2), torch.sum(diff_c ** 2), torch.sum(norm_err / norm_v)


def ae_error_lowmem(s_list, V_tensor, w, n_latent, sp_chunk=2000):
    """
    Computes the ae_error (truncated-SVD variance ratio in Laplace space) without
    materializing U_hat for all cases at once.

    Gram matrix G[k, i, j] = sum_spatial Re(U[i,sp,k] * conj(U[j,sp,k]))
    is accumulated over spatial chunks — memory = n_cases * sp_chunk * K * 8 B.
    With sp_chunk=2000, n_cases=100, K=20 that is ~32 MB.
    """
    s_exp = torch.exp(-s_list[:, None] * t)          # (K, Nt)
    F_re  = (dt * w[None, :] * s_exp.real)            # (K, Nt)
    F_im  = (dt * w[None, :] * s_exp.imag)

    n_cases, N_spatial, _ = V_tensor.shape
    K = s_list.shape[0]

    G = torch.zeros(K, n_cases, n_cases, dtype=torch.float64, device=s_list.device)

    for sp in range(0, N_spatial, sp_chunk):
        V_sp   = V_tensor[:, sp:sp + sp_chunk, :]    # (n_cases, sp_size, Nt)
        U_re_sp = V_sp @ F_re.T                       # (n_cases, sp_size, K)
        U_im_sp = V_sp @ F_im.T
        # G[k, i, j] += einsum over spatial: U_re[i,sp,k]*U_re[j,sp,k] + U_im[...]
        G += (torch.einsum('isk,jsk->kij', U_re_sp, U_re_sp) +
              torch.einsum('isk,jsk->kij', U_im_sp, U_im_sp))

    ev       = torch.linalg.eigvalsh(G)               # (K, n_cases)
    n        = ev.shape[-1]
    r        = min(n_latent, n)
    tail_sq  = ev[:, :n - r].sum(dim=-1)
    total_sq = ev.sum(dim=-1)
    return (tail_sq / total_sq.clamp(min=1e-30)).sqrt().mean()


def amplification_factor(LU, pivots, F_full):
    FH     = torch.conj(F_full).T
    amp_re = torch.linalg.lu_solve(LU, pivots, FH.real)
    amp_im = torch.linalg.lu_solve(LU, pivots, FH.imag)
    return torch.linalg.matrix_norm(torch.cat([amp_re, amp_im], dim=0), ord='fro')


@torch.no_grad()
def reconstruct_lowmem(s_list, V_tensor, alpha_t, lam, w, case_chunk=2):
    """Round-trip V → Laplace → inverse, chunk by chunk. For baseline/eval."""
    c_mask, F_full, FH, LU, pivots = _compute_laplace_matrices(s_list, alpha_t, lam, w)
    K       = s_list.shape[0]
    F_base  = F_full[:K]                              # (K, Nt) complex
    F_re, F_im   = F_base.real, F_base.imag
    FH_re, FH_im = FH.real, FH.imag

    n_cases, N_spatial, _ = V_tensor.shape
    K_full = F_full.shape[0]
    chunks = []

    for i in range(0, n_cases, case_chunk):
        V_c  = V_tensor[i:i + case_chunk]
        cc   = V_c.shape[0]
        U_re = V_c @ F_re.T
        U_im = V_c @ F_im.T
        U_re_full = torch.cat([U_re,  U_re[..., c_mask]],  dim=-1)
        U_im_full = torch.cat([U_im, -U_im[..., c_mask]], dim=-1)
        U_flat_re = U_re_full.reshape(cc * N_spatial, K_full)
        U_flat_im = U_im_full.reshape(cc * N_spatial, K_full)
        RHS = FH_re @ U_flat_re.T - FH_im @ U_flat_im.T
        V_recon_c = torch.linalg.lu_solve(LU, pivots, RHS).T.reshape(cc, N_spatial, Nt)
        chunks.append(V_recon_c)

    return torch.cat(chunks, dim=0)


def _log_s_scatter(s_cur, s_ref, epoch):
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
K       = 20
gamma   = 0.0
lambda_diff = 2.0
step     = 1
t_frame  = 50
seed = 42
n_cases = 100
n_latent = 64
lambda_ae = 1.25
gamma_min = -0.05
lr = 5e-3
n_epochs = 100
case_chunk = 2   # cases per checkpoint call
sp_chunk   = 2000  # spatial points per Gram-accumulation step

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t       = torch.arange(Nt, dtype=torch.float64, device=device) * dt
_Dt     = (torch.diag(torch.ones(Nt - 1), 1) - torch.eye(Nt, dtype=torch.float64))[:Nt - 1, :]
_DtTDt  = (_Dt.T @ _Dt).to(device)
_eye_Nt = torch.eye(Nt, dtype=torch.float64, device=device)

wandb.init(
    project="convdiff",
    name="laplace_opti_s",
    config=dict(K=K, Nt=Nt, dt=dt, gamma=gamma, lambda_diff=lambda_diff,
                step=step, n_cases=n_cases, n_latent=n_latent, lambda_ae=lambda_ae,
                gamma_min=gamma_min, lr_init=lr, n_epochs=n_epochs, seed=seed,
                case_chunk=case_chunk, sp_chunk=sp_chunk),
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
C_cases = C_full[cases_idx, :, ::step, ::step].copy().astype(np.float64)
n_cases_, Nt_, H_sub, W_sub = C_cases.shape
N_spatial = H_sub * W_sub
V = C_cases.transpose(0, 2, 3, 1).reshape(n_cases, N_spatial, Nt)
V_tensor = torch.tensor(V, dtype=torch.float64, device=device)
print(f"Loaded {n_cases} cases: V {V_tensor.shape}  device={device}  (N_spatial={N_spatial})")

torch.manual_seed(seed)
optimizer = torch.optim.AdamW([s_list, lam, alpha_t], lr=lr, weight_decay=1e-4)
w = torch.ones(Nt, dtype=torch.float64, device=device); w[0] = 0.5; w[-1] = 0.5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

norm_V_tensor  = torch.norm(V_tensor)
norm_diff_V    = torch.norm(V_tensor[:, :, 1:] - V_tensor[:, :, :-1])

# ── Baseline avant optimisation ───────────────────────────────────────────────
s_init = s_list.detach().clone()
with torch.no_grad():
    V_recon0 = reconstruct_lowmem(s_list, V_tensor, alpha_t, lam, w, case_chunk=case_chunk)
    l2rel_init = (
        torch.norm(V_recon0 - V_tensor, dim=(1, 2))
        / torch.norm(V_tensor, dim=(1, 2))
    ).cpu().numpy()
    node   = N_spatial // 2
    v_init = V_recon0[0, node].cpu().numpy()
    del V_recon0
    torch.cuda.empty_cache()

# ── Boucle d'optimisation ─────────────────────────────────────────────────────
pbar = tqdm(range(n_epochs), desc="Optim s-points", unit="epoch")
for epoch in pbar:
    optimizer.zero_grad()

    # Pre-compute Nt×Nt matrices once per epoch (tiny memory, ~Nt^2 = 22 500 doubles)
    c_mask, F_full, FH, LU, pivots = _compute_laplace_matrices(s_list, alpha_t, lam, w)
    K_      = s_list.shape[0]
    F_base  = F_full[:K_]                      # (K, Nt) complex
    F_re, F_im   = F_base.real, F_base.imag
    FH_re, FH_im = FH.real, FH.imag

    sq_err     = torch.zeros(1, dtype=torch.float64, device=device)
    sq_diff    = torch.zeros(1, dtype=torch.float64, device=device)
    sum_l2rel  = torch.zeros(1, dtype=torch.float64, device=device)

    for i in range(0, n_cases, case_chunk):
        V_c = V_tensor[i:i + case_chunk]   # no grad — only s_list/lam/alpha_t require grad
        sq_e, sq_d, l2r = checkpoint.checkpoint(
            _fused_chunk,
            V_c, F_re, F_im, FH_re, FH_im, LU, pivots, c_mask,
            use_reentrant=False,
        )
        sq_err    = sq_err    + sq_e
        sq_diff   = sq_diff   + sq_d
        sum_l2rel = sum_l2rel + l2r

    biais_loss_l2   = torch.sqrt(sq_err)   / norm_V_tensor
    biais_loss_diff = torch.sqrt(sq_diff)  / norm_diff_V

    amp     = amplification_factor(LU, pivots, F_full)
    ae      = ae_error_lowmem(s_list, V_tensor, w, n_latent, sp_chunk=sp_chunk)
    var_los = amp * ae / lambda_ae
    loss    = (biais_loss_l2 + lambda_diff * biais_loss_diff) / (lambda_diff + 1.0) + var_los
    loss.backward()

    torch.nn.utils.clip_grad_norm_([s_list, lam, alpha_t], max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.detach())

    with torch.no_grad():
        s_list.real.clamp_(min=gamma_min)
        lam.clamp_(min=1e-10, max=1e-2)
        alpha_t.clamp_(min=1e-10, max=1e-2)
        s_list.imag.clamp_(min=0)

    l2rel_val = (sum_l2rel / n_cases).item()
    cur_lr    = optimizer.param_groups[0]['lr']

    pbar.set_postfix(
        loss=f"{loss.item():.3e}",
        biais_l2=f"{biais_loss_l2.item():.3e}",
        biais_diff=f"{biais_loss_diff.item():.3e}",
        var=f"{var_los.item():.3e}",
        l2rel=f"{l2rel_val:.4f}",
        lr=f"{cur_lr:.1e}",
    )

    log = {
        "loss/total":      loss.item(),
        "loss/biais_l2":   biais_loss_l2.item(),
        "loss/biais_diff": biais_loss_diff.item(),
        "loss/var":        var_los.item(),
        "metrics/l2rel":   l2rel_val,
        "metrics/amp":     amp.item(),
        "metrics/ae":      ae.item(),
        "optim/lr":        cur_lr,
        "params/lam":      lam.item(),
        "params/alpha_t":  alpha_t.item(),
    }

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        log["s_points/scatter"] = _log_s_scatter(s_list, s_init, epoch)

    wandb.log(log, step=epoch)

# ── Résultats finaux ──────────────────────────────────────────────────────────
s_opt = s_list.detach().clone()
with torch.no_grad():
    V_recon_opt = reconstruct_lowmem(s_list, V_tensor, alpha_t, lam, w, case_chunk=case_chunk)
    l2rel_opt = (
        torch.norm(V_recon_opt - V_tensor, dim=(1, 2))
        / torch.norm(V_tensor, dim=(1, 2))
    ).cpu().numpy()

print("Optimized s points:", ", ".join(
    f"{z.real:.4f}+{z.imag:.4f}j" for z in s_opt.numpy().tolist()))
print("Optimized lam: ", lam.item())
print("Optimized alpha_t: ", alpha_t.item())

# ── Figures finales loggées sur wandb ─────────────────────────────────────────
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

t_ax    = np.arange(Nt) * dt
v_true  = V_tensor[0, node].cpu().numpy()
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
