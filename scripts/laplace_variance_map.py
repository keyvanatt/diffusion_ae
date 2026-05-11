"""
laplace_variance_map.py — Relative variance of the Laplace transform in the (Re(s), Im(s)) plane.

For each point s = re + i*im on a 2-D grid, computes two statistics over the dataset:
  - relative_variance(s) : variance / energy  (θ-sensitivity normalised by signal strength)
  - energy(s)            : mean pixel-wise expected squared magnitude of hatU(s)

Grid matches laplace_complex_plane.py (filtered to |Re(s)| ≤ 0.1, Im(s) ≤ Nyquist).

All N_grid points are computed in a single pass over the simulations via a GPU batched
matmul (2*N_grid, Nt) × (Nt, B*HW). Kernels in float32, accumulators in float64.

Run:
    .conda/bin/python transient/laplace_variance_map.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


# ── GPU vectorised computation ────────────────────────────────────────────────

def compute_all_stats(U_sims: np.ndarray, re_vals: np.ndarray, im_vals: np.ndarray,
                      dt: float, device: str = 'cuda', sim_batch: int = 100) -> dict:
    """
    One-pass computation of variance, energy, rel_var for ALL (re, im) grid points.

    U_sims   : (N_sim, Nt, H, W) float32 — pre-loaded in RAM
    Returns results dict keyed by (ri, ci).

    Peak GPU memory ≈ sim_batch × (Nt×HW×4 + 2*N_grid×HW×8) bytes.
    With sim_batch=100, N_grid=~40, H=W=200: ~8 GB → tune sim_batch to VRAM.
    """
    N_sim, Nt, H, W = U_sims.shape
    N_re, N_im = len(re_vals), len(im_vals)
    N_grid = N_re * N_im
    HW = H * W

    # ── Pre-compute kernels for all grid points (float32 on GPU) ──────────────
    t      = torch.arange(Nt, dtype=torch.float32, device=device)
    w      = torch.ones(Nt, device=device); w[0] = 0.5; w[-1] = 0.5
    re_t   = torch.tensor(re_vals, dtype=torch.float32, device=device)
    im_t   = torch.tensor(im_vals, dtype=torch.float32, device=device)
    re_all = re_t.repeat_interleave(N_im)   # (N_grid,)
    im_all = im_t.repeat(N_re)              # (N_grid,)
    amp    = dt * w[None, :] * torch.exp(-re_all[:, None] * t[None, :])  # (N_grid, Nt)
    phase  = im_all[:, None] * t[None, :]                                 # (N_grid, Nt)
    # Stack Re and Im kernels: (2*N_grid, Nt)
    k = torch.cat([amp * torch.cos(phase), -amp * torch.sin(phase)], dim=0)

    # ── Running sums in float64 on GPU ────────────────────────────────────────
    sum1 = torch.zeros(2 * N_grid, HW, dtype=torch.float64, device=device)
    sum2 = torch.zeros(2 * N_grid, HW, dtype=torch.float64, device=device)
    n = 0

    for start in tqdm(range(0, N_sim, sim_batch), desc='Simulations'):
        U_b   = torch.from_numpy(U_sims[start:start + sim_batch]).to(device)  # (B, Nt, H, W)
        B     = U_b.shape[0]
        # Reshape to (Nt, B*HW) for a single matmul
        U_t   = U_b.reshape(B, Nt, HW).permute(1, 0, 2).reshape(Nt, B * HW)
        # hatU float32: sum first, then square in-place → single allocation, halves peak VRAM
        hatU  = torch.mm(k, U_t).reshape(2 * N_grid, B, HW)   # float32
        sum1 += hatU.sum(dim=1).double()
        hatU.pow_(2)                                            # in-place: no extra tensor
        sum2 += hatU.sum(dim=1).double()
        n    += B

    # ── Statistics ────────────────────────────────────────────────────────────
    mean    = sum1 / n
    var_map = torch.clamp(sum2 / n - mean ** 2, min=0.0)  # (2*N_grid, HW)

    # Sum Re and Im contributions, then average over pixels
    variance = (var_map[:N_grid] + var_map[N_grid:]).mean(dim=1).cpu().numpy()  # (N_grid,)
    energy   = ((sum2[:N_grid]   + sum2[N_grid:]) / n).mean(dim=1).cpu().numpy()
    rel_var  = np.where(energy > 1e-30, variance / energy, 0.0)

    results = {}
    for ri in range(N_re):
        for ci in range(N_im):
            g = ri * N_im + ci
            results[(ri, ci)] = {
                'variance':          float(variance[g]),
                'energy':            float(energy[g]),
                'relative_variance': float(rel_var[g]),
                's': complex(float(re_vals[ri]), float(im_vals[ci])),
            }
    return results


# ── SVD residual ─────────────────────────────────────────────────────────────

def compute_svd_residuals(U_sims: np.ndarray, re_vals: np.ndarray, im_vals: np.ndarray,
                          dt: float, n_latent: int,
                          device: str = 'cuda', sim_batch: int = 20,
                          g_batch: int = 10) -> np.ndarray:
    """
    For each grid point s, compute the residual SVD error at rank n_latent:

        residual(s) = sqrt( sum_{i > n_latent} σ_i(s)^2 )

    where D(s) = [Re(hatU(s)) | Im(hatU(s))] is the (N_sim, 2*HW) data matrix
    and σ_i are its singular values (equiv. eigenvalues of D @ D.T).

    G = D_re @ D_re.T + D_im @ D_im.T  (N_sim × N_sim, float64)
    avoids forming the full (N_sim, 2*HW) matrix.

    Returns (N_grid,) float64 array, ordered row-major over (re_vals, im_vals).
    """
    N_sim, Nt, H, W = U_sims.shape
    N_re, N_im = len(re_vals), len(im_vals)
    N_grid = N_re * N_im
    HW = H * W

    t     = torch.arange(Nt, dtype=torch.float32, device=device)
    w     = torch.ones(Nt, device=device); w[0] = 0.5; w[-1] = 0.5
    re_all = torch.tensor([re_vals[ri] for ri in range(N_re) for _ in range(N_im)],
                          dtype=torch.float32, device=device)
    im_all = torch.tensor([im_vals[ci] for _ in range(N_re) for ci in range(N_im)],
                          dtype=torch.float32, device=device)

    residuals = np.zeros(N_grid, dtype=np.float64)

    for g_start in tqdm(range(0, N_grid, g_batch), desc='SVD residuals'):
        g_end = min(g_start + g_batch, N_grid)
        gb    = g_end - g_start

        re_g  = re_all[g_start:g_end]
        im_g  = im_all[g_start:g_end]
        amp   = dt * w[None, :] * torch.exp(-re_g[:, None] * t[None, :])  # (gb, Nt)
        phase = im_g[:, None] * t[None, :]
        k_g   = torch.cat([amp * torch.cos(phase),
                           -amp * torch.sin(phase)], dim=0)                # (2*gb, Nt)

        # D[g] = Re(hatU) for grid point g  (N_sim, HW)
        # D[gb+g] = Im(hatU)               (N_sim, HW)
        D = torch.zeros(2 * gb, N_sim, HW, dtype=torch.float32, device=device)
        for s in range(0, N_sim, sim_batch):
            U_b  = torch.from_numpy(U_sims[s:s + sim_batch]).to(device)
            B    = U_b.shape[0]
            U_t  = U_b.reshape(B, Nt, HW).permute(1, 0, 2).reshape(Nt, B * HW)
            hatU = torch.mm(k_g, U_t).reshape(2 * gb, B, HW)
            D[:, s:s + B, :] = hatU

        # Batched Gram: G[g] = D_re[g] @ D_re[g].T + D_im[g] @ D_im[g].T
        # One bmm call per component, no inner loop over g
        D_re = D[:gb].double()                                      # (gb, N_sim, HW)
        G    = torch.bmm(D_re, D_re.permute(0, 2, 1))              # (gb, N_sim, N_sim)
        del D_re
        D_im = D[gb:].double()
        G   += torch.bmm(D_im, D_im.permute(0, 2, 1))
        del D_im, D

        # Batched eigvalsh + vectorised relative residual
        sigma_sq   = torch.linalg.eigvalsh(G)                            # (gb, N_sim) ascending
        tail       = sigma_sq[:, :-n_latent].clamp(min=0).sum(dim=1)     # (gb,)
        total      = sigma_sq.clamp(min=0).sum(dim=1)                    # (gb,) = ||D||_F²
        rel_res    = (tail / total.clamp(min=1e-30)).sqrt()
        residuals[g_start:g_end] = rel_res.cpu().numpy()

    return residuals


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(results: dict, sim_idx: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'results': results, 'sim_idx': sim_idx}, f)


def load_ckpt(path: str):
    if Path(path).exists():
        with open(path, 'rb') as f:
            d = pickle.load(f)
        print(f"Found checkpoint: {path}  ({len(d['results'])} points)")
        return d['results'], d.get('sim_idx')
    return {}, None


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(results: dict, re_vals: np.ndarray, im_vals: np.ndarray,
                 stat_key: str, title: str, cbar_label: str, path: str,
                 nyquist: float = None):
    Nr, Ni = len(re_vals), len(im_vals)
    grid = np.full((Nr, Ni), np.nan)
    for (ri, ci), v in results.items():
        val = v.get(stat_key, np.nan)
        if np.isfinite(val) and val > 0:
            grid[ri, ci] = val

    vals = grid[np.isfinite(grid) & (grid > 0)]
    if len(vals) == 0:
        print(f"  [skip] no valid data for {stat_key}")
        return
    norm = mcolors.LogNorm(vmin=float(vals.min()), vmax=float(vals.max()))

    fig, ax = plt.subplots(figsize=(9, 5))
    pm = ax.pcolormesh(re_vals, im_vals, grid.T,
                       cmap='plasma', norm=norm, shading='nearest')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_yscale('log')
    ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.6)
    if nyquist is not None:
        ax.axhline(nyquist, color='white', lw=0.8, ls=':', alpha=0.6,
                   label=f'Nyquist ω={nyquist:.2f}')
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('Re(s) = γ')
    ax.set_ylabel('Im(s) = ω')
    ax.set_title(title)
    plt.colorbar(pm, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Config ────────────────────────────────────────────────────────────────
    DATA_PATH  = 'dataset/ch4_rotated.npy'
    dt         = 1.0
    Nt         = 150
    N_NEG      = 10
    N_POS      = 15
    N_OMEGA    = 20
    N_sim      = 1000
    SIM_BATCH  = 20    # peak VRAM ≈ 2*N_grid × B × HW × 4 bytes
    RESULTS    = 'plots/laplace_variance_map.pkl'
    FIG_RELVAR = 'plots/laplace_variance_map.png'
    FIG_ENERGY = 'plots/laplace_energy_map.png'
    SEED       = 42
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Grid (same points as laplace_complex_plane.py, filtered to our limits) ─
    T          = Nt * dt
    nyquist    = np.pi / dt
    re_max_pos = float(-np.log(np.finfo(np.float64).eps) / dt)  # ≈ 36 for dt=1
    re_neg  = -np.logspace(np.log10(0.05), np.log10(0.1), N_NEG)[::-1]  # [-0.1…-0.01]
    re_zero = np.array([0.0])
    re_pos  = np.logspace(-3, 0, N_POS)                                  # [0.001…1.0]
    re_vals = np.concatenate([re_neg, re_zero, re_pos])

    im_vals = np.logspace(np.log10(1.0 / T), np.log10(nyquist), N_OMEGA)

    N_re, N_im = len(re_vals), len(im_vals)
    print(f"Device : {DEVICE}")
    print(f"Grid   : {N_re} Re(s) × {N_im} Im(s) = {N_re * N_im} points")
    print(f"  Re(s): {np.array2string(re_vals, precision=4)}")
    print(f"  Im(s): [{im_vals[0]:.4f} … {im_vals[-1]:.4f}]  (Nyquist={nyquist:.4f})")

    # ── Data ──────────────────────────────────────────────────────────────────
    _, saved_idx = load_ckpt(RESULTS)
    U_raw = np.load(DATA_PATH, mmap_mode='r')        # (ns, Nt, H, W)
    rng   = np.random.default_rng(SEED)
    sim_idx = (saved_idx if saved_idx is not None
               else np.sort(rng.choice(U_raw.shape[0], N_sim, replace=False)))

    print(f"\nDataset: {U_raw.shape}  —  loading {N_sim} simulations into RAM...")
    U_sims = U_raw[sim_idx].astype(np.float32)       # (N_sim, Nt, H, W) in RAM
    print(f"RAM usage: {U_sims.nbytes / 1e9:.1f} GB")

    # ── n_latent from surrogate checkpoint ───────────────────────────────────
    SURROGATE_CKPT = 'checkpoints/LaplaceLatentModel_finetuned.pt'
    try:
        ckpt_s = torch.load(SURROGATE_CKPT, map_location='cpu', weights_only=False)
        n_latent = int(ckpt_s['latent_dim'])
        print(f"n_latent = {n_latent}  (from {SURROGATE_CKPT})")
    except Exception as e:
        n_latent = 64
        print(f"Could not load surrogate ckpt ({e}). Using n_latent={n_latent}")

    # ── Compute variance / energy stats (one pass over all grid points) ───────
    torch.cuda.empty_cache()
    results = compute_all_stats(U_sims, re_vals, im_vals, dt,
                                device=DEVICE, sim_batch=SIM_BATCH)

    # ── Compute SVD residual at rank n_latent ─────────────────────────────────
    torch.cuda.empty_cache()
    svd_res = compute_svd_residuals(U_sims, re_vals, im_vals, dt, n_latent,
                                    device=DEVICE, sim_batch=SIM_BATCH, g_batch=10)
    for ri in range(N_re):
        for ci in range(N_im):
            results[(ri, ci)]['svd_residual'] = float(svd_res[ri * N_im + ci])

    # ── Save & plot ───────────────────────────────────────────────────────────
    save_ckpt(results, sim_idx, RESULTS)
    plot_heatmap(results, re_vals, im_vals, 'relative_variance',
                 'Relative variance  $\\mathrm{Var}[\\hat{U}(s)] / \\mathbb{E}[|\\hat{U}(s)|^2]$',
                 'Var / Energy', FIG_RELVAR, nyquist)
    plot_heatmap(results, re_vals, im_vals, 'energy',
                 'Mean energy of $\\hat{U}(s)$  [mean $|\\hat{U}|^2$ per pixel]',
                 'Energy', FIG_ENERGY, nyquist)
    plot_heatmap(results, re_vals, im_vals, 'svd_residual',
                 f'Relative SVD residual at rank $n={n_latent}$'
                 r'  $\left(\sum_{i>n}\sigma_i^2\,/\,\sum_i\sigma_i^2\right)^{1/2}$',
                 f'Relative residual (rank {n_latent})', 'plots/laplace_svd_residual.png', nyquist)
    print("Done.")
