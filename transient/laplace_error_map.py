"""
laplace_error_map.py — Grid search of AE reconstruction error in the Laplace domain.

For each point s_k = re + i*im on a log-spaced Nr×Ni grid, trains a small conv
autoencoder on hat_U(s_k) fields and records the median relative L2 val error.
Produces an error heatmap and a field preview grid (Re(hat_U) per s_k).

Checkpoints to plots/error_map.pkl after every grid point (safe to interrupt/resume).

Run:
    .conda/bin/python transient/laplace_error_map.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleConvAE(nn.Module):
    """
    Compact conv AE with split Re/Im refinement heads in the decoder,
    mirroring the LaplaceDecoder structure from models/laplace_ae_surrogate.py.
    No FiLM conditioning — one fresh instance per s_k.

    For N=200: base=25, enc_flat=64*625=40000, dec_flat=32*625=20000.
    Encoder: 200→100→50→25 (3 stride-2 conv blocks).
    Decoder: 25→50→100→200 (3 stride-2 ConvTranspose blocks + 2 heads).
    """
    def __init__(self, N: int = 200, dz: int = 32):
        super().__init__()
        base     = N // 8
        enc_flat = 64 * base * base
        dec_flat = 32 * base * base

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 16, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_flat, 2 * dz), nn.ReLU(),
            nn.Linear(2 * dz, dz),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(dz, 2 * dz), nn.ReLU(),
            nn.Linear(2 * dz, dec_flat), nn.ReLU(),
        )
        self._base = base

        # Shared upsampling trunk
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16,  8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d( 8,  8, 4, 2, 1), nn.ReLU(),
        )

        # Two separate refinement heads (Re and Im)
        self.refine_re = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
        self.refine_im = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.encoder_fc(self.encoder_conv(x))
        h     = self.decoder_fc(z).view(x.size(0), 32, self._base, self._base)
        trunk = self.decoder_deconv(h)
        return torch.cat([self.refine_re(trunk), self.refine_im(trunk)], dim=1)


# ── Laplace transform — GPU-batched over (im_vals × sim_batch) ───────────────

def compute_laplace_row(U_sims: np.ndarray, re_val: float, im_vals: np.ndarray,
                        dt: float, device: str, sim_batch: int = 50) -> np.ndarray:
    """
    Compute hat_U(s_k) for one fixed Re(s)=re_val and all Im(s) in im_vals.

    Uses a single GPU matmul over the (2*n_im, Nt) kernel matrix so all
    frequencies are processed simultaneously — avoids re-reading U_sims
    once per frequency.

    Parameters
    ----------
    U_sims   : (N_sim, Nt, H, W) float16 in RAM
    re_val   : scalar Re(s)
    im_vals  : (n_im,) array of Im(s) values
    sim_batch: number of simulations per GPU chunk (tune to VRAM)

    Returns
    -------
    (n_im, N_sim, 2, H, W) float32
    """
    N_sim, Nt, H, W = U_sims.shape
    n_im = len(im_vals)
    HW   = H * W

    t   = torch.arange(Nt, dtype=torch.float32, device=device)
    w   = torch.ones(Nt, device=device); w[0] = 0.5; w[-1] = 0.5
    amp = dt * w * torch.exp(torch.tensor(-re_val, dtype=torch.float32, device=device) * t)

    im_t  = torch.tensor(im_vals, dtype=torch.float32, device=device)  # (n_im,)
    phase = im_t[:, None] * t[None, :]                                  # (n_im, Nt)
    k_re  =  amp[None, :] * torch.cos(phase)                           # (n_im, Nt)
    k_im  = -amp[None, :] * torch.sin(phase)
    k     = torch.cat([k_re, k_im], dim=0)                             # (2*n_im, Nt)

    # Accumulate into CPU buffer to keep GPU memory small
    out = np.empty((2 * n_im, N_sim, HW), dtype=np.float32)
    for s in range(0, N_sim, sim_batch):
        U_b  = torch.from_numpy(
            np.asarray(U_sims[s:s + sim_batch], dtype=np.float32)
        ).to(device)                                        # (B, Nt, H, W)
        B    = U_b.shape[0]
        U_t  = U_b.reshape(B, Nt, HW).permute(1, 0, 2).reshape(Nt, B * HW)
        hatU = torch.mm(k, U_t).reshape(2 * n_im, B, HW)
        out[:, s:s + B, :] = hatU.cpu().numpy()

    out_re = out[:n_im].reshape(n_im, N_sim, H, W)
    out_im = out[n_im:].reshape(n_im, N_sim, H, W)
    return np.stack([out_re, out_im], axis=2)               # (n_im, N_sim, 2, H, W)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(results: dict, fields_demo: dict, sim_idx: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'results': results, 'fields_demo': fields_demo,
                     'sim_idx': sim_idx}, f)


def load_ckpt(path: str):
    if Path(path).exists():
        with open(path, 'rb') as f:
            d = pickle.load(f)
        print(f"Resuming from {path}  ({len(d['results'])} / {15*15} points done)")
        return d['results'], d['fields_demo'], d.get('sim_idx')
    return {}, {}, None


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation(results: dict, x_key: str, y_key: str,
                     xlabel: str, ylabel: str, title: str, path: str):
    """Log-log scatter of y_key vs x_key across all grid points, with power-law fit."""
    xs, ys = [], []
    for v in results.values():
        x = v.get(x_key)
        y = v.get(y_key)
        if x is not None and y is not None and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
    if not xs:
        print(f"  [skip] no data for {x_key} vs {y_key}")
        return
    xs, ys = np.array(xs), np.array(ys)
    log_x, log_y = np.log(xs), np.log(ys)
    r = float(np.corrcoef(log_x, log_y)[0, 1])
    slope, intercept = np.polyfit(log_x, log_y, 1)

    x_fit = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 200)
    y_fit = np.exp(intercept) * x_fit ** slope

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=45, alpha=0.85, edgecolors='k', linewidths=0.4, zorder=3)
    ax.plot(x_fit, y_fit, 'r--', lw=1.8,
            label=f'slope = {slope:.2f},  r = {r:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_error_heatmap(results: dict, re_vals: np.ndarray, im_vals: np.ndarray,
                       path: str):
    Nr, Ni = len(re_vals), len(im_vals)
    grid = np.full((Nr, Ni), np.nan)
    for (ri, ci), v in results.items():
        grid[ri, ci] = v['val_error']

    fig, ax = plt.subplots(figsize=(7, 5))
    pm = ax.pcolormesh(re_vals, im_vals, grid.T, cmap='viridis', shading='nearest')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_yscale('log')
    ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Re(s)'); ax.set_ylabel('Im(s)')
    ax.set_title('Median relative L2 error — AE reconstruction in Laplace domain')

    plt.colorbar(pm, ax=ax, label='Median rel. L2 error')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_field_grid(fields_demo: dict, re_vals: np.ndarray, im_vals: np.ndarray,
                    path: str):
    Nr, Ni = len(re_vals), len(im_vals)
    fig, axes = plt.subplots(Nr, Ni, figsize=(Ni * 1.8, Nr * 1.8), squeeze=False)
    for ri in range(Nr):
        for ci in range(Ni):
            ax  = axes[ri, ci]
            img = fields_demo.get((ri, ci))
            if img is not None:
                vmax = float(np.abs(img).max()) or 1.0
                ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(f're={re_vals[ri]:.2f}\nim={im_vals[ci]:.2f}', fontsize=4.5)
            ax.axis('off')
    plt.suptitle('Re(hat_U(s_k))  — demo simulation', fontsize=10, y=1.005)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Config ────────────────────────────────────────────────────────────────
    DATA_PATH  = 'dataset/ch4_rotated.npy'
    dt         = 1.0
    Nt         = 150
    N=200
    N_POS      = 6
    N_NEG      = N_POS*2//3
    N_OMEGA    = 10
    N_sim      = 1000           # random subset of simulations to use
    SIM_BATCH  = 50            # sims per GPU chunk in compute_laplace_row
    dz         = 32            # AE latent dimension
    epochs     = 300
    patience   = 20
    batch_size = 32
    lr         = 1e-3
    RESULTS    = 'plots/error_map.pkl'
    FIG_ERROR  = 'plots/error_map.png'
    FIG_FIELDS = 'plots/error_map_fields.png'
    FIG_CORR_SVD    = 'plots/laplace_error_svd_correlation.png'
    FIG_CORR_RELVAR = 'plots/laplace_error_variance_correlation.png'
    VAR_MAP_PKL     = 'plots/laplace_variance_map.pkl'
    SEED       = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Grid ──────────────────────────────────────────────────────────────────
    T          = Nt * dt
    nyquist    = np.pi / dt
    re_neg  = -np.logspace(-3, -2, N_NEG)[::-1]  # [-0.05…-0.01]
    re_zero = np.array([0.0])
    re_pos  = np.logspace(-3, -1, N_POS)                                  # [0.001…1.0]
    re_vals = np.concatenate([re_neg, re_zero, re_pos])

    im_vals = np.logspace(np.log10(1.0 / T), np.log10(nyquist), N_OMEGA)

    # ── Variance map (pre-computed by laplace_variance_map.py) ───────────────
    var_map_data = {}
    if Path(VAR_MAP_PKL).exists():
        with open(VAR_MAP_PKL, 'rb') as f:
            _vm = pickle.load(f)
        var_map_data = _vm.get('results', {})
        print(f"Loaded variance map: {len(var_map_data)} points")
    else:
        print(f"Warning: {VAR_MAP_PKL} not found — correlation plots will be skipped")

    # ── Data / checkpoint ─────────────────────────────────────────────────────
    results, fields_demo, sim_idx_ckpt = load_ckpt(RESULTS)

    # Enrich already-computed results with variance map data (resume-safe)
    for key, v in results.items():
        if 'svd_residual' not in v and key in var_map_data:
            v['svd_residual']      = var_map_data[key]['svd_residual']
            v['relative_variance'] = var_map_data[key]['relative_variance']
    U_raw = np.load(DATA_PATH, mmap_mode='r')           # (ns, Nt, H, W)
    rng   = np.random.default_rng(SEED)

    if sim_idx_ckpt is not None:
        sim_idx = sim_idx_ckpt
    else:
        sim_idx = np.sort(rng.choice(U_raw.shape[0], N_sim, replace=False))

    n_train = int(0.8 * N_sim)
    n_val   = N_sim - n_train

    print(f"Grid: {N_NEG+N_POS+1}×{N_OMEGA} = {len(re_vals) * len(im_vals)} points")
    print(f"Simulations: {N_sim}  (train={n_train}, val={n_val})")

    print(f"Loading {N_sim} simulations into RAM (float16, "
          f"{N_sim*Nt*N*N*2/1e9:.1f} GB)...")
    U_sims = U_raw[sim_idx].astype(np.float16)          # (N_sim, Nt, H, W)
    print("  done.")

    # ── Main grid loop ────────────────────────────────────────────────────────
    # Outer loop: one Re(s) row at a time.
    # Per row: one GPU matmul computes all pending Im(s) simultaneously,
    # then AEs are trained from the precomputed fields.
    for ri, re in enumerate(tqdm(re_vals, desc='Re(s)', position=0)):

        # Identify which Im(s) values still need computation in this row
        pending_ci = [ci for ci in range(len(im_vals)) if (ri, ci) not in results]

        if pending_ci:
            # One GPU call for all pending im values in this row
            row_fields = compute_laplace_row(
                U_sims, re, im_vals[np.array(pending_ci)], dt, device, SIM_BATCH
            )  # (len(pending_ci), N_sim, 2, H, W)

            for j, ci in enumerate(tqdm(pending_ci, desc='  Im(s)', position=1, leave=False)):
                key    = (ri, ci)
                fields = row_fields[j]              # (N_sim, 2, H, W)
                s      = complex(re, im_vals[ci])

                fields_demo[key] = fields[0, 0].copy()   # Re(hat_U) of first sim

                # ── Dead zone check ───────────────────────────────────────
                norms = np.linalg.norm(fields.reshape(N_sim, -1), axis=1)
                if float(np.median(norms)) < 1e-10:
                    entry = {'val_error': 1.0, 's': s}
                    if key in var_map_data:
                        entry['svd_residual']      = var_map_data[key]['svd_residual']
                        entry['relative_variance'] = var_map_data[key]['relative_variance']
                    results[key] = entry
                    save_ckpt(results, fields_demo, sim_idx, RESULTS)
                    continue

                # ── Normalise (train-split stats) ─────────────────────────
                mean        = fields[:n_train].mean(axis=0, keepdims=True)
                std         = fields[:n_train].std(axis=0, keepdims=True) + 1e-8
                fields_norm = (fields - mean) / std

                # ── DataLoaders ───────────────────────────────────────────
                X            = torch.from_numpy(fields_norm).float()
                train_loader = DataLoader(TensorDataset(X[:n_train]),
                                          batch_size=batch_size, shuffle=True,
                                          num_workers=0)
                val_loader   = DataLoader(TensorDataset(X[n_train:]),
                                          batch_size=batch_size, num_workers=0)

                # ── Train fresh AE with early stopping ───────────────────
                model     = SimpleConvAE(N=N, dz=dz).to(device)
                opt       = torch.optim.Adam(model.parameters(), lr=lr)
                crit      = nn.MSELoss()
                best_val  = float('inf')
                best_state = None
                wait       = 0

                for _ in tqdm(range(epochs), desc='    train', position=2, leave=False):
                    model.train()
                    for (xb,) in train_loader:
                        xb = xb.to(device)
                        opt.zero_grad()
                        crit(model(xb), xb).backward()
                        opt.step()

                    model.eval()
                    with torch.no_grad():
                        val_loss = sum(
                            crit(model(xb.to(device)), xb.to(device)).item()
                            for (xb,) in val_loader
                        ) / len(val_loader)

                    if val_loss < best_val - 1e-6:
                        best_val   = val_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        wait       = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break

                model.load_state_dict(best_state)

                # ── Validation error ──────────────────────────────────────
                model.eval()
                preds_list = []
                with torch.no_grad():
                    for (xb,) in val_loader:
                        preds_list.append(model(xb.to(device)).cpu())
                preds_norm = torch.cat(preds_list).numpy()
                preds_raw  = preds_norm * std + mean
                tgts_raw   = fields[n_train:]
                diffs      = np.linalg.norm((preds_raw - tgts_raw).reshape(n_val, -1), axis=1)
                norms_val  = np.linalg.norm(tgts_raw.reshape(n_val, -1), axis=1)
                val_error  = float(np.median(diffs / (norms_val + 1e-10)))

                entry = {'val_error': val_error, 's': s}
                if key in var_map_data:
                    entry['svd_residual']      = var_map_data[key]['svd_residual']
                    entry['relative_variance'] = var_map_data[key]['relative_variance']
                results[key] = entry
                save_ckpt(results, fields_demo, sim_idx, RESULTS)

        # Intermediate heatmap after each Re(s) row
        plot_error_heatmap(results, re_vals, im_vals, FIG_ERROR)

    # ── Final plots ───────────────────────────────────────────────────────────
    print("\nPlotting final figures...")
    plot_error_heatmap(results, re_vals, im_vals, FIG_ERROR)
    plot_field_grid(fields_demo, re_vals, im_vals, FIG_FIELDS)
    plot_correlation(
        results,
        x_key='svd_residual', y_key='val_error',
        xlabel='Relative SVD residual  (rank-$n$ tail energy)',
        ylabel='AE val error  (median rel. L2)',
        title='AE reconstruction error vs. SVD residual',
        path=FIG_CORR_SVD,
    )
    plot_correlation(
        results,
        x_key='relative_variance', y_key='val_error',
        xlabel=r'Relative variance  $\mathrm{Var}[\hat{U}(s)]\,/\,\mathbb{E}[|\hat{U}|^2]$',
        ylabel='AE val error  (median rel. L2)',
        title='AE reconstruction error vs. relative variance',
        path=FIG_CORR_RELVAR,
    )
    print("Done.")
