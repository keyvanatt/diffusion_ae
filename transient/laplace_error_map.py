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


# ── Laplace transform for arbitrary s ────────────────────────────────────────

def compute_laplace_fields(U_batch: np.ndarray, s: complex, dt: float) -> np.ndarray:
    """
    Compute hat_U(s) = dt * sum_n w_n * U_i[n] * exp(-s * t_n) for a batch of sims.

    Parameters
    ----------
    U_batch : (B, Nt, H, W) float32
    s       : complex scalar  (s = re + i*im)
    dt      : time step

    Returns
    -------
    (B, 2, H, W) float32 — [Re(hat_U), Im(hat_U)]
    """
    Nt = U_batch.shape[1]
    t  = np.arange(Nt, dtype=np.float64) * dt
    w  = np.ones(Nt); w[0] = 0.5; w[-1] = 0.5          # trapezoidal weights
    amp  = dt * w * np.exp(-s.real * t)                  # exp(-Re(s)*t) part
    k_re = (amp *  np.cos(s.imag * t)).astype(np.float32)   # Re(exp(-s*t))
    k_im = (amp * -np.sin(s.imag * t)).astype(np.float32)   # Im(exp(-s*t))
    M_re = np.einsum('t,bthw->bhw', k_re, U_batch, optimize=True)
    M_im = np.einsum('t,bthw->bhw', k_im, U_batch, optimize=True)
    return np.stack([M_re, M_im], axis=1)                    # (B, 2, H, W)


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

def plot_error_heatmap(results: dict, re_vals: np.ndarray, im_vals: np.ndarray,
                       path: str):
    Nr, Ni = len(re_vals), len(im_vals)
    grid = np.full((Nr, Ni), np.nan)
    for (ri, ci), v in results.items():
        grid[ri, ci] = v['val_error']

    fig, ax = plt.subplots(figsize=(7, 5))
    pm = ax.pcolormesh(re_vals, im_vals, grid.T, cmap='viridis', shading='nearest')
    ax.set_xscale('log'); ax.set_yscale('log')
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
    N          = 200
    Nr         = 15            # grid points along Re(s)
    Ni         = 15            # grid points along Im(s)
    N_sim      = 500           # random subset of simulations to use
    LOAD_BATCH = 50            # sims loaded per batch when computing Laplace fields
    dz         = 32            # AE latent dimension
    epochs     = 30
    batch_size = 32
    lr         = 1e-3
    RESULTS    = 'plots/error_map.pkl'
    FIG_ERROR  = 'plots/error_map.png'
    FIG_FIELDS = 'plots/error_map_fields.png'
    SEED       = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Grid ──────────────────────────────────────────────────────────────────
    T       = Nt * dt                                   # total duration
    re_vals = np.logspace(-2, np.log10(10.0 / dt), Nr)
    im_vals = np.logspace(np.log10(1.0 / T), np.log10(np.pi / dt), Ni)

    # ── Data / checkpoint ─────────────────────────────────────────────────────
    results, fields_demo, sim_idx_ckpt = load_ckpt(RESULTS)
    U_raw = np.load(DATA_PATH, mmap_mode='r')           # (ns, Nt, H, W)
    rng   = np.random.default_rng(SEED)

    if sim_idx_ckpt is not None:
        sim_idx = sim_idx_ckpt
    else:
        sim_idx = np.sort(rng.choice(U_raw.shape[0], N_sim, replace=False))

    n_train = int(0.8 * N_sim)
    n_val   = N_sim - n_train

    print(f"Grid: {Nr}×{Ni} = {Nr * Ni} points")
    print(f"Simulations: {N_sim}  (train={n_train}, val={n_val})")

    # ── Main grid loop ────────────────────────────────────────────────────────
    for ri, re in enumerate(tqdm(re_vals, desc='Re(s)', position=0)):
        for ci, im in enumerate(tqdm(im_vals, desc='  Im(s)', position=1, leave=False)):
            key = (ri, ci)
            if key in results:
                continue

            s = complex(re, im)

            # ── Compute Laplace fields ─────────────────────────────────────
            batches = []
            n_batches = (N_sim + LOAD_BATCH - 1) // LOAD_BATCH
            for start in tqdm(range(0, N_sim, LOAD_BATCH),
                              desc='    load', total=n_batches,
                              position=2, leave=False):
                idx    = sim_idx[start:start + LOAD_BATCH]
                U_b    = U_raw[idx].astype(np.float32)      # (B, Nt, H, W)
                batches.append(compute_laplace_fields(U_b, s, dt))
            fields = np.concatenate(batches, axis=0)        # (N_sim, 2, H, W)

            # Store Re(hat_U) of first sim for the field preview grid
            fields_demo[key] = fields[0, 0].copy()          # (H, W)

            # ── Dead zone check ────────────────────────────────────────────
            norms = np.linalg.norm(fields.reshape(N_sim, -1), axis=1)
            if float(np.median(norms)) < 1e-10:
                results[key] = {'val_error': 1.0, 's': s}
                save_ckpt(results, fields_demo, sim_idx, RESULTS)
                continue

            # ── Normalise (stats on train split only) ─────────────────────
            mean        = fields[:n_train].mean(axis=0, keepdims=True)   # (1,2,H,W)
            std         = fields[:n_train].std(axis=0, keepdims=True) + 1e-8
            fields_norm = (fields - mean) / std                           # (N_sim,2,H,W)

            # ── DataLoaders ────────────────────────────────────────────────
            X            = torch.from_numpy(fields_norm).float()
            train_loader = DataLoader(TensorDataset(X[:n_train]),
                                      batch_size=batch_size, shuffle=True,
                                      num_workers=0)
            val_loader   = DataLoader(TensorDataset(X[n_train:]),
                                      batch_size=batch_size, num_workers=0)

            # ── Train fresh AE ─────────────────────────────────────────────
            model = SimpleConvAE(N=N, dz=dz).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            crit  = nn.MSELoss()

            for _ in tqdm(range(epochs), desc='    train', position=2, leave=False):
                model.train()
                for (xb,) in train_loader:
                    xb = xb.to(device)
                    opt.zero_grad()
                    crit(model(xb), xb).backward()
                    opt.step()

            # ── Validation error ───────────────────────────────────────────
            model.eval()
            preds_list = []
            with torch.no_grad():
                for (xb,) in val_loader:
                    preds_list.append(model(xb.to(device)).cpu())
            preds_norm = torch.cat(preds_list).numpy()       # (Nval, 2, H, W)
            preds_raw  = preds_norm * std + mean             # denormalise
            tgts_raw   = fields[n_train:]
            diffs      = np.linalg.norm((preds_raw - tgts_raw).reshape(n_val, -1), axis=1)
            norms_val  = np.linalg.norm(tgts_raw.reshape(n_val, -1), axis=1)
            val_error  = float(np.median(diffs / (norms_val + 1e-10)))

            results[key] = {'val_error': val_error, 's': s}
            save_ckpt(results, fields_demo, sim_idx, RESULTS)

        # Intermediate error heatmap after each Re(s) row
        plot_error_heatmap(results, re_vals, im_vals, FIG_ERROR)

    # ── Final plots ───────────────────────────────────────────────────────────
    print("\nPlotting final figures...")
    plot_error_heatmap(results, re_vals, im_vals, FIG_ERROR)
    plot_field_grid(fields_demo, re_vals, im_vals, FIG_FIELDS)
    print("Done.")
