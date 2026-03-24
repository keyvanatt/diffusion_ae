# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surrogate model for a 2D convection-diffusion PDE using a Conditional Variational Autoencoder (CVAE). Given physical parameters θ = (D, bx, by, x0x, x0y, f), the model predicts the steady-state solution U on an N×N grid.

## Commands

**Generate dataset** (run from `utils/` directory — `sim.py` is imported directly):
```bash
cd utils && python dataset_generator.py
```

**Train:**
```bash
wandb login
python utils/train.py
```

**Inference:**
```bash
python main.py --theta 1.0 0.5 0.3 0.8 0.2 0.6
python main.py --theta 1.0 0.5 0.3 0.8 0.2 0.6 --n_samples 4 --out pred.npy --plot
```

**Run a single FEM simulation (standalone test):**
```bash
cd utils && python sim.py
```

## Architecture

### Data pipeline
- `utils/sim.py`: FEM solver using DOLFINx + PETSc. Solves `-D∇²u + b·∇u = f·δ(x-x₀)` on the unit square with Dirichlet BC=0. Uses SUPG stabilization for convection-dominated regimes. Outputs a DOLFINx `Function`.
- `utils/dataset_generator.py`: Calls `simulate()` + `to_grid()` to produce (θ, U) pairs and saves to `dataset/dataset.npz`. θ is stored raw and z-score normalized. D is sampled log-uniformly.
- `utils/dataset.py` (`ConvDiffDataset`): Loads `.npz`; **requires calling `dataset.fit(train_indices)` before use** to compute U_min/U_max on train split only (prevents leakage). U is min-max normalized to [-1, 1].

### Models
Both models are in `models/`. Grid resolution N must be divisible by 16 (base = N//16).

- **`CVAE`**: Encoder takes `(U, θ)` — concatenates θ as a spatial feature map to U channels. Decoder takes `(z, θ)`.
- **`CVAELight`**: Encoder and decoder condition only on (bx, by) (indices 1 and 2 of θ). The latent z must therefore encode D, f, x0x, x0y, which discourages posterior collapse. Reconstruction loss is variance-normalized: `MSE / (var(U) + 1e-6)`.

Both share the same 4-layer strided-conv encoder / transposed-conv decoder structure. Loss = reconstruction MSE + spatial gradient MSE (`lambda_grad` weight) + KL with free-bits.

### Checkpoint format
Saved by `utils/train.py`, consumed by `main.py`:
```python
{
  'model_type': 'cvae_light' | 'cvae',
  'model_state': ...,
  'config': CONFIG dict,
  'U_min', 'U_max': float  # for denormalization
  'theta_mean', 'theta_std': tensor  # for normalizing raw θ at inference
}
```

`main.py` infers N from the checkpoint by reading `dec_fc.0.weight` (CVAELight) or `decoder.fc.0.weight` (CVAE) shape.

### Training config
All hyperparameters are in the `CONFIG` dict at the top of `utils/train.py`. Key defaults: `latent_dim=16`, `beta=0.5`, `lambda_grad=15.0`, `epochs=500`, `patience=40`. Uses AdamW + ReduceLROnPlateau (scheduler steps on val recon loss). Logs to W&B project `cvae-light-convdiff`.

## Dependencies
- PyTorch, NumPy, tqdm, wandb, matplotlib
- DOLFINx, PETSc, mpi4py, UFL (FEM stack — installed in `.conda/`)
- scipy (for `griddata` interpolation from FEM mesh to regular grid)

The conda environment is at `.conda/` in the project root.
