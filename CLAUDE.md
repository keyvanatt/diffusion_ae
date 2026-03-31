# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local conda environment at `.conda/`. Always run scripts with:
```bash
.conda/bin/python <script>
```

All scripts add their parent directory to `sys.path`, so they must be run from the repo root.

## Running scripts

```bash
# Generate the physics dataset (requires FEniCS/DOLFINx)
.conda/bin/python utils/dataset_generator.py

# Train the VAE on the physics dataset
.conda/bin/python utils/train_ae.py

# Train the VAE on MNIST
.conda/bin/python utils/train_ae_mnist.py

# Train a direct decoder (theta → U, no latent)
.conda/bin/python utils/train_decoder.py

# Run inference
.conda/bin/python main.py --theta 0.02 0.5 0.3 10.0 --plot

# Launch the Streamlit visualization app
.conda/bin/streamlit run app.py
```

## Architecture overview

The project learns to emulate a 2D convection-diffusion PDE solver: given physics parameters θ = (D, bx, by, f), predict the solution field U on an N×N grid.

**Two-stage pipeline:**
1. **AE stage** (`utils/train_ae.py`): Train a VAE to compress U fields into a latent space z.
2. **Decoder stage** (`utils/train_decoder.py`): Train a decoder that maps θ → U (either directly, or via θ → z → U using the frozen VAE decoder).

**Model hierarchy:**
- `models/base.py` — `BaseAutoEncoder` and `BaseDecoder` abstract base classes. All models must implement `loss()`.
- `models/variationalAutoEncoder.py` — `VAE` (encoder + decoder), plus `IndirectDecoder` which projects θ into the VAE's latent space and reuses its frozen decoder.
- `models/direct_decoder.py` — `DirectDecoder` and `DirectDecoderDenseOut` (θ → U directly, no AE).

**Data flow:**
- `utils/sim.py` — FEniCS/DOLFINx solver that produces FEM solutions, interpolated to N×N grids via scipy.
- `utils/dataset_generator.py` — samples random θ, runs the simulator, saves to `dataset/dataset.npz`.
- `utils/dataset.py` — `ConvDiffDataset`: loads `.npz`, applies train-set-fitted normalization. Must call `dataset.fit(train_indices)` before training. Returns `(theta_norm, U_norm)` pairs.

**Normalization:**
U is center-normalized (subtract per-pixel train mean, then min-max scale to [-1, 1]). Stats are stored in checkpoints. θ is z-score normalized using stats pre-computed during dataset generation.

## Decoder architecture constraint

Both `VAE.Decoder` and `DirectDecoderDenseOut` use `base = N // 32`, so **N must be a multiple of 32** (e.g. N=32 for MNIST, N=64 for the physics dataset). `DirectDecoder` uses `base = N // 16`, so N must be a multiple of 16.

## Inference and visualization

`main.py` exposes two functions that must be reused by any new script needing inference:
- `load_model(ckpt_path, device)` — auto-detects model type from checkpoint, reconstructs the model.
- `run_inference(theta_raw, model, ckpt, device)` — normalizes θ, runs the decoder, denormalizes U back to physical values.

**Always import and reuse these functions** rather than reimplementing model loading or denormalization logic.

`app.py` is a Streamlit app that wraps these two functions. It exposes sliders for θ (D, |b|, angle, f — the app converts polar to (bx, by) internally), lets the user pick a checkpoint, and optionally compares the prediction against the nearest dataset sample. The app uses `@st.cache_resource` for the model and `@st.cache_data` for the dataset.

## Experiment tracking

All training scripts log to [Weights & Biases](https://wandb.ai) (project `convdiff`). Checkpoints are saved to `checkpoints/<ModelName>_best.pt`. The best model is selected on `val/recon` if available, else `val/loss`.

## SVD-based surrogate pipeline (Amine dataset)

A second pipeline operates on a separate dataset of CH4 concentration fields `(ns, T, H, W)` with parameters `doe` = `(k, A, C, theta)`.

**Data:** `dataset/Results/CH4.npy` (or `ch4_rotated.npy`), `dataset/Results/doe.npy` (or `doe_rotated.npy`)

**Step 1 — Tucker SVD decomposition** (`utils/learn_svd.py`):
- Spatial subsampling (default `step=5`), reshape to `HH (nr, ns, Nt)`
- `svd_3d_gpu` from `utils/SVD_Amine_3D.py` decomposes HH into `F (nr, nf_eff)`, `G (ns, nf_eff)`, `P (Nt, nf_eff)`, `alph (nf_eff,)`
- `G` encodes per-simulation coefficients — this is what the surrogate must learn to predict from theta
- Saves `dataset/Results/svd_train.npz`; also saves a comparison GIF in `plots/` for visual check

**Step 2 — Surrogate training** (two versions):
- `utils/train_surrogate_svd.py` — MLP PyTorch (`SVDSurrogate` in `models/svd_surrogate.py`), logs to W&B, saves `test_idx` in checkpoint
- `utils/train_surrogate_svd_sklearn.py` — sklearn Pipeline (`PolynomialFeatures + StandardScaler + Ridge`), saved with `joblib`

**Evaluation** (both scripts):
- Metrics computed against two references: SVD reconstruction (G_true → field) and original concentration
- Key metric: **L2 relative error** (comparable to `learn_svd` output)
- Histograms (log x-scale) and animations (best/median/worst) saved in `plots/`

**Known limitation:** With ~150 samples, both surrogates give ~60% L2rel error vs ~13-17% for SVD alone. The bottleneck is likely that G does not vary smoothly with theta at this sample size. Consider per-simulation SVD or more samples.

**SVD implementation note:** In `svd_3d_gpu`, denominators must be recomputed sequentially after each R/S/T update (not precomputed) to avoid NaN divergence.


