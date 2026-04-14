# diffusion_ae

Émulation de solveurs EDP (convection-diffusion stationnaire et transitoire CH4) par apprentissage automatique.

## Structure

```
models/               — architectures PyTorch
  base.py             — BaseAutoEncoder, BaseDecoder
  variationalAutoEncoder.py
  direct_decoder.py   — DirectDecoder, DirectDecoderDenseOut
  laplace_surrogate.py         — LaplaceModel (Pipeline 2)
  laplace_ae_surrogate.py      — LaplaceLatentModel (Pipeline 3)
  svd_surrogate.py             — SVDSurrogate (Pipeline 1)
  correction_ae.py             — CorrectionAE, CorrectedPipeline (Pipeline 4)

stationary/           — champ stationnaire (θ → U)
transient/            — champs transitoires (θ → U(t))
utils/                — solveur FEM, génération dataset, SVD, Laplace, animations
dataset/              — données brutes (.npy / .npz)
checkpoints/          — modèles sauvegardés (.pt)
```

## Partie 1 — Stationnaire

Prédit le champ stationnaire de convection-diffusion 2D depuis θ = (D, bx, by, f).

```bash
.conda/bin/python stationary/train_ae.py       # VAE
.conda/bin/python stationary/train_decoder.py  # décodeur direct θ→U
.conda/bin/python stationary/main.py           # inférence
.conda/bin/streamlit run stationary/app.py     # app interactive
```

## Partie 2 — Transitoire

Prédit les champs CH4 transitoires U(t) depuis θ = (k, A, C).
Dataset principal : `dataset/ch4_rotated.npy` (8 100 sims × 150 pas × 200×200).

### Pipeline 1 — SVD Tucker

```bash
.conda/bin/python transient/learn_svd.py
.conda/bin/python transient/train_surrogate_svd.py
```

### Pipeline 2 — Surrogate Laplace

```bash
.conda/bin/python transient/train_laplace.py
```

### Pipeline 3 — AE latent Laplace (3 étapes)

```bash
.conda/bin/python transient/train_ae_laplace.py
.conda/bin/python transient/train_laplace.py      # mode LaplaceLatentSurrogate
.conda/bin/python transient/finetune_decoder_laplace.py
```

### Pipeline 4 — CorrectionAE (post-traitement)

Corrige les artefacts oscillatoires du surrogate frame-par-frame via un UNet résiduel.
Pré-calcule les paires (U_pred, U_true) une seule fois, puis entraîne le UNet.

```bash
.conda/bin/python transient/train_correction_ae.py
```

Inférence via `CorrectedPipeline` (surrogate + correction enchaînés) :

```bash
.conda/bin/python transient/main.py   # ckpt_path = checkpoints/CorrectionAE_best.pt
```

## Inférence générale

```python
from transient.main import load_model, run_inference

model, ckpt = load_model('checkpoints/CorrectionAE_best.pt', device)
U_pred = run_inference(theta_raw, model, ckpt, device)  # (B, Nt, N, N)
```

`load_model` détecte automatiquement le type de modèle (`LaplaceModel`, `LaplaceLatentModel`, `SVDSurrogate`, `CorrectionAE`) et charge le surrogate associé si besoin.

## Experiment tracking

Tous les scripts loggent sur [Weights & Biases](https://wandb.ai) (projet `convdiff`).
Checkpoints dans `checkpoints/<ModelName>_best.pt`.