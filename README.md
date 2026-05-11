# diffusion_ae

Émulation de solveurs EDP (convection-diffusion stationnaire et transitoire CH4) par apprentissage automatique.

## Structure

```
models/                        — architectures PyTorch
  base.py                      — BaseAutoEncoder, BaseDecoder
  variationalAutoEncoder.py
  direct_decoder.py            — DirectDecoder, DirectDecoderDenseOut
  laplace_surrogate.py         — LaplaceModel (pipeline 3)
  laplace_svd_surrogate.py     — LaplaceSVDModel (pipeline 2)
  laplace_ae_surrogate.py      — LaplaceLatentModel (pipeline 4)
  svd_surrogate.py             — SVDSurrogate (pipeline 1)
  correction_ae.py             — CorrectionAE, CorrectedPipeline (pipeline 5)

stationary/                    — champ stationnaire (θ → U)

transient/                     — champs transitoires (θ → U(t))
  dataset.py                   — TransientDataset
  main.py                      — load_model, run_inference
  app.py                       — Streamlit
  benchmark.py
  tucker/                      — pipeline 1 : Tucker POD (domaine temporel)
    learn_svd.py
    train_surrogate.py
  laplace_svd/                 — pipeline 2 : SVD linéaire par fréquence
    train.py
  laplace_ae/                  — pipelines 3-5 : AE Laplace + surrogate + correction
    train_ae.py
    train_surrogate.py
    finetune.py
    train_correction.py

utils/                         — solveur FEM, SVD, Laplace, animations
scripts/                       — analyses, visualisation, one-off
dataset/                       — données brutes (.npy / .npz)
checkpoints/                   — modèles sauvegardés (.pt)
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

### Pipeline 1 — Tucker POD (domaine temporel)

```bash
.conda/bin/python transient/tucker/learn_svd.py
.conda/bin/python transient/tucker/train_surrogate.py
```

### Pipeline 2 — SVD linéaire par fréquence Laplace

```bash
.conda/bin/python transient/laplace_svd/train.py
```

### Pipeline 3 — Surrogate direct dans l'espace de Laplace

```bash
.conda/bin/python transient/laplace_ae/train_surrogate.py
```

### Pipeline 4 — AE latent Laplace (3 étapes)

```bash
.conda/bin/python transient/laplace_ae/train_ae.py
.conda/bin/python transient/laplace_ae/train_surrogate.py  # mode LaplaceLatentSurrogate
.conda/bin/python transient/laplace_ae/finetune.py
```

### Pipeline 5 — CorrectionAE (post-traitement)

Corrige les artefacts oscillatoires du surrogate frame-par-frame via un UNet résiduel.
Pré-calcule les paires (U_pred, U_true) une seule fois, puis entraîne le UNet.

```bash
.conda/bin/python transient/laplace_ae/train_correction.py
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