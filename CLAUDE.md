# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local conda environment at `.conda/`. Always run scripts with:
```bash
.conda/bin/python <script>
```

All scripts add their parent directory to `sys.path`, so they must be run from the repo root.

---

# Partie 1 — Dataset stationnaire (convection-diffusion 2D)

Objectif : émuler un solveur EDP de convection-diffusion 2D. Donnés des paramètres physiques θ = (D, bx, by, f), prédire le champ de solution stationnaire U sur une grille N×N.

## Données

- `utils/sim.py` — solveur FEniCS/DOLFINx, produit des solutions FEM interpolées sur une grille N×N via scipy.
- `utils/dataset_generator.py` — échantillonne des θ aléatoires, lance le simulateur, sauvegarde dans `dataset/dataset.npz`.
- `stationary/dataset.py` — `ConvDiffDataset` : charge `.npz`, applique une normalisation ajustée sur le train. Appeler `dataset.fit(train_indices)` avant l'entraînement. Retourne des paires `(theta_norm, U_norm)`.

**Normalisation :** U est centré-normalisé (soustraction de la moyenne par pixel du train, puis min-max vers [-1, 1]). Les stats sont stockées dans les checkpoints. θ est normalisé z-score avec des stats pré-calculées lors de la génération.

## Pipelines

**Pipeline VAE + décodeur (deux étapes) :**
1. **Étape AE** (`stationary/train_ae.py`) : entraîne un VAE pour compresser les champs U dans un espace latent z.
2. **Étape décodeur** (`stationary/train_decoder.py`) : entraîne un décodeur θ → U (directement, ou via θ → z → U avec le décodeur VAE gelé).

**Modèles :**
- `models/base.py` — classes abstraites `BaseAutoEncoder` et `BaseDecoder`. Tous les modèles doivent implémenter `loss()`.
- `models/variationalAutoEncoder.py` — `VAE` (encodeur + décodeur), plus `IndirectDecoder` qui projette θ dans l'espace latent du VAE et réutilise son décodeur gelé.
- `models/direct_decoder.py` — `DirectDecoder` et `DirectDecoderDenseOut` (θ → U directement, sans AE).

**Contrainte d'architecture :** `VAE.Decoder` et `DirectDecoderDenseOut` utilisent `base = N // 32` (N doit être multiple de 32, ex. N=32 pour MNIST, N=64 pour le dataset physique). `DirectDecoder` utilise `base = N // 16` (N multiple de 16).

## Inférence et visualisation

`stationary/main.py` expose deux fonctions à réutiliser dans tout nouveau script d'inférence :
- `load_model(ckpt_path, device)` — détecte automatiquement le type de modèle depuis le checkpoint.
- `run_inference(theta_raw, model, ckpt, device)` — normalise θ, lance le décodeur, dénormalise U vers les valeurs physiques.

**Toujours importer et réutiliser ces fonctions** plutôt que de réimplémenter le chargement de modèle ou la dénormalisation.

`stationary/app.py` est une app Streamlit qui enveloppe ces deux fonctions. Elle expose des sliders pour θ (D, |b|, angle, f — l'app convertit polaire → (bx, by) en interne), permet de choisir un checkpoint, et compare optionnellement la prédiction avec l'échantillon le plus proche du dataset. Utilise `@st.cache_resource` pour le modèle et `@st.cache_data` pour le dataset.

## Commandes

```bash
# Générer le dataset physique (requiert FEniCS/DOLFINx)
.conda/bin/python utils/dataset_generator.py

# Entraîner le VAE sur le dataset physique
.conda/bin/python stationary/train_ae.py

# Entraîner le VAE sur MNIST
.conda/bin/python stationary/train_ae_mnist.py

# Entraîner un décodeur direct (theta → U, sans latent)
.conda/bin/python stationary/train_decoder.py

# Inférence stationnaire
.conda/bin/python stationary/main.py

# App Streamlit
.conda/bin/streamlit run stationary/app.py
```

## Experiment tracking

Tous les scripts loggent sur [Weights & Biases](https://wandb.ai) (projet `convdiff`). Les checkpoints sont sauvegardés dans `checkpoints/<ModelName>_best.pt`. Le meilleur modèle est sélectionné sur `val/recon` si disponible, sinon `val/loss`.

---

# Partie 2 — Dataset transitoire (champs temporels)

Objectif : émuler des champs de concentration CH4 transitoires `U(t)` en fonction de paramètres θ. Deux datasets disponibles, même objectif :

- **ch4_rotated (~8 100 samples)** : `dataset/ch4_rotated.npy` + `dataset/doe_rotated.npy`. Format `(8100, 150, 200, 200)`. Paramètres `doe = (k, A, C)` (le champ `theta` du doe est l'angle de rotation, déjà appliqué — on l'ignore). **Dataset principal à utiliser.**
- **dataset_transient (~5 000 samples)** : `dataset/dataset_transient.npz`. Contient `U (ns, Nt, N, N)`, `theta (ns, theta_dim)`, `dt`.

## Dataset

- `transient/dataset.py` — `TransientDataset(data_path, laplace=False, gamma, rule, dt, doe_path)` :
  - Si `data_path` se termine par `.npy` : charge `ch4_rotated.npy` + `doe_rotated.npy` (déduit automatiquement), theta_dim=3 `(k, A, C)`, `dt` à passer en paramètre.
  - Sinon : ancien format `.npz` (clés `U`, `theta`, `dt`).
  - `laplace=False` : `__getitem__` retourne `(theta_norm, U)` de shape `(Nt, N, N)`
  - `laplace=True` : pré-calcule la transformée de Laplace, `__getitem__` retourne `(theta_norm, U_laplace_norm)` de shape `(Nt_half, 2, N, N)`
  - Appeler `dataset.fit(train_indices)` avant l'entraînement pour calculer les stats de normalisation.

## Modèles

Les surrogates héritent de `BaseDecoder` et exposent la même interface :
- `forward(theta_norm)` → représentation intermédiaire normalisée
- `generate(theta_norm, ...)` → `U_pred (B, Nt, N, N)` en valeurs physiques

**`models/laplace_surrogate.py`** :
- `LaplaceSurrogate(N, theta_dim, s)` — MLP → ConvTranspose2d, prédit `(Re(Û), Im(Û))` soit `(B, 2, N, N)` pour une fréquence. Architecture : `base = N // 16`.
- `LaplaceModel(N_freq, N_half, N, theta_dim)` — encapsule `N_half` surrogates.
  - `forward(theta_norm)` → spectre normalisé `(B, N_freq, N, N)` complexe
  - `_generate(theta_norm, dt, gamma, rule)` → `U_pred (B, Nt, N, N)` (dénorm + inverse Laplace)
  - `target_mean/std` stockés comme buffers (remplis via `set_normalization()` avant sauvegarde)

**`models/svd_surrogate.py`** :
- `SVDSurrogate(nr, Nt, nf_eff, theta_dim)` — MLP, prédit les coefficients G normalisés `(B, nf_eff)`
  - `_generate(theta_norm)` → `U_pred (B, Nt, Hsub, Wsub)` (dénorm G + reconstruction SVD)
  - `F`, `P`, `alph`, `G_mean`, `G_std` stockés comme buffers (remplis via `set_bases()` avant sauvegarde)

**`models/laplace_ae_surrogate.py`** (Pipeline 3) :
- `SinusoidalFreqEncoding(L, hidden_dim, out_dim)` — positional encoding sinusoïdal (style NeRF) pour freq_ratio ∈ [0, 1], utilisé pour le conditionnement FiLM.
- `LaplaceEncoder(N, latent_dim, freq_L)` — Conv2d U→z (3 downsampling), conditionné sur freq_ratio via FiLM.
- `LaplaceDecoder(N, latent_dim, freq_L)` — ConvTranspose2d z→Û (3 upsampling + raffinement séparé Re/Im), conditionné sur freq_ratio via FiLM. `base = N // 8`.
- `LaplaceAE(N, latent_dim, beta, freq_L)` — autoencoder déterministe (Encoder + Decoder) + ridge loss (L2 sur les sorties). Conditionné sur freq_ratio.
- `LaplaceLatentSurrogate(latent_dim, theta_dim, freq_ratio)` — MLP θ→z (`proj` : 4 couches Linear + LayerNorm). Utilise un `shared_decoder` injecté par référence (non ré-enregistré comme sous-module). Appeler `set_decoder(decoder)` avant `forward()`.
- `LaplaceLatentModel(N_freq, N_half, N, theta_dim, latent_dim, freq_L)` — hérite de `LaplaceModel`. Contient N_half `LaplaceLatentSurrogate` + un `shared_decoder` (LaplaceDecoder) gelé partagé.
  - `set_ae_decoder(ae)` — charge les poids du décodeur AE dans `shared_decoder`
  - `load_state_dict(...)` — ré-injecte le décodeur dans tous les surrogates après chargement

## Pipeline 1 — SVD Tucker + surrogate

**Principe :** décomposer les champs via Tucker SVD pour extraire des coefficients `G` par simulation, puis apprendre θ → G.

**Étape 1 — Décomposition Tucker SVD** (`transient/tucker/learn_svd.py`) :
- Sous-échantillonnage spatial (défaut `step=5`), reshape en `HH (nr, ns, Nt)`
- `svd_3d_gpu` de `utils/SVD_Amine_3D.py` décompose HH en `F (nr, nf_eff)`, `G (ns, nf_eff)`, `P (Nt, nf_eff)`, `alph (nf_eff,)`
- Sauvegarde `dataset/svd_train_diff.npz` ; GIF de comparaison dans `plots/`

**Étape 2 — Entraînement surrogate** (`transient/tucker/train_surrogate.py`) :
- Utilise `TransientDataset` pour theta et ses stats de normalisation
- Sauvegarde `checkpoints/SVDSurrogate_best.pt` avec `model_state` (inclut les buffers F, P, alph, G_mean, G_std), `theta_mean/std`, `test_idx`

**Note d'implémentation SVD :** dans `svd_3d_gpu`, les dénominateurs doivent être recalculés séquentiellement après chaque mise à jour R/S/T pour éviter la divergence NaN.

```bash
.conda/bin/python transient/tucker/learn_svd.py
.conda/bin/python transient/tucker/train_surrogate.py
```

## Pipeline 2 — SVD linéaire dans l'espace de Laplace

**Principe :** SVD tronquée par fréquence sur les champs Laplace, puis MLP θ → coefficients SVD pour chaque fréquence.

**Fichier clé :** `transient/laplace_svd/train.py` — pour chaque fréquence k : SVD tronquée (`torch.svd_lowrank`) sur Re_k et Im_k du train set, puis entraîne un `LaplaceSVDSurrogate` par composante.

**Checkpoint :** `checkpoints/LaplaceSVDModel.pt`

```bash
.conda/bin/python transient/laplace_svd/train.py
```

## Pipeline 3 — Surrogate direct dans l'espace de Laplace

**Principe :** prédire la transformée de Laplace numérique $\hat{U}(s_k)$ pour chaque fréquence $s_k = \gamma + i\omega_k$, puis reconstruire U(t) par transformée inverse.

**Fichiers clés :**
- `utils/laplace.py` — `laplace_forward` et `laplace_inverse`
- `transient/laplace_ae/train_surrogate.py` — trois fonctions :
  - `train_one(k, s_k, ...)` → entraîne un `LaplaceSurrogate` (ou `LaplaceLatentSurrogate` si `vae` fourni) pour la fréquence k, sauvegarde dans `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt`
  - `train_all(dataset, train_idx, val_idx, test_idx, ...)` → boucle sur toutes les fréquences, puis appelle `assemble_model`
  - `assemble_model(dataset, ckpt_dir, test_idx, save_dir)` → charge les N checkpoints individuels, assemble un `LaplaceModel` unique, sauvegarde dans `save_dir/LaplaceModel.pt`

**Checkpoints :**
- `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt` — un par fréquence (pour reprise d'entraînement)
- `checkpoints/LaplaceModel.pt` — modèle assemblé pour l'inférence, contient `model_state` (avec buffers `target_mean/std`), `theta_mean/std`, `dt`, `gamma`, `test_idx`

**Symétrie conjuguée :** seules les `Nt_half = Nt//2 + 1` premières fréquences sont entraînées. Le spectre complet est reconstruit par `M[Nt-k] = conj(M[k])`.

```bash
.conda/bin/python transient/laplace_ae/train_surrogate.py
```

## Pipeline 4 — AE latent dans l'espace de Laplace (3 étapes)

**Principe :** entraîner d'abord un autoencoder `LaplaceAE` sur tous les champs Laplace (conditionné sur la fréquence via FiLM), puis apprendre θ → z (espace latent) pour chaque fréquence avec le décodeur gelé, et enfin finetuner end-to-end.

**Étape 1 — Entraînement de l'AE** (`transient/laplace_ae/train_ae.py`) :
- Dataset aplati `(simulation, fréquence)` : chaque paire est un échantillon.
- `_LaplaceFlatDataset` : ordre sim-first, `reshuffle()` à chaque epoch pour mélanger les sims.
- Sauvegarde `checkpoints/LaplaceAE_best.pt` avec `model_state`, `N`, `latent_dim`, `val_loss`.

**Étape 2 — Entraînement des surrogates θ→z** (`transient/laplace_ae/train_surrogate.py` avec `vae` fourni) :
- Utilise `train_one(..., vae=ae)` : crée un `LaplaceLatentSurrogate`, injecte le décodeur gelé, entraîne seulement `proj` (MLP θ→z).
- Sauvegarde `checkpoints/laplace_latent/LatentSurrogate_freq{k:03d}.pt` par fréquence.
- `assemble_model(...)` assemble l'ensemble en `LaplaceLatentModel`, sauvegarde `checkpoints/LaplaceLatentModel.pt`.

**Étape 3 — Finetune end-to-end** (`transient/laplace_ae/finetune.py`) :
- Charge `LaplaceLatentModel.pt`, dégèle le `shared_decoder`.
- `_FinetuneDataset` : ordre freq-first → 1 seule fréquence par batch → forward batché rapide (1 proj + 1 decoder call).
- LR différentiel : `lr_surrogate > lr_decoder` (typiquement 5e-5 / 1e-5).
- Log wandb par fréquence (`val_freq/loss_k`) toutes les 5 epochs.
- Sauvegarde `checkpoints/LaplaceLatentModel_finetuned.pt` (avec flag `finetuned=True`).

**Checkpoints :**
- `checkpoints/LaplaceAE_best.pt` — autoencoder Laplace entraîné
- `checkpoints/laplace_latent/LatentSurrogate_freq{k:03d}.pt` — surrogate par fréquence
- `checkpoints/LaplaceLatentModel.pt` — modèle assemblé (avant finetune)
- `checkpoints/LaplaceLatentModel_finetuned.pt` — modèle finetuné end-to-end

```bash
.conda/bin/python transient/laplace_ae/train_ae.py
# puis modifier train_surrogate.py pour utiliser le mode LaplaceLatentSurrogate (vae fourni)
.conda/bin/python transient/laplace_ae/train_surrogate.py
.conda/bin/python transient/laplace_ae/finetune.py
```

## Pipeline 5 — CorrectionAE (post-traitement frame-par-frame)

**Principe :** corriger les artefacts oscillatoires du surrogate en appliquant un UNet résiduel frame-par-frame. Pré-calcul unique des paires (U_pred, U_true) sous forme de memmaps, puis entraînement du UNet sans appel au surrogate.

**Fichiers clés :**
- `models/correction_ae.py` :
  - `CorrectionAE(N, base_ch)` — UNet résiduel (encodeur 3 niveaux + bottleneck + décodeur skip). `out_conv` zero-init → identité au démarrage. `forward()` normalise chaque frame per-sample avant le UNet et rescale le résidu en sortie (Option B). `loss(U_corr, U_true, U_pred)` porte sur le résidu normalisé par son écart-type (Option A) + terme gradient spatial.
  - `CorrectedPipeline(surrogate, correction_ae)` — hérite de `LaplaceLatentModel`. Partage les modules du surrogate par référence (pas de duplication mémoire). Surcharge `_generate()` : appelle `super()._generate()` puis applique la correction par chunks de 64 frames.
- `transient/laplace_ae/train_correction.py` :
  - `precompute()` — génère des memmaps `(ns, kt, N, N) × 2` dans `cache_dir`. Libère le surrogate GPU (`del surrogate; torch.cuda.empty_cache()`) avant l'entraînement.
  - `_FrameDataset` — lit directement depuis les memmaps, `num_workers=4`.
  - `train()` — deux barres tqdm (epoch + batch), wandb avec images toutes les 5 epochs.

**Hyperparamètres clés :** `kt=20` (frames par sim), `base_ch=16` (481K params), `lambda_grad=10.0`.

**Loss :** MSE résidu normalisé + `lambda_grad` × MSE gradients spatiaux du résidu normalisé.

**Checkpoint :**
- `checkpoints/CorrectionAE_best.pt` — contient `model_state`, `model_type='CorrectionAE'`, `N`, `base_ch`, `surrogate_ckpt` (chemin vers le surrogate associé), `test_idx`.

```bash
.conda/bin/python transient/laplace_ae/train_correction.py
```

## Inférence transitoire

`transient/main.py` miroir de `stationary/main.py` :
- `load_model(ckpt_path, device)` — charge `LaplaceModel`, `LaplaceLatentModel`, `SVDSurrogate` ou `CorrectionAE` depuis un fichier `.pt` (détection automatique via `model_type`). Pour `CorrectionAE` : charge le surrogate depuis `ckpt['surrogate_ckpt']`, construit un `CorrectedPipeline`, et remonte `theta_mean/std/dt/gamma` du surrogate dans le ckpt.
- `run_inference(theta_raw, model, ckpt, device, dt, gamma, rule)` — normalise θ depuis le checkpoint, appelle `model.generate(theta_norm, ...)`, retourne `U_pred (B, Nt, N, N)`
- `predict(theta_raw, ckpt_path, ...)` — combine les deux
- `evaluate(U, theta, ckpt_path, ...)` — métriques L2rel (en %), histogramme et animations GIFs dans `plots/`

**Toujours passer un fichier `.pt` direct** à ces fonctions (jamais un répertoire).

```bash
# Inférence
.conda/bin/python transient/main.py

# Évaluation (modifier do_evaluation=True dans main())
.conda/bin/python transient/main.py
```

---

# Rapport LaTeX

Le rapport est dans `report/text/main.tex`. Les figures sont dans `plots/` (chemin relatif configuré via `\graphicspath{{../../plots/}}`).

## Compilation

```bash
cd report/text && pdflatex -interaction=nonstopmode main.tex
```

Lancer deux fois si les références croisées changent (avertissement "Label(s) may have changed").

## Structure du document

- **Section 2** — Dataset stationnaire (convection-diffusion 2D) : VAE + décodeur indirect
- **Section 3** — Dataset transitoire CH4 :
  - `\subsection{Dataset and Temporal Representation}` : description des données, transformée de Laplace
  - `\subsection{Latent Space Representation}` :
    - `\subsubsection{Tucker POD Surrogate}` : algorithme ALS, application, entraînement θ→G, limitations (mémoire + erreur de représentation ~10%)
    - `\subsubsection{SVD Surrogate in the Laplace Domain}` : SVD par fréquence, limitations
    - `\subsubsection{Autoencoder-Based Latent Space}` : FiLM, encodage sinusoïdal, évaluation par fréquence
  - `\subsection{Surrogate and Fine-Tuning}` : surrogate θ→z, finetune end-to-end
  - `\subsection{Post-Processing and Results}` : UNet résiduel, tableau comparatif

## Commandes LaTeX personnalisées

- `\vtheta` → **θ** (gras)
- `\R`, `\C`, `\E`, `\KL`, `\diag`

## Figures clés

| Label | Fichier | Introduit dans |
|---|---|---|
| `fig:svd`(a) | `svd_convergence.png` | Application to CH4 dataset |
| `fig:svd`(b) | `SVDSurrogate_l2rel_hist.png` | Limitations (Tucker POD) |
| `fig:ae_freq_error` | `ae_study_frequency_error.png` | Autoencoder performance |
| `fig:ae_truncated_error` | `ae_study_laplace_truncated_error.png` | Choix de k_max |
| `fig:latentmodel_baseline` | `laplacelatentmodel_l2rel_hist.png` | Surrogate baseline |
| `fig:laplacelatent_finetuned` | `laplacelatentmodel_finetuned_l2rel_hist.png` | Fine-tuning |
| `fig:correction_unet` | `correctionae_unet_l2rel_hist.png` | Résultats correction |
| `fig:violin_benchmark` | `benchmark_l2rel.png` | Comparaison finale |
