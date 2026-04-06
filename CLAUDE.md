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

Les deux surrogates héritent de `BaseDecoder` et exposent la même interface :
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

## Pipeline 1 — SVD Tucker + surrogate

**Principe :** décomposer les champs via Tucker SVD pour extraire des coefficients `G` par simulation, puis apprendre θ → G.

**Étape 1 — Décomposition Tucker SVD** (`transient/learn_svd.py`) :
- Sous-échantillonnage spatial (défaut `step=5`), reshape en `HH (nr, ns, Nt)`
- `svd_3d_gpu` de `utils/SVD_Amine_3D.py` décompose HH en `F (nr, nf_eff)`, `G (ns, nf_eff)`, `P (Nt, nf_eff)`, `alph (nf_eff,)`
- Sauvegarde `dataset/svd_train_diff.npz` ; GIF de comparaison dans `plots/`

**Étape 2 — Entraînement surrogate** (`transient/train_surrogate_svd.py`) :
- Utilise `TransientDataset` pour theta et ses stats de normalisation
- Sauvegarde `checkpoints/SVDSurrogate_best.pt` avec `model_state` (inclut les buffers F, P, alph, G_mean, G_std), `theta_mean/std`, `test_idx`

**Note d'implémentation SVD :** dans `svd_3d_gpu`, les dénominateurs doivent être recalculés séquentiellement après chaque mise à jour R/S/T pour éviter la divergence NaN.

```bash
.conda/bin/python transient/learn_svd.py
.conda/bin/python transient/train_surrogate_svd.py
```

## Pipeline 2 — Surrogate dans l'espace de Laplace

**Principe :** prédire la transformée de Laplace numérique $\hat{U}(s_k)$ pour chaque fréquence $s_k = \gamma + i\omega_k$, puis reconstruire U(t) par transformée inverse.

**Fichiers clés :**
- `utils/laplace.py` — `laplace_forward` et `laplace_inverse`
- `transient/train_laplace.py` — trois fonctions :
  - `train_one(k, s_k, ...)` → entraîne un `LaplaceSurrogate` pour la fréquence k, sauvegarde dans `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt`
  - `train_all(dataset, train_idx, val_idx, test_idx, ...)` → boucle sur toutes les fréquences, puis appelle `assemble_model`
  - `assemble_model(dataset, ckpt_dir, test_idx, save_dir)` → charge les N checkpoints individuels, assemble un `LaplaceModel` unique, sauvegarde dans `save_dir/LaplaceModel.pt`

**Checkpoints :**
- `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt` — un par fréquence (pour reprise d'entraînement)
- `checkpoints/LaplaceModel.pt` — modèle assemblé pour l'inférence, contient `model_state` (avec buffers `target_mean/std`), `theta_mean/std`, `dt`, `gamma`, `test_idx`

**Symétrie conjuguée :** seules les `Nt_half = Nt//2 + 1` premières fréquences sont entraînées. Le spectre complet est reconstruit par `M[Nt-k] = conj(M[k])`.

```bash
.conda/bin/python transient/train_laplace.py
```

## Inférence transitoire

`transient/main.py` miroir de `stationary/main.py` :
- `load_model(ckpt_path, device)` — charge `LaplaceModel` ou `SVDSurrogate` depuis un fichier `.pt`
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
