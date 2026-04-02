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

`transient/main.py` est l'équivalent pour le dataset transitoire : `predict(theta, ckpt, ...)` et `evaluate(U, theta, ckpt, ...)` avec détection automatique du backend (Laplace ou SVD).

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
.conda/bin/python stationary/main.py --theta 0.02 0.5 0.3 10.0 --plot

# App Streamlit
.conda/bin/streamlit run stationary/app.py
```

## Experiment tracking

Tous les scripts loggent sur [Weights & Biases](https://wandb.ai) (projet `convdiff`). Les checkpoints sont sauvegardés dans `checkpoints/<ModelName>_best.pt`. Le meilleur modèle est sélectionné sur `val/recon` si disponible, sinon `val/loss`.

---

# Partie 2 — Dataset transitoire (champs temporels)

Objectif : émuler des champs de concentration CH4 transitoires `U(t)` en fonction de paramètres θ. Deux datasets disponibles, même objectif :

- **CH4 (ancien, ~150 samples)** : `dataset/Results/CH4.npy` + `dataset/Results/doe.npy` (ou versions `_rotated`). Paramètres `doe = (k, A, C, theta)`. Format `(ns, T, H, W)`.
- **dataset_transient (~5 000 samples)** : `dataset/dataset_transient.npz`. Contient `U (ns, Nt, N, N)`, `theta (ns, theta_dim)`, `dt`. À préférer — le gain en samples réduit significativement l'erreur surrogate.

## Pipeline 1 — SVD Tucker + surrogate

**Principe :** décomposer les champs via Tucker SVD pour extraire des coefficients `G` par simulation, puis apprendre θ → G.

**Étape 1 — Décomposition Tucker SVD** (`transient/learn_svd.py`) :
- Sous-échantillonnage spatial (défaut `step=5`), reshape en `HH (nr, ns, Nt)`
- `svd_3d_gpu` de `utils/SVD_Amine_3D.py` décompose HH en `F (nr, nf_eff)`, `G (ns, nf_eff)`, `P (Nt, nf_eff)`, `alph (nf_eff,)`
- `G` encode les coefficients par simulation — c'est ce que le surrogate doit apprendre à prédire depuis theta
- Sauvegarde `dataset/Results/svd_train.npz` ; sauvegarde aussi un GIF de comparaison dans `plots/`

**Étape 2 — Entraînement surrogate** :
- `transient/train_surrogate_svd.py` — MLP PyTorch (`SVDSurrogate` dans `models/svd_surrogate.py`), logs W&B, sauvegarde `F`, `P`, `test_idx` dans le checkpoint

**Évaluation** : via `transient/main.py evaluate(...)` — métriques L2 relative, histogramme et animations best/median/worst sauvegardés dans `plots/`.

**Limitation connue (CH4.npy) :** avec ~150 samples, les deux surrogates donnent ~60% d'erreur L2rel vs ~13-17% pour le SVD seul. Le goulot d'étranglement est que G ne varie pas lissément avec theta à cette taille.

**Note d'implémentation SVD :** dans `svd_3d_gpu`, les dénominateurs doivent être recalculés séquentiellement après chaque mise à jour R/S/T (pas pré-calculés) pour éviter la divergence NaN.

```bash
.conda/bin/python transient/learn_svd.py
.conda/bin/python transient/train_surrogate_svd.py
```

## Pipeline 2 — Surrogate dans l'espace de Laplace

**Principe :** au lieu de prédire U(t) directement, prédire sa transformée de Laplace numérique $\hat{U}(s_k)$ pour chaque fréquence complexe $s_k = \gamma + i\omega_k$, puis reconstruire U(t) par transformée inverse. Un MLP indépendant est entraîné par fréquence.

**Fichiers clés :**
- `utils/laplace.py` — `laplace_forward` et `laplace_inverse` : quadrature trapézoïdale ou rectangulaire sur le signal discret.
- `models/laplace_surrogate.py` — `LaplaceSurrogate(s, N, theta_dim)` : MLP → ConvTranspose2d, prédit `(Re(Û), Im(Û))` soit `(B, 2, N, N)`. Architecture : `base = N // 16` (N doit être multiple de 16). Une instance par fréquence.
- `transient/dataset.py` — `TransientDataset` : charge `dataset_transient.npz`, option `laplace=True` qui pré-calcule la transformée de Laplace. Appeler `dataset.fit(train_indices)` avant l'entraînement.
- `transient/train_laplace.py` — deux fonctions :
  - `train_one(k, s_k, ...)` → entraîne un `LaplaceSurrogate` pour la fréquence k, sauvegarde dans `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt`.
  - `train_all(dataset, train_idx, val_idx, ...)` → boucle sur toutes les fréquences, log W&B global.

**Checkpoints :** `checkpoints/laplace/LaplaceSurrogate_freq{k:03d}.pt`, un par fréquence. `test_idx.npy` dans le même répertoire. Chaque checkpoint stocke `theta_mean/std`, `target_mean/std`, `N`, `Nt`, `theta_dim`, `freq_idx`, `s_k_real/imag`.

**Normalisation :** theta z-score global (stats train) ; target z-score par fréquence (moyenne sur pixels et batch du train set), calculé par `dataset.fit()`.

**Symétrie conjuguée :** seules les `Nt_half = Nt//2 + 1` premières fréquences sont entraînées. À l'inférence, le spectre complet est reconstruit par `M[Nt-k] = conj(M[k])`.

```bash
.conda/bin/python transient/train_laplace.py

# Inférence / évaluation transitoire
.conda/bin/python transient/main.py --ckpt checkpoints/laplace --theta 1.0 0.5 0.3 2.0 --plot
.conda/bin/python transient/main.py --ckpt checkpoints/laplace --evaluate --data dataset/dataset_transient.npz
```
