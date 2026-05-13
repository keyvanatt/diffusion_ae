"""
train_full_pipeline.py — Pipeline complet Laplace (5 étapes séquentielles).

Étapes :
  1. Optimisation du chemin de Laplace           → s_opt  (laplace_opti.py)
  2. Entraînement du LaplaceAE                   → LaplaceAE_best.pt  (train_ae.py)
  3. Entraînement des surrogates θ→z             → LaplaceLatentModel.pt  (train_surrogate.py)
  4. Finetune end-to-end                         → LaplaceLatentModel_finetuned.pt  (finetune.py)
  5. CorrectionAE frame-par-frame                → CorrectionAE_best.pt  (train_correction.py)

Les runs wandb utilisent les mêmes noms que dans les fichiers respectifs :
  "laplace_opti_s", "LaplaceAE", "LaplaceLatentSurrogate_Nt{Nt}", "LaplaceLatentModel_finetune_e2e", "CorrectionAE"

Seuls les indices train (issus du split) sont utilisés pour l'optimisation et l'entraînement.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gc
import time
import numpy as np
import torch

from transient.laplace_ae.laplace_opti import optimize_laplace_path
from transient.laplace_ae.train_ae import train_ae
from transient.laplace_ae.train_surrogate import train_all
from transient.laplace_ae.finetune import finetune
from transient.laplace_ae.train_correction import precompute, train as train_correction
from transient.dataset import TransientDataset


def _make_splits(split_path, seed):
    """
    Charge dataset/split.npz, retourne (train_idx, val_idx, test_idx).
    train_idx / val_idx sont tirés du pool train du split (80/20).
    """
    _split   = np.load(split_path)
    test_idx = _split['test_idx'].tolist()
    non_test = _split['train_idx'].tolist()   # déjà le pool train global

    g = torch.Generator()
    g.manual_seed(seed)
    perm    = torch.randperm(len(non_test), generator=g)
    n_train = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]
    return train_idx, val_idx, test_idx


def main(
    # --- Data ---
    data_path    = os.path.join('dataset', 'ch4_rotated.npy'),
    split_path   = os.path.join('dataset', 'split.npz'),
    cache_dir    = '/Data/KAT',
    seed         = 42,
    dt           = 1.0,
    rule         = 'trap',
    interp_size  = 128,
    project      = 'convdiff',
    ns_max       = None,   # si fourni, tronque le dataset à ns_max samples (smoke test)

    # --- Stage 1 : Laplace path optimization ---
    K            = 20,     # nombre de points s optimisés
    gamma        = 0.0,
    lambda_diff  = 0.5,
    lambda_x     = 0.5,
    step         = 1,
    n_cases_opti = 100,
    n_latent     = 64,
    lambda_ae_o  = 1.25,
    gamma_min    = -0.05,
    lr_opti      = 5e-3,
    n_epochs_opti = 500,
    case_chunk   = 10,
    sp_chunk     = 2000,

    # --- Stage 2 : LaplaceAE ---
    latent_dim   = 64,
    freq_L       = 8,
    epochs_ae    = 100,
    batch_size_ae = 256,
    lr_ae        = 5e-4,
    beta         = 0,
    patience_ae  = 30,
    k_max        = None,   # None → toutes les K fréquences optimisées

    # --- Stage 3 : Surrogates θ→z ---
    epochs_surr    = 1000,
    batch_size_surr = 512,
    lr_surr        = 1e-3,
    patience_surr  = 50,
    hidden_dim     = 512,

    # --- Stage 4 : Finetune ---
    epochs_ft      = 50,
    batch_size_ft  = 16,
    lr_surrogate   = 5e-5,
    lr_decoder     = 1e-5,
    patience_ft    = 15,
    # Valeurs optimales de l'inverseur Laplace (sorties de Stage 1).
    # À renseigner explicitement quand resume_from >= 2.
    alpha_t        = 0.092214,
    lam            = 0.32193,

    # --- Stage 5 : CorrectionAE ---
    epochs_corr    = 100,
    batch_size_corr = 64,
    kt             = 20,
    base_ch        = 16,
    lr_corr        = 1e-3,
    patience_corr  = 15,
    lambda_grad    = 10.0,

    # --- Dirs ---
    ckpt_dir       = 'checkpoints',
    surr_ckpt_dir  = os.path.join('checkpoints', 'laplace_latent'),

    # --- Reprise ---
    # 1 = pipeline complet, 2 = depuis AE, 3 = depuis surrogates, 4 = depuis finetune, 5 = depuis correction
    resume_from    = 1,
):
    t_pipeline = time.perf_counter()

    # -----------------------------------------------------------------------
    # Splits
    # -----------------------------------------------------------------------
    print("=== Splits ===")
    train_idx, val_idx, test_idx = _make_splits(split_path, seed)
    if ns_max is not None:
        train_idx = [i for i in train_idx if i < ns_max]
        val_idx   = [i for i in val_idx   if i < ns_max]
        test_idx  = [i for i in test_idx  if i < ns_max]
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    assembled_ckpt = os.path.join(ckpt_dir, 'LaplaceLatentModel.pt')
    finetuned_ckpt = os.path.join(ckpt_dir, 'LaplaceLatentModel_finetuned.pt')
    ae         = None   # initialisé dans stage 2 ou 3
    dataset_ae = None   # initialisé si resume_from <= 4

    # -----------------------------------------------------------------------
    # Stage 1 — Optimisation du chemin de Laplace (ou chargement depuis ckpt)
    # -----------------------------------------------------------------------
    if resume_from <= 1:
        print("\n=== Stage 1 : Optimisation du chemin de Laplace ===")
        _data    = np.load(data_path, mmap_mode='r')
        _Nt_data = int(_data.shape[1])
        _doe_path    = os.path.join(os.path.dirname(data_path), 'doe_rotated.npy')
        _doe         = np.load(_doe_path)
        _A_vals      = _doe['A'] if ns_max is None else _doe['A'][:ns_max]
        _opti_indices = [i for i in train_idx if _A_vals[i] == 0]
        print(f"  Indices pour opti (A=0) : {len(_opti_indices)} / {len(train_idx)} train")
        s_opt, _lam_opt, _alpha_t_opt = optimize_laplace_path(
            Nt=_Nt_data, dt=dt, K=K, gamma=gamma,
            lambda_diff=lambda_diff, lambda_x=lambda_x,
            step=step, n_cases=n_cases_opti, n_latent=n_latent,
            lambda_ae=lambda_ae_o, gamma_min=gamma_min,
            lr=lr_opti, n_epochs=n_epochs_opti,
            case_chunk=case_chunk, sp_chunk=sp_chunk,
            seed=seed, data_path=data_path, log_wandb=True,
            indices=_opti_indices,
        )
        s_list = s_opt.numpy().astype(np.complex128)
        if k_max is not None:
            s_list = s_list[:k_max + 1]
        print(f"s_list ({len(s_list)} pts) — lam={_lam_opt:.5f}  alpha_t={_alpha_t_opt:.6f}")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n=== Stage 1 ignoré (resume_from={resume_from}) ===")

        s_list = (1j * 2 * np.pi * np.fft.rfftfreq(150, d=dt))[:20]
        _lam_opt     = float(lam)
        _alpha_t_opt = float(alpha_t)
        print(f"  s_list ({len(s_list)} pts) chargé depuis {assembled_ckpt}")
        print(f"  lam={_lam_opt:.5f}  alpha_t={_alpha_t_opt:.6f}")

    # -----------------------------------------------------------------------
    # Dataset Laplace (stages 2-4) — cache disque, rapide si déjà calculé
    # -----------------------------------------------------------------------
    if resume_from <= 4:
        print("\nChargement dataset Laplace...", end=' ', flush=True)
        dataset_ae = TransientDataset(
            data_path, laplace=True, s_list=s_list,
            rule=rule, interp_size=interp_size, dt=dt, cache_dir=cache_dir,
            ns_max=ns_max,
        )
        dataset_ae.fit(train_idx)
        print("Chargement en RAM...", end=' ', flush=True)
        t0    = time.perf_counter()
        U_lap = np.ascontiguousarray(dataset_ae.U_laplace)
        dataset_ae.U_laplace = U_lap
        print(f"OK — {U_lap.nbytes/1e9:.1f} Go, {time.perf_counter()-t0:.1f}s")
        os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Stage 2 — Entraînement du LaplaceAE
    # -----------------------------------------------------------------------
    if resume_from <= 2:
        print("\n=== Stage 2 : Entraînement LaplaceAE ===")
        ae = train_ae(
            dataset_ae, train_idx, val_idx,
            latent_dim=latent_dim,
            epochs=epochs_ae, batch_size=batch_size_ae,
            lr=lr_ae, beta=beta, patience=patience_ae,
            freq_L=freq_L, k_max=k_max,
            ckpt_dir=ckpt_dir, project=project,
        )
    elif resume_from == 3:
        print("\n=== Stage 2 ignoré — chargement LaplaceAE ===")
        from models.laplace_ae_surrogate import LaplaceAE
        _ae_ckpt = torch.load(os.path.join(ckpt_dir, 'LaplaceAE_best.pt'),
                              map_location='cpu', weights_only=False)
        ae = LaplaceAE(N=interp_size, latent_dim=latent_dim)
        ae.load_state_dict(_ae_ckpt['model_state'])
        ae.eval()
        print(f"  LaplaceAE chargé ({latent_dim=})")

    # -----------------------------------------------------------------------
    # Stage 3 — Entraînement des surrogates θ→z
    # -----------------------------------------------------------------------
    if resume_from <= 3:
        print("\n=== Stage 3 : Entraînement des surrogates θ→z ===")
        os.makedirs(surr_ckpt_dir, exist_ok=True)
        train_all(
            dataset_ae, train_idx, val_idx, test_idx,
            epochs=epochs_surr, batch_size=batch_size_surr,
            lr=lr_surr, patience=patience_surr,
            ckpt_dir=surr_ckpt_dir, project=project,
            ae=ae, k_max=k_max, hidden_dim=hidden_dim, freq_L=freq_L,
            alpha_t=_alpha_t_opt, lam=_lam_opt,
        )
        del ae
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 4 — Finetune end-to-end
    # -----------------------------------------------------------------------
    if resume_from <= 4:
        print("\n=== Stage 4 : Finetune end-to-end ===")
        finetune(
            dataset_ae, train_idx, val_idx, test_idx,
            ckpt_path=assembled_ckpt,
            epochs=epochs_ft, batch_size=batch_size_ft,
            lr_surrogate=lr_surrogate, lr_decoder=lr_decoder,
            patience=patience_ft, save_dir=ckpt_dir, project=project,
            dt=dt, rule=rule, alpha_t=_alpha_t_opt, lam=_lam_opt,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 5 — CorrectionAE
    # -----------------------------------------------------------------------
    print("\n=== Stage 5 : CorrectionAE ===")
    dataset_corr = TransientDataset(
        data_path, laplace=False,
        interp_size=interp_size, dt=dt, cache_dir=cache_dir,
        ns_max=ns_max,
    )
    dataset_corr.fit(train_idx)

    U_pred, U_true = precompute(
        dataset_corr, finetuned_ckpt, kt=kt, cache_dir=cache_dir,
        batch_size=32, dt=dt, alpha_t=_alpha_t_opt, lam=_lam_opt, rule=rule, seed=seed,
    )
    train_correction(
        U_pred, U_true,
        train_idx, val_idx, test_idx,
        surrogate_ckpt=finetuned_ckpt,
        epochs=epochs_corr, batch_size=batch_size_corr, kt=kt,
        base_ch=base_ch, lr=lr_corr, patience=patience_corr,
        lambda_grad=lambda_grad,
        save_dir=ckpt_dir, project=project,
        N=dataset_corr.N, Nt=dataset_corr.Nt,
    )

    elapsed = (time.perf_counter() - t_pipeline) / 60
    print(f"\n=== Pipeline complet terminé en {elapsed:.1f} min ===")
    print(f"  CorrectionAE → {os.path.join(ckpt_dir, 'CorrectionAE_best.pt')}")


if __name__ == '__main__':
    main(resume_from=2)
