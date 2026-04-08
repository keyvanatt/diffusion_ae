"""
train_laplace.py — Entraînement des surrogates dans l'espace de Laplace
=======================================================================
Un LaplaceSurrogate indépendant est entraîné par fréquence de Laplace.
La transformation des champs en espace de Laplace est assurée par TransientDataset.

Usage :
    python transient/train_laplace.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from models.laplace_surrogate import LaplaceSurrogate, LaplaceModel
from models.laplace_ae_surrogate import LaplaceLatentSurrogate


def train_one(
    k,
    s_k,
    train_loader,
    val_loader,
    theta_mean,
    theta_std,
    target_mean,
    target_std,
    N,
    Nt,
    epochs       = 300,
    lr           = 1e-3,
    patience     = 30,
    theta_dim    = 4,
    device       = None,
    ckpt_dir     = 'checkpoints',
    global_step  = 0,
    vae          = None,   # si fourni : LaplaceLatentSurrogate avec décodeur VAE gelé
    freq_ratio   = 0.0,    # k / (Nt_half - 1), pour le conditionnement FiLM
):
    """
    Entraîne un LaplaceSurrogate (ou LaplaceLatentSurrogate si ae fourni) pour la fréquence k.

    Retour
    ------
    best_val : float – meilleure val loss atteinte
    n_epochs : int   – nombre d'époques réalisées
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if vae is not None:
        model = LaplaceLatentSurrogate(latent_dim=vae.latent_dim, theta_dim=theta_dim, freq_ratio=freq_ratio).to(device)
        vae.decoder.to(device).requires_grad_(False)
        model.set_decoder(vae.decoder)
        params_to_train = model.proj.parameters()  # proj seulement, décodeur gelé
    else:
        model           = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim, freq_ratio=freq_ratio).to(device)
        params_to_train = model.parameters()

    model = model  # torch.compile inutile sur des MLP tiny (overhead compilation > gain)

    optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_val   = float('inf')
    best_state = None
    best_epoch = 0
    patience_  = 0
    epoch      = 0
    prefix    = 'LatentSurrogate' if vae is not None else 'LaplaceSurrogate'
    ckpt_path = os.path.join(ckpt_dir, f'{prefix}_freq{k:03d}.pt')

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"  freq {k:03d}  s={s_k.imag:.3g}j",
        leave=False,
        position=1,
    )
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss  = torch.zeros(1, device=device)
        train_l2rel = torch.zeros(1, device=device)
        for th, u in train_loader:
            th, u = th.to(device), u.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                u_hat = model(th)
                loss  = model.loss(u_hat, u)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss  += loss.detach()
            train_l2rel += ((u_hat.detach() - u).flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-8)).mean()
        train_loss  = train_loss.item()  / len(train_loader)
        train_l2rel = train_l2rel.item() / len(train_loader)

        # --- Val ---
        model.eval()
        val_loss  = torch.zeros(1, device=device)
        val_l2rel = torch.zeros(1, device=device)
        with torch.no_grad():
            for th, u in val_loader:
                th, u = th.to(device), u.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    u_hat = model(th)
                    val_loss  += model.loss(u_hat, u)
                val_l2rel += ((u_hat - u).flatten(1).norm(dim=1) / (u.flatten(1).norm(dim=1) + 1e-8)).mean()
        val_loss  = val_loss.item()  / len(val_loader)
        val_l2rel = val_l2rel.item() / len(val_loader)

        scheduler.step(val_loss)
        epoch_time = time.perf_counter() - t0
        epoch_bar.set_postfix(train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", l2=f"{val_l2rel:.2%}")

        wandb.log({
            'train/loss'   : train_loss,
            'train/l2rel'  : train_l2rel,
            'val/loss'     : val_loss,
            'val/l2rel'    : val_l2rel,
            'lr'           : optimizer.param_groups[0]['lr'],
            'epoch_time_s' : epoch_time,
        }, step=global_step + epoch)

        if val_loss < best_val:
            best_val       = val_loss
            best_epoch     = epoch
            best_state     = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            patience_      = 0
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.set_postfix(
                    train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", stop=f"ep{epoch}"
                )
                break

    # Sauvegarde unique en fin de training
    assert best_state is not None, f"Aucune epoch n'a amélioré la val loss pour freq {k}"
    torch.save({
        'model_state': best_state,
        'epoch': best_epoch,
        'val_loss': best_val,
        'theta_mean':  theta_mean.numpy(),
        'theta_std':   theta_std.numpy(),
        'target_mean': target_mean.numpy(),
        'target_std':  target_std.numpy(),
        'N': N, 'Nt': Nt, 'theta_dim': theta_dim,
        'freq_idx': k, 's_k_real': s_k.real, 's_k_imag': s_k.imag,
    }, ckpt_path)

    if wandb.run is not None:
        wandb.run.summary[f'best_val/freq_{k:03d}'] = best_val
    return best_val, epoch


def precompute_latents(vae, U_laplace, target_mean, target_std, Nt_half, device, batch_size=256):
    """
    Encode tous les champs Laplace normalisés avec vae.encoder.
    Retourne Z de shape (Nt_half, ns, latent_dim) en CPU RAM (~150MB).

    Z[k, i] = encoder(U_laplace_norm[k, i], freq_ratio=k/(Nt_half-1))
    """
    ns         = U_laplace.shape[0]
    latent_dim = vae.latent_dim
    encoder    = vae.encoder.to(device)
    encoder.eval()

    Z = torch.zeros(Nt_half, ns, latent_dim)
    with torch.no_grad():
        for k in tqdm(range(Nt_half), desc="Pré-calcul latents", leave=False):
            freq_ratio = k / max(Nt_half - 1, 1)
            tm = target_mean[k].to(device)   # (2, 1, 1)
            ts = target_std[k].to(device)
            for start in range(0, ns, batch_size):
                end = min(start + batch_size, ns)
                if isinstance(U_laplace, np.ndarray):
                    u = torch.from_numpy(U_laplace[start:end, k].copy()).float().to(device)
                else:
                    u = U_laplace[start:end, k].to(device)
                u_n = (u - tm) / ts
                Z[k, start:end] = encoder(u_n, freq_ratio).cpu()
    return Z   # (Nt_half, ns, latent_dim)


def train_chunk(
    ks,            # liste d'indices de fréquences à entraîner simultanément
    s_values,      # liste de complex — fréquences correspondantes
    th_train,      # (n_train, theta_dim) sur GPU
    u_trains,      # liste de K tenseurs sur GPU :
                   #   mode pixel  → (n_train, 2, N, N)
                   #   mode latent → (n_train, latent_dim)
    th_val,        # (n_val,   theta_dim) sur GPU
    u_vals,        # liste de K tenseurs sur GPU (même format que u_trains)
    theta_mean, theta_std,
    target_means, target_stds,  # listes de K tenseurs (2,1,1) — pour checkpoint
    N, Nt,
    batch_size   = 256,
    epochs       = 300,
    lr           = 1e-3,
    patience     = 15,
    theta_dim    = 4,
    device       = None,
    ckpt_dir     = 'checkpoints',
    global_step  = 0,
    vae          = None,
    latent_mode  = False,  # True → targets sont des z (n, latent_dim), loss en espace latent
):
    """
    Entraîne K surrogates simultanément (une fréquence par modèle).

    latent_mode=True : les u_trains/u_vals contiennent des codes latents pré-calculés.
    Le forward ne passe plus par le décodeur → 20-50× plus rapide par batch.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    K        = len(ks)
    Nt_half  = Nt // 2 + 1
    n_train  = th_train.shape[0]
    n_val    = th_val.shape[0]
    n_train_b = (n_train + batch_size - 1) // batch_size
    n_val_b   = (n_val   + batch_size - 1) // batch_size
    prefix    = 'LatentSurrogate' if vae is not None else 'LaplaceSurrogate'

    # ------------------------------------------------------------------
    # Initialisation des K modèles
    # ------------------------------------------------------------------
    # En mode latent, le décodeur n'est pas utilisé pendant l'entraînement
    if vae is not None and not latent_mode:
        vae.decoder.to(device).requires_grad_(False)

    models, optimizers, schedulers, scalers = [], [], [], []
    for i, k in enumerate(ks):
        freq_ratio = k / max(Nt_half - 1, 1)
        if vae is not None:
            m = LaplaceLatentSurrogate(latent_dim=vae.latent_dim, theta_dim=theta_dim,
                                       freq_ratio=freq_ratio).to(device)
            if not latent_mode:
                m.set_decoder(vae.decoder)
            params = m.proj.parameters()
        else:
            m      = LaplaceSurrogate(s=s_values[i], N=N, theta_dim=theta_dim,
                                      freq_ratio=freq_ratio).to(device)
            params = m.parameters()
        models.append(m)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=15, min_lr=1e-6))
        scalers.append(torch.cuda.amp.GradScaler(enabled=device.type == 'cuda'))

    best_vals      = [float('inf')] * K
    best_states: list[dict | None] = [None] * K
    best_epochs    = [0] * K
    patience_ctrs  = [0] * K
    active         = [True] * K

    epoch_bar = tqdm(range(1, epochs + 1),
                     desc=f"  freqs {ks[0]:03d}-{ks[-1]:03d}", leave=False, position=1)
    for epoch in epoch_bar:
        if not any(active):
            break

        # Permutation partagée entre toutes les fréquences du chunk
        perm = torch.randperm(n_train, device=device)

        # --- Train ---
        for m in models: m.train()
        train_losses = [0.0] * K

        for start in range(0, n_train, batch_size):
            idx  = perm[start:start + batch_size]
            th_b = th_train[idx]
            for i in range(K):
                if not active[i]:
                    continue
                tgt_b = u_trains[i][idx]
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    if latent_mode:
                        pred = models[i].proj(th_b)          # (B, latent_dim)
                        loss = F.mse_loss(pred, tgt_b)
                    else:
                        pred = models[i](th_b)               # (B, 2, N, N)
                        loss = models[i].loss(pred, tgt_b)
                optimizers[i].zero_grad()
                scalers[i].scale(loss).backward()
                scalers[i].unscale_(optimizers[i])
                torch.nn.utils.clip_grad_norm_(models[i].parameters(), 1.0)
                scalers[i].step(optimizers[i])
                scalers[i].update()
                train_losses[i] += loss.detach().item()

        # --- Val ---
        for m in models: m.eval()
        val_losses = [0.0] * K
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                end  = min(start + batch_size, n_val)
                idx  = torch.arange(start, end, device=device)
                th_b = th_val[idx]
                for i in range(K):
                    if not active[i]:
                        continue
                    tgt_b = u_vals[i][idx]
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                        if latent_mode:
                            pred = models[i].proj(th_b)
                            val_losses[i] += F.mse_loss(pred, tgt_b).item()
                        else:
                            pred = models[i](th_b)
                            val_losses[i] += models[i].loss(pred, tgt_b).item()

        # --- Early stopping + log (un seul wandb.log par epoch) ---
        log_dict = {}
        for i in range(K):
            tl = train_losses[i] / n_train_b
            vl = val_losses[i]   / n_val_b
            schedulers[i].step(vl)
            log_dict[f'freq{ks[i]:03d}/train_loss'] = tl
            log_dict[f'freq{ks[i]:03d}/val_loss']   = vl
            log_dict[f'freq{ks[i]:03d}/lr']         = optimizers[i].param_groups[0]['lr']
            if vl < best_vals[i]:
                best_vals[i]   = vl
                best_epochs[i] = epoch
                best_states[i] = {k_: v.cpu().clone() for k_, v in models[i].state_dict().items()}
                patience_ctrs[i] = 0
            else:
                patience_ctrs[i] += 1
                if patience_ctrs[i] >= patience:
                    active[i] = False
        wandb.log(log_dict, step=global_step + epoch)

        epoch_bar.set_postfix(active=len([a for a in active if a]),
                              best=f"{min(best_vals):.2e}")

    # --- Sauvegarde ---
    for i, k in enumerate(ks):
        assert best_states[i] is not None
        torch.save({
            'model_state': best_states[i],
            'epoch': best_epochs[i],
            'val_loss': best_vals[i],
            'theta_mean':  theta_mean.numpy(),
            'theta_std':   theta_std.numpy(),
            'target_mean': target_means[i].numpy(),
            'target_std':  target_stds[i].numpy(),
            'N': N, 'Nt': Nt, 'theta_dim': theta_dim,
            'freq_idx': k, 's_k_real': s_values[i].real, 's_k_imag': s_values[i].imag,
        }, os.path.join(ckpt_dir, f'{prefix}_freq{k:03d}.pt'))
        if wandb.run is not None:
            wandb.run.summary[f'best_val/freq_{k:03d}'] = best_vals[i]

    return best_vals, [e for e in best_epochs]


def train_all(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    epochs         = 300,
    batch_size     = 256,
    lr             = 1e-3,
    patience       = 15,
    ckpt_dir       = 'checkpoints/laplace',
    project        = 'convdiff',
    vae            = None,   # si fourni : utilise LaplaceLatentSurrogate
    parallel_freqs = 4,      # nombre de fréquences entraînées en parallèle
):
    """
    Entraîne un surrogate par fréquence de Laplace, en traitant parallel_freqs
    fréquences simultanément sur le même GPU pour un meilleur remplissage du pipeline CUDA.

    Paramètres
    ----------
    dataset        : TransientDataset avec laplace=True et fit() déjà appelé
    train_idx      : liste d'indices train
    val_idx        : liste d'indices val
    parallel_freqs : nombre de fréquences dans chaque chunk GPU (défaut=4)
    """
    Nt_half   = dataset.Nt_half
    N         = dataset.N
    Nt        = dataset.Nt
    theta_dim = dataset.theta_dim
    ns        = len(dataset)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device : {device}   ns={ns}  Nt={Nt} (Nt_half={Nt_half})  N={N}  "
          f"theta_dim={theta_dim}  parallel_freqs={parallel_freqs}")
    os.makedirs(ckpt_dir, exist_ok=True)

    theta_n = (dataset.theta - dataset.theta_mean) / dataset.theta_std  # (ns, theta_dim)

    if vae is not None:
        _dummy   = LaplaceLatentSurrogate(latent_dim=vae.latent_dim, theta_dim=theta_dim)
        run_name = f'LaplaceLatentSurrogate_Nt{Nt}_P{parallel_freqs}'
    else:
        _dummy   = LaplaceSurrogate(s=0j, N=N, theta_dim=theta_dim)
        run_name = f'LaplaceSurrogate_Nt{Nt}_P{parallel_freqs}'
    n_params = sum(p.numel() for p in _dummy.parameters())
    wandb.init(project=project, name=run_name, config=dict(
        Nt=Nt, Nt_half=Nt_half, N=N, ns=ns, theta_dim=theta_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        n_params_surrogate=n_params, use_vae=vae is not None,
        parallel_freqs=parallel_freqs,
    ))

    # theta sur GPU (taille négligeable)
    th_train_gpu = theta_n[train_idx].to(device)
    th_val_gpu   = theta_n[val_idx].to(device)

    # ------------------------------------------------------------------
    # Mode latent : pré-calcul de Z = encoder(U_laplace_norm) pour toutes
    # les fréquences et tous les samples (~150MB RAM, ~30s).
    # L'entraînement utilise ensuite MSE(proj(theta), z) sans décodeur.
    # ------------------------------------------------------------------
    latent_mode = vae is not None
    Z: torch.Tensor | None = None
    if latent_mode:
        print("Pré-calcul des codes latents (encoder)…")
        Z = precompute_latents(
            vae, dataset.U_laplace, dataset.target_mean, dataset.target_std,
            Nt_half, device, batch_size=batch_size,
        )   # (Nt_half, ns, latent_dim) en CPU RAM
        print(f"Z calculé : {tuple(Z.shape)}  ({Z.nbytes / 1e6:.0f} MB)")

    best_vals: list[float] = [float('inf')] * Nt_half
    global_step = 0

    chunk_bar = tqdm(range(0, Nt_half, parallel_freqs), desc="Chunks", position=0, leave=True)
    for k_start in chunk_bar:
        ks       = list(range(k_start, min(k_start + parallel_freqs, Nt_half)))
        s_values = [complex(dataset.s[k]) for k in ks]
        tmeans   = [dataset.target_mean[k] for k in ks]
        tstds    = [dataset.target_std[k]  for k in ks]

        if latent_mode and Z is not None:
            # Pousse les slices z (~K × 7MB) sur GPU — trivial
            targets_train = [Z[k][train_idx].to(device, non_blocking=True) for k in ks]
            targets_val   = [Z[k][val_idx  ].to(device, non_blocking=True) for k in ks]
        else:
            # Charge K slices pixel depuis le mmap (~K × 1GB)
            targets_train, targets_val = [], []
            for k in ks:
                tm = dataset.target_mean[k]
                ts = dataset.target_std[k]
                if isinstance(dataset.U_laplace, np.ndarray):
                    sl = torch.from_numpy(dataset.U_laplace[:, k].copy())
                else:
                    sl = dataset.U_laplace[:, k]
                tn = (sl - tm) / ts
                targets_train.append(tn[train_idx].to(device, non_blocking=True))
                targets_val.append(  tn[val_idx  ].to(device, non_blocking=True))

        chunk_best, _ = train_chunk(
            ks, s_values,
            th_train_gpu, targets_train, th_val_gpu, targets_val,
            dataset.theta_mean, dataset.theta_std,
            tmeans, tstds,
            N=N, Nt=Nt, batch_size=batch_size, epochs=epochs,
            lr=lr, patience=patience, theta_dim=theta_dim,
            device=device, ckpt_dir=ckpt_dir,
            global_step=global_step, vae=vae, latent_mode=latent_mode,
        )
        for i, k in enumerate(ks):
            best_vals[k] = chunk_best[i]
        global_step += epochs
        chunk_bar.set_postfix(freqs=f"{ks[0]}-{ks[-1]}", best=f"{min(chunk_best):.2e}")

        del targets_train, targets_val
        torch.cuda.empty_cache()

    wandb.log({
        'summary/best_val_mean' : float(np.mean(best_vals)),
        'summary/best_val_max'  : float(np.max(best_vals)),
        'summary/best_val_min'  : float(np.min(best_vals)),
    })
    wandb.finish()
    print(f"\nEntraînement terminé — val loss  mean={np.mean(best_vals):.3e}"
          f"  max={np.max(best_vals):.3e}  min={np.min(best_vals):.3e}")
    assemble_model(dataset, ckpt_dir, test_idx, save_dir=os.path.dirname(ckpt_dir), vae=vae)
    return best_vals


def assemble_model(dataset, ckpt_dir: str, test_idx, save_dir: str = 'checkpoints/', vae=None):
    """
    Charge les checkpoints individuels et assemble LaplaceModel ou LaplaceLatentModel.
    Sauvegarde dans <save_dir>/LaplaceModel.pt ou LaplaceLatentModel.pt.
    """
    Nt_half   = dataset.Nt_half
    N         = dataset.N
    Nt        = dataset.Nt
    theta_dim = dataset.theta_dim
    gamma     = float(dataset.s[0].real) if hasattr(dataset, 's') else 0.0

    if vae is not None:
        from models.laplace_ae_surrogate import LaplaceLatentModel
        latent_dim  = vae.latent_dim
        model       = LaplaceLatentModel(N_freq=Nt, N_half=Nt_half, N=N,
                                         theta_dim=theta_dim, latent_dim=latent_dim)
        model.set_ae_decoder(vae)
        prefix      = 'LatentSurrogate'
        model_type  = 'LaplaceLatentModel'
        out_name    = 'LaplaceLatentModel.pt'
        extra       = {'latent_dim': latent_dim}
    else:
        model      = LaplaceModel(N_freq=Nt, N_half=Nt_half, N=N, theta_dim=theta_dim)
        prefix     = 'LaplaceSurrogate'
        model_type = 'LaplaceModel'
        out_name   = 'LaplaceModel.pt'
        extra      = {}

    for k in range(Nt_half):
        path_k = os.path.join(ckpt_dir, f'{prefix}_freq{k:03d}.pt')
        ckpt_k = torch.load(path_k, map_location='cpu', weights_only=False)
        model.surrogates[k].load_state_dict(ckpt_k['model_state'])

    model.set_normalization(dataset.target_mean, dataset.target_std)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)
    torch.save({
        'model_state': model.state_dict(),
        'model_type':  model_type,
        'N_freq':      Nt,
        'N_half':      Nt_half,
        'N':           N,
        'theta_dim':   theta_dim,
        'dt':          dataset.dt,
        'gamma':       gamma,
        'theta_mean':  dataset.theta_mean,
        'theta_std':   dataset.theta_std,
        'test_idx':    np.asarray(test_idx),
        **extra,
    }, out_path)
    print(f"{model_type} assemblé -> {out_path}")


if __name__ == '__main__':
    from transient.dataset import TransientDataset
    from models.laplace_ae_surrogate import LaplaceAE

    data_path    = os.path.join("dataset", "ch4_rotated.npy")
    ae_ckpt     = os.path.join("checkpoints", "LaplaceAE_best.pt")
    ckpt_dir     = os.path.join("checkpoints", "laplace_latent")
    epochs, batch_size, lr, patience = 300, 256, 1e-3, 15
    gamma, rule, seed = 0.0, 'trap', 42
    interp_size  = 128   # doit correspondre au N utilisé pour entraîner le VAE
    dt           = 1.0
    latent_dim   = 64    # doit correspondre au latent_dim du VAE

    # Charger le AE pré-entraîné
    ae_ckpt_data = torch.load(ae_ckpt, map_location='cpu', weights_only=False)
    ae = LaplaceAE(N=interp_size, latent_dim=latent_dim)
    ae.load_state_dict(ae_ckpt_data['model_state'])
    ae.eval()
    print(f"LaplaceAE chargé depuis {ae_ckpt}")

    dataset = TransientDataset(data_path, laplace=True, gamma=gamma, rule=rule,
                               interp_size=interp_size, dt=dt)

    torch.manual_seed(seed)
    idx       = torch.randperm(len(dataset))
    n_train   = int(0.8 * len(dataset))
    n_val     = int(0.1 * len(dataset))
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].numpy()

    dataset.fit(train_idx)
    Nt_half = dataset.Nt_half
    print(f"Dataset : {tuple(dataset.U_laplace.shape)}  (Nt_half={Nt_half}/{dataset.Nt})")

    os.makedirs(ckpt_dir, exist_ok=True)
    train_all(dataset, train_idx, val_idx, test_idx,
              epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
              ckpt_dir=ckpt_dir, vae=ae, parallel_freqs=Nt_half)
