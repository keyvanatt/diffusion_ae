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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from models.laplace_surrogate import LaplaceSurrogate, LaplaceModel
from models.laplace_ae_surrogate import LaplaceAE, LaplaceLatentSurrogate, LaplaceLatentModel


def train_one(
    k,
    s_k,
    train_loader,
    val_loader,
    theta_mean,
    theta_std,
    N,
    Nt,
    epochs       = 300,
    lr           = 1e-3,
    patience     = 30,
    theta_dim    = 4,
    device       = None,
    ckpt_dir     = 'checkpoints',
    global_step  = 0,
    ae           = None,        # si fourni → LaplaceLatentSurrogate avec décodeur gelé
    freq_ratio   = 0.0,         # k / (K - 1), pour le conditionnement FiLM
    latent_mode  = False,       # True → targets sont des z pré-calculés, loss MSE en espace latent
    val_loader_u = None,        # loader (theta_n, u_norm) pour L2rel sur U en mode latent
    hidden_dim   = 256,
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

    if ae is not None:
        model = LaplaceLatentSurrogate(
            latent_dim=ae.latent_dim, theta_dim=theta_dim, freq_ratio=freq_ratio,
            hidden_dim=hidden_dim,
        ).to(device)
        ae.decoder.to(device).requires_grad_(False)
        model.set_decoder(ae.decoder)
        params_to_train = model.proj.parameters()  # décodeur gelé
    else:
        model           = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim, freq_ratio=freq_ratio).to(device)
        params_to_train = model.parameters()

    optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6,
    )
    use_amp = device.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val    = float('inf')
    best_state  = None
    best_epoch  = 0
    no_improve  = 0
    prefix      = 'LatentSurrogate' if ae is not None else 'LaplaceSurrogate'
    ckpt_path   = os.path.join(ckpt_dir, f'{prefix}_freq{k:03d}.pt')

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"  freq {k:03d}  s={s_k.imag:.3g}j",
        leave=False,
        position=1,
    )
    epoch = 0
    latent_mode = ae is not None
    for epoch in epoch_bar:
        t0 = time.perf_counter()

        # --- Train ---
        model.train()
        train_loss  = torch.zeros(1, device=device)
        train_l2rel = torch.zeros(1, device=device)
        for th, target in train_loader:
            th, target = th.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if latent_mode:
                    target_hat = model.proj(th)  # type: ignore[operator]
                    loss  = F.mse_loss(target_hat, target)
                else:
                    target_hat = model(th)
                    loss  = model.loss(target_hat, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss  += loss.detach()
            train_l2rel += ((target_hat.detach() - target).flatten(1).norm(dim=1)
                            / (target.flatten(1).norm(dim=1) + 1e-8)).mean()
        train_loss  = train_loss.item()  / len(train_loader)
        train_l2rel = train_l2rel.item() / len(train_loader)

        # --- Val ---
        model.eval()
        val_loss  = torch.zeros(1, device=device)
        val_l2rel = torch.zeros(1, device=device)
        with torch.no_grad():
            for th, target in val_loader:
                th, target = th.to(device), target.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if latent_mode:
                        target_hat    = model.proj(th)  # type: ignore[operator]
                        loss_val = F.mse_loss(target_hat, target)
                    else:
                        target_hat    = model(th)
                        loss_val = model.loss(target_hat, target)
                val_loss  += loss_val
                val_l2rel += ((target_hat - target).flatten(1).norm(dim=1)
                              / (target.flatten(1).norm(dim=1) + 1e-8)).mean()
        val_loss  = val_loss.item()  / len(val_loader)
        val_l2rel = val_l2rel.item() / len(val_loader)

        # --- Val L2rel sur U (mode latent uniquement) ---
        val_l2rel_U = None
        if latent_mode and ae is not None and val_loader_u is not None:
            decoder = ae.decoder.to(device)
            acc = torch.zeros(1, device=device)
            with torch.no_grad():
                for th, u_norm in val_loader_u:
                    th, u_norm = th.to(device), u_norm.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        z_hat = model.proj(th)       # type: ignore[operator]
                        u_hat = decoder(z_hat, freq_ratio)
                    acc += ((u_hat - u_norm).flatten(1).norm(dim=1)
                            / (u_norm.flatten(1).norm(dim=1) + 1e-8)).mean()
            val_l2rel_U = acc.item() / len(val_loader_u)

        scheduler.step(val_loss)
        epoch_bar.set_postfix(
            train=f"{train_loss:.3e}", val=f"{val_loss:.3e}",
            l2U=f"{val_l2rel_U:.2%}" if val_l2rel_U is not None else f"{val_l2rel:.2%}",
        )
        log_dict = {
            'train/loss'  : train_loss,
            'train/l2rel' : train_l2rel,
            'val/loss'    : val_loss,
            'val/l2rel'   : val_l2rel,
            'lr'          : optimizer.param_groups[0]['lr'],
            'epoch_time_s': time.perf_counter() - t0,
        }
        if val_l2rel_U is not None:
            log_dict['val/l2rel_U'] = val_l2rel_U
        wandb.log(log_dict, step=global_step + epoch)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {key: v.cpu().clone() for key, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                epoch_bar.set_postfix(
                    train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", stop=f"ep{epoch}",
                )
                break

    assert best_state is not None, f"Aucune epoch n'a amélioré la val loss pour freq {k}"
    torch.save({
        'model_state': best_state,
        'epoch':       best_epoch,
        'val_loss':    best_val,
        'theta_mean':  theta_mean.numpy(),
        'theta_std':   theta_std.numpy(),
        'N': N, 'Nt': Nt, 'theta_dim': theta_dim,
        'freq_idx': k, 's_k_real': s_k.real, 's_k_imag': s_k.imag,
        'hidden_dim': hidden_dim,
    }, ckpt_path)

    if wandb.run is not None:
        wandb.run.summary[f'best_val/freq_{k:03d}'] = best_val
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return best_val, epoch


def precompute_latents(ae, U_laplace, K, device, batch_size=256):
    """
    Encode tous les champs Laplace (déjà normalisés) avec ae.encoder.
    Retourne Z de shape (K, ns, latent_dim) en CPU.

    Z[k, i] = encoder(U_laplace[k, i], freq_ratio=k/(K-1))
    """
    ns      = U_laplace.shape[0]
    encoder = ae.encoder.to(device).eval()
    Z       = torch.zeros(K, ns, ae.latent_dim)

    with torch.no_grad():
        for k in tqdm(range(K), desc="Pré-calcul latents", leave=False):
            freq_ratio = k / max(K - 1, 1)
            for start in range(0, ns, batch_size):
                end = min(start + batch_size, ns)
                if isinstance(U_laplace, np.ndarray):
                    u = torch.from_numpy(U_laplace[start:end, k].copy()).float().to(device)
                else:
                    u = U_laplace[start:end, k].to(device)
                Z[k, start:end] = encoder(u, freq_ratio).cpu()
    return Z  # (K, ns, latent_dim)


def train_all(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    epochs     = 300,
    batch_size = 256,
    lr         = 1e-3,
    patience   = 15,
    ckpt_dir    = 'checkpoints/laplace',
    project     = 'convdiff',
    ae          = None,
    k_max       = None,   # fréquences k > k_max → pas d'entraînement, prédiction = moyenne
    hidden_dim  = 256,
    freq_L      = 8,
    alpha_t     = 0.0,
    lam         = 1e-6,
):
    """
    Entraîne un surrogate par fréquence de Laplace, séquentiellement.

    Si ae est fourni, utilise LaplaceLatentSurrogate en mode latent :
    les codes z sont pré-calculés une fois, puis chaque surrogate apprend
    proj(theta) → z via MSE (sans décodeur pendant l'entraînement).
    La L2rel sur U est calculée séparément en validation via ae.decoder.

    Si k_max est fourni, les fréquences k > k_max ne sont pas entraînées :
    le modèle assemblé prédit la moyenne (target_mean[k]) pour ces fréquences.
    """
    K           = dataset.K
    N           = dataset.N
    Nt          = dataset.Nt
    theta_dim   = dataset.theta_dim
    latent_mode = ae is not None
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device : {device}   ns={len(dataset)}  Nt={Nt} (K={K})  N={N}  theta_dim={theta_dim}")
    os.makedirs(ckpt_dir, exist_ok=True)

    theta_n = (dataset.theta - dataset.theta_mean) / dataset.theta_std  # (ns, theta_dim)

    if latent_mode:
        dummy    = LaplaceLatentSurrogate(latent_dim=ae.latent_dim, theta_dim=theta_dim, hidden_dim=hidden_dim)
        run_name = f'LaplaceLatentSurrogate_Nt{Nt}'
    else:
        dummy    = LaplaceSurrogate(s=0j, N=N, theta_dim=theta_dim)
        run_name = f'LaplaceSurrogate_Nt{Nt}'
    n_params = sum(p.numel() for p in dummy.parameters())

    wandb.init(project=project, name=run_name, config=dict(
        Nt=Nt, K=K, N=N, ns=len(dataset), theta_dim=theta_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        n_params_surrogate=n_params, use_ae=latent_mode,
        hidden_dim=hidden_dim, k_max=k_max,
    ))

    n_train_freqs = (min(k_max, K - 1) + 1) if k_max is not None else K

    Z: torch.Tensor | None = None
    if latent_mode:
        print("Pré-calcul des codes latents (encoder)…")
        Z = precompute_latents(
            ae, dataset.U_laplace,
            n_train_freqs, device, batch_size=batch_size,
        )
        print(f"Z calculé : {tuple(Z.shape)}  ({Z.nbytes / 1e6:.0f} MB)")

    best_vals   = []
    global_step = 0

    for k in tqdm(range(n_train_freqs), desc='Fréquences', position=0, leave=True):

        s_k        = complex(dataset.s[k])
        freq_ratio = k / max(K - 1, 1)

        # Targets U — déjà normalisés (Laplace de U normalisé)
        if isinstance(dataset.U_laplace, np.ndarray):
            sl = torch.from_numpy(dataset.U_laplace[:, k].copy()).float()
        else:
            sl = dataset.U_laplace[:, k].float()
        u_norm = sl  # (ns, 2, N, N), already normalized

        if latent_mode:
            assert Z is not None
            targets = Z[k]  # (ns, latent_dim)
            val_loader_u = DataLoader(
                TensorDataset(theta_n[val_idx], u_norm[val_idx]),
                batch_size=batch_size, shuffle=False, num_workers=0,
            )
        else:
            targets      = u_norm
            val_loader_u = None

        train_loader = DataLoader(
            TensorDataset(theta_n[train_idx], targets[train_idx]),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(theta_n[val_idx], targets[val_idx]),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        best_val, n_epochs = train_one(
            k, s_k, train_loader, val_loader,
            theta_mean   = dataset.theta_mean,
            theta_std    = dataset.theta_std,
            N=N, Nt=Nt, epochs=epochs, lr=lr, patience=patience,
            theta_dim=theta_dim, device=device, ckpt_dir=ckpt_dir,
            global_step=global_step, ae=ae, freq_ratio=freq_ratio, val_loader_u=val_loader_u,
            hidden_dim=hidden_dim,
        )
        best_vals.append(best_val)
        global_step += n_epochs
        del targets, u_norm
        torch.cuda.empty_cache()

    wandb.log({
        'summary/best_val_mean': float(np.mean(best_vals)),
        'summary/best_val_max' : float(np.max(best_vals)),
        'summary/best_val_min' : float(np.min(best_vals)),
    })
    values = [[k, float(v)] for k, v in enumerate(best_vals)]
    tab = wandb.Table(data=values, columns=['frequency_k', 'best_val_loss'])
    wandb.log({'best_vals': wandb.plot.bar(tab, 'frequency_k', 'best_val_loss', title='Best Val Loss par Fréquence')})
    wandb.finish()
    print(f"\nEntraînement terminé — val loss  mean={np.mean(best_vals):.3e}"
          f"  max={np.max(best_vals):.3e}  min={np.min(best_vals):.3e}")

    assemble_model(dataset, ckpt_dir, test_idx, save_dir=os.path.dirname(ckpt_dir), ae=ae, k_max=k_max, hidden_dim=hidden_dim, freq_L=freq_L, alpha_t=alpha_t, lam=lam)
    return best_vals


def assemble_model(dataset, ckpt_dir: str, test_idx, save_dir: str = 'checkpoints/', ae=None, k_max=None, hidden_dim=256, freq_L=8, alpha_t=0.0, lam=1e-6):
    """
    Charge les checkpoints individuels et assemble LaplaceModel ou LaplaceLatentModel.
    Sauvegarde dans <save_dir>/LaplaceModel.pt ou LaplaceLatentModel.pt.
    """
    K         = dataset.K
    N         = dataset.N
    Nt        = dataset.Nt
    theta_dim = dataset.theta_dim
    if ae is not None:
        # Lire hidden_dim depuis le checkpoint pour éviter les mismatches
        _ckpt0 = torch.load(os.path.join(ckpt_dir, 'LatentSurrogate_freq000.pt'), map_location='cpu', weights_only=False)
        hidden_dim = _ckpt0.get('hidden_dim', hidden_dim)
        model      = LaplaceLatentModel(K=K, Nt=Nt, N=N,
                                        theta_dim=theta_dim, latent_dim=ae.latent_dim, k_max=k_max, hidden_dim=hidden_dim, freq_L=freq_L)
        model.set_ae_decoder(ae)
        prefix     = 'LatentSurrogate'
        model_type = 'LaplaceLatentModel'
        out_name   = 'LaplaceLatentModel.pt'
        extra      = {'latent_dim': ae.latent_dim}
    else:
        model      = LaplaceModel(K=K, Nt=Nt, N=N, theta_dim=theta_dim, k_max=k_max)
        prefix     = 'LaplaceSurrogate'
        model_type = 'LaplaceModel'
        out_name   = 'LaplaceModel.pt'
        extra      = {}

    n_load = (min(k_max, K - 1) + 1) if k_max is not None else K
    for k in range(n_load):
        path_k = os.path.join(ckpt_dir, f'{prefix}_freq{k:03d}.pt')
        ckpt_k = torch.load(path_k, map_location='cpu', weights_only=False)
        model.surrogates[k].load_state_dict(ckpt_k['model_state'])
    if k_max is not None and k_max < K - 1:
        print(f"  k_max={k_max} : fréquences {k_max+1}..{K-1} → prédiction = moyenne")

    model.set_normalization(dataset.U_mean, dataset.U_std)
    model.set_s_list(dataset.s)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)
    torch.save({
        'model_state': model.state_dict(),
        'model_type':  model_type,
        'K':           K,
        'Nt':          Nt,
        'N':           N,
        'theta_dim':   theta_dim,
        'dt':          dataset.dt,
        'theta_mean':  dataset.theta_mean,
        'theta_std':   dataset.theta_std,
        'test_idx':    np.asarray(test_idx),
        'k_max':       k_max,
        'hidden_dim':  hidden_dim,
        'freq_L':      freq_L,
        'alpha_t':     alpha_t,
        'lam':         lam,
        **extra,
    }, out_path)
    print(f"{model_type} assemblé → {out_path}")


if __name__ == '__main__':
    from transient.dataset import TransientDataset

    data_path   = os.path.join("dataset", "ch4_rotated.npy")
    ae_ckpt     = os.path.join("checkpoints", "LaplaceAE_best.pt")
    ckpt_dir    = os.path.join("checkpoints", "laplace_latent")
    epochs, batch_size, lr, patience = 1000, 512, 1e-3, 50
    rule        = 'trap'
    seed        = 42
    interp_size = 128  # doit correspondre au N utilisé pour entraîner le AE
    dt          = 1.0
    latent_dim  = 64   # doit correspondre au latent_dim du AE
    k_max       = 20   # None = toutes les fréquences
    hidden_dim  = 512

    # Points s : k_max+1 premières fréquences FFT (gamma=0)
    # Nt=150 pour ch4_rotated, rfftfreq donne 76 valeurs
    Nt_data  = 150
    K_total  = k_max + 1
    s_list   = (1j * 2 * np.pi * np.fft.rfftfreq(Nt_data, d=dt))[:K_total]

    ae_ckpt_data = torch.load(ae_ckpt, map_location='cpu', weights_only=False)
    ae = LaplaceAE(N=interp_size, latent_dim=latent_dim)
    ae.load_state_dict(ae_ckpt_data['model_state'])
    ae.eval()
    print(f"LaplaceAE chargé depuis {ae_ckpt}")

    dataset = TransientDataset(data_path, laplace=True, s_list=s_list, rule=rule,
                               interp_size=interp_size, dt=dt)

    _split   = np.load('dataset/split.npz')
    test_idx = _split['test_idx']
    non_test = [i for i in range(len(dataset)) if i not in set(test_idx.tolist())]
    torch.manual_seed(seed)
    perm     = torch.randperm(len(non_test))
    n_train  = int(0.8 * len(non_test))
    train_idx = [non_test[i] for i in perm[:n_train].tolist()]
    val_idx   = [non_test[i] for i in perm[n_train:].tolist()]

    dataset.fit(train_idx)
    print(f"Dataset : {tuple(dataset.U_laplace.shape)}  (K={dataset.K}  Nt={dataset.Nt})")

    os.makedirs(ckpt_dir, exist_ok=True)
    train_all(dataset, train_idx, val_idx, test_idx,
              epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
              ckpt_dir=ckpt_dir, ae=ae, k_max=k_max, hidden_dim=hidden_dim)
