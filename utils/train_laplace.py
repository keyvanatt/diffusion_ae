import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
import wandb

from utils.laplace import laplace_forward
from models.laplace_surrogate import LaplaceSurrogate
from utils.animate import animate_comparaison

def to_laplace(U_train, dt, gamma=0.0, rule='trap'):
    """
    Convertit un dataset de la forme (theta, U) en (theta, U_hat) où U_hat est la représentation de U dans l'espace de Laplace.
    Entrée :
        - U_train : torch.Tensor (ns, Nt, N, N)
        - dt : float, pas de temps entre les échantillons temporels de U
        - gamma : float, amortissement pour la transformée de Laplace (0 = DFT pure)
        - rule : 'rect' ou 'trap', règle de quadrature temporelle utilisée dans laplace_forward
    Sortie :
        - U_laplace : torch.Tensor (ns, Nt, 2, N, N)
        - s : ndarray (Nt,) complex – fréquences s_k = gamma + i*omega_k
    """
    ns, Nt, N, _ = U_train.shape
    Nt_half = Nt // 2 + 1                                                  # symétrie conjuguée : seules ces fréquences sont indépendantes
    U_laplace = torch.zeros(ns, Nt_half, 2, N, N)
    s = None
    for i in tqdm(range(ns), desc="Transformée de Laplace", leave=False):
        U_i = U_train[i].numpy()                                           # (Nt, N, N)
        C_i = U_i.transpose(1, 2, 0).reshape(N * N, Nt)                   # (N², Nt)
        M, s, _ = laplace_forward(C_i, dt=dt, gamma=gamma, rule=rule)     # (N², Nt) complex
        M_half  = M[:, :Nt_half]                                           # (N², Nt_half)
        stacked = np.stack([M_half.real, M_half.imag], axis=1)            # (N², 2, Nt_half)
        U_laplace[i] = torch.from_numpy(stacked).permute(2, 1, 0).reshape(Nt_half, 2, N, N)
    return U_laplace, s[:Nt_half], Nt                                      # Nt original nécessaire pour l'inversion


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
):
    """
    Entraîne un LaplaceSurrogate pour la fréquence k.

    Retour
    ------
    best_val : float – meilleure val loss atteinte
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    best_val  = float('inf')
    patience_ = 0
    ckpt_path = os.path.join(ckpt_dir, f'LaplaceSurrogate_freq{k:03d}.pt')

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"  freq {k:03d}  s={s_k.imag:.3g}j",
        leave=False,
        position=1,
    )
    for epoch in epoch_bar:
        # --- Train ---
        model.train()
        train_loss = 0.
        for th, u in train_loader:
            th, u = th.to(device), u.to(device)
            loss = model.loss(model(th), u)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Val ---
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for th, u in val_loader:
                th, u = th.to(device), u.to(device)
                val_loss += model.loss(model(th), u).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        epoch_bar.set_postfix(train=f"{train_loss:.3e}", val=f"{val_loss:.3e}")

        wandb.log({
            'freq': k,
            'train/loss': train_loss,
            'val/loss':   val_loss,
            'lr':         optimizer.param_groups[0]['lr'],
        }, step=global_step + epoch)

        if val_loss < best_val:
            best_val  = val_loss
            patience_ = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'theta_mean':  theta_mean.numpy(),
                'theta_std':   theta_std.numpy(),
                'target_mean': target_mean.numpy(),
                'target_std':  target_std.numpy(),
                'N': N, 'Nt': Nt, 'theta_dim': theta_dim,
                'freq_idx': k, 's_k_real': s_k.real, 's_k_imag': s_k.imag,
            }, ckpt_path)
        else:
            patience_ += 1
            if patience_ >= patience:
                epoch_bar.set_postfix(
                    train=f"{train_loss:.3e}", val=f"{val_loss:.3e}", stop=f"ep{epoch}"
                )
                break

    wandb.log({f'best_val/freq_{k:03d}': best_val})
    return best_val, epoch


def train_all(
    U_laplace,
    theta,
    s,
    N,
    Nt,
    epochs     = 300,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 30,
    seed       = 42,
    theta_dim  = 4,
    ckpt_dir   = 'checkpoints/laplace',
    project    = 'convdiff',
):
    """
    Entraîne un LaplaceSurrogate par fréquence de Laplace.

    Entrées
    -------
    U_laplace : torch.Tensor (ns, Nt, 2, N, N)
    theta     : ndarray (ns, theta_dim)
    s         : ndarray (Nt,) complex – fréquences de Laplace
    N         : int – résolution spatiale
    """
    ns, Nt_half, _, _, _ = U_laplace.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}   ns={ns}  Nt={Nt} (Nt_half={Nt_half})  N={N}  theta_dim={theta_dim}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Split train / val / test (identique pour toutes les fréquences)
    torch.manual_seed(seed)
    idx     = torch.randperm(ns)
    n_train = int(0.8 * ns)
    n_val   = int(0.1 * ns)
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()

    # Normalisation theta (z-score sur le train)
    theta_t    = torch.tensor(theta, dtype=torch.float32)
    theta_mean = theta_t[train_idx].mean(0)
    theta_std  = theta_t[train_idx].std(0) + 1e-8
    theta_n    = (theta_t - theta_mean) / theta_std        # (ns, theta_dim)

    n_params = sum(p.numel() for p in LaplaceSurrogate(s=0j, N=N, theta_dim=theta_dim).parameters())
    wandb.init(project=project, name=f'LaplaceSurrogate_Nt{Nt}', config=dict(
        Nt=Nt, Nt_half=Nt_half, N=N, ns=ns, theta_dim=theta_dim, epochs=epochs,
        batch_size=batch_size, lr=lr, patience=patience, n_params=n_params,
    ))

    best_vals  = []
    global_step = 0

    freq_bar = tqdm(range(Nt_half), desc="Fréquences", position=0, leave=True)
    for k in freq_bar:
        s_k      = complex(s[k])
        target_k = U_laplace[:, k]                               # (ns, 2, N, N)

        # Normalisation z-score par fréquence (stats sur le train uniquement)
        target_mean = target_k[train_idx].mean(dim=(0, 2, 3), keepdim=True).squeeze(0)  # (2, 1, 1)
        target_std  = target_k[train_idx].std(dim=(0, 2, 3),  keepdim=True).squeeze(0) + 1e-8
        target_k_n  = (target_k - target_mean) / target_std

        dataset      = TensorDataset(theta_n, target_k_n)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size)

        best_val, n_epochs = train_one(
            k, s_k, train_loader, val_loader, theta_mean, theta_std,
            target_mean=target_mean, target_std=target_std,
            N=N, Nt=Nt, epochs=epochs, lr=lr, patience=patience,
            theta_dim=theta_dim, device=device, ckpt_dir=ckpt_dir,
            global_step=global_step,
        )

        best_vals.append(best_val)
        global_step += n_epochs
        freq_bar.set_postfix(freq=k, best_val=f"{best_val:.3e}", s=f"{s_k:.3g}")

    wandb.log({'best_val/mean': np.mean(best_vals)})
    wandb.finish()
    print(f"\nEntraînement terminé. Val loss moyenne : {np.mean(best_vals):.3e}")
    return best_vals, idx[n_train + n_val:].numpy()



def predict(theta, ckpt_dir='checkpoints/laplace', dt=1.0, gamma=0.0, rule='trap', device=None):
    """
    Reconstruit les champs U(t) pour un batch de theta en passant par tous les modèles entraînés.

    Paramètres
    ----------
    theta    : ndarray ou Tensor (B, theta_dim)
    ckpt_dir : répertoire contenant les LaplaceSurrogate_freq{k:03d}.pt
    dt, gamma, rule : doivent correspondre aux paramètres utilisés dans to_laplace

    Retour
    ------
    U_pred : Tensor (B, Nt, N, N)
    """
    from utils.laplace import laplace_inverse

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_files = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith('LaplaceSurrogate_freq') and f.endswith('.pt')
    ])
    Nt = len(ckpt_files)
    assert Nt > 0, f"Aucun checkpoint trouvé dans {ckpt_dir}"

    theta_t = torch.tensor(np.asarray(theta), dtype=torch.float32)
    B = theta_t.shape[0]

    # Lire N et theta_dim depuis le premier checkpoint
    ckpt0     = torch.load(os.path.join(ckpt_dir, ckpt_files[0]), map_location='cpu', weights_only=False)
    N         = ckpt0['N']
    theta_dim = ckpt0['theta_dim']

    Nt_half = len(ckpt_files)
    Nt      = ckpt0.get("Nt",100)                   # taille temporelle originale

    # M_half (B, N², Nt_half) : on remplit fréquence par fréquence
    M_half = np.zeros((B, N * N, Nt_half), dtype=np.complex64)

    freq_bar = tqdm(enumerate(ckpt_files), total=Nt_half, desc="Prédiction Laplace", leave=True)
    for k, fname in freq_bar:
        ckpt = torch.load(os.path.join(ckpt_dir, fname), map_location=device, weights_only=False)

        # Normalisation theta
        theta_mean = torch.tensor(ckpt['theta_mean'], device=device)
        theta_std  = torch.tensor(ckpt['theta_std'],  device=device)
        theta_n    = (theta_t.to(device) - theta_mean) / theta_std

        # Chargement et inférence du modèle
        s_k   = complex(ckpt['s_k_real'], ckpt['s_k_imag'])
        model = LaplaceSurrogate(s=s_k, N=N, theta_dim=theta_dim).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        with torch.no_grad():
            pred = model(theta_n)                                            # (B, 2, N, N)

        # Dénormalisation
        target_mean = torch.tensor(ckpt['target_mean'], device=device)      # (2, 1, 1)
        target_std  = torch.tensor(ckpt['target_std'],  device=device)
        pred = pred * target_std + target_mean                               # (B, 2, N, N)

        pred_np = pred.cpu().numpy()                                         # (B, 2, N, N)
        M_half[:, :, k] = (pred_np[:, 0] + 1j * pred_np[:, 1]).reshape(B, N * N)

    # Reconstruction du spectre complet par symétrie conjuguée : M[Nt-k] = conj(M[k])
    M_full = np.zeros((B, N * N, Nt), dtype=np.complex64)
    M_full[:, :, :Nt_half] = M_half
    M_full[:, :, Nt_half:] = np.conj(M_half[:, :, Nt - Nt_half:0:-1])

    # Reconstruction temporelle par Laplace inverse
    U_pred = np.zeros((B, Nt, N, N), dtype=np.float32)
    for b in tqdm(range(B), desc="Inverse Laplace", leave=True):
        C_b, _ = laplace_inverse(M_full[b], dt, Nt, rule=rule, gamma=gamma)  # (N², Nt)
        U_pred[b] = C_b.reshape(N, N, Nt).transpose(2, 0, 1).astype(np.float32)

    return torch.from_numpy(U_pred)


def evaluate(U, theta, test_idx, ckpt_dir='checkpoints/laplace', dt=1.0, gamma=0.0,
             rule='trap', device=None):
    """
    Calcule l'erreur L2 relative sur le test set.

    Paramètres
    ----------
    U        : Tensor (ns, Nt, N, N) – champs complets
    theta    : ndarray (ns, theta_dim)
    test_idx : ndarray – indices du test set (retourné par train_all)

    Retour
    ------
    l2rel : ndarray (n_test,) – erreur L2 relative par simulation
    """

    print(f"Test set : {len(test_idx)} simulations")

    U_true = U[test_idx].numpy()                           # (n_test, Nt, N, N)
    theta_test = theta[test_idx]                           # (n_test, theta_dim)

    U_pred = predict(theta_test, ckpt_dir=ckpt_dir, dt=dt,
                     gamma=gamma, rule=rule, device=device).numpy()   # (n_test, Nt, N, N)
    
    #Erreur MSE globale
    mse = np.mean((U_pred - U_true) ** 2)
    print(f"MSE global sur le test set : {mse:.4e}")

    # Erreur L2 relative par simulation
    diff  = U_pred - U_true                                            # (n_test, Nt, N, N)
    norms_err  = np.linalg.norm(diff.reshape(len(test_idx), -1), axis=1)
    norms_true = np.linalg.norm(U_true.reshape(len(test_idx), -1), axis=1) + 1e-12
    l2rel = norms_err / norms_true

    print(f"L2 relative error — mean : {l2rel.mean():.4e}  std : {l2rel.std():.4e}")
    print(f"                    min  : {l2rel.min():.4e}  max : {l2rel.max():.4e}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(l2rel, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('L2 Relative Error')
    plt.ylabel('Frequency')
    plt.title('L2 Relative Error Distribution (Test Set)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('plots', 'l2_relative_error_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved: plots/l2_relative_error_hist.png")

    return l2rel


if __name__ == '__main__':
    data_path = os.path.join("dataset", "dataset_transient.npz")

    EVALUATE = True

    epochs     = 300
    batch_size = 64
    lr         = 1e-3
    patience   = 30
    gamma      = 0.0
    rule       = 'trap'

    data  = np.load(data_path)
    U     = torch.tensor(data['U'], dtype=torch.float32)   # (ns, Nt, N, N)
    theta = data['theta']                                   # (ns, theta_dim)
    dt    = float(data['dt'][0])
    ns, Nt, N, _ = U.shape
    print(f"Dataset : {tuple(U.shape)}  dt={dt:.4f}")

    if not EVALUATE:


        print()
        U_laplace, s, Nt = to_laplace(U, dt=dt, gamma=gamma, rule=rule)
        print(f"U_laplace : {tuple(U_laplace.shape)}  (Nt_half={U_laplace.shape[1]}/{Nt})  s dtype={s.dtype}")



        _, test_idx = train_all(
            U_laplace  = U_laplace,
            theta      = theta,
            s          = s,
            N          = N,
            Nt         = Nt,
            epochs     = epochs,
            batch_size = batch_size,
            lr         = lr,
            patience   = patience,
        )
    else:
        torch.manual_seed(42)
        idx = torch.randperm(ns)
        test_idx = idx[int(0.8 * ns) + int(0.1 * ns):].numpy()  # indices du test set (derniers 10% des données)

    evaluate(U, theta, test_idx, dt=dt, gamma=gamma, rule=rule)

    n_exemples = min(5, len(test_idx))
    for i in range(n_exemples):
        si = test_idx[i]
        print(f"\nExemple {i+1}/{n_exemples} — index {si}")
        U_pred = predict(theta[si:si+1], dt=dt, gamma=gamma, rule=rule).squeeze(0).numpy()  # (Nt, N, N)
        U_true = U[si].numpy()                                                            # (Nt, N, N)
        mse_ex = np.mean((U_pred - U_true) ** 2)
        print(f"  MSE : {mse_ex:.4e}")
        l2_rel_ex = np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true) + 1e-12)
        print(f"  L2 relative error : {l2_rel_ex:.4e}")
        animate_comparaison(
            U_true, U_pred,
            output_path=os.path.join('plots', f'laplace_surrogate_example_{i+1}.gif'),
            title_fn=lambda t: f"Exemple {i+1} — t={t:.3f}s",
        )
        print(f"  Animation sauvegardée : plots/laplace_surrogate_example_{i+1}.gif")

