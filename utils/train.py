"""
train.py — Entraînement CVAE avec Weights & Biases
====================================================
Usage :
    pip install wandb torch
    wandb login
    python train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wandb

from models.CVAE import CVAE
from utils.dataset import ConvDiffDataset

from tqdm import tqdm


CONFIG = {
    # Données
    'dataset'       : 'dataset/dataset.npz',

    # Modèle
    'latent_dim'    : 16,
    'beta'          : 0.5,
    'free_bits'     : 0.5,     # KL min par dimension latente (anti-collapse)
    'beta_warmup'   : 80,      # époques pour monter beta de 0 → beta

    # Entraînement
    'epochs'        : 500,
    'batch_size'    : 32,
    'lr'            : 1e-3,
    'patience'      : 40,
    'seed'          : 42,

    # Logging
    'project'       : 'cvae-convdiff',
    'run_name'      : None,    # None → wandb génère un nom automatique
    'ckpt_dir'      : 'checkpoints',
    'log_img_every' : 10,      # envoyer des images toutes les N époques
}




def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    metrics = dict(loss=0., recon=0., kl=0., grad=0.)

    for theta, U in loader:
        theta, U   = theta.to(device), U.to(device)
        model.beta = beta

        U_hat, mu, logvar     = model(U, theta)
        loss, recon, kl, grad = model.elbo(U, U_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics['loss']  += loss.item()
        metrics['recon'] += recon.item()
        metrics['kl']    += kl.item()
        metrics['grad']  += grad.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def val_epoch(model, loader, device, beta):
    model.eval()
    metrics = dict(loss=0., recon=0., kl=0., grad=0.)

    for theta, U in loader:
        theta, U   = theta.to(device), U.to(device)
        model.beta = beta

        U_hat, mu, logvar     = model(U, theta)
        loss, recon, kl, grad = model.elbo(U, U_hat, mu, logvar)

        metrics['loss']  += loss.item()
        metrics['recon'] += recon.item()
        metrics['kl']    += kl.item()
        metrics['grad']  += grad.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}



@torch.no_grad()
def log_reconstructions(model, loader, dataset, device, n_show=4):
    model.eval()
    theta, U = next(iter(loader))
    theta, U = theta[:n_show].to(device), U[:n_show].to(device)

    U_hat, _, _ = model(U, theta)

    U_phys     = dataset.denorm_U(U.cpu())
    U_hat_phys = dataset.denorm_U(U_hat.cpu())
    err        = (U_phys - U_hat_phys).abs()

    def to_img(arr):
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        return (arr * 255).astype(np.uint8)

    images = []
    for i in range(n_show):
        row = np.concatenate([
            to_img(U_phys[i, 0].numpy()),
            to_img(U_hat_phys[i, 0].numpy()),
            to_img(err[i, 0].numpy()),
        ], axis=1)
        images.append(wandb.Image(
            row, caption=f'sample {i} | truth / pred / |err|'
        ))

    return images



def train():
    wandb.init(
        project = CONFIG['project'],
        name    = CONFIG['run_name'],
        config  = CONFIG,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Dataset + splits
    dataset = ConvDiffDataset(CONFIG['dataset'])
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )

    # Normalisation basée uniquement sur les stats train (pas de leakage)
    dataset.fit(train_set.indices)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2)

    print(f'Train: {n_train}  Val: {n_val}  Test: {n_test}')

    # Modèle
    model = CVAE(
        N          = dataset.N,
        theta_dim  = 6,
        latent_dim = CONFIG['latent_dim'],
        beta       = CONFIG['beta'],
        free_bits  = CONFIG['free_bits'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres entraînables : {n_params:,}')
    wandb.config.update({'n_params': n_params, 'device': str(device)})
    wandb.watch(model, log='gradients', log_freq=50)

    # Optimiseur + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    # Checkpoint
    ckpt_dir  = Path(CONFIG['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'cvae_best.pt'

    best_val  = float('inf')
    patience_ = 0

    for epoch in tqdm(range(1, CONFIG['epochs'] + 1)):
        t0   = time.perf_counter()
        beta = CONFIG['beta'] * min(1.0, epoch / CONFIG['beta_warmup'])

        tr = train_epoch(model, train_loader, optimizer, device, beta)
        va = val_epoch(model, val_loader, device, beta)
        scheduler.step(va['recon'])

        epoch_time = time.perf_counter() - t0
        lr_now     = optimizer.param_groups[0]['lr']

        log = {
            'epoch'        : epoch,
            'beta'         : beta,
            'lr'           : lr_now,
            'epoch_time_s' : epoch_time,
            'train/loss'   : tr['loss'],
            'train/recon'  : tr['recon'],
            'train/kl'     : tr['kl'],
            'train/grad'   : tr['grad'],
            'val/loss'     : va['loss'],
            'val/recon'    : va['recon'],
            'val/kl'       : va['kl'],
            'val/grad'     : va['grad'],
        }

        if epoch % CONFIG['log_img_every'] == 0:
            log['reconstructions'] = log_reconstructions(
                model, val_loader, dataset, device
            )

        wandb.log(log)

        if va['recon'] < best_val:
            best_val  = va['recon']
            patience_ = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_loss'   : best_val,
                'config'     : CONFIG,
                'U_min'      : dataset.U_min,
                'U_max'      : dataset.U_max,
                'theta_mean' : dataset.theta_mean,
                'theta_std'  : dataset.theta_std,
            }, best_path)
            wandb.save(str(best_path))
        else:
            patience_ += 1

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    te = val_epoch(model, test_loader, device, beta=CONFIG['beta'])

    all_U, all_U_hat = [], []
    with torch.no_grad():
        for theta, U in test_loader:
            theta, U = theta.to(device), U.to(device)
            U_hat, _, _ = model(U, theta)
            all_U.append(dataset.denorm_U(U.cpu()))
            all_U_hat.append(dataset.denorm_U(U_hat.cpu()))

    U_all     = torch.cat(all_U)
    U_hat_all = torch.cat(all_U_hat)
    ss_res = ((U_all - U_hat_all) ** 2).sum().item()
    ss_tot = ((U_all - U_all.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / ss_tot

    print(f'\nTest  loss={te["loss"]:.4f}  recon={te["recon"]:.4f}  R²={r2:.4f}')
    wandb.log({
        'test/loss'  : te['loss'],
        'test/recon' : te['recon'],
        'test/kl'    : te['kl'],
        'test/grad'  : te['grad'],
        'test/r2'    : r2,
    })

    wandb.finish()


if __name__ == '__main__':
    train()