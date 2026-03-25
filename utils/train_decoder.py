"""
train_decoder.py — Entraînement du Decoder direct (theta → U)
==============================================================
Usage :
    python utils/train_decoder.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import wandb

from models.Decoder import DirectDecoder
from utils.dataset import ConvDiffDataset

from tqdm import tqdm


CONFIG = {
    # Données
    'dataset'       : 'dataset/dataset.npz',

    # Modèle
    'lambda_grad'   : 1.0,    # poids du terme gradient dans la loss

    # Entraînement
    'epochs'        : 500,
    'batch_size'    : 32,
    'lr'            : 1e-3,
    'patience'      : 40,
    'seed'          : 42,

    # Logging
    'project'       : 'decoder-convdiff',
    'run_name'      : None,
    'ckpt_dir'      : 'checkpoints',
    'log_img_every' : 10,
}


def train_epoch(model, loader, optimizer, device, lambda_grad):
    model.train()
    metrics = dict(loss=0., recon=0., grad=0.)

    for theta, U in loader:
        theta, U = theta.to(device), U.to(device)

        U_hat                   = model(theta)
        loss, recon, grad       = model.loss(U_hat, U, lambda_grad)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics['loss']  += loss.item()
        metrics['recon'] += recon.item()
        metrics['grad']  += grad.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def val_epoch(model, loader, device, lambda_grad):
    model.eval()
    metrics = dict(loss=0., recon=0., grad=0.)

    for theta, U in loader:
        theta, U = theta.to(device), U.to(device)

        U_hat             = model(theta)
        loss, recon, grad = model.loss(U_hat, U, lambda_grad)

        metrics['loss']  += loss.item()
        metrics['recon'] += recon.item()
        metrics['grad']  += grad.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def log_reconstructions(model, loader, dataset, device, n_show=4):
    model.eval()
    theta, U = next(iter(loader))
    theta, U = theta[:n_show].to(device), U[:n_show].to(device)

    U_hat = model(theta)

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

    dataset.fit(train_set.indices)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2)

    print(f'Train: {n_train}  Val: {n_val}  Test: {n_test}')

    model = DirectDecoder(N=dataset.N, theta_dim=4).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres entraînables : {n_params:,}')
    wandb.config.update({'n_params': n_params, 'device': str(device)})
    wandb.watch(model, log='gradients', log_freq=50)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    ckpt_dir  = Path(CONFIG['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / 'decoder_best.pt'

    best_val  = float('inf')
    patience_ = 0

    for epoch in tqdm(range(1, CONFIG['epochs'] + 1)):
        t0 = time.perf_counter()

        tr = train_epoch(model, train_loader, optimizer, device, CONFIG['lambda_grad'])
        va = val_epoch(model, val_loader, device, CONFIG['lambda_grad'])
        scheduler.step(va['recon'])

        epoch_time = time.perf_counter() - t0
        lr_now     = optimizer.param_groups[0]['lr']

        log = {
            'epoch'        : epoch,
            'lr'           : lr_now,
            'epoch_time_s' : epoch_time,
            'train/loss'   : tr['loss'],
            'train/recon'  : tr['recon'],
            'train/grad'   : tr['grad'],
            'val/loss'     : va['loss'],
            'val/recon'    : va['recon'],
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
                'model_type' : 'decoder',
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
            if patience_ >= CONFIG['patience']:
                print(f'Early stopping à l\'époque {epoch}')
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    te = val_epoch(model, test_loader, device, CONFIG['lambda_grad'])

    all_U, all_U_hat = [], []
    with torch.no_grad():
        for theta, U in test_loader:
            theta, U = theta.to(device), U.to(device)
            U_hat    = model(theta)
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
        'test/grad'  : te['grad'],
        'test/r2'    : r2,
    })

    wandb.finish()


if __name__ == '__main__':
    train()
