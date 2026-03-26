"""
train_decoder.py — Entraînement générique pour tout BaseDecoder
=============================================================
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

from models.base import BaseDecoder
from models.direct_decoder import DirectDecoder, DirectDecoderDenseOut
from models.variationalAutoEncoder import VAE, IndirectDecoder
from models.AE_SVD import AutoencoderSVD, IndirectDecoderSVD, compute_fixed_svd_basis
from utils.dataset import ConvDiffDataset

from tqdm import tqdm


def train_epoch(model: BaseDecoder, loader, optimizer, device):
    model.train()
    metrics = dict(loss=0., recon=0., grad=0.)

    for theta, U in loader:
        theta, U = theta.to(device), U.to(device)

        U_hat             = model(theta)
        loss, recon, grad = model.loss(U_hat, U)

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
def val_epoch(model: BaseDecoder, loader, device):
    model.eval()
    metrics = dict(loss=0., recon=0., grad=0.)

    for theta, U in loader:
        theta, U = theta.to(device), U.to(device)

        U_hat             = model(theta)
        loss, recon, grad = model.loss(U_hat, U)

        metrics['loss']  += loss.item()
        metrics['recon'] += recon.item()
        metrics['grad']  += grad.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def log_reconstructions(model: BaseDecoder, loader, dataset, device, n_show=4):
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


def train(
    model       : BaseDecoder,
    dataset_path: str   = 'dataset/dataset.npz',
    epochs      : int   = 500,
    batch_size  : int   = 32,
    lr          : float = 1e-3,
    patience    : int   = 40,
    seed        : int   = 42,
    project     : str   = 'convdiff',
    ckpt_dir    : str   = 'checkpoints',
    prefix      : str   = "",
    log_img_every: int  = 10,
):
    model_name = type(model).__name__
    run_name= f'{prefix}_{model_name}_{time.strftime("%Y%m%d-%H%M%S")}'
    wandb.init(
        project = project,
        name    = run_name,
        config  = dict(
            dataset_path  = dataset_path,
            epochs        = epochs,
            batch_size    = batch_size,
            lr            = lr,
            patience      = patience,
            seed          = seed,
            model         = model_name,
        ),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    model = model.to(device)

    # Dataset + splits
    dataset = ConvDiffDataset(dataset_path)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    dataset.fit(train_set.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    print(f'Train: {n_train}  Val: {n_val}  Test: {n_test}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres entraînables : {n_params:,}')
    wandb.config.update({'n_params': n_params, 'device': str(device)})
    wandb.watch(model, log='gradients', log_freq=50)

    if model.__class__.__name__ == "IndirectDecoderSVD":
        U_train = dataset.U[train_set.indices]                  # (N_train, 1, N, N)
        U_train = dataset.denorm_U(U_train.cpu()).numpy()       # type: ignore
        model.compute_and_set_fixed_basis(U_train)              # type: ignore
        optimizer = torch.optim.AdamW([
            {'params': model.theta_proj.parameters(), 'lr': lr},
            {'params': model.decoder.parameters(),    'lr': lr * 0.01},
        ], weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    ckpt_dir_ = Path(ckpt_dir)
    ckpt_dir_.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir_ / f'{prefix}_{model_name}_best.pt'

    best_val  = float('inf')
    patience_ = 0

    for epoch in tqdm(range(1, epochs + 1)):
        t0 = time.perf_counter()

        tr = train_epoch(model, train_loader, optimizer, device)
        va = val_epoch(model, val_loader, device)
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

        if epoch % log_img_every == 0:
            log['reconstructions'] = log_reconstructions(
                model, val_loader, dataset, device
            )

        wandb.log(log)

        if va['recon'] < best_val:
            best_val  = va['recon']
            patience_ = 0
            torch.save({
                'model_type' : model_name,
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_loss'   : best_val,
                'U_mean'     : dataset.U_mean,
                'U_std'      : dataset.U_std,
                'theta_mean' : dataset.theta_mean,
                'theta_std'  : dataset.theta_std,
            }, best_path)
            wandb.save(str(best_path))
        else:
            patience_ += 1
            if patience_ >= patience:
                print(f'Early stopping à l\'époque {epoch}')
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    te = val_epoch(model, test_loader, device)

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
    dataset = ConvDiffDataset('dataset/dataset.npz')
    trained_AE = AutoencoderSVD(N=64, latent_dim=32, kmax=3)
    trained_AE.load_state_dict(torch.load('checkpoints/AutoencoderSVD_best.pt')['model_state'])
    model = IndirectDecoderSVD(
        N=64,
        kmax=3,
        theta_dim=4,
        latent_dim=32,
        trained_autoencoder=trained_AE,
    )
    train(
        model,
        dataset_path  = 'dataset/dataset.npz',
        epochs        = 500,
        batch_size    = 128,
        lr            = 1e-3,
        patience      = 100,
        seed          = 42,
        project       = 'convdiff',
        ckpt_dir      = 'checkpoints',
        prefix = "finetune",
        log_img_every = 50,
    )
