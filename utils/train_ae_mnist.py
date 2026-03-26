"""
train_ae_mnist.py — Entraînement générique pour tout BaseAutoEncoder sur MNIST
==============================================================================
Usage :
    python utils/train_ae_mnist.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import wandb
from torchvision import datasets, transforms

from models.base import BaseAutoEncoder
from models.variationalAutoEncoder import VAE

from tqdm import tqdm


def train_epoch(model: BaseAutoEncoder, loader, optimizer, device):
    model.train()
    totals = {}

    for imgs, _ in loader:
        imgs = imgs.to(device)

        out            = model(imgs)
        total, metrics = model.loss(imgs, *out)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals['loss'] = totals.get('loss', 0.) + total.item()
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.) + v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model: BaseAutoEncoder, loader, device):
    model.eval()
    totals = {}

    for imgs, _ in loader:
        imgs = imgs.to(device)

        out            = model(imgs)
        total, metrics = model.loss(imgs, *out)

        totals['loss'] = totals.get('loss', 0.) + total.item()
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.) + v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def log_reconstructions(model: BaseAutoEncoder, loader, device, n_show=4):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:n_show].to(device)

    imgs_hat = model(imgs)[0]

    imgs     = imgs.cpu()
    imgs_hat = imgs_hat.cpu()
    err      = (imgs - imgs_hat).abs()

    def to_img(arr):
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        return (arr * 255).astype(np.uint8)

    images = []
    for i in range(n_show):
        row = np.concatenate([
            to_img(imgs[i, 0].numpy()),
            to_img(imgs_hat[i, 0].numpy()),
            to_img(err[i, 0].numpy()),
        ], axis=1)
        images.append(wandb.Image(
            row, caption=f'sample {i} | truth / pred / |err|'
        ))

    return images


def train(
    model        : BaseAutoEncoder,
    data_root    : str   = 'dataset/mnist',
    epochs       : int   = 100,
    batch_size   : int   = 128,
    lr           : float = 1e-3,
    patience     : int   = 20,
    seed         : int   = 42,
    project      : str   = 'mnist-ae',
    ckpt_dir     : str   = 'checkpoints',
    log_img_every: int   = 10,
    beta_warmup  : int   = 20,
    img_size     : int   = 32,
):
    model_name = type(model).__name__
    run_name   = f'{model_name}_mnist_{time.strftime("%Y%m%d-%H%M%S")}'
    model_hparams = {k: v for k, v in vars(model).items()
                     if isinstance(v, (int, float, bool, str))}
    wandb.init(
        project = project,
        name    = run_name,
        config  = dict(
            dataset       = 'MNIST',
            img_size      = img_size,
            epochs        = epochs,
            batch_size    = batch_size,
            lr            = lr,
            patience      = patience,
            seed          = seed,
            beta_warmup   = beta_warmup,
            model         = model_name,
            **model_hparams,
        ),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    train_full = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    n_val   = int(0.1 * len(train_full))
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'Train: {n_train}  Val: {n_val}  Test: {len(test_set)}')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres entraînables : {n_params:,}')
    wandb.config.update({'n_params': n_params, 'device': str(device)})
    wandb.watch(model, log='gradients', log_freq=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    ckpt_dir_ = Path(ckpt_dir)
    ckpt_dir_.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir_ / f'{model_name}_mnist_best.pt'

    best_val  = float('inf')
    patience_ = 0

    beta_target = getattr(model, 'beta', None)

    for epoch in tqdm(range(1, epochs + 1)):
        t0 = time.perf_counter()

        if beta_target is not None:
            model.beta = beta_target * min(1.0, epoch / beta_warmup)

        tr = train_epoch(model, train_loader, optimizer, device)
        va = val_epoch(model, val_loader, device)

        val_metric = va.get('recon', va['loss'])
        scheduler.step(val_metric)

        epoch_time = time.perf_counter() - t0
        lr_now     = optimizer.param_groups[0]['lr']

        log = {
            'epoch'        : epoch,
            'lr'           : lr_now,
            'epoch_time_s' : epoch_time,
            **({'beta': model.beta} if beta_target is not None else {}),
            **{f'train/{k}': v for k, v in tr.items()},
            **{f'val/{k}':   v for k, v in va.items()},
        }

        if epoch % log_img_every == 0:
            log['reconstructions'] = log_reconstructions(model, val_loader, device)

        wandb.log(log)

        if val_metric < best_val:
            best_val  = val_metric
            patience_ = 0
            torch.save({
                'model_type' : model_name,
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_loss'   : best_val,
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

    all_imgs, all_imgs_hat = [], []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            imgs_hat = model(imgs)[0]
            all_imgs.append(imgs.cpu())
            all_imgs_hat.append(imgs_hat.cpu())

    imgs_all     = torch.cat(all_imgs)
    imgs_hat_all = torch.cat(all_imgs_hat)
    ss_res = ((imgs_all - imgs_hat_all) ** 2).sum().item()
    ss_tot = ((imgs_all - imgs_all.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / ss_tot

    print(f'\nTest  loss={te["loss"]:.4f}  R²={r2:.4f}')
    wandb.log({**{f'test/{k}': v for k, v in te.items()}, 'test/r2': r2})

    wandb.finish()


if __name__ == '__main__':
    model = VAE(N=32, 
            latent_dim=64,
            beta=1.0,
            free_bits=0.1
            )
    train(
        model,
        data_root     = 'dataset/mnist',
        epochs        = 100,
        batch_size    = 128,
        lr            = 1e-3,
        patience      = 20,
        seed          = 42,
        project       = 'mnist-ae',
        ckpt_dir      = 'checkpoints',
        log_img_every = 10,
        img_size      = 32,
    )
