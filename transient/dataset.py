import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from utils.laplace import laplace_forward

class TransientDataset(Dataset):
    """
    Dataset pour les champs transitoires (ns, Nt, N, N).

    Modes
    -----
    laplace=False (défaut) : __getitem__ retourne (theta_norm, U)
    laplace=True           : applique la transformée de Laplace au chargement,
                             __getitem__ retourne (theta_norm, U_laplace_norm)
                             où U_laplace_norm a la forme (Nt_half, 2, N, N).

    Usage
    -----
    # Données boris (ch4_rotated)
    dataset = TransientDataset('dataset/ch4_rotated.npy', dt=1.0, laplace=True)
    # Ancien format
    dataset = TransientDataset('dataset/dataset_transient.npz', laplace=True)
    dataset.fit(train_indices)   # calcule les stats de normalisation
    theta_n, target_n = dataset[i]
    """

    def __init__(self, data_path: str, laplace: bool = False,
                 gamma: float = 0.0, rule: str = 'trap', dt: float = 1.0,
                 doe_path: str | None = None, interp_size: int | None = None):
        if data_path.endswith('.npy'):
            # Données boris : ch4_rotated.npy + doe_rotated.npy
            # Gardé en mmap — pas de matérialisation en RAM
            U_raw = np.load(data_path, mmap_mode='r')                 # (ns, Nt, H, W)
            if doe_path is None:
                doe_path = str(Path(data_path).parent / 'doe_rotated.npy')
            doe = np.load(doe_path)                                    # structured (ns,)
            theta_np = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)
            self.theta = torch.tensor(theta_np, dtype=torch.float32)  # (ns, 3)
            self.dt    = dt
            ns, Nt, H, W = U_raw.shape
            self.ns, self.Nt = ns, Nt
            self.N = interp_size if interp_size is not None else H
            self._U_raw = U_raw     # mmap — lu sample par sample dans _to_laplace
        else:
            # Ancien format : dataset_transient.npz
            data   = np.load(data_path)
            self.U = torch.tensor(data['U'],     dtype=torch.float32)
            self.theta = torch.tensor(data['theta'], dtype=torch.float32)
            dt_raw     = data['dt']
            self.dt    = float(dt_raw[0]) if hasattr(dt_raw, '__len__') else float(dt_raw)
            self.ns, self.Nt, self.N, _ = self.U.shape
            self._U_raw: np.ndarray | None = None

        self.theta_dim = self.theta.shape[1]
        self.Nt_half   = self.Nt // 2 + 1
        self.interp_size = interp_size

        self.laplace = laplace
        if laplace:
            self.U_laplace, self.s = self._to_laplace(gamma, rule)
            # U_laplace : (ns, Nt_half, 2, N, N)
            # s         : (Nt_half,) complex — fréquences s_k = gamma + i*omega_k

        print(f"TransientDataset : ns={self.ns}  Nt={self.Nt}  N={self.N}  "
              f"theta_dim={self.theta_dim}  laplace={laplace}")

    # ------------------------------------------------------------------
    # Transformée de Laplace
    # ------------------------------------------------------------------

    def _to_laplace(self, gamma: float, rule: str):

        ns, Nt, N = self.ns, self.Nt, self.N
        Nt_half   = self.Nt_half
        U_laplace = torch.zeros(ns, Nt_half, 2, N, N)
        s = None
        for i in tqdm(range(ns), desc="Transformée de Laplace", leave=False):
            if self._U_raw is not None:
                # Lecture mmap : copie d'un seul sample pour éviter la matérialisation totale
                U_i = torch.from_numpy(self._U_raw[i].copy()).float()  # (Nt, H, W)
                if self.interp_size is not None:
                    U_i = F.interpolate(
                        U_i.unsqueeze(0),                              # (1, Nt, H, W) → 4D
                        size=(self.interp_size, self.interp_size),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)                                        # (Nt, N, N)
                U_np = U_i.numpy()
            else:
                U_np = self.U[i].numpy()                               # (Nt, N, N)
            C_i    = U_np.transpose(1, 2, 0).reshape(N * N, Nt)       # (N², Nt)
            M, s, _ = laplace_forward(C_i, dt=self.dt, gamma=gamma, rule=rule)
            M_half  = M[:, :Nt_half]                                   # (N², Nt_half)
            stacked = np.stack([M_half.real, M_half.imag], axis=1)    # (N², 2, Nt_half)
            U_laplace[i] = torch.from_numpy(stacked).permute(2, 1, 0).reshape(Nt_half, 2, N, N)
        return U_laplace, s[:Nt_half]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def fit(self, train_indices):
        """
        Calcule les stats de normalisation sur le train set.

        - theta     : z-score global  → theta_mean (theta_dim,), theta_std (theta_dim,)
        - U_laplace : z-score par fréquence → target_mean (Nt_half, 2, 1, 1),
                                               target_std  (Nt_half, 2, 1, 1)
        """
        self.theta_mean = self.theta[train_indices].mean(0)        # (theta_dim,)
        self.theta_std  = self.theta[train_indices].std(0) + 1e-8

        if self.laplace:
            target_train = self.U_laplace[train_indices]           # (n_train, Nt_half, 2, N, N)
            # moyenne/écart-type sur (batch, pixels) par fréquence et par composante (Re/Im)
            self.target_mean = (target_train.mean(dim=(0, 3, 4))   # (Nt_half, 2)
                                .unsqueeze(-1).unsqueeze(-1))       # (Nt_half, 2, 1, 1)
            self.target_std  = (target_train.std(dim=(0, 3, 4))
                                .unsqueeze(-1).unsqueeze(-1) + 1e-8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        theta_n = (self.theta[idx] - self.theta_mean) / self.theta_std
        if self.laplace:
            target_n = (self.U_laplace[idx] - self.target_mean) / self.target_std
            return theta_n, target_n                               # (Nt_half, 2, N, N)
        return theta_n, self.U[idx]                                # (Nt, N, N)

    def denorm_target(self, target_norm: torch.Tensor,
                      k: int | None = None) -> torch.Tensor:
        """Dénormalise la sortie du surrogate. Si k fourni, fréquence k uniquement."""
        if self.laplace:
            if k is not None:
                return target_norm * self.target_std[k] + self.target_mean[k]
            return target_norm * self.target_std + self.target_mean
        return target_norm

