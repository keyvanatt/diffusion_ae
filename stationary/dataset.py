import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import Dataset


class ConvDiffDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)

        U = data['U'].astype(np.float32)
        self.U_raw = torch.from_numpy(U[:, None, :, :])  # non normalisé
        self.theta = torch.from_numpy(data['theta_norm'].astype(np.float32))

        self.theta_mean = torch.from_numpy(data['theta_mean'].astype(np.float32))
        self.theta_std  = torch.from_numpy(data['theta_std'].astype(np.float32))

        self.U_mean : torch.Tensor
        self.U_std  : torch.Tensor
        self.U      : torch.Tensor  # rempli après fit()

        self.N = U.shape[-1]
        print(f'Dataset : {len(self.U_raw)} samples  |  N={self.N}')

    def fit(self, train_indices):
        """Calcule la grille moyenne et l'écart-type global sur le train, puis standardise."""
        train_U     = self.U_raw[train_indices]
        self.U_mean = train_U.mean(dim=0, keepdim=True)  # (1, 1, N, N)
        U_centered  = self.U_raw - self.U_mean
        self.U_std  = U_centered[train_indices].std() + 1e-8
        self.U      = U_centered / self.U_std

    def __len__(self):
        return len(self.U_raw)

    def __getitem__(self, idx):
        return self.theta[idx], self.U[idx]

    def denorm_U(self, U_norm: torch.Tensor) -> torch.Tensor:
        return U_norm * self.U_std + self.U_mean
