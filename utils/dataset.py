
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

        self.U_min  = None
        self.U_max  = None
        self.U_mean = None
        self.U      = None          # rempli après fit()

        self.N = U.shape[-1]
        print(f'Dataset : {len(self.U_raw)} samples  |  N={self.N}')

    def fit(self, train_indices):
        """Calcule U_mean/U_min/U_max sur les seuls indices train, puis normalise tout."""
        train_U     = self.U_raw[train_indices]
        self.U_mean = train_U.mean(dim=0, keepdim=True)   # (1, 1, N, N)
        U_centered  = self.U_raw - self.U_mean
        train_U_c   = U_centered[train_indices]
        self.U_min  = float(train_U_c.min())
        self.U_max  = float(train_U_c.max())
        self.U = 2.0 * (U_centered - self.U_min) / (self.U_max - self.U_min) - 1.0

    def __len__(self):
        return len(self.U_raw)

    def __getitem__(self, idx):
        return self.theta[idx], self.U[idx]

    def denorm_U(self, U_norm: torch.Tensor) -> torch.Tensor:
        return (U_norm + 1.0) / 2.0 * (self.U_max - self.U_min) + self.U_min + self.U_mean