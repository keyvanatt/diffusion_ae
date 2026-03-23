
import torch
import numpy as np
from torch.utils.data import Dataset


class ConvDiffDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)

        U = data['U'].astype(np.float32)
        self.U     = torch.from_numpy(U[:, None, :, :])
        self.theta = torch.from_numpy(data['theta_norm'].astype(np.float32))

        self.theta_mean = torch.from_numpy(data['theta_mean'].astype(np.float32))
        self.theta_std  = torch.from_numpy(data['theta_std'].astype(np.float32))

        self.U_min = float(self.U.min())
        self.U_max = float(self.U.max())
        self.U = 2.0 * (self.U - self.U_min) / (self.U_max - self.U_min) - 1.0

        self.N = U.shape[-1]
        print(f'Dataset : {len(self.U)} samples  |  N={self.N}')

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.theta[idx], self.U[idx]

    def denorm_U(self, U_norm: torch.Tensor) -> torch.Tensor:
        return (U_norm + 1.0) / 2.0 * (self.U_max - self.U_min) + self.U_min