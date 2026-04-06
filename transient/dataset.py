import sys
import hashlib
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
                             __getitem __ retourne (theta_norm, U_laplace_norm)
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
                 doe_path: str | None = None, interp_size: int | None = None,
                 cache_dir: str = '/Data/KAT'):
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
            self._U_raw     = U_raw        # mmap — lu sample par sample dans _to_laplace
            self._data_path = data_path
            self._cache_dir = Path(cache_dir)
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

        if self._U_raw is not None:
            # Gros dataset : écriture dans un memmap sur disque (évite l'OOM)
            stem   = Path(self._data_path).stem #dataset_path sans extension
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            fpath  = self._cache_dir / f"{stem}_laplace_N{N}_g{gamma:.3f}_{rule}.npy"
            spath  = self._cache_dir / f"{stem}_s_Nt{Nt}.npy"

            if fpath.exists() and spath.exists():
                print(f"Cache Laplace trouvé : {fpath}")
                return np.load(str(fpath), mmap_mode='r'), np.load(str(spath))[:Nt_half]

            print(f"Calcul Laplace → {fpath}  ({ns} samples, N={N})")
            U_mmap = np.lib.format.open_memmap(
                str(fpath), mode='w+', dtype=np.float32, shape=(ns, Nt_half, 2, N, N)
            )
            s = None
            for i in tqdm(range(ns), desc="Transformée de Laplace"):
                U_i = torch.from_numpy(self._U_raw[i].copy()).float()  # (Nt, H, W)
                if self.interp_size is not None:
                    U_i = F.interpolate(
                        U_i.unsqueeze(0), size=(N, N),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                C_i     = U_i.numpy().transpose(1, 2, 0).reshape(N * N, Nt)
                M, s, _ = laplace_forward(C_i, dt=self.dt, gamma=gamma, rule=rule)
                M_half  = M[:, :Nt_half]
                stacked = np.stack([M_half.real, M_half.imag], axis=1)  # (N², 2, Nt_half)
                U_mmap[i] = stacked.transpose(2, 1, 0).reshape(Nt_half, 2, N, N)
            assert s is not None
            np.save(str(spath), s)
            return np.load(str(fpath), mmap_mode='r'), s[:Nt_half]

        else:
            # Petit dataset (.npz) : tout en mémoire comme avant
            U_laplace = torch.zeros(ns, Nt_half, 2, N, N)
            s = None
            for i in tqdm(range(ns), desc="Transformée de Laplace", leave=False):
                U_np    = self.U[i].numpy()
                C_i     = U_np.transpose(1, 2, 0).reshape(N * N, Nt)
                M, s, _ = laplace_forward(C_i, dt=self.dt, gamma=gamma, rule=rule)
                M_half  = M[:, :Nt_half]
                stacked = np.stack([M_half.real, M_half.imag], axis=1)
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
            if isinstance(self.U_laplace, np.ndarray):
                # Clé de cache : hash des indices + paramètres du dataset
                idx_hash  = hashlib.md5(np.array(sorted(train_indices)).tobytes()).hexdigest()[:8]
                stem      = Path(self._data_path).stem
                stats_path = self._cache_dir / f"{stem}_stats_N{self.N}_idx{idx_hash}.pt"

                if stats_path.exists():
                    print(f"Cache stats Laplace trouvé : {stats_path}")
                    saved = torch.load(str(stats_path), weights_only=True)
                    self.target_mean = saved['target_mean']
                    self.target_std  = saved['target_std']
                    return

                # Mmap : calcul par fréquence pour éviter de tout charger (~800 MB/freq)
                means, stds = [], []
                for k in tqdm(range(self.Nt_half), desc="Stats Laplace", leave=False):
                    chunk = torch.from_numpy(self.U_laplace[train_indices, k].copy())  # (n_train, 2, N, N)
                    means.append(chunk.mean(dim=(0, 2, 3)))          # (2,)
                    stds.append(chunk.std(dim=(0, 2, 3)) + 1e-8)
                self.target_mean = torch.stack(means).unsqueeze(-1).unsqueeze(-1)  # (Nt_half, 2, 1, 1)
                self.target_std  = torch.stack(stds).unsqueeze(-1).unsqueeze(-1)

                torch.save({'target_mean': self.target_mean,
                            'target_std':  self.target_std}, str(stats_path))
                print(f"Cache stats Laplace sauvegardé : {stats_path}")
            else:
                # Tensor en mémoire : calcul global direct
                target_train = self.U_laplace[train_indices]
                self.target_mean = (target_train.mean(dim=(0, 3, 4))
                                    .unsqueeze(-1).unsqueeze(-1))
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
            if isinstance(self.U_laplace, np.ndarray):
                target = torch.from_numpy(self.U_laplace[idx].copy()).float()
            else:
                target = self.U_laplace[idx]
            target_n = (target - self.target_mean) / self.target_std
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

