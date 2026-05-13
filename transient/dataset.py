import sys
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from utils.laplace import laplace_forward_tik

class TransientDataset(Dataset):
    """
    Dataset pour les champs transitoires (ns, Nt, N, N).

    Modes
    -----
    laplace=False (défaut) : __getitem__ retourne (theta_norm, U)
    laplace=True           : applique la transformée de Laplace au chargement,
                             __getitem__ retourne (theta_norm, U_laplace_norm)
                             où U_laplace_norm a la forme (K, 2, N, N).

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
                 s_list=None, rule: str = 'trap', dt: float = 1.0,
                 doe_path: str | None = None, interp_size: int | None = None,
                 cache_dir: str = '/Data/KAT', ns_max: int | None = None):
        if data_path.endswith('.npy'):
            # Données boris : ch4_rotated.npy + doe_rotated.npy
            # Gardé en mmap — pas de matérialisation en RAM
            U_raw = np.load(data_path, mmap_mode='r')                 # (ns, Nt, H, W)
            if doe_path is None:
                doe_path = str(Path(data_path).parent / 'doe_rotated.npy')
            doe = np.load(doe_path)                                    # structured (ns,)
            theta_np = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)
            if ns_max is not None:
                U_raw    = U_raw[:ns_max]
                theta_np = theta_np[:ns_max]
            self.theta = torch.tensor(theta_np, dtype=torch.float32)  # (ns, 3)
            self.dt    = dt
            ns, Nt, H, W = U_raw.shape
            self.ns, self.Nt = ns, Nt
            self.N = interp_size if interp_size is not None else H
            self._U_raw     = U_raw        # mmap — lu sample par sample dans _to_laplace
            self.U = U_raw
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
        self.interp_size = interp_size

        self.laplace = laplace
        if laplace:
            if s_list is None:
                raise ValueError("s_list est requis quand laplace=True")
            s_list = np.asarray(s_list, dtype=np.complex128)
            self._s_hash = hashlib.md5(
                np.round(s_list, 2).astype(np.complex64).tobytes()
            ).hexdigest()[:8]
            self.U_laplace, self.s = self._to_laplace(s_list, rule)
            self.K = len(s_list)

        print(f"TransientDataset : ns={self.ns}  Nt={self.Nt}  N={self.N}  "
              f"theta_dim={self.theta_dim}  laplace={laplace}")

    # ------------------------------------------------------------------
    # Transformée de Laplace
    # ------------------------------------------------------------------

    def _to_laplace(self, s_list: np.ndarray, rule: str):
        ns, Nt, N = self.ns, self.Nt, self.N
        K = len(s_list)
        s_torch = torch.tensor(s_list, dtype=torch.complex128)

        if self._U_raw is not None:
            # Gros dataset : écriture dans un memmap sur disque (évite l'OOM)
            stem = Path(self._data_path).stem
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            fpath = self._cache_dir / f"{stem}_laplace_N{N}_s{self._s_hash}_{rule}.npy"
            spath = self._cache_dir / f"{stem}_slist_s{self._s_hash}.npy"

            if fpath.exists() and spath.exists():
                print(f"Cache Laplace trouvé : {fpath}")
                return np.load(str(fpath), mmap_mode='r'), np.load(str(spath))

            print(f"Calcul Laplace → {fpath}  ({ns} samples, N={N}, K={K})")
            U_mmap = np.lib.format.open_memmap(
                str(fpath), mode='w+', dtype=np.float32, shape=(ns, K, 2, N, N)
            )
            for i in tqdm(range(ns), desc="Transformée de Laplace"):
                U_i = torch.from_numpy(self._U_raw[i].copy()).float()  # (Nt, H, W)
                if self.interp_size is not None:
                    U_i = F.interpolate(
                        U_i.unsqueeze(0), size=(N, N),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                C_i   = U_i.double().reshape(Nt, N * N).T            # (N², Nt)
                U_hat = laplace_forward_tik(C_i, s_torch, dt=self.dt, rule=rule)  # (N², K)
                Re = U_hat.real.float().T.reshape(K, N, N)           # (K, N, N)
                Im = U_hat.imag.float().T.reshape(K, N, N)
                U_mmap[i] = torch.stack([Re, Im], dim=1).numpy()     # (K, 2, N, N)
            np.save(str(spath), s_list)
            return np.load(str(fpath), mmap_mode='r'), s_list

        else:
            # Petit dataset (.npz) : tout en mémoire
            U_laplace = torch.zeros(ns, K, 2, N, N, dtype=torch.float32)
            for i in tqdm(range(ns), desc="Transformée de Laplace", leave=False):
                U_np  = self.U[i].numpy()                            # (Nt, N, N)
                C_i   = torch.from_numpy(U_np).double().reshape(Nt, N * N).T  # (N², Nt)
                U_hat = laplace_forward_tik(C_i, s_torch, dt=self.dt, rule=rule)  # (N², K)
                Re = U_hat.real.float().T.reshape(K, N, N)
                Im = U_hat.imag.float().T.reshape(K, N, N)
                U_laplace[i] = torch.stack([Re, Im], dim=1)          # (K, 2, N, N)
            return U_laplace, s_list

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def fit(self, train_indices):
        """
        Calcule les stats de normalisation sur le train set.

        - theta     : z-score global  → theta_mean (theta_dim,), theta_std (theta_dim,)
        - U_laplace : z-score par fréquence → target_mean (K, 2, N, N),
                                               target_std  (K, 2, N, N)
        """
        self.theta_mean = self.theta[train_indices].mean(0)        # (theta_dim,)
        self.theta_std  = self.theta[train_indices].std(0) + 1e-8

        if self.laplace:
            if isinstance(self.U_laplace, np.ndarray):
                idx_hash   = hashlib.md5(np.array(sorted(train_indices)).tobytes()).hexdigest()[:8]
                stem       = Path(self._data_path).stem
                stats_path = self._cache_dir / f"{stem}_stats_N{self.N}_s{self._s_hash}_idx{idx_hash}_vnorm.pt"

                if stats_path.exists():
                    print(f"Cache stats trouvé : {stats_path}")
                    saved = torch.load(str(stats_path), weights_only=True)
                    self.target_mean = saved['target_mean']
                    self.target_std  = saved['target_std']
                    return

                # target_mean : moyenne de U_laplace sur le train (par linéarité = F(V_mean))
                means = []
                for k in tqdm(range(self.K), desc="Laplace mean", leave=False):
                    chunk = torch.from_numpy(self.U_laplace[train_indices, k].copy())
                    means.append(chunk.mean(dim=0))          # (2, N, N)
                self.target_mean = torch.stack(means)        # (K, 2, N, N)

                # target_std : std de V par pixel sur (n_train × Nt), broadcasté sur (K, 2, N, N)
                sum_   = torch.zeros(self.N, self.N, dtype=torch.float64)
                sum_sq = torch.zeros(self.N, self.N, dtype=torch.float64)
                count  = 0
                for i in tqdm(train_indices, desc="V std", leave=False):
                    V_i = torch.from_numpy(self._U_raw[i].copy()).float()  # (Nt, H, W)
                    if self.interp_size is not None:
                        V_i = F.interpolate(
                            V_i.unsqueeze(0), size=(self.N, self.N),
                            mode='bilinear', align_corners=False,
                        ).squeeze(0)
                    V_i = V_i.double()
                    sum_   += V_i.sum(0)
                    sum_sq += (V_i ** 2).sum(0)
                    count  += V_i.shape[0]
                V_std_px = ((sum_sq / count - (sum_ / count) ** 2).clamp(min=0).sqrt() + 1e-8).float()
                self.target_std = V_std_px[None, None].expand(self.K, 2, self.N, self.N).clone()

                torch.save({'target_mean': self.target_mean,
                            'target_std':  self.target_std}, str(stats_path))
                print(f"Cache stats sauvegardé : {stats_path}")
            else:
                # Tensor en mémoire : calcul direct depuis U (champ temporel)
                target_train     = self.U_laplace[train_indices]          # (n_train, K, 2, N, N)
                self.target_mean = target_train.mean(dim=0)               # (K, 2, N, N)
                V_train          = self.U[train_indices]                  # (n_train, Nt, N, N)
                n_train, Nt_     = V_train.shape[:2]
                V_std_px = (V_train.reshape(n_train * Nt_, self.N, self.N)
                            .double().std(dim=0) + 1e-8).float()          # (N, N)
                self.target_std = V_std_px[None, None].expand(self.K, 2, self.N, self.N).clone()

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
            return theta_n, target_n                               # (K, 2, N, N)
        return theta_n, self.U[idx]                                # (Nt, N, N)

    def denorm_target(self, target_norm: torch.Tensor,
                      k: int | None = None) -> torch.Tensor:
        """Dénormalise la sortie du surrogate. Si k fourni, fréquence k uniquement."""
        if self.laplace:
            if k is not None:
                return target_norm * self.target_std[k] + self.target_mean[k]
            return target_norm * self.target_std + self.target_mean
        return target_norm

