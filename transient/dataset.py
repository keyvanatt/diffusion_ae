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
            self._s_list = s_list
            self._rule   = rule
            self.K       = len(s_list)
            self.s       = s_list
            self.U_laplace = None   # calculé dans fit() après normalisation

        print(f"TransientDataset : ns={self.ns}  Nt={self.Nt}  N={self.N}  "
              f"theta_dim={self.theta_dim}  laplace={laplace}")

    # ------------------------------------------------------------------
    # Transformée de Laplace
    # ------------------------------------------------------------------

    def _to_laplace(self, s_list: np.ndarray, rule: str, idx_hash: str):
        """
        Calcule la transformée de Laplace sur U normalisé (U - U_mean) / U_std.
        Doit être appelé après _u_stats (self.U_mean et self.U_std déjà calculés).
        Le cache inclut idx_hash car la normalisation dépend du train set.
        """
        ns, Nt, N = self.ns, self.Nt, self.N
        K = len(s_list)
        s_torch = torch.tensor(s_list, dtype=torch.complex128)
        U_mean = self.U_mean.double()   # (N, N)
        U_std  = self.U_std.double()    # (N, N)

        if self._U_raw is not None:
            stem = Path(self._data_path).stem
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            fpath = self._cache_dir / f"{stem}_laplace_N{N}_s{self._s_hash}_{rule}_idx{idx_hash}.npy"
            spath = self._cache_dir / f"{stem}_slist_s{self._s_hash}.npy"

            if fpath.exists() and spath.exists():
                print(f"Cache Laplace trouvé : {fpath}")
                return np.load(str(fpath), mmap_mode='r')

            print(f"Calcul Laplace (normalisé) → {fpath}  ({ns} samples, N={N}, K={K})")
            U_mmap = np.lib.format.open_memmap(
                str(fpath), mode='w+', dtype=np.float32, shape=(ns, K, 2, N, N)
            )
            for i in tqdm(range(ns), desc="Transformée de Laplace"):
                U_i = torch.from_numpy(self._U_raw[i].copy()).float()
                if self.interp_size is not None:
                    U_i = F.interpolate(
                        U_i.unsqueeze(0), size=(N, N),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                U_i  = (U_i.double() - U_mean) / U_std              # normalisation
                C_i  = U_i.reshape(Nt, N * N).T                     # (N², Nt)
                U_hat = laplace_forward_tik(C_i, s_torch, dt=self.dt, rule=rule)
                Re = U_hat.real.float().T.reshape(K, N, N)
                Im = U_hat.imag.float().T.reshape(K, N, N)
                U_mmap[i] = torch.stack([Re, Im], dim=1).numpy()
            np.save(str(spath), s_list)
            return np.load(str(fpath), mmap_mode='r')

        else:
            U_laplace = torch.zeros(ns, K, 2, N, N, dtype=torch.float32)
            for i in tqdm(range(ns), desc="Transformée de Laplace", leave=False):
                U_i  = (self.U[i].double() - U_mean) / U_std        # normalisation
                C_i  = U_i.reshape(Nt, N * N).T
                U_hat = laplace_forward_tik(C_i, s_torch, dt=self.dt, rule=rule)
                Re = U_hat.real.float().T.reshape(K, N, N)
                Im = U_hat.imag.float().T.reshape(K, N, N)
                U_laplace[i] = torch.stack([Re, Im], dim=1)
            return U_laplace

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _u_stats(self, train_indices, idx_hash: str | None = None) -> tuple:
        """
        Calcule la moyenne et l'écart-type pixel-par-pixel de U sur le train set.
        Retourne (U_mean, U_std) de shape (N, N) float32.
        Pour le format mmap, le résultat est mis en cache sur disque (clé idx_hash).
        """
        if self._U_raw is not None:
            if idx_hash is not None:
                stem  = Path(self._data_path).stem
                fpath = self._cache_dir / f"{stem}_ustats_N{self.N}_idx{idx_hash}.npz"
                if fpath.exists():
                    print(f"Cache U stats trouvé : {fpath}")
                    d = np.load(str(fpath))
                    return torch.from_numpy(d['mean']), torch.from_numpy(d['std'])

            sum_   = torch.zeros(self.N, self.N, dtype=torch.float64)
            sum_sq = torch.zeros(self.N, self.N, dtype=torch.float64)
            count  = 0
            for i in tqdm(train_indices, desc="U stats", leave=False):
                V_i = torch.from_numpy(self._U_raw[i].copy()).float()
                if self.interp_size is not None:
                    V_i = F.interpolate(
                        V_i.unsqueeze(0), size=(self.N, self.N),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                V_i = V_i.double()
                sum_   += V_i.sum(0)
                sum_sq += (V_i ** 2).sum(0)
                count  += V_i.shape[0]
            mean = (sum_ / count).float()
            std  = ((sum_sq / count - (sum_ / count) ** 2).clamp(min=0).sqrt() + 1e-8).float()

            if idx_hash is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(str(fpath), mean=mean.numpy(), std=std.numpy())
                print(f"Cache U stats sauvegardé : {fpath}")
        else:
            V_train = self.U[train_indices]
            n, Nt_  = V_train.shape[:2]
            V_flat  = V_train.reshape(n * Nt_, self.N, self.N).double()
            mean = V_flat.mean(0).float()
            std  = (V_flat.std(0) + 1e-8).float()
        return mean, std

    def fit(self, train_indices):
        """
        Calcule les stats de normalisation sur le train set.
        Toujours : theta_mean/std, U_mean/std (z-score pixel-par-pixel, (N, N)).
        Si laplace=True : calcule U_laplace sur U normalisé (pas de post-normalisation dans __getitem__).
        """
        self.theta_mean = self.theta[train_indices].mean(0)
        self.theta_std  = self.theta[train_indices].std(0) + 1e-8
        idx_hash = hashlib.md5(np.array(sorted(train_indices)).tobytes()).hexdigest()[:8]
        self.U_mean, self.U_std = self._u_stats(train_indices, idx_hash=idx_hash)
        if self.laplace:
            self.U_laplace = self._to_laplace(self._s_list, self._rule, idx_hash)

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
            return theta_n, target                                 # (K, 2, N, N) déjà normalisé
        return theta_n, self.U[idx]                                # (Nt, N, N)


