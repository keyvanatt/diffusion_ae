"""
For each entry in CH4 (shape: N_samples x T x H x W), generate N_ROTATIONS rotated copies
with uniformly spaced rotation angles. Saves:
  - dataset/ch4_rotated.npy  : shape (N_samples * N_ROTATIONS, T, H, W), float32, written via memmap
  - dataset/doe_rotated.npy  : structured array (k, A, C, theta), shape (N_samples * N_ROTATIONS,)

Acceleration:
  - cv2.warpAffine instead of scipy.ndimage.rotate (~5-10x faster)
  - joblib.Parallel over samples (multiprocessing, each writes to its own memmap slice)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Config ---
DATA_DIR = "dataset"
SAVE_DIR = "/Data/KAT/"
N_ROTATIONS = 36
N_JOBS = 8  # parallel workers (set to os.cpu_count() for max)

# --- Load data ---
ch4 = np.load(os.path.join(DATA_DIR, "CH4.npy"))   # (N, T, H, W)
doe = np.load(os.path.join(DATA_DIR, "doe.npy"))    # structured: (N,) with fields k, A, C

N, T, H, W = ch4.shape
print(f"CH4 shape: {ch4.shape}, dtype: {ch4.dtype}")
print(f"doe shape: {doe.shape}, fields: {doe.dtype.names}")

# --- Uniform rotation angles ---
angles = np.linspace(0, 360, N_ROTATIONS, endpoint=False)
print(f"Rotation angles (degrees): {angles}")

# --- Precompute rotation matrices (one per angle) ---
center = (W / 2.0, H / 2.0)
rot_matrices = [cv2.getRotationMatrix2D(center, float(a), 1.0) for a in angles]

# --- Output: memmap to disk (float32) ---
out_path = os.path.join(SAVE_DIR, "ch4_rotated.npy")
out_shape = (N * N_ROTATIONS, T, H, W)
fp = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=out_shape)
print(f"Allocated memmap {out_path}  shape={out_shape}  dtype=float32")

out_doe = np.empty(N * N_ROTATIONS, dtype=np.dtype([
    ('k', '<f8'), ('A', '<f8'), ('C', '<f8')
]))


def process_sample(i):
    """Rotate sample i for all angles, write to memmap slice [i*N_ROTATIONS : (i+1)*N_ROTATIONS]."""
    sample = ch4[i].astype(np.float32)  # (T, H, W)
    k_val = float(doe['k'][i])
    A_val = float(doe['A'][i])
    if A_val != 0.0:
        print(f"Warning: A={A_val} non nul pour i={i} — vérifie les données d'entrée.")

    C_val = float(doe['C'][i])

    for j, (angle, M) in enumerate(zip(angles, rot_matrices)):
        idx = i * N_ROTATIONS + j
        rotated = np.stack([
            cv2.warpAffine(sample[t], M, (W, H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            for t in range(T)
        ])  # (T, H, W)
        fp[idx] = rotated
        out_doe[idx] = (k_val, angle, C_val)

    fp.flush()


# --- Parallel execution over samples ---
Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(process_sample)(i) for i in tqdm(range(N), desc="Samples")
)

print(f"Saved {out_path}")

doe_path = os.path.join(SAVE_DIR, "doe_rotated.npy")
np.save(doe_path, out_doe)
print(f"Saved {doe_path}")
print("Done.")
