"""
For each entry in CH4 (shape: N_samples x T x H x W), generate 50 rotated copies
with uniformly spaced rotation angles. Saves:
  - dataset/Results/ch4_rotated.npy  : shape (N_samples * 50, T, H, W), float32, written via memmap
  - dataset/Results/doe_augmented.npy: structured array (k, A, C, theta), shape (N_samples * 50,)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm

# --- Config ---
DATA_DIR = "dataset/Results"
N_ROTATIONS = 2

# --- Load data ---
ch4 = np.load(os.path.join(DATA_DIR, "CH4.npy"))   # (N, T, H, W)
doe = np.load(os.path.join(DATA_DIR, "doe.npy"))    # structured: (N,) with fields k, A, C

N, T, H, W = ch4.shape
print(f"CH4 shape: {ch4.shape}, dtype: {ch4.dtype}")
print(f"doe shape: {doe.shape}, fields: {doe.dtype.names}")

# --- Uniform rotation angles ---
angles = np.linspace(0, 360, N_ROTATIONS, endpoint=False) 

# --- Output: memmap to disk (float32 to halve memory) ---
out_path = os.path.join(DATA_DIR, "ch4_rotated.npy")
out_shape = (N * N_ROTATIONS, T, H, W)

# Write the .npy header manually, then open as memmap
fp = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=out_shape)
print(f"Allocated memmap {out_path}  shape={out_shape}  dtype=float32")

out_doe = np.empty(N * N_ROTATIONS, dtype=np.dtype([
    ('k', '<f8'), ('A', '<f8'), ('C', '<f8'), ('theta', '<f8')
]))

# --- Rotate and write sample by sample ---
for i in tqdm(range(N), desc="Samples"):
    k_val = float(doe['k'][i])
    A_val = float(doe['A'][i])
    C_val = float(doe['C'][i])
    sample = ch4[i].astype(np.float32)  # (T, H, W)

    for j, angle in enumerate(tqdm(angles, desc=f"  Rotations", leave=False)):
        idx = i * N_ROTATIONS + j

        rotated = np.stack([
            rotate(sample[t], angle=angle, axes=(0, 1),
                   reshape=False, order=1, mode='nearest')
            for t in range(T)
        ])  # (T, H, W)

        fp[idx] = rotated
        out_doe[idx] = (k_val, A_val, C_val, angle)

    fp.flush()

print(f"Saved {out_path}")

doe_path = os.path.join(DATA_DIR, "doe_rotated.npy")
np.save(doe_path, out_doe)
print(f"Saved {doe_path}")
print("Done.")
