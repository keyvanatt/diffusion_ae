"""
For each entry in CH4 (shape: N_samples x T x H x W), generate N_ROTATIONS rotated copies
with uniformly spaced rotation angles. The rotation is performed at native resolution (200x200),
then downsampled to OUTPUT_SIZE x OUTPUT_SIZE (128x128) via INTER_AREA to avoid aliasing.

Rotating before downsampling avoids:
  - black corner artifacts (would appear at native res, then get averaged away)
  - aliasing from rotating an already-coarse grid

Saves:
  - {SAVE_DIR}/ch4_rotated.npy  : shape (N_samples * N_ROTATIONS, T, 128, 128), float32, memmap
  - {SAVE_DIR}/doe_rotated.npy  : structured array (k, A, C, theta), shape (N_samples * N_ROTATIONS,)

Acceleration:
  - cv2.warpAffine + cv2.resize instead of scipy.ndimage.rotate (~5-10x faster)
  - joblib.Parallel over samples (threads, each writes to its own memmap slice)
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
SAVE_DIR = "F:/keyvan/"
N_ROTATIONS = 36
N_JOBS = 8
OUTPUT_SIZE = 128  # final spatial resolution after downsampling

# --- Load data ---
ch4 = np.load(os.path.join(DATA_DIR, "CH4.npy"))   # (N, T, H, W)
doe = np.load(os.path.join(DATA_DIR, "doe.npy"))    # structured: (N,) with fields k, A, C

N, T, H, W = ch4.shape
assert H == 200 and W == 200, f"Expected 200x200 input, got {H}x{W}"
assert OUTPUT_SIZE < H, "OUTPUT_SIZE must be smaller than native resolution for downsampling"
print(f"CH4 shape: {ch4.shape}, dtype: {ch4.dtype}")
print(f"doe shape: {doe.shape}, fields: {doe.dtype.names}")
print(f"Pipeline: rotate at {H}x{W} → downsample to {OUTPUT_SIZE}x{OUTPUT_SIZE}")

# --- Uniform rotation angles ---
angles = np.linspace(0, 360, N_ROTATIONS, endpoint=False)
print(f"Rotation angles (degrees): {angles}")

# --- Precompute rotation matrices at native resolution ---
# Rotate around the true center of the 200x200 grid.
# BORDER_REFLECT_101 mirrors pixels at the border → no black corners,
# which then vanish cleanly after INTER_AREA downsampling.
center = (W / 2.0, H / 2.0)
rot_matrices = [cv2.getRotationMatrix2D(center, float(a), 1.0) for a in angles]

# --- Output: memmap to disk (float32) ---
out_path = os.path.join(SAVE_DIR, "ch4_rotated.npy")
out_shape = (N * N_ROTATIONS, T, OUTPUT_SIZE, OUTPUT_SIZE)
fp = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=out_shape)
print(f"Allocated memmap {out_path}  shape={out_shape}  dtype=float32")

out_doe = np.empty(N * N_ROTATIONS, dtype=np.dtype([
    ('k', '<f8'), ('A', '<f8'), ('C', '<f8')
]))


def rotate_and_downsample(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Rotate a single (H, W) frame at native resolution, then downsample to OUTPUT_SIZE.

    - BORDER_REFLECT_101: mirrors pixels at the edge (no black corners).
      This is safe for physical fields that vary smoothly near the boundary.
      If your field has hard boundaries (e.g. wall = 0), use BORDER_REPLICATE instead.
    - INTER_LINEAR for rotation: subpixel-accurate, fast.
    - INTER_AREA for downsampling: equivalent to proper anti-aliased averaging
      when shrinking; avoids moiré and aliasing that INTER_LINEAR would introduce.
    """
    rotated = cv2.warpAffine(
        frame, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    downsampled = cv2.resize(
        rotated, (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return downsampled


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
        rotated_downsampled = np.stack([
            rotate_and_downsample(sample[t], M)
            for t in range(T)
        ])  # (T, OUTPUT_SIZE, OUTPUT_SIZE)
        fp[idx] = rotated_downsampled
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