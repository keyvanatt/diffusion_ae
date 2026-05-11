import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

if __name__ == '__main__':
    doe_path  = os.path.join('dataset', 'doe_rotated.npy')
    out_path  = os.path.join('dataset', 'split.npz')
    test_frac = 0.2
    seed      = 42

    doe = np.load(doe_path)
    ns  = len(doe)

    rng      = np.random.default_rng(seed)
    idx      = rng.permutation(ns)
    n_test   = int(test_frac * ns)
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    np.savez(out_path, train_idx=train_idx, test_idx=test_idx)
    print(f"Split sauvegardé : {out_path}")
    print(f"  train : {len(train_idx)} samples  ({100*(1-test_frac):.0f}%)")
    print(f"  test  : {len(test_idx)} samples  ({100*test_frac:.0f}%)")
    print(f"  seed  : {seed}")
