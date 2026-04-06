import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tqdm import tqdm
import numpy as np
from utils.SVD_Amine_3D import svd_3d_gpu, svd_inverse_3d


def learn_svd(concentration_path, step=5, erreur=1e-8, nf=None):
    """
    Charge un fichier .npy de concentration (ns, T, H, W), applique une décomposition
    Tucker rang-1 (SVD 3D), et sauvegarde le résultat dans le même répertoire.
    """
    if concentration_path.endswith('.npy'):
        concentration = np.load(concentration_path, mmap_mode='r')  # (ns, T, H, W)
    else:
        data = np.load(concentration_path)
        concentration = data['U']  # (ns, T, H, W)
    ns, Nt, H, W = concentration.shape

    Hsub = len(range(0, H, step))
    Wsub = len(range(0, W, step))
    nr = Hsub * Wsub
    nf = min(nr, Nt) if nf is None else nf

    print(f"Fichier           : {concentration_path}")
    print(f"Shape             : {concentration.shape}")
    print(f"Grille sous-échantillonnée : {Hsub}×{Wsub} = {nr} nœuds")
    print(f"Tenseur HH        : ({nr}, {ns}, {Nt})  |  nf={nf}")
    print(f"Mémoire HH        : {nr * ns * Nt * 4 / 1e9:.2f} Go")

    HH = np.empty((nr, ns, Nt), dtype=np.float32)
    for i in tqdm(range(ns), desc="Préparation du tenseur HH"):
        sub = concentration[i, :, ::step, ::step]  # (Nt, Hsub, Wsub) — lu depuis disque
        HH[:, i, :] = sub.reshape(Nt, nr).T

    print(f"SVD 3D  (nf={nf}, erreur={erreur})...")
    F, G, P, alph, Hist_ErrL2 = svd_3d_gpu(HH, nf=nf, erreur=erreur)
    nf_eff = len(alph)
    print(f"  Modes retenus      : {nf_eff}")
    print(f"  Erreur L2 finale   : {Hist_ErrL2[-1]:.6e}")

    HH_rec = svd_inverse_3d(F, G, P, alph)
    rel_err = np.linalg.norm(HH_rec - HH) / (np.linalg.norm(HH) + 1e-12)
    print(f"  Erreur L2 globale  : {rel_err:.6e}")

    save_path = os.path.join(os.path.dirname(concentration_path), 'svd_train_diff.npz')
    np.savez(save_path, F=F, G=G, P=P, alph=alph, Hist_ErrL2=Hist_ErrL2)
    print(f"Décomposition sauvegardée : {save_path}")

    return F, G, P, alph, Hist_ErrL2, HH, HH_rec, Hsub, Wsub


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')

    concentration_path = os.path.join('/Data/KAT/ch4_rotated.npy')
    step = 5
    erreur = 1e-5
    example_idx = 2  # indice de simulation à sauvegarder pour vérification

    F, G, P, alph, Hist_ErrL2, HH, HH_rec, Hsub, Wsub = learn_svd(
        concentration_path, step=step, erreur=erreur, nf=300
    )

    from utils.animate import animate_comparaison

    # HH shape : (nr, ns, Nt) → extraire l'exemple example_idx et remettre en (Nt, Hsub, Wsub)
    orig = HH[:, example_idx, :].reshape(Hsub, Wsub, -1).transpose(2, 0, 1)   # (Nt, Hsub, Wsub)
    rec  = HH_rec[:, example_idx, :].reshape(Hsub, Wsub, -1).transpose(2, 0, 1)

    err_ex = np.linalg.norm(rec - orig) / (np.linalg.norm(orig) + 1e-12)
    print(f"Erreur L2 exemple #{example_idx} : {err_ex:.6e}")

    animate_comparaison(
        orig, rec,
        output_path=os.path.join('plots', f'svd_example_{example_idx}.gif'),
        title_fn=lambda t: f"Exemple #{example_idx} — t = {t}",
    )
