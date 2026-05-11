"""
transient/benchmark.py — Comparaison de plusieurs checkpoints sur le test set
==============================================================================
Génère un violin plot des L2rel losses pour plusieurs modèles.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transient.main import load_model, run_inference


def count_parameters(model):
    """Compte le nombre total de paramètres dans un modèle (entraînables et freezés)."""
    return sum(p.numel() for p in model.parameters())


def benchmark(
    checkpoints: dict[str, str],
    data_path: str = '/Data/KAT/ch4_rotated.npy',
    dt: float | None = None,
    alpha_t: float = 0.0,
    lam: float = 1e-6,
    rule: str = 'trap',
    batch_size: int = 32,
    device_str: str = 'auto',
    k_max: int | None = None,
    output_path: str = 'plots/benchmark_l2rel.png',
):
    """
    Évalue plusieurs checkpoints et produit un violin plot.

    Paramètres
    ----------
    checkpoints  : dict {label → ckpt_path}
                   ex. {'LaplaceModel': 'checkpoints/LaplaceModel.pt', ...}
    data_path    : chemin vers ch4_rotated.npy
    dt, alpha_t, lam, rule, k_max : paramètres d'inférence
    batch_size   : nombre de sims par batch
    device_str   : 'auto', 'cpu' ou 'cuda'
    output_path  : où sauver le violin plot
    """
    import torch

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"Device : {device}")
    print(f"Checkpoints à évaluer : {list(checkpoints.keys())}")

    # Charger les données
    print(f"Chargement des données depuis {data_path}...")
    U_data = np.load(data_path, mmap_mode='r')
    doe_path = str(Path(data_path).parent / 'doe_rotated.npy')
    doe = np.load(doe_path)
    theta_data = np.stack([doe['k'], doe['A'], doe['C']], axis=1).astype(np.float32)

    # Évaluer chaque checkpoint
    results = {}  # {label → l2rel_array}
    model_params = {}  # {label → nombre de paramètres}

    for label, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Évaluation : {label}")
        print(f"Checkpoint : {ckpt_path}")
        print(f"{'='*60}")

        model, ckpt = load_model(ckpt_path, device)
        model_type = ckpt.get('model_type', 'Unknown')
        print(f"Type : {model_type}")

        # Compter les paramètres
        n_params = count_parameters(model)
        model_params[label] = n_params
        print(f"Paramètres : {n_params:,}")

        if 'test_idx' not in ckpt:
            raise ValueError(f"test_idx absent du checkpoint {ckpt_path}")
        test_idx = ckpt['test_idx']

        theta_arr = np.asarray(theta_data, dtype=np.float32)
        n_test = len(test_idx)
        print(f"Test set : {n_test} simulations")

        l2rel_list = []

        for start in tqdm(range(0, n_test, batch_size), desc=f'Éval {label}'):
            pos_batch = list(range(start, min(start + batch_size, n_test)))
            idx_batch = [int(test_idx[p]) for p in pos_batch]

            # Prédiction
            U_pred_b = run_inference(
                theta_arr[idx_batch], model, ckpt, device,
                dt=dt, alpha_t=alpha_t, lam=lam, rule=rule, k_max=k_max
            )  # (B, Nt, H_pred, W_pred)

            # Vérité terrain
            H_pred, W_pred = U_pred_b.shape[-2], U_pred_b.shape[-1]
            slices = []
            for i in idx_batch:
                u = np.array(U_data[i], dtype=np.float32)
                if u.shape[-2] != H_pred or u.shape[-1] != W_pred:
                    u_t = torch.from_numpy(u).unsqueeze(0)
                    u_t = torch.nn.functional.interpolate(
                        u_t, size=(H_pred, W_pred), mode='bilinear', align_corners=False
                    )
                    u = u_t.squeeze(0).numpy()
                slices.append(u)
            U_true_b = np.stack(slices)

            # Calcul L2rel
            diff = U_pred_b - U_true_b
            norms_err = np.linalg.norm(diff.reshape(len(pos_batch), -1), axis=1)
            norms_true = np.linalg.norm(U_true_b.reshape(len(pos_batch), -1), axis=1) + 1e-12
            l2rel_list.append(norms_err / norms_true)

        l2rel = np.concatenate(l2rel_list)
        results[label] = l2rel

        print(f"L2rel — mean: {l2rel.mean()*100:.2f}%  std: {l2rel.std()*100:.2f}%")
        print(f"        min : {l2rel.min()*100:.2f}%   max: {l2rel.max()*100:.2f}%")

    # Créer les histogrammes individuels
    print(f"\n{'='*60}")
    print("Création des histogrammes individuels...")
    print(f"{'='*60}")

    for label, l2rel in results.items():
        plt.figure(figsize=(10, 5))
        plt.hist(l2rel * 100, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlim((0, 100))
        plt.xlabel('L2 Relative Error (%)')
        plt.ylabel('Fréquence')
        n_params = model_params[label]
        plt.title(f'L2 Relative Error — {label}\n'
                  f'mean={l2rel.mean()*100:.2f}%  median={np.median(l2rel)*100:.2f}%  std={l2rel.std()*100:.2f}%  params={n_params:,}')
        plt.grid(True, alpha=0.3)
        # Nettoyer le nom pour créer le fichier
        clean_name = label.replace(' ', '_').replace('(', '').replace(')', '').replace('_', '_').lower()
        # Simplifier les doubles underscores
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')
        hist_path = os.path.join('plots', f'{clean_name}_l2rel_hist.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → {hist_path}")

    # Créer le violin plot
    print(f"\n{'='*60}")
    print("Création du violin plot...")
    print(f"{'='*60}")

    os.makedirs('plots', exist_ok=True)

    # Préparer les données pour seaborn
    data_for_plot = []
    for label, l2rel_arr in results.items():
        for val in l2rel_arr * 100:  # Convertir en %
            data_for_plot.append({'Model': label, 'L2 Relative Error (%)': val})

    import pandas as pd
    df = pd.DataFrame(data_for_plot)

    # Créer le violinplot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Model', y='L2 Relative Error (%)', palette='Set2')
    plt.title('Distribution L2 Relative Error — Comparaison Modèles', fontsize=14, fontweight='bold')
    plt.ylabel('L2 Relative Error (%)', fontsize=12)
    plt.xlabel('Modèle', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)

    # Ajouter les statistiques en texte
    stats_text = "\n".join([
        f"{label}\n"
        f"  Params: {model_params[label]:,}\n"
        f"  μ={results[label].mean()*100:.2f}%, median={np.median(results[label])*100:.2f}%, σ={results[label].std()*100:.2f}%"
        for label in results.keys()
    ])
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Violin plot sauvegardé -> {output_path}")

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    for label in results.keys():
        l2rel = results[label]
        n_params = model_params[label]
        print(f"{label:35s}")
        print(f"  Paramètres     : {n_params:,}")
        print(f"  Mean           : {l2rel.mean()*100:6.2f}%")
        print(f"  Median         : {np.median(l2rel)*100:6.2f}%")
        print(f"  Std            : {l2rel.std()*100:6.2f}%")
        print(f"  Range          : [{l2rel.min()*100:5.2f}% - {l2rel.max()*100:5.2f}%]")
        print()

    return results


if __name__ == '__main__':
    # Checkpoints à comparer
    checkpoints = {
        'LaplaceLatentModel': 'checkpoints/LaplaceLatentModel.pt',
        'LaplaceLatentModel (finetuned)': 'checkpoints/LaplaceLatentModel_finetuned.pt',
        'CorrectionAE (UNet)': 'checkpoints/CorrectionAE_best.pt',
    }

    results = benchmark(
        checkpoints=checkpoints,
        data_path='/Data/KAT/ch4_rotated.npy',
        k_max=20,
    )
