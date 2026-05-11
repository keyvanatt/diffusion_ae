"""
Génère un exemple du dataset CH4 à plusieurs temps pour visualiser l'évolution temporelle
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# Charger le dataset
data_path = "/Data/KAT/ch4_rotated.npy"
doe_path = "/Data/KAT/doe_rotated.npy"

print("Chargement du dataset...")
U = np.load(data_path, mmap_mode='r')  # (8100, 150, 200, 200)
doe = np.load(doe_path)

print(f"Dataset shape: {U.shape}")
print(f"Parameters: k, A, C")

# Sélectionner une simulation aléatoire
np.random.seed(42)
sim_idx = np.random.randint(0, len(U))

print(f"\nSimulation {sim_idx}")
print(f"  k = {doe['k'][sim_idx]:.4f}")
print(f"  A = {doe['A'][sim_idx]:.4f}")
print(f"  C = {doe['C'][sim_idx]:.4f}")

# Sélectionner plusieurs pas de temps
Nt = 150
time_indices = [0, 25, 50, 75, 100, 125, 149]  # Début, et régulièrement espacés

# Créer la figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, t_idx in enumerate(time_indices):
    ax = axes[i]

    field = U[sim_idx, t_idx]  # (200, 200)

    im = ax.imshow(field, cmap='viridis')
    ax.set_title(f't = {t_idx}/150', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('[ppm]', fontsize=9)

# Masquer le dernier subplot
axes[-1].axis('off')

fig.suptitle(f'CH4 Concentration Field Evolution (Simulation #{sim_idx})\n'
             f'k={doe["k"][sim_idx]:.3f}, A={doe["A"][sim_idx]:.3f}, C={doe["C"][sim_idx]:.3f}',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/ch4_dataset_preview.png', dpi=150, bbox_inches='tight')
print("\n✓ Sauvegardé : plots/ch4_dataset_preview.png")
plt.close()

# Créer une deuxième figure avec une bande temporelle
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

time_points = [0, 75, 149]
titles = ['Initial (t=0)', 'Mid (t=75)', 'Final (t=149)']

for ax, t_idx, title in zip(axes, time_points, titles):
    field = U[sim_idx, t_idx]

    im = ax.imshow(field, cmap='viridis')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('[ppm]', fontsize=10)

fig.suptitle(f'CH4 Transient Field (Simulation #{sim_idx}, k={doe["k"][sim_idx]:.3f}, A={doe["A"][sim_idx]:.3f}, C={doe["C"][sim_idx]:.3f})',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/ch4_dataset_timeseries.png', dpi=150, bbox_inches='tight')
print("✓ Sauvegardé : plots/ch4_dataset_timeseries.png")
plt.close()

print("\nDone!")
