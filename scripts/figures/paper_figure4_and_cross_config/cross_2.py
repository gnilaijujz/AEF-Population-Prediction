#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Configuration Domain Shift Distance Similarity Analysis
Compare L2 distance matrices across different configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"results")
OUTPUT_DIR = Path(r"paper_figures\cross_config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define configs to include (those that have domain_shift matrices)
CONFIGS = ['AEF', 'Building', 'Impervious', 'NDVI', 'Nightlight', 
           'POI', 'POI_diversity', 'Road', 'Slope', 'Waterdis', 'GIS', 'GIS9']

LABEL_MAP = {
    'AEF': 'AEF',
    'Building': 'AEF+BLD',
    'Impervious': 'AEF+IMP',
    'NDVI': 'AEF+NDVI',
    'Nightlight': 'AEF+NTL',
    'POI': 'AEF+POI',
    'POI_diversity': 'AEF+POId',
    'Road': 'AEF+ROAD',
    'Slope': 'AEF+SLP',
    'Waterdis': 'AEF+WAT',
    'GIS': 'PureGIS',
    'GIS9': 'AEF+9GIS',
}

# Hard-coded paths for each configuration's L2 distance matrix
def get_l2_path(config):
    base_domain = BASE / "domain_shift"
    if config == 'AEF':
        return base_domain / "aef" / "domain_shift_L2_Dist_matrix.csv"
    elif config == 'Building':
        return base_domain / "aef_plus_building" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'Impervious':
        return base_domain / "aef_plus_impervious" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'NDVI':
        return base_domain / "aef_plus_ndvi" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'Nightlight':
        return base_domain / "aef_plus_nighttime_lights" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'POI':
        return base_domain / "aef_plus_poi" / "domain_shift_L2_Dist_matrix.csv"
    elif config == 'POI_diversity':
        return base_domain / "aef_plus_poi_diversity" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'Road':
        return base_domain / "aef_plus_roads" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'Slope':
        return base_domain / "aef_plus_slope" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'Waterdis':
        return base_domain / "aef_plus_water_distance" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    elif config == 'GIS':
        return base_domain / "gis" / "domain_shift_L2_Dist_matrix.csv"
    elif config == 'GIS9':
        return base_domain / "aef_plus_gis9" / "with_gis" / "domain_shift_Weighted_L2_matrix.csv"
    else:
        return None

# Load all L2 matrices
l2_dict = {}
for cfg in CONFIGS:
    path = get_l2_path(cfg)
    if path is not None and path.exists():
        mat = pd.read_csv(path, index_col=0)
        mat = mat.astype(float)
        l2_dict[cfg] = mat
        print(f"Loaded {cfg} from {path}")
    else:
        print(f"Warning: {path} not found. Skipping {cfg}.")

# Find common cities
common_cities = None
for mat in l2_dict.values():
    if common_cities is None:
        common_cities = set(mat.index) & set(mat.columns)
    else:
        common_cities &= (set(mat.index) & set(mat.columns))
common_cities = sorted(common_cities)
print(f"Common cities across all matrices: {len(common_cities)}")

# Compute correlations
configs = list(l2_dict.keys())
n = len(configs)
corr_l2 = np.zeros((n, n))
for i, cfg1 in enumerate(configs):
    for j, cfg2 in enumerate(configs):
        mat1 = l2_dict[cfg1].loc[common_cities, common_cities]
        mat2 = l2_dict[cfg2].loc[common_cities, common_cities]
        upper_idx = np.triu_indices_from(mat1, k=1)
        vec1 = mat1.values[upper_idx]
        vec2 = mat2.values[upper_idx]
        mask = ~(np.isnan(vec1) | np.isnan(vec2))
        if mask.sum() > 10:
            corr_l2[i, j] = np.corrcoef(vec1[mask], vec2[mask])[0, 1]
        else:
            corr_l2[i, j] = np.nan

df_corr_l2 = pd.DataFrame(corr_l2, index=[LABEL_MAP[c] for c in configs],
                          columns=[LABEL_MAP[c] for c in configs])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr_l2, annot=True, fmt='.2f', cmap='coolwarm', center=0.8,
            vmin=0.5, vmax=1.0, square=True,
            cbar_kws={'label': 'Pearson r'},
            ax=ax, linewidths=0.5)
ax.set_title('Similarity of Domain Shift Distance Matrices Across Configurations\n(Pearson correlation of upper-triangular entries)', fontsize=14)
ax.set_xlabel('Configuration')
ax.set_ylabel('Configuration')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "L2_distance_matrix_correlation.png", dpi=300)
plt.close()