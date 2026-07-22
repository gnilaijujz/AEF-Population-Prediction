#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨配置城市排名一致性分析
Compare source city ranking (by avg_CDS) across different feature configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BASE = Path(r"results")
OUTPUT_DIR = Path(r"paper_figures\cross_config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define configurations and their transferability folder names
CONFIG_MAP = {
    'AEF': 'AEF',
    'Building': 'aef_plus_building',   # AEF + building_density
    'Impervious': 'impervious',                # AEF + impervious
    'NDVI': 'ndvi',                           # AEF + ndvi
    'Nightlight': 'nighttime_lights',               # AEF + nighttime_lights
    'POI': 'poi',                             # AEF + POI
    'POI_diversity': 'poi_diversity',         # AEF + POI diversity
    'Road': 'roads',                          # AEF + road
    'Slope': 'slope',                         # AEF + slope
    'Waterdis': 'water_distance',                   # AEF + water distance
    'GIS': 'GIS',                             # pure GIS (no AEF)
    'GIS9': 'aef_plus_gis9',                  # AEF + 9 GIS variables
}

# We will use short labels for display
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

# ==================== Load Data ====================
def load_avg_cds(config_key):
    """
    Load CDS matrix for a given configuration and compute average CDS per source city.
    Returns a Series: city -> avg_CDS (excluding self).
    """
    folder = CONFIG_MAP[config_key]
    cds_path = BASE / "transferability" / folder / "CDS_matrix_robust_entropy.csv"
    if not cds_path.exists():
        print(f"Warning: {cds_path} not found. Skipping {config_key}.")
        return None
    cds = pd.read_csv(cds_path, index_col=0)
    avg_cds = {}
    for city in cds.index:
        vals = cds.loc[city].drop(index=city, errors='ignore').dropna().values
        avg_cds[city] = np.mean(vals) if len(vals) > 0 else np.nan
    return pd.Series(avg_cds).dropna()

# Load all configs
avg_cds_dict = {}
for key in CONFIG_MAP:
    ser = load_avg_cds(key)
    if ser is not None:
        avg_cds_dict[key] = ser

# Combine into a DataFrame, align by city
df_ranks = pd.DataFrame(avg_cds_dict).dropna()
# Sort rows by AEF avg_CDS for consistent ordering
df_ranks = df_ranks.sort_values('AEF', ascending=True)

# Rename columns for plotting
df_ranks.columns = [LABEL_MAP.get(c, c) for c in df_ranks.columns]

# ==================== Visualization 1: Heatmap of avg_CDS ====================
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_ranks, cmap='RdBu_r', center=0, annot=True, fmt='.3f',
            cbar_kws={'label': 'Avg. Transfer Loss (avg_CDS)'},
            ax=ax, linewidths=0.5)
ax.set_title('Source City Ranking by Transferability Across Configurations\n(lower = better teacher)', fontsize=14)
ax.set_xlabel('Feature Configuration')
ax.set_ylabel('Source City')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_CDS_heatmap_cross_config.png", dpi=300)
plt.close()

# ==================== Visualization 2: Ranking Correlation Matrix ====================
# Compute Spearman correlations between configurations
corr_rank = df_ranks.corr(method='spearman')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_rank, annot=True, fmt='.2f', cmap='coolwarm', center=1,
            vmin=0.8, vmax=1.0, square=True,
            cbar_kws={'label': 'Spearman ρ'},
            ax=ax, linewidths=0.5)
ax.set_title('Correlation of City Rankings Across Configurations\n(Spearman ρ)', fontsize=14)
ax.set_xlabel('Configuration')
ax.set_ylabel('Configuration')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ranking_correlation_matrix.png", dpi=300)
plt.close()

# ==================== Print Summary ====================
print("Cross-configuration ranking analysis complete.")
print(f"Number of configurations: {len(df_ranks.columns)}")
print(f"Number of common cities: {len(df_ranks)}")
print("\nAverage ranking correlations (Spearman):")
print(corr_rank.mean().sort_values(ascending=False))
