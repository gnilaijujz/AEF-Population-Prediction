#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域偏移距离贡献分解（针对Building配置）
分析AEF和GIS特征在加权欧氏距离中的各自贡献。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"results")
OUTPUT_DIR = Path(r"paper_figures\cross_config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load AEF_L2 and GIS_L2 and Weighted_L2 for the Building configuration
building_dir = BASE / "domain_shift" / "aef_plus_building" / "with_gis"
aef_l2 = pd.read_csv(building_dir / "domain_shift_AEF_L2_matrix.csv", index_col=0)
gis_l2 = pd.read_csv(building_dir / "gis_L2_matrix.csv", index_col=0)
weighted_l2 = pd.read_csv(building_dir / "domain_shift_Weighted_L2_matrix.csv", index_col=0)

# Convert to numeric
aef_l2 = aef_l2.astype(float)
gis_l2 = gis_l2.astype(float)
weighted_l2 = weighted_l2.astype(float)

# We want to show that Weighted_L2 ≈ sqrt( wAEF * AEF_L2^2 + wGIS * GIS_L2^2 )
# But the actual contributions may be dominated by AEF.
# Compute the squared contributions (variance explained) in the distance matrix.

# Take upper-triangular values
upper_idx = np.triu_indices_from(aef_l2, k=1)
aef_vec = aef_l2.values[upper_idx]
gis_vec = gis_l2.values[upper_idx]
w_vec = weighted_l2.values[upper_idx]

# Remove NaNs
mask = ~(np.isnan(aef_vec) | np.isnan(gis_vec) | np.isnan(w_vec))
aef_vec = aef_vec[mask]
gis_vec = gis_vec[mask]
w_vec = w_vec[mask]

# Compute squared distances
aef_sq = aef_vec**2
gis_sq = gis_vec**2
w_sq = w_vec**2

# Since the weight is applied to the squared components in Weighted_L2,
# we can estimate the contribution of each component to the weighted distance
# by computing the fraction of the weighted squared distance contributed by AEF.
# Weighted_L2^2 = wAEF * AEF_L2^2 + wGIS * GIS_L2^2  (approximately, with weights)
# But we don't have weights explicitly; we can treat the empirical contributions.

# Directly compute the ratio of AEF squared to total squared (i.e., how much of the total variance is explained by AEF)
aef_contrib = aef_sq / (aef_sq + gis_sq + 1e-12)
gis_contrib = gis_sq / (aef_sq + gis_sq + 1e-12)

# Plot the distribution of contributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of AEF contribution ratio
axes[0].hist(aef_contrib, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(np.median(aef_contrib), color='red', linestyle='--', label=f'Median: {np.median(aef_contrib):.3f}')
axes[0].set_xlabel('AEF Contribution Ratio')
axes[0].set_ylabel('Frequency')
axes[0].set_title('AEF vs GIS Contribution to Squared Distance\n(AEF+BLD configuration)')
axes[0].legend()

# Pie chart: average contributions
avg_aef = np.mean(aef_contrib)
avg_gis = np.mean(gis_contrib)
axes[1].pie([avg_aef, avg_gis], labels=['AEF', 'GIS'], autopct='%1.1f%%',
            colors=['steelblue', 'lightcoral'], explode=(0.05, 0))
axes[1].set_title('Average Contribution to Total Squared Distance')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_contribution_pie.png", dpi=300)
plt.close()

# Also create a bar plot showing the actual squared distances per city-pair (scatter)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(aef_vec, w_vec, alpha=0.3, s=5, label='Weighted_L2')
ax.scatter(aef_vec, np.sqrt(aef_sq + gis_sq), alpha=0.2, s=3, label='sqrt(AEF^2+GIS^2)')
ax.set_xlabel('AEF L2 Distance')
ax.set_ylabel('Distance')
ax.set_title('Comparison of AEF Distance vs Weighted Distance')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aef_vs_weighted_scatter.png", dpi=300)
plt.close()

print("Distance contribution analysis complete.")