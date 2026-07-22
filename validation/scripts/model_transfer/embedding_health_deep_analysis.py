#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extra figures:
1. Custom correlation heatmap (exclude NYC/Duluth ONLY for neighbor_consistency)
2. Boxplot of city size (natural breaks) vs. avg_CDS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------- Fonts (English only) -------------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ------------------------- Configuration -------------------------
DEEP_METRICS_PATH = Path(r"validation\results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv")
OUTPUT_DIR = Path(r"validation/results/embedding_health_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DEEP_METRICS_PATH, index_col=0)
print(f"Loaded {len(df)} cities")

# Define metric columns
metric_cols = ['effective_rank', 'isotropy', 'neighbor_consistency',
               'inter_intra_ratio', 'pca_5_explained', 'n_nodes', 'avg_CDS', 'R2_self']
metric_cols = [c for c in metric_cols if c in df.columns]

# ------------------------- 1. Custom Correlation Heatmap -------------------------
def conditional_correlation_matrix(df, metrics, target_col='neighbor_consistency', exclude_cities=['newyork', 'Duluth']):
    """
    Compute correlations.
    If target_col is involved, exclude the specified outliers.
    Otherwise, use all cities.
    """
    n = len(metrics)
    corr_mat = pd.DataFrame(np.ones((n, n)), index=metrics, columns=metrics)
    
    for i in range(n):
        for j in range(n):
            if i >= j:  # we fill both upper and lower, so only compute once
                continue
            m1, m2 = metrics[i], metrics[j]
            
            # Determine if we need to exclude outliers
            if target_col in [m1, m2]:
                data = df[~df.index.isin(exclude_cities)]
            else:
                data = df
            
            # Drop NaNs for these specific columns
            clean_data = data[[m1, m2]].dropna()
            if len(clean_data) > 2:
                r = clean_data.corr().iloc[0, 1]
                corr_mat.loc[m1, m2] = r
                corr_mat.loc[m2, m1] = r
            else:
                corr_mat.loc[m1, m2] = np.nan
                corr_mat.loc[m2, m1] = np.nan
    return corr_mat

corr_custom = conditional_correlation_matrix(df, metric_cols)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_custom, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
plt.title('Correlation Matrix (NYC & Duluth excluded only for neighbor_consistency)', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap_conditional.png', dpi=300)
plt.close()
print("Saved: correlation_heatmap_conditional.png")

# ------------------------- 2. Boxplot: City Size (Natural Breaks) -------------------------
# Natural breaks based on real-world city scale
# Small: < 200 nodes, Medium: 200-500, Large: > 500
bins = [0, 200, 500, 10000]
labels = ['Small (<200)', 'Medium (200-500)', 'Large (>500)']
df['size_group'] = pd.cut(df['n_nodes'], bins=bins, labels=labels, right=False)

# Reorder groups
df['size_group'] = pd.Categorical(df['size_group'], categories=['Small (<200)', 'Medium (200-500)', 'Large (>500)'], ordered=True)

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='size_group', y='avg_CDS', data=df, palette='Set3')
sns.stripplot(x='size_group', y='avg_CDS', data=df, color='black', alpha=0.5, jitter=True)
plt.xlabel('City Size (Number of Nodes / Tracts)')
plt.ylabel('avg_CDS (Transfer Loss)')
plt.title('City Size vs. Transferability (Natural Breaks)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'size_boxplot_natural_breaks.png', dpi=300)
plt.close()
print("Saved: size_boxplot_natural_breaks.png")

# Print composition for verification
print("\nGroup composition (Natural Breaks):")
for group in ['Small (<200)', 'Medium (200-500)', 'Large (>500)']:
    cities = df[df['size_group'] == group].index.tolist()
    sizes = df[df['size_group'] == group]['n_nodes'].tolist()
    print(f"{group}: {cities} (n_nodes: {sizes})")

print("\nAll figures saved to:", OUTPUT_DIR)