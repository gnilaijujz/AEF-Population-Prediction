#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-configuration Self-prediction vs Cross-domain Transferability
with regression lines for each configuration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

BASE = Path(r"results")
OUTPUT_DIR = Path(r"paper_figures\cross_config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_MAP = {
    'AEF': 'AEF',
    'Building': 'aef_plus_building',
    'GIS': 'GIS',
    'GIS9': 'aef_plus_gis9',
}

LABEL_MAP = {
    'AEF': 'AEF',
    'Building': 'AEF+BLD',
    'GIS': 'PureGIS',
    'GIS9': 'AEF+9GIS',
}

COLOR_MAP = {'AEF': 'blue', 'Building': 'green', 'GIS': 'red', 'GIS9': 'orange'}
MARKER_MAP = {'AEF': 'o', 'Building': 's', 'GIS': '^', 'GIS9': 'D'}

def load_self_and_cds(config_key):
    folder = CONFIG_MAP[config_key]
    cds_path = BASE / "transferability" / folder / "CDS_matrix_robust_entropy.csv"
    scores_path = BASE / "transferability" / folder / "full_scores_robust.csv"
    if not cds_path.exists() or not scores_path.exists():
        print(f"Warning: Missing files for {config_key}")
        return None
    cds = pd.read_csv(cds_path, index_col=0)
    scores = pd.read_csv(scores_path)
    avg_cds = {}
    for city in cds.index:
        vals = cds.loc[city].drop(index=city, errors='ignore').dropna().values
        avg_cds[city] = np.mean(vals) if len(vals) > 0 else np.nan
    avg_cds = pd.Series(avg_cds).dropna()
    self_rows = scores[scores['source'] == scores['target']]
    self_scores = self_rows.set_index('source')['Score'].to_dict()
    common_cities = list(set(avg_cds.index) & set(self_scores.keys()))
    if not common_cities:
        return None
    # Build DataFrame
    df = pd.DataFrame({
        'avg_CDS': [avg_cds[c] for c in common_cities],
        'Self_Score': [self_scores[c] for c in common_cities]
    }, index=common_cities)
    return df

# Load data
data_dict = {}
for key in CONFIG_MAP:
    df = load_self_and_cds(key)
    if df is not None and not df.empty:
        data_dict[key] = df

if not data_dict:
    raise ValueError("No data loaded for any configuration.")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter points
for key, df in data_dict.items():
    ax.scatter(df['Self_Score'], df['avg_CDS'], label=LABEL_MAP[key],
               c=COLOR_MAP[key], marker=MARKER_MAP[key], s=60, alpha=0.7)
    if key == 'AEF':
        for city, row in df.iterrows():
            ax.annotate(city, (row['Self_Score'], row['avg_CDS']),
                        fontsize=8, xytext=(5,5), textcoords='offset points')

# Regression lines for each configuration
x_min, x_max = ax.get_xlim()
x_range = np.linspace(x_min, x_max, 100)

for key, df in data_dict.items():
    if len(df) < 2:
        continue
    X = sm.add_constant(df['Self_Score'])
    model = sm.OLS(df['avg_CDS'], X).fit()
    slope = model.params['Self_Score']
    intercept = model.params['const']
    r2 = model.rsquared
    p_val = model.pvalues['Self_Score']
    y_pred = intercept + slope * x_range
    ax.plot(x_range, y_pred, color=COLOR_MAP[key], linestyle='--', linewidth=2,
            label=f"{LABEL_MAP[key]} (slope={slope:.3f}, R²={r2:.3f})")

ax.set_xlabel('Self-prediction Score (Entropy-weighted)')
ax.set_ylabel('Avg. Transfer Loss (avg_CDS)')
ax.set_title('Self-prediction vs Cross-domain Transferability\nAcross Configurations')
ax.legend(loc='best')
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "self_vs_cross_with_regression.png", dpi=300)
plt.close()

print("Self vs cross plot with regression lines completed.")
