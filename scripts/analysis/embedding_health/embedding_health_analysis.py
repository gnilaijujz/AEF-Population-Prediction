#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding Health Analysis and Visualization
Based on deep health metrics (embedding_health_deep_metrics.csv).
Generates all key figures for "Which source cities are good teachers?"

Modified: Self-prediction score (R2_self) is now read directly from
full_scores_robust.csv (entropy-weighted Score), not from the CDS matrix diagonal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== Configuration ========================
# Set this to match your experiment configuration (e.g., 'AEF', 'Buildings', 'GIS')
CONFIG_NAME = 'AEF'  # Change as needed

# Base result directory
BASE_RESULT = Path(r"results")

# Paths relative to CONFIG_NAME
DEEP_METRICS_PATH = BASE_RESULT / "embedding_health_deep_analysis" / "embedding_health_deep_metrics.csv"
FULL_SCORES_PATH = BASE_RESULT / "transferability" / CONFIG_NAME / "full_scores_robust.csv"
OUTPUT_DIR = BASE_RESULT / "embedding_health_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================== Global font settings (English only) ========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ======================== Load Data ========================
# 1. Load deep health metrics
health_df = pd.read_csv(DEEP_METRICS_PATH, index_col=0)
print(f"Loaded {len(health_df)} cities from {DEEP_METRICS_PATH}")

# 2. Load self-prediction scores (entropy-weighted Score) from full_scores_robust.csv
if FULL_SCORES_PATH.exists():
    df_scores = pd.read_csv(FULL_SCORES_PATH)
    # Filter for source == target (self-prediction)
    self_rows = df_scores[df_scores['source'] == df_scores['target']]
    self_scores = {}
    for _, row in self_rows.iterrows():
        city = row['source']
        if 'Score' in row:
            self_scores[city] = row['Score']
        else:
            print(f"Warning: 'Score' column missing for {city}, setting NaN")
            self_scores[city] = np.nan
    # Override R2_self column in health_df with these scores
    health_df['R2_self'] = health_df.index.map(lambda c: self_scores.get(c, np.nan))
    print(f"Updated R2_self from {FULL_SCORES_PATH}")
else:
    print(f"Warning: {FULL_SCORES_PATH} not found. R2_self remains as in CSV.")

# Ensure required columns exist
required_cols = ['avg_CDS', 'R2_self', 'neighbor_consistency', 'effective_rank', 'isotropy',
                 'inter_intra_ratio', 'pca_5_explained', 'n_nodes']
missing = [col for col in required_cols if col not in health_df.columns]
if missing:
    print(f"Warning: missing columns {missing}, using available ones.")

# Sort by avg_CDS ascending (best teacher first)
health_df = health_df.sort_values('avg_CDS', ascending=True)

# Define metric columns for plotting and regression
metric_cols = [col for col in ['neighbor_consistency', 'effective_rank', 'isotropy',
                               'inter_intra_ratio', 'pca_5_explained', 'n_nodes']
               if col in health_df.columns]

# ======================== Regression and Correlation ========================
scaler = StandardScaler()
health_df_scaled = health_df.copy()
health_df_scaled[metric_cols] = scaler.fit_transform(health_df[metric_cols])

# Correlation matrix
corr = health_df[metric_cols + ['avg_CDS', 'R2_self']].corr()
print("\nCorrelation with avg_CDS:")
print(corr['avg_CDS'].sort_values(ascending=False))

# Univariate regression
univariate_results = []
for col in metric_cols:
    X = sm.add_constant(health_df_scaled[[col]])
    model = sm.OLS(health_df_scaled['avg_CDS'], X).fit()
    univariate_results.append({
        'Metric': col,
        'Coefficient': model.params[col],
        'P_value': model.pvalues[col],
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj
    })
df_uni = pd.DataFrame(univariate_results).sort_values('P_value')
print("\nUnivariate regression results (predicting avg_CDS):")
print(df_uni)

# Multivariate regression
X_multi = sm.add_constant(health_df_scaled[metric_cols])
model_multi = sm.OLS(health_df_scaled['avg_CDS'], X_multi).fit()
print("\nMultivariate regression summary:")
print(model_multi.summary())

# ======================== Helper Functions ========================
def add_regression_line(ax, x_data, y_data, label_prefix='', color='red', linestyle='--'):
    """Add regression line with R² and p-value annotation"""
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    if len(x_clean) < 3:
        return
    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()
    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_range = model.params['const'] + model.params[x_clean.name] * x_range
    p_val = model.pvalues[x_clean.name]
    r2 = model.rsquared
    ax.plot(x_range, y_range, color=color, linestyle=linestyle, linewidth=2,
            label=f"{label_prefix}slope={model.params[x_clean.name]:.3f}, $R^2$={r2:.3f}, p={p_val:.4f}")

# ======================== Plotting ========================

# ---- 1. City ranking bar plot ----
fig, ax = plt.subplots(figsize=(12, 6))
sorted_df = health_df.sort_values('avg_CDS')
colors_bar = ['#2ecc71' if i < 3 else '#f1c40f' if i < 6 else '#e74c3c' for i in range(len(sorted_df))]
ax.bar(sorted_df.index, sorted_df['avg_CDS'], color=colors_bar, edgecolor='black', linewidth=0.5)
ax.axhline(health_df['avg_CDS'].mean(), color='gray', linestyle='--', linewidth=1.5, label='Global mean')
ax.set_ylabel('Avg. Transfer Loss (avg_CDS)')
ax.set_xlabel('Source City')
ax.set_title('Source City Ranking by Transferability (best→worst)')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_avg_CDS_ranking.png", dpi=300)
plt.close()

# ---- 2. Best univariate metric scatter ----
best_metric = df_uni.iloc[0]['Metric']
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df[best_metric], health_df['avg_CDS'], s=80, alpha=0.7, c='steelblue', edgecolors='black')
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, best_metric], health_df.loc[city, 'avg_CDS']),
                fontsize=8, xytext=(5,5), textcoords='offset points')
add_regression_line(ax, health_df[best_metric], health_df['avg_CDS'])
ax.set_xlabel(best_metric)
ax.set_ylabel('Avg. Transfer Loss (avg_CDS)')
ax.set_title(f'Best Single Metric vs. Transferability: {best_metric}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_health_metric_scatter.png", dpi=300)
plt.close()

# ---- 3. Self-prediction vs Cross-domain (now using corrected R2_self) ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df['R2_self'], health_df['avg_CDS'], s=80, alpha=0.7, c='darkorange', edgecolors='black')
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'R2_self'], health_df.loc[city, 'avg_CDS']),
                fontsize=8, xytext=(5,5), textcoords='offset points')
if len(health_df['R2_self'].unique()) > 1:
    add_regression_line(ax, health_df['R2_self'], health_df['avg_CDS'], color='red')
else:
    ax.text(0.5, 0.9, 'R2_self too flat for regression', transform=ax.transAxes, ha='center')
ax.set_xlabel('Self-prediction Score (Entropy-weighted)')
ax.set_ylabel('Cross-domain avg. CDS (lower is better)')
ax.set_title('Self-prediction vs. Cross-domain Transferability')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "self_vs_cross_transfer.png", dpi=300)
plt.close()

# ---- 4. Boxplots of health metrics ----
n_cols = 3
n_rows = (len(metric_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()
for i, col in enumerate(metric_cols):
    sns.boxplot(y=health_df[col], ax=axes[i], color='lightblue', fliersize=4)
    axes[i].set_title(col, fontsize=12)
    axes[i].set_ylabel('')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "health_metrics_boxplots.png", dpi=300)
plt.close()

# ---- 5. Correlation heatmap ----
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix: Health Metrics vs. Transferability')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# ---- 6. Neighbor Consistency vs avg_CDS (NY excluded) ----
df_nc = health_df[health_df.index != 'newyork'].copy()
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_nc['neighbor_consistency'], df_nc['avg_CDS'], s=80, alpha=0.7, c='#3498db', edgecolors='black')
for city in df_nc.index:
    ax.annotate(city, (df_nc.loc[city, 'neighbor_consistency'], df_nc.loc[city, 'avg_CDS']),
                fontsize=8, xytext=(5,5), textcoords='offset points')
add_regression_line(ax, df_nc['neighbor_consistency'], df_nc['avg_CDS'], label_prefix='(NY excluded) ', color='red')
ax.set_xlabel('Neighbor Consistency')
ax.set_ylabel('avg_CDS')
ax.set_title('Neighbor Consistency vs. Transfer Loss (NY excluded)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "neighbor_consistency_vs_CDS_noNY.png", dpi=300)
plt.close()

# ---- 7. Effective Rank vs avg_CDS ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df['effective_rank'], health_df['avg_CDS'], s=80, alpha=0.7, c='#27ae60', edgecolors='black')
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'effective_rank'], health_df.loc[city, 'avg_CDS']),
                fontsize=8, xytext=(5,5), textcoords='offset points')
add_regression_line(ax, health_df['effective_rank'], health_df['avg_CDS'])
ax.set_xlabel('Effective Rank')
ax.set_ylabel('avg_CDS')
ax.set_title('Effective Rank vs. Transfer Loss')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "effective_rank_vs_CDS.png", dpi=300)
plt.close()

# ---- 8. Number of nodes vs avg_CDS (colored by neighbor consistency) ----
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(health_df['n_nodes'], health_df['avg_CDS'],
                c=health_df['neighbor_consistency'], cmap='viridis', s=80, edgecolors='black', alpha=0.7)
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'n_nodes'], health_df.loc[city, 'avg_CDS']),
                fontsize=8, xytext=(5,5), textcoords='offset points')
cbar = plt.colorbar(sc)
cbar.set_label('Neighbor Consistency')
ax.set_xlabel('Number of Nodes (City Size)')
ax.set_ylabel('avg_CDS')
ax.set_title('City Size vs. Transfer Loss (colored by Neighbor Consistency)')
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "n_nodes_vs_CDS_colored.png", dpi=300)
plt.close()

# ---- 9. City size group boxplot ----
health_df['size_group'] = pd.cut(health_df['n_nodes'], bins=3, labels=['Small (<200)', 'Medium (200-1000)', 'Large (>1000)'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='size_group', y='avg_CDS', data=health_df, palette='Set2', ax=ax)
ax.set_xlabel('City Size (Number of Nodes)')
ax.set_ylabel('Average Transfer Loss (avg_CDS)')
ax.set_title('City Size vs. Transferability')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "size_group_boxplot.png", dpi=300)
plt.close()

# ---- 10. Radar chart: Good vs Bad teachers ----
sorted_by_cds = health_df.sort_values('avg_CDS')
good_cities = sorted_by_cds.head(3).index.tolist()
bad_cities = sorted_by_cds.tail(3).index.tolist()
radar_metrics = ['effective_rank', 'isotropy', 'neighbor_consistency', 'inter_intra_ratio', 'pca_5_explained']
radar_metrics = [m for m in radar_metrics if m in health_df.columns]
df_radar = health_df[radar_metrics].copy()
df_radar = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min() + 1e-10)

angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for group, cities, color in [('Good Teachers', good_cities, 'blue'), ('Bad Teachers', bad_cities, 'red')]:
    values = df_radar.loc[cities].mean().values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=group, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Good vs Bad Teachers: Embedding Health Radar')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "radar_good_vs_bad.png", dpi=300)
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}")
print("Generated plots:")
print("1. City ranking bar plot")
print("2. Best single metric scatter")
print("3. Self-prediction vs cross-domain (using entropy-weighted Score)")
print("4. Health metrics boxplots")
print("5. Correlation heatmap")
print("6. Neighbor consistency vs avg_CDS (NY excluded)")
print("7. Effective rank vs avg_CDS")
print("8. City size vs avg_CDS (colored by neighbor consistency)")
print("9. City size group boxplot")
print("10. Good vs Bad teachers radar chart")