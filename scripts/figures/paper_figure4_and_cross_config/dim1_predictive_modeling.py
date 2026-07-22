#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成果4 综合分析与可视化（修正自预测得分、中文字体、$R^2$ 渲染）
输出所有图表至 paper_figures/figure4_embedding_health/figure4_panels/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from utils_health import get_health_df, get_city_order, get_config_paths, load_matrix, safe_linregress

# ==================== 配置 ====================
CONFIG_NAME = 'AEF'                # 可选: 'AEF', 'Buildings', 'GIS' 等
OUTPUT_BASE = r"paper_figures\figure4_embedding_health\figure4_panels"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ---------- 强制指定中文字体（Windows 通用） ----------
# 方法1：直接指定黑体（Windows 标配）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==================== 加载数据 ====================
health_df = get_health_df(CONFIG_NAME)
city_order = get_city_order(CONFIG_NAME)
health_df = health_df.loc[city_order]

# 加载域偏移矩阵（用于调节效应，此处仅演示，未使用）
# paths = get_config_paths(CONFIG_NAME)
# l2_mat = load_matrix(paths['l2_path'])

# ==================== 1. 源城市平均迁移损失排序 ====================
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if i < 3 else 'orange' if i < 6 else 'red' for i in range(len(health_df))]
ax.bar(health_df.index, health_df['avg_CDS'], color=colors)
ax.axhline(health_df['avg_CDS'].mean(), color='gray', linestyle='--', label='均值')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_xlabel('源城市')
ax.set_title('源城市平均可迁移性排序（好→差）')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "1_Avg_CDS_Ranking.png"), dpi=300)
plt.close()

# ==================== 2. 跨目标迁移性能箱线图 ====================
# 需从CDS矩阵读取每个源城市的所有目标值
cds_path = os.path.join(r"results", "transferability", "aef_plus_building" if CONFIG_NAME=='Buildings' else CONFIG_NAME, "CDS_matrix_robust_entropy.csv")
cds_mat = pd.read_csv(cds_path, index_col=0)
box_data = [cds_mat.loc[city].drop(index=city, errors='ignore').dropna().values for city in health_df.index if city in cds_mat.index]

fig, ax = plt.subplots(figsize=(14, 6))
bp = ax.boxplot(box_data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
ax.set_xticklabels(health_df.index, rotation=45, ha='right')
ax.axhline(health_df['avg_CDS'].mean(), color='red', linestyle='--', label='总体均值')
ax.set_ylabel('迁移损失 (CDS)')
ax.set_xlabel('源城市')
ax.set_title('各源城市跨目标迁移性能分布')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "2_CDS_Boxplot.png"), dpi=300)
plt.close()

# ==================== 3. 自预测综合得分 vs 跨域损失 ====================
fig, ax = plt.subplots(figsize=(8, 8))
for city in health_df.index:
    ax.scatter(health_df.loc[city, 'Self_Score'], health_df.loc[city, 'avg_CDS'], s=80)
    ax.annotate(city, (health_df.loc[city, 'Self_Score'], health_df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=8)
slope, intercept, r, p, _ = safe_linregress(health_df['Self_Score'], health_df['avg_CDS'])
if not np.isnan(slope):
    x_vals = np.linspace(health_df['Self_Score'].min(), health_df['Self_Score'].max(), 50)
    ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'$R^2$={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('自预测综合得分 (熵权法 Score)')
ax.set_ylabel('平均跨域迁移损失 (avg_CDS)')
ax.set_title('源域自预测能力 vs 跨域迁移能力')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "3_Self_vs_Cross.png"), dpi=300)
plt.close()

# ==================== 4. 邻居一致性 vs avg_CDS（核心图） ====================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df['neighbor_consistency'], health_df['avg_CDS'], s=80)
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'neighbor_consistency'], health_df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=8)
slope, intercept, r, p, _ = safe_linregress(health_df['neighbor_consistency'], health_df['avg_CDS'])
x_vals = np.linspace(health_df['neighbor_consistency'].min(), health_df['neighbor_consistency'].max(), 50)
ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'$R^2$={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('邻居一致性')
ax.set_ylabel('avg_CDS')
ax.set_title('邻居一致性 vs 迁移损失')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "4_NeighborConsistency_vs_CDS.png"), dpi=300)
plt.close()

# ==================== 5. 有效秩 vs avg_CDS ====================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df['effective_rank'], health_df['avg_CDS'], s=80)
slope, intercept, r, p, _ = safe_linregress(health_df['effective_rank'], health_df['avg_CDS'])
x_vals = np.linspace(health_df['effective_rank'].min(), health_df['effective_rank'].max(), 50)
ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'$R^2$={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('有效秩')
ax.set_ylabel('avg_CDS')
ax.set_title('有效秩 vs 迁移损失')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "5_EffectiveRank_vs_CDS.png"), dpi=300)
plt.close()

# ==================== 6. 节点数分箱箱线图 ====================
health_df['size_group'] = pd.cut(health_df['n_nodes'], bins=3, labels=['小型(<200)', '中型(200-1000)', '大型(>1000)'])
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='size_group', y='avg_CDS', data=health_df, ax=ax)
ax.set_xlabel('城市规模 (节点数)')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_title('城市规模与迁移性能的关系')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "6_Size_Boxplot.png"), dpi=300)
plt.close()

# ==================== 7. 节点数 vs avg_CDS（颜色=邻居一致性） ====================
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(health_df['n_nodes'], health_df['avg_CDS'], c=health_df['neighbor_consistency'], cmap='viridis', s=80)
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'n_nodes'], health_df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=8)
ax.set_xlabel('节点数')
ax.set_ylabel('avg_CDS')
ax.set_title('城市规模 vs 迁移损失 (颜色表示邻居一致性)')
cbar = plt.colorbar(sc)
cbar.set_label('邻居一致性')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "7_Size_vs_CDS_colored.png"), dpi=300)
plt.close()

# ==================== 8. 散点图矩阵（所有健康度指标） ====================
cols = ['n_nodes', 'effective_rank', 'isotropy', 'neighbor_consistency', 'inter_intra_ratio', 'pca_5_explained', 'avg_CDS']
sns.pairplot(health_df[cols], diag_kind='kde')
plt.savefig(os.path.join(OUTPUT_BASE, "8_Pairplot.png"), dpi=300)
plt.close()

# ==================== 9. 健康度指标与 avg_CDS 相关性热力图 ====================
corr = health_df[['effective_rank', 'isotropy', 'neighbor_consistency', 'inter_intra_ratio', 'pca_5_explained', 'avg_CDS']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, ax=ax)
ax.set_title('健康度指标与迁移损失的相关性')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "9_Correlation_Heatmap.png"), dpi=300)
plt.close()

# ==================== 10. 好老师 vs 差老师雷达图 ====================
sorted_by_cds = health_df.sort_values('avg_CDS')
good_cities = sorted_by_cds.head(3).index.tolist()
bad_cities = sorted_by_cds.tail(3).index.tolist()
radar_metrics = ['effective_rank', 'isotropy', 'neighbor_consistency', 'inter_intra_ratio', 'pca_5_explained']
df_radar = health_df[radar_metrics].copy()
df_radar = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min() + 1e-10)

angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for group, cities, color in [('好老师', good_cities, 'blue'), ('差老师', bad_cities, 'red')]:
    values = df_radar.loc[cities].mean().values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=group, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_ylim(0, 1)
ax.set_title('"好老师" vs "差老师" 嵌入健康度雷达图')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, "10_Radar_Good_vs_Bad.png"), dpi=300)
plt.close()

print("成果4 所有图表已生成至:", OUTPUT_BASE)
