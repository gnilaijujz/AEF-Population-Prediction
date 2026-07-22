#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成果4：嵌入空间特征与可迁移性关系
生成所有指定图表（稳健版，兼容所有异常）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BASE_DIR = r"result"
OUTPUT_DIR = r"paper_figures\figure4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")
import matplotlib.pyplot as plt

# 方法1：直接指定（Windows 常用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ==================== 辅助函数：安全回归 ====================
def safe_linregress(x, y):
    """
    安全的线性回归，当 x 或 y 方差为零时返回 (nan, nan, nan, nan, nan)
    """
    x_clean = x.dropna()
    y_clean = y.dropna()
    # 对齐索引
    common_idx = x_clean.index.intersection(y_clean.index)
    if len(common_idx) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    x_vals = x_clean.loc[common_idx]
    y_vals = y_clean.loc[common_idx]
    if np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return linregress(x_vals, y_vals)

# ==================== 加载数据 ====================
health_df = pd.read_csv(os.path.join(BASE_DIR, "embedding_health_deep_analysis", "embedding_health_deep_metrics.csv"), index_col=0)
cds_mat = pd.read_csv(os.path.join(BASE_DIR, "transferability", "aef_plus_building", "CDS_matrix_robust_entropy.csv"), index_col=0)

# 计算 avg_CDS 和 CDS_std
avg_cds = {}
cds_std = {}
for city in cds_mat.index:
    row = cds_mat.loc[city]
    vals = row.drop(index=city, errors='ignore').dropna().values
    if len(vals) > 0:
        avg_cds[city] = np.mean(vals)
        cds_std[city] = np.std(vals)
    else:
        avg_cds[city] = np.nan
        cds_std[city] = np.nan

health_df['avg_CDS'] = health_df.index.map(lambda x: avg_cds.get(x, np.nan))
health_df['CDS_std'] = health_df.index.map(lambda x: cds_std.get(x, np.nan))
health_df['R2_self'] = health_df.index.map(lambda x: cds_mat.loc[x, x] if not np.isnan(cds_mat.loc[x, x]) else np.nan)

# 按 avg_CDS 升序排序
city_order = health_df.sort_values('avg_CDS', ascending=True).index.tolist()
health_df = health_df.loc[city_order]

# ==================== 1. 源城市排序柱状图 ====================
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if i < 3 else 'orange' if i < 6 else 'red' for i in range(len(city_order))]
ax.bar(city_order, health_df['avg_CDS'], color=colors)
ax.axhline(health_df['avg_CDS'].mean(), color='gray', linestyle='--', label='平均值')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_xlabel('源城市')
ax.set_title('源城市平均可迁移性排序')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_1_Avg_CDS_Ranking.png"), dpi=300)
plt.close()

# ==================== 2. 箱线图 ====================
box_data = []
for city in city_order:
    vals = cds_mat.loc[city].drop(index=city, errors='ignore').dropna().values
    box_data.append(vals if len(vals) > 0 else [np.nan])

fig, ax = plt.subplots(figsize=(14, 6))
bp = ax.boxplot(box_data, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                boxprops=dict(facecolor='lightblue'))
ax.axhline(health_df['avg_CDS'].mean(), color='red', linestyle='--', label='总体均值')
ax.set_ylabel('迁移损失 (CDS)')
ax.set_xlabel('源城市')
ax.set_title('各源城市跨目标迁移性能分布')
ax.set_xticklabels(city_order, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_2_CDS_Boxplot.png"), dpi=300)
plt.close()

# ==================== 3. 自预测 vs 跨域 ====================
fig, ax = plt.subplots(figsize=(8, 8))
for city in city_order:
    if not np.isnan(health_df.loc[city, 'R2_self']) and not np.isnan(health_df.loc[city, 'avg_CDS']):
        ax.scatter(health_df.loc[city, 'R2_self'], health_df.loc[city, 'avg_CDS'], s=100)
        ax.annotate(city, (health_df.loc[city, 'R2_self'], health_df.loc[city, 'avg_CDS']),
                    xytext=(5,5), textcoords='offset points', fontsize=9)

# 回归线（使用 safe_linregress）
slope, intercept, r, p, se = safe_linregress(health_df['R2_self'], health_df['avg_CDS'])
if not np.isnan(slope):
    x_vals = np.linspace(health_df['R2_self'].min(), health_df['R2_self'].max(), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, 'r--', label=f'趋势线 ($R$2={r**2:.3f})')
else:
    ax.text(0.5, 0.9, '回归线不可用（数据方差为零）', transform=ax.transAxes, ha='center')
ax.set_xlabel('自预测 $R$2')
ax.set_ylabel('平均跨域迁移损失 (avg_CDS)')
ax.set_title('源城市自预测能力 vs 跨域迁移能力')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_3_Self_vs_Cross.png"), dpi=300)
plt.close()

# ==================== 4. 热力图（含汇总值） ====================
cds_sorted = cds_mat.reindex(index=city_order, columns=city_order)
row_means = cds_sorted.mean(axis=1)
col_means = cds_sorted.mean(axis=0)
cds_with_summary = cds_sorted.copy()
cds_with_summary['源均值'] = row_means
cds_with_summary.loc['目标均值'] = col_means

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cds_with_summary, cmap='RdBu_r', center=0, square=True,
            cbar_kws={'label': '迁移损失 (CDS)'}, ax=ax,
            annot=True, fmt='.2f', annot_kws={'size': 6})
ax.set_title('可迁移性模糊矩阵（行=源城市，列=目标城市）\n右侧和底部为汇总均值')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_4_CDS_Heatmap_With_Summary.png"), dpi=300)
plt.close()

# ==================== 5. 有效秩排序柱状图 ====================
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if i < 3 else 'orange' if i < 6 else 'red' for i in range(len(city_order))]
ax.bar(city_order, health_df['effective_rank'], color=colors)
ax.set_ylabel('有效秩 (Effective Rank)')
ax.set_xlabel('源城市')
ax.set_title('源城市嵌入空间有效秩排序')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_5_Effective_Rank_Ranking.png"), dpi=300)
plt.close()

# ==================== 6. 有效秩 vs avg_CDS ====================
fig, ax = plt.subplots(figsize=(8, 8))
for city in city_order:
    ax.scatter(health_df.loc[city, 'effective_rank'], health_df.loc[city, 'avg_CDS'], s=100)
    ax.annotate(city, (health_df.loc[city, 'effective_rank'], health_df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=9)

slope, intercept, r, p, se = safe_linregress(health_df['effective_rank'], health_df['avg_CDS'])
if not np.isnan(slope):
    x_vals = np.linspace(health_df['effective_rank'].min(), health_df['effective_rank'].max(), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, 'r--', label=f'$R$2={r**2:.3f}, p={p:.4f}')
else:
    ax.text(0.5, 0.9, '回归线不可用', transform=ax.transAxes, ha='center')
ax.set_xlabel('有效秩')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_title('有效秩 vs 平均迁移性能')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_6_EffectiveRank_vs_CDS.png"), dpi=300)
plt.close()

# ==================== 7. 各向同性 vs 迁移方差 ====================
fig, ax = plt.subplots(figsize=(8, 8))
for city in city_order:
    if not np.isnan(health_df.loc[city, 'isotropy']) and not np.isnan(health_df.loc[city, 'CDS_std']):
        ax.scatter(health_df.loc[city, 'isotropy'], health_df.loc[city, 'CDS_std'], s=100)
        ax.annotate(city, (health_df.loc[city, 'isotropy'], health_df.loc[city, 'CDS_std']),
                    xytext=(5,5), textcoords='offset points', fontsize=9)

slope, intercept, r, p, se = safe_linregress(health_df['isotropy'], health_df['CDS_std'])
if not np.isnan(slope):
    x_vals = np.linspace(health_df['isotropy'].min(), health_df['isotropy'].max(), 100)
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, 'r--', label=f'$R$2={r**2:.3f}, p={p:.4f}')
else:
    ax.text(0.5, 0.9, '回归线不可用', transform=ax.transAxes, ha='center')
ax.set_xlabel('各向同性得分 (1/条件数)')
ax.set_ylabel('迁移性能标准差 (CDS_std)')
ax.set_title('各向同性 vs 迁移稳定性')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_7_Isotropy_vs_Stability.png"), dpi=300)
plt.close()

# ==================== 8. PCA累计方差柱状图 ====================
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(city_order, health_df['pca_5_explained'], color='steelblue')
ax.set_ylabel('前5主成分累计方差解释率')
ax.set_xlabel('源城市')
ax.set_title('各源城市嵌入空间低维结构清晰度')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_8_PCA_Cumulative_Bar.png"), dpi=300)
plt.close()

# ==================== 9. t-SNE 可视化（若嵌入文件存在） ====================
try:
    emb_dir = r"results/embeddings"
    good_cities = city_order[:3]
    bad_cities = city_order[-3:]
    selected_cities = good_cities + bad_cities
    embeddings = []
    labels = []
    for city in selected_cities:
        emb_path = os.path.join(emb_dir, f"embedding_{city}.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            if emb.shape[0] > 200:
                idx = np.random.choice(emb.shape[0], 200, replace=False)
                emb = emb[idx]
            embeddings.append(emb)
            labels.extend([city] * emb.shape[0])
        else:
            print(f"跳过 {city}: 嵌入文件不存在")
            raise FileNotFoundError
    if embeddings:
        X = np.vstack(embeddings)
        pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        fig, ax = plt.subplots(figsize=(10, 8))
        for city in selected_cities:
            mask = np.array(labels) == city
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=city, alpha=0.7, s=10)
        ax.set_title('好老师 vs 差老师嵌入空间 t-SNE投影')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_9_tSNE_Good_vs_Bad.png"), dpi=300)
        plt.close()
    else:
        print("t-SNE跳过：无嵌入数据")
except Exception as e:
    print(f"t-SNE图跳过: {e}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

X = health_df[['neighbor_consistency', 'effective_rank', 'isotropy', 'inter_intra_ratio', 'pca_5_explained']]
y = health_df['avg_CDS']

loo = LeaveOneOut()
preds, actuals = [], []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = LinearRegression().fit(X_train, y_train)
    preds.append(model.predict(X_test)[0])
    actuals.append(y_test.values[0])

# 绘图
fig, ax = plt.subplots()
ax.scatter(actuals, preds)
ax.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
ax.set_xlabel('实际 avg_CDS'); ax.set_ylabel('预测 avg_CDS')
ax.set_title(f'LOO 交叉验证 (R² = {r2_score(actuals, preds):.3f})')

import statsmodels.api as sm

# 合并数据：每个源城市的 avg_CDS、平均 L2 距离、邻居一致性
l2_mat = pd.read_csv(...)  # Weighted_L2 矩阵
avg_l2 = {city: l2_mat.loc[city].drop(city).mean() for city in city_order}
health_df['avg_L2'] = health_df.index.map(avg_l2)

X = sm.add_constant(health_df[['avg_L2', 'neighbor_consistency']])
X['interaction'] = X['avg_L2'] * X['neighbor_consistency']
model = sm.OLS(health_df['avg_CDS'], X).fit()
print(model.summary())

print("所有图表生成完成！")
