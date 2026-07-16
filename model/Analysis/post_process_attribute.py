#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归因分析增强版：计算多种距离指标，并测试 Spearman 相关性，
自动生成更丰富的诊断图表。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, linregress
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ----- 配置 -----
R2_MATRIX_FILE = r"GNN_output/transfer_results_experiments/transfer_matrix_r2_null_calibrated.csv"
EMB_DIST_FILE = r"GNN_output/transfer_results_experiments/transfer_embedding_distances_gnn.csv"  # 由 run_single_transfer 生成
OUTPUT_PREFIX = "GNN_output/attribute/attribution"

# ----- 1. 读取数据 -----
df_r2 = pd.read_csv(R2_MATRIX_FILE, index_col=0)
df_long = df_r2.stack().reset_index()
df_long.columns = ["Source", "Target", "R2_cal"]
df_long = df_long.dropna(subset=["R2_cal"])

# 加载嵌入距离（必须存在）
try:
    df_emb = pd.read_csv(EMB_DIST_FILE)
    df_emb = df_emb.rename(columns={"source": "Source", "target": "Target"})
    # 合并
    df_merged = df_long.merge(df_emb, on=["Source", "Target"], how="inner")
    print(f"合并后有效样本数: {len(df_merged)}")
except FileNotFoundError:
    print(f"错误：未找到嵌入距离文件 {EMB_DIST_FILE}，请先运行迁移实验生成。")
    exit()

# 检查距离列名（可能是 EmbeddingDistance 或 distance）
dist_col = None
for col in df_merged.columns:
    if 'distance' in col.lower() or 'dist' in col.lower():
        dist_col = col
        break
if dist_col is None:
    raise ValueError("未找到距离列，请检查 CSV 列名。")

# ----- 2. 计算多种相关性 -----
# 2.1 Pearson 线性回归（全量）
reg_all = linregress(df_merged[dist_col], df_merged["R2_cal"])
# 2.2 Spearman 秩相关
spear_all, p_spear = spearmanr(df_merged[dist_col], df_merged["R2_cal"])

print("=== 全量数据相关性 ===")
print(f"Pearson r = {reg_all.rvalue:.4f}, p = {reg_all.pvalue:.4e}, R² = {reg_all.rvalue**2:.4f}")
print(f"Spearman ρ = {spear_all:.4f}, p = {p_spear:.4e}")

# ----- 3. 异常值剔除后再次计算（IQR）-----
# 使用之前定义的 IQR 剔除
vals = df_merged["R2_cal"].values
Q1 = np.percentile(vals, 25)
Q3 = np.percentile(vals, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
mask_normal = (vals > lower) & (vals < upper)
df_normal = df_merged[mask_normal]

reg_norm = linregress(df_normal[dist_col], df_normal["R2_cal"])
spear_norm, p_spear_norm = spearmanr(df_normal[dist_col], df_normal["R2_cal"])

print("\n=== 剔除异常后相关性 ===")
print(f"Pearson r = {reg_norm.rvalue:.4f}, p = {reg_norm.pvalue:.4e}, R² = {reg_norm.rvalue**2:.4f}")
print(f"Spearman ρ = {spear_norm:.4f}, p = {p_spear_norm:.4e}")

# 保存结果
summary = pd.DataFrame({
    "数据集": ["全量", "剔除异常"],
    "Pearson_r": [reg_all.rvalue, reg_norm.rvalue],
    "Pearson_p": [reg_all.pvalue, reg_norm.pvalue],
    "Spearman_rho": [spear_all, spear_norm],
    "Spearman_p": [p_spear, p_spear_norm],
})
summary.to_csv(f"{OUTPUT_PREFIX}_correlation_summary.csv", index=False)

# ----- 4. 可视化 -----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：散点图 + 线性回归（全量 + 剔除异常）
ax = axes[0]
ax.scatter(df_merged[dist_col], df_merged["R2_cal"], alpha=0.5, label="全量")
# 标记异常点
outlier_mask = ~mask_normal
if outlier_mask.sum() > 0:
    ax.scatter(df_merged[outlier_mask][dist_col], df_merged[outlier_mask]["R2_cal"],
               facecolors='none', edgecolors='red', s=80, label="异常 (IQR)")

# 全量回归线
x_range = np.linspace(df_merged[dist_col].min(), df_merged[dist_col].max(), 100)
y_all = reg_all.intercept + reg_all.slope * x_range
ax.plot(x_range, y_all, 'b--', label=f"Pearson r={reg_all.rvalue:.3f}")

# 剔除异常回归线（如果样本足够）
if len(df_normal) > 1:
    x_norm = np.linspace(df_normal[dist_col].min(), df_normal[dist_col].max(), 100)
    y_norm = reg_norm.intercept + reg_norm.slope * x_norm
    ax.plot(x_norm, y_norm, 'g-', label=f"剔除异常 r={reg_norm.rvalue:.3f}")

ax.set_xlabel("嵌入距离 (MMD)")
ax.set_ylabel("校准后 R² (零模型)")
ax.legend()
ax.grid(True)

# 右：残差图（检查线性假设）
ax = axes[1]
residuals = df_merged["R2_cal"] - (reg_all.intercept + reg_all.slope * df_merged[dist_col])
ax.scatter(df_merged[dist_col], residuals, alpha=0.5)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("嵌入距离 (MMD)")
ax.set_ylabel("残差")
ax.set_title("残差图")
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_scatter_residual.png", dpi=300, bbox_inches="tight")
plt.close()

# ----- 5. 尝试非线性关系：分箱分析 -----
df_merged['dist_bin'] = pd.qcut(df_merged[dist_col], q=10, labels=False)
bin_stats = df_merged.groupby('dist_bin').agg(
    mean_r2=('R2_cal', 'mean'),
    std_r2=('R2_cal', 'std'),
    count=('R2_cal', 'count')
).reset_index()
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(bin_stats['dist_bin'], bin_stats['mean_r2'], yerr=bin_stats['std_r2'],
            fmt='o-', capsize=5, color='blue')
ax.set_xlabel("距离分箱 (等频)")
ax.set_ylabel("平均校准后 R²")
ax.set_title("嵌入距离与迁移性能的非线性趋势")
ax.grid(True)
plt.savefig(f"{OUTPUT_PREFIX}_binned_trend.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n归因分析图表已保存，前缀 {OUTPUT_PREFIX}")