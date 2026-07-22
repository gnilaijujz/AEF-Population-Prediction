#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建域偏移指标矩阵（含熵权法 CDSI）
从 transfer_embedding_distances_gnn.csv 提取各偏移指标，
计算每对城市之间的偏移量，并通过熵权法合成综合域偏移指数 (CDSI)。
输出：各指标矩阵 CSV 及 CDSI 矩阵 CSV。
"""
import os

import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ======================== 配置 ========================
CSV_PATH = r"results/transfer_results/aef\transfer_embedding_distances_gnn.csv"
OUT_DIR = r"results/domain_shift/aef"   # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 需要提取的指标列（依据实际列名）
METRIC_COLS = [
    'L2_Dist',          # 均值 L2
    'MMD',              # 核 MMD
    'CORAL',            # 协方差差
    'KL_Div',           # 高斯 KL
    'Spectral_Dist',    # 图谱距离
    'Degree_Diff',      # 平均度差
    'L1_Dist',          # 均值 L1（可选）
    'Cos_Sim'           # 余弦相似度（>0 表示越相似）
]

# 是否将相似性指标转为距离（1 - Cos_Sim）
CONVERT_SIM_TO_DIST = True

# ======================== 数据读取 ========================
df = pd.read_csv(CSV_PATH)
required_cols = ['Source', 'Target'] + METRIC_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少列: {missing}")

# 转换 Cos_Sim -> Cos_Dist
if 'Cos_Sim' in METRIC_COLS and CONVERT_SIM_TO_DIST:
    df['Cos_Dist'] = 1 - df['Cos_Sim']
    METRIC_COLS = [col if col != 'Cos_Sim' else 'Cos_Dist' for col in METRIC_COLS]
    print("已将 Cos_Sim 转换为 Cos_Dist (1 - Cos_Sim)")

# ======================== 构建每个指标的矩阵 ========================
cities = sorted(set(df['Source'].unique()) | set(df['Target'].unique()))
print(f"发现 {len(cities)} 个城市: {cities}")

matrix_dict = {}   # 存储每个指标的 DataFrame

for metric in METRIC_COLS:
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for src, tgt in itertools.product(cities, repeat=2):
        row = df[(df['Source'] == src) & (df['Target'] == tgt)]
        if len(row) == 1:
            mat.loc[src, tgt] = row.iloc[0][metric]
        else:
            mat.loc[src, tgt] = np.nan
    matrix_dict[metric] = mat
    mat.to_csv(f"{OUT_DIR}/domain_shift_{metric}_matrix.csv")
    print(f"已保存 {metric} 矩阵")

# ======================== 熵权法计算 CDSI ========================
print("\n===== 使用熵权法计算 CDSI =====")

# 收集所有非对角线有效值，构成长格式
long_data = []
for src, tgt in itertools.product(cities, repeat=2):
    if src == tgt:
        continue
    row = {'Source': src, 'Target': tgt}
    for metric in METRIC_COLS:
        val = matrix_dict[metric].loc[src, tgt]
        if not np.isnan(val):
            row[metric] = val
        else:
            row[metric] = np.nan
    long_data.append(row)
long_df = pd.DataFrame(long_data).dropna(subset=METRIC_COLS)
print(f"有效城市对样本数: {len(long_df)}")

# ---- 数据正向化与标准化 ----
# 所有指标已为“越大表示偏移越大”（距离类），因此无需反转方向
data_for_entropy = long_df[METRIC_COLS].copy()

# 稳健标准化：Z-score 后 Min-Max 至 [0,1]，避免量纲影响且对异常值鲁棒
z_scored = data_for_entropy.apply(zscore, nan_policy='omit')
min_vals = z_scored.min()
max_vals = z_scored.max()
norm_data = (z_scored - min_vals) / (max_vals - min_vals + 1e-12)
norm_data = norm_data.clip(lower=1e-10)   # 避免 log(0)

# ---- 熵权法 ----
def entropy_weight(data):
    p = data.div(data.sum(axis=0), axis=1)
    n = len(data)
    k = 1 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    w = d / d.sum()
    return w

weights = entropy_weight(norm_data)
print("\n熵权法权重：")
for col, w in weights.items():
    print(f"  {col}: {w:.4f}")
weights_df = pd.DataFrame({'指标': weights.index, '权重': weights.values})
weights_df.to_csv(f"{OUT_DIR}/entropy_weights.csv", index=False)

# ---- 计算 CDSI ----
long_df['CDSI_raw'] = 0.0
for col in weights.index:
    long_df['CDSI_raw'] += norm_data[col] * weights[col]

# 映射至 [0,1] 便于解释
min_cdsi = long_df['CDSI_raw'].min()
max_cdsi = long_df['CDSI_raw'].max()
long_df['CDSI'] = (long_df['CDSI_raw'] - min_cdsi) / (max_cdsi - min_cdsi + 1e-12)

# 构建 CDSI 矩阵
cdsi_mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
for _, row in long_df.iterrows():
    src, tgt = row['Source'], row['Target']
    cdsi_mat.loc[src, tgt] = row['CDSI']
# 对角线设为 0
for c in cities:
    cdsi_mat.loc[c, c] = 0.0

cdsi_mat.to_csv(f"{OUT_DIR}/domain_shift_CDSI_matrix.csv")
print(f"CDSI 矩阵已保存至 {OUT_DIR}/domain_shift_CDSI_matrix.csv")

# ======================== 可视化热力图（可选） ========================
try:
    plot_metrics = ['L2_Dist', 'MMD', 'CORAL', 'Spectral_Dist', 'CDSI']
    n = len(plot_metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, plot_metrics):
        if metric == 'CDSI':
            mat = cdsi_mat
        else:
            mat = matrix_dict.get(metric)
            if mat is None:
                continue
        sns.heatmap(mat, annot=True, fmt='.2f', cmap='Reds',
                    square=True, cbar_kws={'label': metric}, ax=ax,
                    annot_kws={'size': 7})
        ax.set_title(metric)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/domain_shift_heatmaps.png", dpi=300)
    plt.close()
    print(f"热力图已保存至 {OUT_DIR}/domain_shift_heatmaps.png")
except Exception as e:
    print(f"绘图跳过: {e}")

print("\n所有矩阵已生成完毕。")