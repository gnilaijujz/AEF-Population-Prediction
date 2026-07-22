#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 GIS 特征计算得到的各距离矩阵，使用熵权法合成综合域偏移指数 (CDSI)。
假设距离矩阵文件名为 gis_{metric}_matrix.csv，
并放置在指定目录中。
"""

import os
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from pathlib import Path

# ======================== 配置 ========================
MATRIX_DIR = r"results/transfer_results/gis"   # 距离矩阵所在目录
OUT_DIR = r"results/domain_shift/gis/cdsi_results"              # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 定义要读取的指标列表（按文件名后缀）
# 这些文件名应与 with_gis.py 输出一致
METRIC_NAMES = [
    'L1',
    'L2',
    'CosineDist',
    'MMD',
    'CORAL',
    'KLDiv',
    'Spectral'
]

# 若有某些指标希望反转方向（如余弦距离，值越小越相似，已经是距离，不需要反转）
# 所有指标都为“越大表示偏移越大”，所以无需反转。

# ======================== 读取所有距离矩阵 ========================
matrix_dict = {}  # {metric: pd.DataFrame}
cities_set = set()

for metric in METRIC_NAMES:
    file_path = Path(MATRIX_DIR) / f"gis_{metric}_matrix.csv"
    if not file_path.exists():
        print(f"警告: 文件 {file_path} 不存在，跳过该指标")
        continue
    mat = pd.read_csv(file_path, index_col=0)
    matrix_dict[metric] = mat
    cities_set.update(mat.index)
    cities_set.update(mat.columns)
    print(f"已读取 {metric} 矩阵，形状 {mat.shape}")

if not matrix_dict:
    raise RuntimeError("未读取到任何矩阵文件，请检查 MATRIX_DIR 路径和文件名匹配。")

cities = sorted(cities_set)
print(f"共有 {len(cities)} 个城市: {cities}")

# 确保所有矩阵的城市顺序一致（按 cities 排序）
for metric in matrix_dict.keys():
    matrix_dict[metric] = matrix_dict[metric].reindex(index=cities, columns=cities)

# ======================== 构成长格式数据 ========================
long_data = []
for src, tgt in itertools.product(cities, repeat=2):
    if src == tgt:
        continue
    row = {'Source': src, 'Target': tgt}
    valid = True
    for metric, mat in matrix_dict.items():
        val = mat.loc[src, tgt]
        if pd.isna(val):
            valid = False
            break
        row[metric] = val
    if valid:
        long_data.append(row)

long_df = pd.DataFrame(long_data)
print(f"有效城市对样本数: {len(long_df)}")
if long_df.empty:
    raise ValueError("没有完整的城市对数据，请检查矩阵中是否包含 NaN 太多。")

# ======================== 熵权法计算权重 ========================
print("\n===== 使用熵权法计算 CDSI =====")

# 数据正向化与标准化（所有指标已为距离，无需反转）
data_raw = long_df[METRIC_NAMES].copy()

# 稳健标准化：Z-score 后 Min-Max 至 [0,1]
z_scored = data_raw.apply(zscore, nan_policy='omit')
min_vals = z_scored.min()
max_vals = z_scored.max()
norm_data = (z_scored - min_vals) / (max_vals - min_vals + 1e-12)
norm_data = norm_data.clip(lower=1e-10)   # 避免 log(0)

# 熵权法函数
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

# 保存权重
weights_df = pd.DataFrame({'指标': weights.index, '权重': weights.values})
weights_df.to_csv(f"{OUT_DIR}/entropy_weights.csv", index=False)

# ---- 计算 CDSI ----
long_df['CDSI_raw'] = 0.0
for col in weights.index:
    long_df['CDSI_raw'] += norm_data[col] * weights[col]

# 映射至 [0,1]
min_cdsi = long_df['CDSI_raw'].min()
max_cdsi = long_df['CDSI_raw'].max()
long_df['CDSI'] = (long_df['CDSI_raw'] - min_cdsi) / (max_cdsi - min_cdsi + 1e-12)

# 构建 CDSI 矩阵
cdsi_mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
for _, row in long_df.iterrows():
    src, tgt = row['Source'], row['Target']
    cdsi_mat.loc[src, tgt] = row['CDSI']
# 对角线置 0
for c in cities:
    cdsi_mat.loc[c, c] = 0.0

cdsi_mat.to_csv(f"{OUT_DIR}/domain_shift_CDSI_matrix.csv")
print(f"CDSI 矩阵已保存至 {OUT_DIR}/domain_shift_CDSI_matrix.csv")

# ======================== 可视化热力图 ========================
try:
    # 选取几个代表性指标 + CDSI 绘制热力图
    plot_metrics = ['L2', 'MMD', 'CORAL', 'Spectral', 'CDSI']
    # 检查哪些指标存在
    valid_plot = [m for m in plot_metrics if m in matrix_dict or m == 'CDSI']
    n = len(valid_plot)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1:
            axes = [axes]
        for ax, metric in zip(axes, valid_plot):
            if metric == 'CDSI':
                mat = cdsi_mat
                label = 'CDSI'
            else:
                mat = matrix_dict[metric]
                label = metric
            sns.heatmap(mat, annot=True, fmt='.2f', cmap='Reds',
                        square=True, cbar_kws={'label': label}, ax=ax,
                        annot_kws={'size': 7})
            ax.set_title(label)
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/domain_shift_heatmaps.png", dpi=300)
        plt.close()
        print(f"热力图已保存至 {OUT_DIR}/domain_shift_heatmaps.png")
    else:
        print("没有可绘制的指标，跳过热力图。")
except Exception as e:
    print(f"绘图跳过: {e}")

print("\n所有结果已生成。")
