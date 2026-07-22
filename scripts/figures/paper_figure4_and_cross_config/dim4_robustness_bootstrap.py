#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
维度四：稳健性检验——通过Bootstrap重采样评估好老师排名的稳定性
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_health import get_city_order, load_matrix

OUTPUT_DIR = r"paper_figures/figure4/dim4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载CDS矩阵
cds_path = r"results/transferability/aef_plus_building/CDS_matrix_robust_entropy.csv"
cds_mat = pd.read_csv(cds_path, index_col=0)
city_order = get_city_order()

# 原始 avg_CDS 排序
avg_cds_orig = {}
for city in city_order:
    vals = cds_mat.loc[city].drop(index=city, errors='ignore').dropna().values
    avg_cds_orig[city] = np.mean(vals) if len(vals) > 0 else np.nan
orig_rank = pd.Series(avg_cds_orig).sort_values().index.tolist()

# Bootstrap: 每次随机抽取 80% 的目标城市（约12个），计算 avg_CDS，重复 1000 次
n_bootstrap = 1000
rank_matrix = []  # 存储每次抽样的排名
for _ in range(n_bootstrap):
    # 随机选择目标城市（不放回抽样）
    target_subset = np.random.choice(cds_mat.columns, size=int(0.8*len(cds_mat.columns)), replace=False)
    avg_cds_tmp = {}
    for city in city_order:
        vals = cds_mat.loc[city, target_subset].drop(index=city, errors='ignore').dropna().values
        avg_cds_tmp[city] = np.mean(vals) if len(vals) > 0 else np.nan
    # 按值排序，记录排名（越小越好）
    sorted_cities = pd.Series(avg_cds_tmp).sort_values().index.tolist()
    rank_dict = {city: i+1 for i, city in enumerate(sorted_cities)}
    rank_matrix.append(rank_dict)

# 统计每个城市的平均排名和排名标准差
rank_df = pd.DataFrame(rank_matrix)
rank_stats = rank_df.mean().sort_values()
rank_std = rank_df.std()

# 绘制排名稳定性图（误差线）
fig, ax = plt.subplots(figsize=(12, 6))
top_cities = rank_stats.head(10).index.tolist()
x_pos = np.arange(len(top_cities))
mean_ranks = rank_stats[top_cities].values
std_ranks = rank_std[top_cities].values
ax.bar(x_pos, mean_ranks, yerr=std_ranks, capsize=5, color='steelblue', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(top_cities, rotation=45, ha='right')
ax.set_ylabel('平均排名 (越小越好)')
ax.set_title('Bootstrap 重采样排名稳定性 (80% 目标城市子集)')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Ranking_Stability.png"), dpi=300)
plt.close()

# 保存排名统计
rank_stats.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_rank_means.csv"))
rank_std.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_rank_stds.csv"))
print("[维度四] 排名稳定性分析完成，图片已保存。")
