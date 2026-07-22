#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
维度二：调节效应——邻居一致性是否缓冲了域偏移（L2距离）的负面影响？
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils_health import get_health_df, get_city_order, get_config_paths, load_matrix

# ===================== 配置 =====================
CONFIG_NAME = 'AEF'  # 修改此处以切换不同特征集
# ===============================================

OUTPUT_DIR = rf"paper_figures/figure4/{CONFIG_NAME}/dim2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
health_df = get_health_df(CONFIG_NAME)
city_order = get_city_order(CONFIG_NAME)
health_df = health_df.loc[city_order]

# 根据配置加载对应的 L2 矩阵
paths = get_config_paths(CONFIG_NAME)
l2_mat = load_matrix(paths['l2_path'])
avg_l2 = {}
for city in city_order:
    if city in l2_mat.index:
        vals = l2_mat.loc[city].drop(index=city, errors='ignore').dropna().values
        avg_l2[city] = np.mean(vals) if len(vals) > 0 else np.nan
health_df['avg_L2'] = health_df.index.map(lambda x: avg_l2.get(x, np.nan))
health_df = health_df.dropna(subset=['avg_L2'])

X = health_df[['avg_L2', 'neighbor_consistency']].copy()
X['interaction'] = X['avg_L2'] * X['neighbor_consistency']
y = health_df['avg_CDS']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print(f"[{CONFIG_NAME}] 交互项系数 = {model.params['interaction']:.4f}, p = {model.pvalues['interaction']:.4f}")
print(model.summary())

# 条件效应图
quantiles = health_df['neighbor_consistency'].quantile([0.25, 0.5, 0.75])
fig, ax = plt.subplots(figsize=(8, 6))
x_vals = np.linspace(health_df['avg_L2'].min(), health_df['avg_L2'].max(), 50)
colors = ['blue', 'green', 'red']; labels = ['低一致性 (Q1)', '中一致性 (Q2)', '高一致性 (Q3)']
b0, b1, b2, b3 = model.params['const'], model.params['avg_L2'], model.params['neighbor_consistency'], model.params['interaction']
for idx, q in enumerate(quantiles):
    cons = quantiles[q]
    y_pred = b0 + b1 * x_vals + b2 * cons + b3 * x_vals * cons
    ax.plot(x_vals, y_pred, color=colors[idx], label=labels[idx], linewidth=2)
ax.set_xlabel('平均域偏移 (avg_L2)'); ax.set_ylabel('预测迁移损失 (avg_CDS)')
ax.set_title(f'[{CONFIG_NAME}] 邻居一致性对域偏移-迁移损失关系的调节效应')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Moderation_Effect_Plot.png"), dpi=300)
plt.close()
