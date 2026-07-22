#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
维度五：异常挖掘——识别实际迁移性能与预测值偏差较大的城市，进行地理归因
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils_health import get_health_df, get_city_order

OUTPUT_DIR = r"paper_figures/figure4/dim5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

health_df = get_health_df()
city_order = get_city_order()
health_df = health_df.loc[city_order]

# 用所有健康度指标预测 avg_CDS
features = ['neighbor_consistency', 'effective_rank', 'isotropy', 'inter_intra_ratio', 'pca_5_explained']
X = health_df[features]
y = health_df['avg_CDS']
model = LinearRegression().fit(X, y)
preds = model.predict(X)
residuals = y - preds  # 残差 = 实际 - 预测，正值表示实际迁移损失大于预测（即实际性能比预测差）

health_df['residual'] = residuals

# 找出残差最大（最差）和最小（最好）的城市
anomaly_df = health_df[['avg_CDS', 'residual'] + features].copy()
anomaly_df['city'] = anomaly_df.index
anomaly_df = anomaly_df.sort_values('residual', ascending=False)  # 最差异常在前

# 绘制残差散点图（标注极端值）
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(preds, y, s=80, alpha=0.7)
# 标注异常点（残差绝对值最大的3个）
top_anomalies = anomaly_df.head(3).index.tolist() + anomaly_df.tail(3).index.tolist()
for city in top_anomalies:
    ax.annotate(city, (preds[health_df.index.get_loc(city)], y[health_df.index.get_loc(city)]),
                xytext=(5,5), textcoords='offset points', fontsize=9, color='red')
min_val = min(min(preds), min(y))
max_val = max(max(preds), max(y))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 理想线')
ax.set_xlabel('预测 avg_CDS')
ax.set_ylabel('实际 avg_CDS')
ax.set_title('异常城市标注 (残差绝对值大的城市)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Anomaly_Scatter.png"), dpi=300)
plt.close()

# 输出异常城市表
anomaly_table = anomaly_df[['city', 'avg_CDS', 'residual'] + features].head(5)
anomaly_table.to_csv(os.path.join(OUTPUT_DIR, "anomaly_cities.csv"), index=False)

# 绘制残差与各指标的平行坐标图（选择几个关键指标）
fig, ax = plt.subplots(figsize=(10, 6))
# 将残差分为三组：高残差（差）、中、低残差（好）
health_df['residual_group'] = pd.qcut(health_df['residual'], q=3, labels=['Good', 'Medium', 'Bad'])
group_means = health_df.groupby('residual_group')[['neighbor_consistency', 'effective_rank', 'isotropy']].mean()
group_means.T.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
ax.set_ylabel('平均值')
ax.set_title('不同残差组的健康度指标均值对比')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Residual_Group_Profile.png"), dpi=300)
plt.close()

print("[维度五] 异常挖掘完成，异常城市列表已保存。")
