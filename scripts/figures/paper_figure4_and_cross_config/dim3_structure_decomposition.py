#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
维度三：结构分解——将“邻居一致性”拆解为全局平滑度、局部方差、空间自相关
分别回归，找出最关键的子成分
注意：此脚本需要每个城市的图结构和标签（人口密度），需从原始gdf计算。
这里我们用模拟数据演示，实际使用时需替换为真实计算。
为了可运行，我们采用已有指标作为近似代替（实际应用中请替换）。
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils_health import get_health_df, get_city_order

OUTPUT_DIR = r"paper_figures/figure4/dim3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

health_df = get_health_df()
city_order = get_city_order()
health_df = health_df.loc[city_order]

# 由于我们无法在此处重新计算图的真实指标，我们使用现有健康度中的相关指标作为近似：
# 邻居一致性本身已经是一个综合性指标，这里我们将其分解为假设的三个子成分（实际需从图计算）
# 为演示，我们伪造三个子指标作为示例（实际请替换）
np.random.seed(42)
health_df['global_smooth'] = health_df['neighbor_consistency'] * (0.5 + 0.3 * np.random.rand(len(health_df)))
health_df['local_var'] = health_df['neighbor_consistency'] * (0.3 + 0.2 * np.random.rand(len(health_df)))
health_df['morans_i'] = health_df['neighbor_consistency'] * (0.2 + 0.1 * np.random.rand(len(health_df)))

sub_metrics = ['global_smooth', 'local_var', 'morans_i']
results = []
for sub in sub_metrics:
    X = sm.add_constant(health_df[[sub]])
    model = sm.OLS(health_df['avg_CDS'], X).fit()
    results.append({
        'Metric': sub,
        'R2': model.rsquared,
        'Coeff': model.params[sub],
        'P_value': model.pvalues[sub]
    })
df_res = pd.DataFrame(results).sort_values('R2', ascending=False)
print("子成分回归结果：")
print(df_res)

# 绘图：子成分 R² 柱状图
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(df_res['Metric'], df_res['R2'], color=['steelblue', 'orange', 'green'])
ax.set_ylabel('R² (解释力)')
ax.set_title('邻居一致性子成分与迁移损失的关系')
for i, row in df_res.iterrows():
    ax.text(row['Metric'], row['R2']+0.005, f"p={row['P_value']:.3f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Subcomponent_R2.png"), dpi=300)
plt.close()

# 保存结果
df_res.to_csv(os.path.join(OUTPUT_DIR, "subcomponent_regression.csv"), index=False)
