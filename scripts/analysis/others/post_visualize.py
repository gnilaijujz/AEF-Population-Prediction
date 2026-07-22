#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终归因可视化脚本：针对 L2 距离与校准后 R² 的显著相关性，
生成论文级图表，并突出异常城市对。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# -------------------------- 中文字体设置 --------------------------
# 设置中文字体（根据系统字体调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set(font='SimHei')  # 若系统中无 SimHei，可改为 'Microsoft YaHei'
# ---------- 配置 ----------
R2_MATRIX_FILE = "results/transfer_results/aef/transfer_matrix_r2_gnn.csv"
DIST_FILE = "results/transfer_results/aef/transfer_embedding_distances_gnn.csv"
OUTPUT_PREFIX = "results/legacy_analysis/attribute/attribute_final"

# ---------- 读取数据 ----------
df_r2 = pd.read_csv(R2_MATRIX_FILE, index_col=0)
df_dist = pd.read_csv(DIST_FILE)

# 将 R² 转为长格式
df_long = df_r2.stack().reset_index()
df_long.columns = ["Source", "Target", "R2_cal"]
df_long = df_long.dropna(subset=["R2_cal"])

# 合并距离
df_merged = df_long.merge(df_dist, on=["Source", "Target"], how="inner")
print(f"合并后有效样本数: {len(df_merged)}")

# ---------- 1. 计算并比较各度量相关性 ----------
metrics = {
    "L1_Dist": "L1 均值差",
    "L2_Dist": "L2 均值差",
    "Cos_Sim": "余弦相似度",
    "MMD": "MMD"  # 如果存在
}
# 只保留存在的列
metrics = {k: v for k, v in metrics.items() if k in df_merged.columns}

corr_results = []
for col, label in metrics.items():
    rho, p = spearmanr(df_merged[col], df_merged["R2_cal"])
    corr_results.append({"度量": label, "Spearman ρ": rho, "p值": p, "样本数": len(df_merged)})

df_corr = pd.DataFrame(corr_results)
df_corr["显著"] = df_corr["p值"].apply(lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns")))
print("\n各度量 Spearman 相关系数:")
print(df_corr.to_string(index=False))

# ---------- 2. IQR 异常识别（针对 R²） ----------
vals = df_merged["R2_cal"].values
Q1 = np.percentile(vals, 25)
Q3 = np.percentile(vals, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_merged["IsOutlier"] = (vals < lower) | (vals > upper)
print(f"异常城市对数量: {df_merged['IsOutlier'].sum()} ({100*df_merged['IsOutlier'].sum()/len(df_merged):.1f}%)")

# 保存异常对列表
outlier_df = df_merged[df_merged["IsOutlier"]][["Source", "Target", "R2_cal", "L2_Dist"]]
outlier_df.to_csv(f"{OUTPUT_PREFIX}_outlier_pairs.csv", index=False)

# ---------- 3. 图表 1: 相关性对比柱状图 ----------
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df_corr["度量"], df_corr["Spearman ρ"], 
              color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(df_corr)])
for bar, p, rho in zip(bars, df_corr["p值"], df_corr["Spearman ρ"]):
    y = bar.get_height()
    if y > 0:
        y_pos = y + 0.02
    else:
        y_pos = y - 0.08
    if p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    elif p < 0.05:
        star = "*"
    else:
        star = "ns"
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, star, ha='center', va='bottom', fontsize=12)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel("Spearman 相关系数 ρ")
ax.set_title("各距离/相似度度量与校准后 R² 的相关性")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_correlation_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 4. 图表 2: L2 距离散点图（区分异常） ----------
fig, ax = plt.subplots(figsize=(8, 6))
normal = df_merged[~df_merged["IsOutlier"]]
outlier = df_merged[df_merged["IsOutlier"]]
ax.scatter(normal["L2_Dist"], normal["R2_cal"], c="blue", label="正常", alpha=0.6, s=40)
ax.scatter(outlier["L2_Dist"], outlier["R2_cal"], c="red", label="异常 (IQR)", alpha=0.8, s=80, marker="x")

reg_normal = linregress(normal["L2_Dist"], normal["R2_cal"])
x_range = np.linspace(normal["L2_Dist"].min(), normal["L2_Dist"].max(), 100)
y_pred = reg_normal.intercept + reg_normal.slope * x_range
ax.plot(x_range, y_pred, color="green", linestyle="--",
        label=f"拟合线 (剔除异常, R²={reg_normal.rvalue**2:.3f})")

reg_all = linregress(df_merged["L2_Dist"], df_merged["R2_cal"])
x_all = np.linspace(df_merged["L2_Dist"].min(), df_merged["L2_Dist"].max(), 100)
y_all = reg_all.intercept + reg_all.slope * x_all
ax.plot(x_all, y_all, color="gray", linestyle=":", label=f"全量 (R²={reg_all.rvalue**2:.3f})")

ax.set_xlabel("L2 均值距离")
ax.set_ylabel("校准后 R² (真实迁移)")
ax.set_title("L2 距离 vs 迁移性能")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_L2_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 5. 图表 3: L2 距离分箱趋势图 ----------
df_merged["L2_bin"] = pd.qcut(df_merged["L2_Dist"], q=10, labels=False)
bin_stats = df_merged.groupby("L2_bin").agg(
    mean_r2=("R2_cal", "mean"),
    std_r2=("R2_cal", "std"),
    count=("R2_cal", "count")
).reset_index()
bin_medians = df_merged.groupby("L2_bin")["L2_Dist"].median().values
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(bin_medians, bin_stats["mean_r2"], yerr=bin_stats["std_r2"],
            fmt='o-', capsize=5, color='darkblue', ecolor='gray')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel("L2 距离 (分箱中位数)")
ax.set_ylabel("平均校准后 R²")
ax.set_title("L2 距离与迁移性能的非线性趋势")
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_L2_binned.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 6. 图表 4: 异常城市对矩阵热图 ----------
df_heat = df_r2.copy()
# 创建布尔矩阵标记异常位置
outlier_mask_matrix = pd.DataFrame(False, index=df_heat.index, columns=df_heat.columns)
for src, tgt in zip(outlier_df["Source"], outlier_df["Target"]):
    if src in outlier_mask_matrix.index and tgt in outlier_mask_matrix.columns:
        outlier_mask_matrix.loc[src, tgt] = True

df_heat_masked = df_heat.copy()
df_heat_masked[outlier_mask_matrix] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(df_heat, ax=axes[0], cmap="RdBu_r", center=0, annot=False, cbar=True,
            square=True, xticklabels=True, yticklabels=True)
axes[0].set_title("全量迁移 R² (真实迁移)")

sns.heatmap(df_heat_masked, ax=axes[1], cmap="RdBu_r", center=0, annot=False, cbar=True,
            square=True, xticklabels=True, yticklabels=True)
axes[1].set_title("剔除异常值后 (白色为异常)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 7. 异常列表表格 ----------
if len(outlier_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(outlier_df)*0.4)))
    ax.axis('tight')
    ax.axis('off')
    table_data = outlier_df[["Source", "Target", "R2_cal", "L2_Dist"]].values
    col_labels = ["源城市", "目标城市", "校准后 R²", "L2 距离"]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center', colColours=["#f5f5f5"]*4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.title("异常城市对列表 (IQR 法)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_outlier_table.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"\n所有图表已保存，前缀 {OUTPUT_PREFIX}")
print("生成文件：")
print(f"  - {OUTPUT_PREFIX}_correlation_bar.png")
print(f"  - {OUTPUT_PREFIX}_L2_scatter.png")
print(f"  - {OUTPUT_PREFIX}_L2_binned.png")
print(f"  - {OUTPUT_PREFIX}_heatmap.png")
print(f"  - {OUTPUT_PREFIX}_outlier_table.png")