#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后处理脚本：对校准后的 R² 矩阵进行 IQR 异常剔除，
并执行“嵌入差异 vs 迁移性”回归分析，生成论文图表。
可同时处理零模型 (null) 和真实迁移 (gnn)。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# -------------------------- 中文字体设置 --------------------------
# 设置中文字体（根据系统字体调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set(font='SimHei')  # 若系统中无 SimHei，可改为 'Microsoft YaHei'


# -------------------------- 配置 --------------------------
# 请根据实际文件名修改，这里假设文件名遵循 transfer_matrix_r2_{mode}_calibrated.csv
# 其中 mode 可以是 'null' 或 'gnn'
MODES = {
    "null": r"results/transfer_results/aef/transfer_matrix_r2.csv",   # 零模型
    "gnn": r"results/transfer_results/aef/transfer_matrix_r2_gnn.csv"      # 真实迁移（如果有）
}
EMBEDDING_DIST_FILE = r"results/transfer_results/aef/transfer_embedding_distances_gnn.csv"  # 嵌入距离文件（由主脚本生成）
OUTPUT_PREFIX = "transfer_analysis"

# 读取所有可用矩阵
dfs = {}
for mode, fname in MODES.items():
    try:
        df = pd.read_csv(fname, index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        dfs[mode] = df
        print(f"Loaded {mode} matrix: {fname}")
    except FileNotFoundError:
        print(f"Warning: {fname} not found, skipping {mode}.")

if len(dfs) == 0:
    raise FileNotFoundError("No R² matrix files found.")

# 读取嵌入距离
try:
    df_emb = pd.read_csv(EMBEDDING_DIST_FILE)
    # 确保列名标准化
    df_emb = df_emb.rename(columns={"Source": "Source", "Target": "Target", "EmbeddingDistance": "EmbeddingDistance"})
    has_embedding = True
except FileNotFoundError:
    print(f"Warning: Embedding distance file {EMBEDDING_DIST_FILE} not found. Skipping regression.")
    has_embedding = False

# -------------------------- 主分析 --------------------------
for mode, df_r2 in dfs.items():
    print(f"\n=== Analyzing {mode} ===")
    vals = df_r2.values.flatten()
    vals = vals[~np.isnan(vals)]
    n_total = len(vals)
    
    # IQR 异常识别
    Q1 = np.percentile(vals, 25)
    Q3 = np.percentile(vals, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outlier_mask = (df_r2 < lower) | (df_r2 > upper)
    outlier_mask = outlier_mask.fillna(False)
    n_outliers = outlier_mask.sum().sum()
    n_normal = n_total - n_outliers
    print(f"异常城市对数: {n_outliers} ({100*n_outliers/n_total:.1f}%)")
    print(f"正常城市对数: {n_normal} ({100*n_normal/n_total:.1f}%)")
    
    # 异常城市对列表
    outlier_pairs = []
    for src in df_r2.index:
        for tgt in df_r2.columns:
            if outlier_mask.loc[src, tgt]:
                outlier_pairs.append((src, tgt, df_r2.loc[src, tgt]))
    if outlier_pairs:
        outlier_df = pd.DataFrame(outlier_pairs, columns=["Source", "Target", "R2_cal"])
        outlier_df.to_csv(f"{OUTPUT_PREFIX}_{mode}_outlier_pairs.csv", index=False)
    
    # 统计摘要
    normal_vals = vals[~((vals < lower) | (vals > upper))]
    stats_all = {
        "样本数": len(vals),
        "均值": np.mean(vals),
        "中位数": np.median(vals),
        "标准差": np.std(vals),
        "最小值": np.min(vals),
        "最大值": np.max(vals),
    }
    stats_normal = {
        "样本数": len(normal_vals),
        "均值": np.mean(normal_vals),
        "中位数": np.median(normal_vals),
        "标准差": np.std(normal_vals),
        "最小值": np.min(normal_vals),
        "最大值": np.max(normal_vals),
    }
    summary_df = pd.DataFrame([stats_all, stats_normal], index=["全量", "剔除异常"])
    summary_df.to_csv(f"{OUTPUT_PREFIX}_{mode}_summary_stats.csv")
    print("\n统计摘要:")
    print(summary_df)
    
    # 长格式转换
    df_long = df_r2.stack().reset_index()
    df_long.columns = ["Source", "Target", "R2_cal"]
    df_long["IsOutlier"] = df_long.apply(
        lambda row: outlier_mask.loc[row["Source"], row["Target"]], axis=1
    )
    df_long = df_long.dropna(subset=["R2_cal"])
    
    # 如果模式是 gnn，且有嵌入距离，则做回归
    if mode == "gnn" and has_embedding and "EmbeddingDistance" in df_emb.columns:
        df_merged = df_long.merge(df_emb, on=["Source", "Target"], how="left")
        df_reg = df_merged.dropna(subset=["EmbeddingDistance", "R2_cal"])
        if len(df_reg) > 2:
            # 全量回归
            reg_all = linregress(df_reg["EmbeddingDistance"], df_reg["R2_cal"])
            # 剔除异常回归
            df_reg_normal = df_reg[~df_reg["IsOutlier"]]
            reg_normal = linregress(df_reg_normal["EmbeddingDistance"], df_reg_normal["R2_cal"])
            print("\n=== 回归分析（嵌入差异 vs 校准后 R²） ===")
            print("全量数据: R² = {:.4f}, p = {:.4e}, 斜率 = {:.4f}".format(
                reg_all.rvalue**2, reg_all.pvalue, reg_all.slope))
            print("剔除异常: R² = {:.4f}, p = {:.4e}, 斜率 = {:.4f}".format(
                reg_normal.rvalue**2, reg_normal.pvalue, reg_normal.slope))
            # 保存回归结果
            reg_results = pd.DataFrame({
                "数据集": ["全量", "剔除异常"],
                "R2": [reg_all.rvalue**2, reg_normal.rvalue**2],
                "p值": [reg_all.pvalue, reg_normal.pvalue],
                "斜率": [reg_all.slope, reg_normal.slope],
                "截距": [reg_all.intercept, reg_normal.intercept],
            })
            reg_results.to_csv(f"{OUTPUT_PREFIX}_{mode}_regression_results.csv", index=False)
        else:
            print("Not enough data for regression.")
    
    # 图表
    plt.rcParams["font.size"] = 10
    # 箱线图
    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = [vals, normal_vals]
    # 兼容不同 matplotlib 版本
    try:
        bp = ax.boxplot(data_to_plot, tick_labels=["全量", "剔除异常"], patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data_to_plot, labels=["全量", "剔除异常"], patch_artist=True)
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("校准后 R²")
    ax.set_title(f"{mode.upper()} - IQR 异常剔除前后对比")
    plt.savefig(f"{OUTPUT_PREFIX}_{mode}_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 热图（掩码异常值为白色）
    df_r2_masked = df_r2.copy()
    df_r2_masked[outlier_mask] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(df_r2, ax=axes[0], cmap="RdBu_r", center=0, cbar=True,
                square=True, xticklabels=True, yticklabels=True)
    axes[0].set_title(f"{mode.upper()} - 全量")
    sns.heatmap(df_r2_masked, ax=axes[1], cmap="RdBu_r", center=0, cbar=True,
                square=True, xticklabels=True, yticklabels=True)
    axes[1].set_title(f"{mode.upper()} - 剔除异常（白色为异常）")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{mode}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 如果模式是 gnn 且有嵌入距离，画散点图
    if mode == "gnn" and has_embedding:
        print(f"\n绘制 {mode} 模式的嵌入差异 vs 校准后 R² 散点图...")
        # 直接合并（假设 df_emb 已定义，且包含 Source, Target, EmbeddingDistance）
        df_plot = df_long.merge(df_emb, on=["Source", "Target"], how="inner")
        df_plot = df_plot.dropna(subset=["EmbeddingDistance", "R2_cal"])
        
        if len(df_plot) > 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            normal_pts = df_plot[~df_plot["IsOutlier"]]
            outlier_pts = df_plot[df_plot["IsOutlier"]]
            ax.scatter(normal_pts["EmbeddingDistance"], normal_pts["R2_cal"],
                    c="blue", label="正常", alpha=0.6, s=30)
            ax.scatter(outlier_pts["EmbeddingDistance"], outlier_pts["R2_cal"],
                    c="red", label="异常", alpha=0.8, s=50, marker="x")
            ax.set_xlabel("嵌入差异 (MMD)")
            ax.set_ylabel("校准后 R²")
            ax.legend()
            # 添加回归线（剔除异常后）
            df_reg_normal = df_plot[~df_plot["IsOutlier"]]
            if len(df_reg_normal) > 1:
                reg = linregress(df_reg_normal["EmbeddingDistance"], df_reg_normal["R2_cal"])
                x_range = np.linspace(df_reg_normal["EmbeddingDistance"].min(),
                                    df_reg_normal["EmbeddingDistance"].max(), 100)
                y_pred = reg.intercept + reg.slope * x_range
                ax.plot(x_range, y_pred, color="green", linestyle="--",
                        label=f"拟合线 (剔除异常, R²={reg.rvalue**2:.3f})")
                ax.legend()
            ax.set_title(f"{mode.upper()} - 嵌入差异 vs 迁移性能")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_PREFIX}_{mode}_scatter.png", dpi=300, bbox_inches="tight")
            print(f"散点图已保存至 {OUTPUT_PREFIX}_{mode}_scatter.png")
        else:
            print("合并后数据不足，无法绘制散点图。")
    else:
        print(f"\n跳过 {mode} 模式的嵌入差异 vs 校准后 R² 散点图绘制。")
print("\n所有分析完成。")