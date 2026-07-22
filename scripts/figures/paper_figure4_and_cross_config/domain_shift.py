#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为每个域偏移指标矩阵和 CDSI 矩阵单独绘制热力图。
要求：
- 输入目录包含 domain_shift_*.csv 文件（由 domain_shifting.py 生成）
- 输出图片保存至同一目录下的 plots/ 子目录
- 每个指标一张图，标题为英文
"""

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ======================== 配置 ========================
INPUT_DIR = r"results/domain_shift/aef"   # 与 domain_shifting.py 的 OUT_DIR 一致
OUTPUT_DIR = os.path.join(INPUT_DIR, "plots")         # 图片输出子目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指标名称映射（用于图标题）
TITLE_MAP = {
    'L2_Dist': 'L2 Distance Matrix',
    'MMD': 'Maximum Mean Discrepancy Matrix',
    'CORAL': 'CORAL Distance Matrix',
    'KL_Div': 'KL Divergence Matrix',
    'Spectral_Dist': 'Spectral Distance Matrix',
    'Degree_Diff': 'Degree Difference Matrix',
    'L1_Dist': 'L1 Distance Matrix',
    'Cos_Dist': 'Cosine Distance Matrix',
    'CDSI': 'Comprehensive Domain Shift Index (CDSI)'
}

# 颜色映射（可选，这里统一使用 'Reds'，也可按需调整）
CMAP = 'Reds'

# ======================== 读取所有矩阵文件 ========================
matrix_files = glob.glob(os.path.join(INPUT_DIR, "domain_shift_*.csv"))
if not matrix_files:
    raise FileNotFoundError(f"在 {INPUT_DIR} 中未找到任何 domain_shift_*.csv 文件")

# 读取并存储为字典 {指标名: DataFrame}
matrices = {}
for file_path in matrix_files:
    filename = os.path.basename(file_path)
    # 提取指标名：去掉 'domain_shift_' 和 '_matrix.csv' 或 'CDSI_matrix.csv'
    if 'CDSI_matrix' in filename:
        metric = 'CDSI'
    else:
        # 如 domain_shift_L2_Dist_matrix.csv -> L2_Dist
        base = filename.replace('domain_shift_', '').replace('_matrix.csv', '')
        metric = base
    df = pd.read_csv(file_path, index_col=0)
    # 确保数值类型为 float（可能含 NaN）
    df = df.astype(float)
    matrices[metric] = df
    print(f"读取: {metric} ({df.shape[0]}x{df.shape[1]})")

# ======================== 绘制每张图 ========================
for metric, mat in matrices.items():
    # 检查是否存在 NaN，若存在则填充为 0 或保留（seaborn 可处理 NaN，但建议填充）
    if mat.isna().any().any():
        print(f"警告: {metric} 矩阵含有 NaN，将用 0 填充（仅用于可视化）")
        mat = mat.fillna(0)

    # 生成标题
    title = TITLE_MAP.get(metric, f"{metric} Matrix")
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat,
        annot=True,          # 显示数值
        fmt='.2f',           # 保留两位小数
        cmap=CMAP,
        square=True,
        cbar_kws={'label': title},
        annot_kws={'size': 8},
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Target City', fontsize=12)
    plt.ylabel('Source City', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, f"{metric}_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

print("\n所有热力图绘制完成。")