#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高水平论文风格可视化（修正版）
直接从原始矩阵文件读取数据，不依赖中间缓存。
输出：paperfigure/ 目录下的独立图表（英文标题、无子图）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ======================== 用户配置（请根据实际目录修改）========================
# 1. CDS 矩阵文件（迁移损失）
CDS_PATH = r"results/transferability/nighttime_lights\CDS_matrix_robust_entropy.csv"

# 2. 域偏移矩阵所在目录（应包含 domain_shift_*_matrix.csv）
MATRICES_DIR = r"results/domain_shift/aef_plus_nighttime_lights/with_gis"

# 3. 从 regression_analysis.py 输出的单变量排名和变量重要性 CSV（如果不存，可跳过相应图）
UNIVARIATE_CSV = r"results/regression/domain_shift_analysis/nighttime_lights\mix_loss_univariate_ranking.csv"
IMPORTANCE_CSV = r"results/regression/domain_shift_analysis/nighttime_lights\mix_loss_variable_importance.csv"

# 4. 输出目录
OUT_DIR = "paperfigure"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================== 全局绘图风格 ========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ======================== 辅助函数：加载 CDS 与指定偏移矩阵 ========================
def load_cds_and_shift(shift_metric=None):
    """
    加载 CDS 矩阵和（可选）单个偏移矩阵。
    如果 shift_metric 为 None，则仅返回 CDS 长格式。
    如果指定，则合并对应的偏移列。
    """
    # 加载 CDS
    cds = pd.read_csv(CDS_PATH, index_col=0)
    cds_long = cds.stack().reset_index()
    cds_long.columns = ["Source", "Target", "CDS_raw"]
    cds_long = cds_long.dropna()
    cds_long['Source'] = cds_long['Source'].astype(str).str.strip()
    cds_long['Target'] = cds_long['Target'].astype(str).str.strip()

    if shift_metric is None:
        return cds_long

    # 加载指定偏移矩阵
    fname = os.path.join(MATRICES_DIR, f"domain_shift_{shift_metric}_matrix.csv")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"文件不存在: {fname}")
    mat = pd.read_csv(fname, index_col=0)
    mat.index = mat.index.astype(str).str.strip()
    mat.columns = mat.columns.astype(str).str.strip()
    long = mat.stack().reset_index()
    long.columns = ["Source", "Target", shift_metric]
    long['Source'] = long['Source'].astype(str).str.strip()
    long['Target'] = long['Target'].astype(str).str.strip()

    # 合并
    merged = cds_long.merge(long, on=["Source", "Target"], how="inner")
    return merged

def load_multiple_shifts(metrics_list):
    """
    加载 CDS 和多个偏移指标，返回包含所有指标的 DataFrame。
    """
    merged = load_cds_and_shift(None)  # 只有 CDS
    for metric in metrics_list:
        fname = os.path.join(MATRICES_DIR, f"domain_shift_{metric}_matrix.csv")
        if not os.path.exists(fname):
            print(f"警告: {fname} 不存在，跳过 {metric}")
            continue
        mat = pd.read_csv(fname, index_col=0)
        mat.index = mat.index.astype(str).str.strip()
        mat.columns = mat.columns.astype(str).str.strip()
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", metric]
        long['Source'] = long['Source'].astype(str).str.strip()
        long['Target'] = long['Target'].astype(str).str.strip()
        merged = merged.merge(long, on=["Source", "Target"], how="inner")
    return merged

# ======================== 1. 单变量排名柱状图 ========================
def plot_univariate_ranking(csv_path):
    if not os.path.exists(csv_path):
        print(f"跳过 {csv_path}，文件不存在")
        return
    df = pd.read_csv(csv_path)
    top10 = df.head(10).copy().sort_values('R2', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top10['Variable'], top10['R2'], color='steelblue', edgecolor='black', linewidth=0.5)
    for bar, r2 in zip(bars, top10['R2']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{r2:.3f}', va='center', ha='left', fontsize=9)
    ax.set_xlabel('Univariate R²', fontsize=12)
    ax.set_ylabel('Domain shift metric', fontsize=12)
    ax.set_title('Top-10 univariate predictors of transfer loss', fontsize=14, fontweight='bold')
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig1_univariate_ranking.png'), dpi=300)
    plt.close()
    print("Saved Fig1_univariate_ranking.png")

# ======================== 2. 变量重要性柱状图 ========================
def plot_variable_importance(csv_path):
    if not os.path.exists(csv_path):
        print(f"跳过 {csv_path}，文件不存在")
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values('Coefficient', key=abs, ascending=False)
    sig = df['P_value'] < 0.05

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df['Variable'], df['Coefficient'],
                  color=['#2E86AB' if p else '#D3D3D3' for p in sig],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Standardized coefficient (β)', fontsize=12)
    ax.set_xlabel('Domain shift metric', fontsize=12)
    ax.set_title('Variable importance in the best multivariate model', fontsize=14, fontweight='bold')
    for bar, p in zip(bars, df['P_value']):
        if p < 0.001:
            label = '***'
        elif p < 0.01:
            label = '**'
        elif p < 0.05:
            label = '*'
        else:
            label = 'n.s.'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(abs(df['Coefficient'])),
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig2_variable_importance.png'), dpi=300)
    plt.close()
    print("Saved Fig2_variable_importance.png")

# ======================== 3. 最佳单变量散点图（含置信带） ========================
def plot_best_single_scatter(univariate_csv):
    if not os.path.exists(univariate_csv):
        print(f"跳过散点图，{univariate_csv} 不存在")
        return
    # 读取最佳变量
    df_uni = pd.read_csv(univariate_csv)
    if df_uni.empty:
        print("单变量排名为空，跳过散点图")
        return
    best_var = df_uni.iloc[0]['Variable']

    # 加载 CDS 与该变量数据
    try:
        df = load_cds_and_shift(best_var)
    except FileNotFoundError as e:
        print(f"加载数据失败: {e}")
        return
    if len(df) == 0:
        print("合并后无数据，跳过散点图")
        return

    # 因变量：绝对损失
    df['y'] = np.abs(df['CDS_raw'])

    # 异常值剔除（IQR）
    vals = df['y'].values
    Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df_clean = df[(vals >= lower) & (vals <= upper)].copy()
    print(f"散点图数据：原始 {len(df)} 行，剔除异常后 {len(df_clean)} 行")

    # 标准化自变量
    scaler = StandardScaler()
    df_clean[best_var] = scaler.fit_transform(df_clean[[best_var]])

    x = df_clean[best_var].values
    y = df_clean['y'].values

    # OLS 回归
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params[best_var]
    pval = model.pvalues[best_var]
    r2 = model.rsquared

    # 置信带
    from scipy.stats import t
    n = len(x)
    dof = n - 2
    t_val = t.ppf(0.975, dof)
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = model.predict(X_pred)
    mse = np.sum((y - model.predict(X))**2) / dof
    X_design = sm.add_constant(x)
    cov = mse * np.linalg.inv(X_design.T @ X_design)
    se_pred = np.sqrt(np.diag(X_pred @ cov @ X_pred.T))
    upper = y_pred + t_val * se_pred
    lower = y_pred - t_val * se_pred

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.6, s=30, color='#2E86AB', edgecolor='k', linewidth=0.3)
    ax.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'β = {slope:.3f} (p = {pval:.3g})')
    ax.fill_between(x_pred, lower, upper, color='red', alpha=0.15, label='95% CI')
    ax.set_xlabel(f'{best_var} (standardized)', fontsize=12)
    ax.set_ylabel('|CDS| (absolute transfer loss)', fontsize=12)
    ax.set_title(f'Best single predictor: {best_var} (R²={r2:.3f})', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig3_best_single_scatter.png'), dpi=300)
    plt.close()
    print("Saved Fig3_best_single_scatter.png")

# ======================== 4. 分层回归（自动选择存在的指标） ========================
def plot_hierarchical_regression():
    # 自动发现所有偏移矩阵（去除 CDS 自身）
    all_files = glob.glob(os.path.join(MATRICES_DIR, "domain_shift_*_matrix.csv"))
    if not all_files:
        print("没有找到任何偏移矩阵，跳过分层回归")
        return
    # 提取指标名
    all_metrics = [os.path.basename(f).replace("domain_shift_", "").replace("_matrix.csv", "") for f in all_files]
    # 选择几个有代表性的指标（例如 L2_Dist, MMD, CORAL, Spectral_Dist 如果存在）
    preferred = ['L2_Dist', 'MMD', 'CORAL', 'Spectral_Dist', 'KL_Div', 'L1_Dist']
    selected = [m for m in preferred if m in all_metrics]
    if not selected:
        # 如果都没有，就用前4个
        selected = all_metrics[:4]
    print(f"分层回归选用指标: {selected}")

    # 加载数据
    df = load_multiple_shifts(selected)
    if len(df) == 0:
        print("加载数据为空，跳过分层回归")
        return

    df['y'] = np.abs(df['CDS_raw'])
    # 剔除缺失
    df = df.dropna(subset=['y'] + selected)
    # 标准化
    scaler = StandardScaler()
    df[selected] = scaler.fit_transform(df[selected])

    # 逐层加入变量
    models = {}
    for i in range(1, len(selected)+1):
        vars_used = selected[:i]
        X = sm.add_constant(df[vars_used])
        model = sm.OLS(df['y'], X).fit()
        models[f'M{i}'] = model

    # ---- 图4a: R² 增量 ----
    labels = [f'M{i}' for i in range(1, len(selected)+1)]
    r2_vals = [models[m].rsquared for m in models]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, r2_vals, color='#1B4F72', edgecolor='black', linewidth=0.5)
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('R² (explained variance)', fontsize=12)
    ax.set_xlabel('Model (cumulative variables)', fontsize=12)
    ax.set_title('Hierarchical regression: incremental R²', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(r2_vals)*1.2 if max(r2_vals)>0 else 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig4a_hierarchical_R2.png'), dpi=300)
    plt.close()
    print("Saved Fig4a_hierarchical_R2.png")

    # ---- 图4b: 最终模型系数 ----
    final_model = models[f'M{len(selected)}']
    coefs = final_model.params[selected]
    pvals = final_model.pvalues[selected]
    sig = pvals < 0.05
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(selected, coefs,
                  color=['#2E86AB' if s else '#D3D3D3' for s in sig],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    for bar, p, var in zip(bars, pvals, selected):
        if p < 0.001:
            label = '***'
        elif p < 0.01:
            label = '**'
        elif p < 0.05:
            label = '*'
        else:
            label = 'n.s.'
        ypos = bar.get_height() + np.sign(bar.get_height())*0.02*max(abs(coefs))
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                label, ha='center', va='bottom' if bar.get_height()>0 else 'top',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Standardized coefficient (β)', fontsize=12)
    ax.set_xlabel('Domain shift component', fontsize=12)
    ax.set_title('Final model coefficients', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Fig4b_hierarchical_coefficients.png'), dpi=300)
    plt.close()
    print("Saved Fig4b_hierarchical_coefficients.png")

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 检查必要的目录
    if not os.path.exists(CDS_PATH):
        print(f"错误: CDS 文件不存在 {CDS_PATH}，请修改 CDS_PATH 变量")
    elif not os.path.exists(MATRICES_DIR):
        print(f"错误: 矩阵目录不存在 {MATRICES_DIR}，请修改 MATRICES_DIR 变量")
    else:
        # 绘制单变量排名
        plot_univariate_ranking(UNIVARIATE_CSV)
        # 绘制变量重要性
        plot_variable_importance(IMPORTANCE_CSV)
        # 绘制最佳单变量散点图
        plot_best_single_scatter(UNIVARIATE_CSV)
        # 绘制分层回归
        plot_hierarchical_regression()

    print(f"\n所有图表已保存至 {OUT_DIR}/ 目录。")