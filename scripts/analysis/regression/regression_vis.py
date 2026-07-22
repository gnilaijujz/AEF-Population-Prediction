#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版回归分析（论文级可视化）
- 自动选择最优域偏移指标（单变量筛选 + 向前逐步回归）
- 分层回归（四变量：Label, AEF, slpoe, Joint）
所有图表：白底、英文、标注 R²、系数、p 值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr, probplot
from sklearn.preprocessing import StandardScaler
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ======================== 全局绘图设置 ========================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ========================== 用户配置 ==========================
CDS_MATRIX_FILE = r"results\transferability\AEF\CDS_matrix_robust_entropy.csv"
MATRICES_DIR = r"results/domain_shift/aef"
OUTPUT_BASE = r"results\regression\domain_shift_analysis\AEF\mix"
os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)

TARGET_STRATEGY = 'loss'   # 'loss', 'abs_loss', 'gain'
AUTO_SELECT = True             # True: 自动选择；False: 使用手动 SELECTED_COLS
SELECTED_COLS = ['L2_Dist', 'Spectral_Dist', 'KL_Div']  # 仅在 AUTO_SELECT=False 时使用

# 分层回归固定变量（需确保矩阵存在）
HIER_VARS = ['Label_Shift', 'AEF_L2', 'GIS_L2', 'Weighted_L2']

# ======================== 第一部分：自动选择模型 ========================
print("\n" + "="*70)
print("PART 1: Automatic Model Selection")
print("="*70)

# ---- 加载 CDS ----
df_cds = pd.read_csv(CDS_MATRIX_FILE, index_col=0)
df_cds_long = df_cds.stack().reset_index()
df_cds_long.columns = ["Source", "Target", "CDS_raw"]
df_cds_long = df_cds_long.dropna(subset=["CDS_raw"])
df_cds_long['Source'] = df_cds_long['Source'].astype(str).str.strip()
df_cds_long['Target'] = df_cds_long['Target'].astype(str).str.strip()

# ---- 发现所有域偏移矩阵 ----
all_files = glob.glob(f"{MATRICES_DIR}/domain_shift_*_matrix.csv")
if not all_files:
    raise FileNotFoundError(f"No domain shift matrices found in {MATRICES_DIR}")
available_cols = []
for fpath in all_files:
    fname = os.path.basename(fpath)
    metric = fname.replace("domain_shift_", "").replace("_matrix.csv", "")
    available_cols.append(metric)
print(f"Available metrics: {available_cols}")

# ---- 合并所有指标 ----
df_merged = df_cds_long.copy()
for col in available_cols:
    fname = f"{MATRICES_DIR}/domain_shift_{col}_matrix.csv"
    try:
        mat = pd.read_csv(fname, index_col=0)
        mat.index = mat.index.astype(str).str.strip()
        mat.columns = mat.columns.astype(str).str.strip()
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", col]
        long['Source'] = long['Source'].astype(str).str.strip()
        long['Target'] = long['Target'].astype(str).str.strip()
        df_merged = df_merged.merge(long, on=["Source", "Target"], how="inner")
        print(f"Loaded {col}, rows: {len(long)}")
    except Exception as e:
        print(f"Failed to load {col}: {e}")

# ---- 定义因变量 ----
if TARGET_STRATEGY == 'loss':
    df_merged['y'] = df_merged['CDS_raw']
    y_label = "CDS (Transfer Loss, lower better)"
    suffix = "_loss"
elif TARGET_STRATEGY == 'abs_loss':
    df_merged['y'] = np.abs(df_merged['CDS_raw'])
    y_label = "|CDS| (Absolute Loss)"
    suffix = "_abs"
elif TARGET_STRATEGY == 'gain':
    df_merged['y'] = -df_merged['CDS_raw']
    y_label = "-CDS (Transfer Retention, higher better)"
    suffix = "_gain"
else:
    raise ValueError("Invalid TARGET_STRATEGY")

OUTPUT_PREFIX_AUTO = f"{OUTPUT_BASE}{suffix}"
os.makedirs(os.path.dirname(OUTPUT_PREFIX_AUTO), exist_ok=True)

# ---- 剔除缺失 ----
df_merged = df_merged.dropna(subset=['y'] + available_cols)
print(f"Effective samples (no missing): {len(df_merged)}")

# ---- 标准化 ----
scaler = StandardScaler()
df_merged[available_cols] = scaler.fit_transform(df_merged[available_cols])

# ---- IQR 异常剔除（仅对 y） ----
vals = df_merged['y'].values
Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_merged['IsOutlier'] = (vals < lower) | (vals > upper)
print(f"Outliers removed: {df_merged['IsOutlier'].sum()} ({100*df_merged['IsOutlier'].sum()/len(df_merged):.1f}%)")
df_clean = df_merged[~df_merged['IsOutlier']].copy()
print(f"Clean samples: {len(df_clean)}")

# ---- 自动选择 ----
if AUTO_SELECT:
    print("\n===== Automatic Model Selection =====")
    # 单变量排名
    univariate_results = []
    for col in available_cols:
        X = sm.add_constant(df_clean[[col]])
        model = sm.OLS(df_clean['y'], X).fit()
        spearman_r, spearman_p = spearmanr(df_clean[col], df_clean['y'])
        univariate_results.append({
            'Variable': col,
            'R2': model.rsquared,
            'Adj_R2': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'P_value': model.pvalues[col],
            'Coeff': model.params[col],
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p
        })
    df_uni = pd.DataFrame(univariate_results).sort_values('R2', ascending=False)
    print("\nTop 5 univariate R²:")
    print(df_uni[['Variable', 'R2', 'P_value', 'Spearman_r', 'Spearman_p']].head())
    best_single_var = df_uni.iloc[0]['Variable']
    best_single_r2 = df_uni.iloc[0]['R2']

    # 向前逐步回归 (AIC)
    def forward_selection(data, response, candidates):
        selected = []
        remaining = candidates.copy()
        best_aic = np.inf
        while remaining:
            aics = []
            for var in remaining:
                X = sm.add_constant(data[selected + [var]])
                model = sm.OLS(response, X).fit()
                aics.append((var, model.aic))
            aics.sort(key=lambda x: x[1])
            best_var, new_aic = aics[0]
            if new_aic < best_aic - 0.01:
                selected.append(best_var)
                remaining.remove(best_var)
                best_aic = new_aic
                print(f"Added {best_var}, AIC = {new_aic:.2f}")
            else:
                break
        return selected

    selected_vars = forward_selection(df_clean, df_clean['y'], available_cols)
    if not selected_vars:
        selected_vars = [best_single_var]
        print("Forward selection returned empty, using best single variable.")
    print(f"\nSelected variables: {selected_vars}")

    # 最终多变量模型
    X_multi = sm.add_constant(df_clean[selected_vars])
    model_multi = sm.OLS(df_clean['y'], X_multi).fit()
    print("\n===== Multivariate Model Summary =====")
    print(model_multi.summary())
    multi_r2 = model_multi.rsquared

    # 模型比较
    comparison = pd.DataFrame({
        'Model': [f'Best Single ({best_single_var})', f'Multivariate ({len(selected_vars)} vars)'],
        'R²': [best_single_r2, multi_r2],
        'Adj_R²': [df_uni[df_uni['Variable']==best_single_var]['Adj_R2'].values[0], model_multi.rsquared_adj],
        'AIC': [df_uni[df_uni['Variable']==best_single_var]['AIC'].values[0], model_multi.aic]
    })
    comparison.to_csv(f"{OUTPUT_PREFIX_AUTO}_model_selection_comparison.csv", index=False)
    df_uni.to_csv(f"{OUTPUT_PREFIX_AUTO}_univariate_ranking.csv", index=False)

    # 变量重要性
    importance = pd.DataFrame({
        'Variable': selected_vars,
        'Coefficient': model_multi.params[selected_vars],
        'P_value': model_multi.pvalues[selected_vars].values,
        'Std_Error': model_multi.bse[selected_vars].values
    }).sort_values('Coefficient', key=abs, ascending=False)
    importance.to_csv(f"{OUTPUT_PREFIX_AUTO}_variable_importance.csv", index=False)
    print("\nVariable importance (standardized coefficients):")
    print(importance)

    # ======================== 可视化（第一部分） ========================
    # 1. 单变量排名柱状图（水平）
    top10 = df_uni.head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10['Variable'], top10['R2'], color='#4A7B9D', edgecolor='none')
    ax.set_xlabel('$R^2$', fontsize=12)
    ax.set_title('Univariate $R^2$ Ranking (Top 10)', fontsize=14)
    for bar, r2 in zip(bars, top10['R2']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{r2:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX_AUTO}_univariate_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 最重要变量的散点回归图（选显著且系数绝对值最大的）
    sig_vars = importance[importance['P_value'] < 0.05]
    if not sig_vars.empty:
        main_var = sig_vars.iloc[0]['Variable']
    else:
        main_var = selected_vars[0]
    X_single = sm.add_constant(df_clean[[main_var]])
    model_single = sm.OLS(df_clean['y'], X_single).fit()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_clean[main_var], df_clean['y'], alpha=0.6, s=30, edgecolor='none')
    x_range = np.linspace(df_clean[main_var].min(), df_clean[main_var].max(), 100)
    y_range = model_single.params['const'] + model_single.params[main_var] * x_range
    ax.plot(x_range, y_range, 'r-', linewidth=2)
    r2_val = model_single.rsquared
    p_val = model_single.pvalues[main_var]
    n = len(df_clean)
    coef = model_single.params[main_var]
    textstr = f'$R^2$ = {r2_val:.3f}\nβ = {coef:.3f}\np = {p_val:.3f}\nn = {n}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    ax.set_xlabel(f'Standardized {main_var}', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'Relationship between {main_var} and Transfer Loss', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX_AUTO}_scatter_regression.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 多变量模型：预测 vs 实际
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df_clean['y'], model_multi.fittedvalues, alpha=0.6, s=30, edgecolor='none')
    min_val = min(df_clean['y'].min(), model_multi.fittedvalues.min())
    max_val = max(df_clean['y'].max(), model_multi.fittedvalues.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    ax.set_xlabel('Observed |CDS|', fontsize=12)
    ax.set_ylabel('Predicted |CDS|', fontsize=12)
    ax.set_title(f'Observed vs Predicted (Multivariate)\n$R^2$ = {model_multi.rsquared:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX_AUTO}_observed_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 残差诊断图
    residuals = model_multi.resid
    fitted = model_multi.fittedvalues
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(fitted, residuals, alpha=0.6, s=30, edgecolor='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Predicted |CDS|', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Fitted', fontsize=14)
    probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX_AUTO}_diagnostic_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPart 1 results saved to {OUTPUT_PREFIX_AUTO}")

else:
    # 手动模式（略，但保留基本功能）
    used_cols = [c for c in SELECTED_COLS if c in available_cols]
    if not used_cols:
        raise ValueError("No valid manual columns")
    X_multi = sm.add_constant(df_clean[used_cols])
    model_multi = sm.OLS(df_clean['y'], X_multi).fit()
    print(model_multi.summary())
    # 此处可添加类似可视化，但为简洁，省略

# ======================== 第二部分：分层回归（固定四变量） ========================
print("\n" + "="*70)
print("PART 2: Hierarchical Regression (Four Fixed Variables)")
print("="*70)

OUTPUT_PREFIX_HIER = f"{OUTPUT_BASE}_hierarchical"
os.makedirs(os.path.dirname(OUTPUT_PREFIX_HIER), exist_ok=True)

# ---- 重新加载 CDS（确保干净） ----
df_cds = pd.read_csv(CDS_MATRIX_FILE, index_col=0)
df_cds_long = df_cds.stack().reset_index()
df_cds_long.columns = ["Source", "Target", "CDS_raw"]
df_cds_long = df_cds_long.dropna(subset=["CDS_raw"])
df_cds_long['Source'] = df_cds_long['Source'].astype(str).str.strip()
df_cds_long['Target'] = df_cds_long['Target'].astype(str).str.strip()

# ---- 加载四个变量 ----
df_hier = df_cds_long.copy()
for var in HIER_VARS:
    fname = f"{MATRICES_DIR}/domain_shift_{var}_matrix.csv"
    try:
        mat = pd.read_csv(fname, index_col=0)
        mat.index = mat.index.astype(str).str.strip()
        mat.columns = mat.columns.astype(str).str.strip()
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", var]
        long['Source'] = long['Source'].astype(str).str.strip()
        long['Target'] = long['Target'].astype(str).str.strip()
        df_hier = df_hier.merge(long, on=["Source", "Target"], how="inner")
        print(f"Loaded {var}, rows: {len(long)}")
    except FileNotFoundError:
        print(f"Warning: {fname} not found, skipping {var}")
        HIER_VARS.remove(var)  # 移除缺失变量

# ---- 因变量（绝对值） ----
df_hier['y'] = np.abs(df_hier['CDS_raw'])
df_hier = df_hier.dropna(subset=['y'] + HIER_VARS)
print(f"Hierarchical samples: {len(df_hier)}")

# 标准化
scaler = StandardScaler()
df_hier[HIER_VARS] = scaler.fit_transform(df_hier[HIER_VARS])

# ---- IQR 剔除（仅对 y） ----
vals = df_hier['y'].values
Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_hier = df_hier[(vals >= lower) & (vals <= upper)]
print(f"Hierarchical samples after outlier removal: {len(df_hier)}")

# ---- 分层回归 ----
models = {}
# 模型1：仅 Label_Shift
X1 = sm.add_constant(df_hier[['Label_Shift']])
models['M1_Label'] = sm.OLS(df_hier['y'], X1).fit()
# 模型2：+ AEF_L2
X2 = sm.add_constant(df_hier[['Label_Shift', 'AEF_L2']])
models['M2_Label+AEF'] = sm.OLS(df_hier['y'], X2).fit()
# 模型3：+ slpoe_L2
X3 = sm.add_constant(df_hier[['Label_Shift', 'AEF_L2', 'GIS_L2']])
models['M3_+slpoe'] = sm.OLS(df_hier['y'], X3).fit()
# 模型4：+ Weighted_L2
X4 = sm.add_constant(df_hier[['Label_Shift', 'AEF_L2', 'GIS_L2', 'Weighted_L2']])
models['M4_+Joint'] = sm.OLS(df_hier['y'], X4).fit()

# ---- 输出比较 ----
comparison_hier = pd.DataFrame({
    'Model': list(models.keys()),
    'R²': [m.rsquared for m in models.values()],
    'Adj_R²': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()]
})
print("\nHierarchical Model Comparison:")
print(comparison_hier)
comparison_hier.to_csv(f"{OUTPUT_PREFIX_HIER}_model_comparison.csv", index=False)

print("\nΔR² (incremental):")
delta = [
    models['M2_Label+AEF'].rsquared - models['M1_Label'].rsquared,
    models['M3_+slpoe'].rsquared - models['M2_Label+AEF'].rsquared,
    models['M4_+Joint'].rsquared - models['M3_+slpoe'].rsquared
]
print(f"M1→M2: {delta[0]:.4f}")
print(f"M2→M3: {delta[1]:.4f}")
print(f"M3→M4: {delta[2]:.4f}")

# ---- 最终模型系数 ----
print("\nFinal Model (M4) Coefficients:")
print(models['M4_+Joint'].summary().tables[1])

# ---- 可视化：系数柱状图（带显著性） ----
fig, ax = plt.subplots(figsize=(8, 5))
coefs = models['M4_+Joint'].params[HIER_VARS]
pvals = models['M4_+Joint'].pvalues[HIER_VARS]
colors = ['#2E86AB' if p < 0.05 else '#D3D3D3' for p in pvals]
bars = ax.bar(HIER_VARS, coefs, color=colors, edgecolor='black')
ax.axhline(0, color='black', linewidth=0.8)
for bar, p, coef in zip(bars, pvals, coefs):
    y_pos = coef + (0.05 if coef >= 0 else -0.05)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, sig, ha='center', va='center', fontsize=10)
ax.set_ylabel('Standardized Coefficient (β)', fontsize=12)
ax.set_title(f'Final Model (M4) Coefficients\n$R^2$ = {models["M4_+Joint"].rsquared:.3f}, Adj.$R^2$ = {models["M4_+Joint"].rsquared_adj:.3f}', fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX_HIER}_coefficients.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPart 2 results saved to {OUTPUT_PREFIX_HIER}")
print("\nAll analyses completed.")