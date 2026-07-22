#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域偏移与迁移性能回归分析（读取矩阵文件）
依赖：CDS_matrix.csv（由 transfer_ability.py 生成）
      以及 domain_shift_matrix_entropy.py 生成的所有矩阵文件。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 中文字体 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set(font='SimHei')

# ========================== 用户配置 ==========================
CDS_MATRIX_FILE = r"CDS_matrix_robust_entropy_poi.csv"                # 由 transfer_ability.py 生成
MATRICES_DIR = "results/domain_shift/aef_plus_poi"             # 即脚本1的输出目录
OUTPUT_PREFIX = r"domain_shift_analysis\aef_plus_poi"

TARGET_STRATEGY = 'loss'   # 'loss', 'abs_loss', 'gain'

# ========================== 加载 CDS 矩阵 ==========================
df_cds = pd.read_csv(CDS_MATRIX_FILE, index_col=0)
df_cds_long = df_cds.stack().reset_index()
df_cds_long.columns = ["Source", "Target", "CDS_raw"]
df_cds_long = df_cds_long.dropna(subset=["CDS_raw"])

# ========================== 加载偏移矩阵 ==========================
# 读取所有偏移指标矩阵，合并为长格式
matrix_files = [
    "domain_shift_L2_Dist_matrix.csv",
    "domain_shift_MMD_matrix.csv",
    "domain_shift_CORAL_matrix.csv",
    "domain_shift_KL_Div_matrix.csv",
    "domain_shift_Spectral_Dist_matrix.csv",
    "domain_shift_Degree_Diff_matrix.csv",
    "domain_shift_L1_Dist_matrix.csv",
    "domain_shift_Cos_Dist_matrix.csv",   # 若存在
    "domain_shift_CDSI_matrix.csv"
]
shift_cols = []
df_merged = df_cds_long.copy()
for fname in matrix_files:
    fpath = f"{MATRICES_DIR}/{fname}"
    try:
        mat = pd.read_csv(fpath, index_col=0)
        metric = fname.replace("domain_shift_", "").replace("_matrix.csv", "")
        # 堆叠为长格式
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", metric]
        df_merged = df_merged.merge(long, on=["Source", "Target"], how="inner")
        shift_cols.append(metric)
        print(f"加载 {metric} 矩阵，有效样本数: {len(long)}")
    except FileNotFoundError:
        print(f"跳过文件 {fname} (不存在)")

print(f"合并后总样本数: {len(df_merged)}")

# ========================== 生成因变量 ==========================
if TARGET_STRATEGY == 'loss':
    df_merged['y'] = df_merged['CDS_raw']
    y_label = "CDS (迁移损失, 越小越好)"
    suffix = "_loss"
elif TARGET_STRATEGY == 'abs_loss':
    df_merged['y'] = np.abs(df_merged['CDS_raw'])
    y_label = "|CDS| (绝对偏移)"
    suffix = "_abs"
elif TARGET_STRATEGY == 'gain':
    df_merged['y'] = -df_merged['CDS_raw']
    y_label = "-CDS (迁移保留度, 越大越好)"
    suffix = "_gain"
else:
    raise ValueError("TARGET_STRATEGY 参数错误")

OUTPUT_PREFIX = f"{OUTPUT_PREFIX}{suffix}"

# 剔除缺失
df_merged = df_merged.dropna(subset=['y'] + shift_cols)
print(f"有效样本数（无缺失）: {len(df_merged)}")

# ========================== 标准化自变量 ==========================
scaler = StandardScaler()
df_merged[shift_cols] = scaler.fit_transform(df_merged[shift_cols])
# 如果存在 CDSI，单独标准化（但 CDSI 已经是归一化的，可再标准化以比较系数）
if 'CDSI' in shift_cols:
    df_merged['CDSI_std'] = StandardScaler().fit_transform(df_merged[['CDSI']])
    cdsi_col = 'CDSI_std'
else:
    cdsi_col = None

# ========================== IQR 异常剔除 ==========================
vals = df_merged['y'].values
Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_merged['IsOutlier'] = (vals < lower) | (vals > upper)
print(f"异常样本数: {df_merged['IsOutlier'].sum()} ({100*df_merged['IsOutlier'].sum()/len(df_merged):.1f}%)")
df_clean = df_merged[~df_merged['IsOutlier']].copy()
print(f"正常样本数: {len(df_clean)}")

# ========================== 回归模型拟合 ==========================
# 1. L2 单变量
X_l2 = sm.add_constant(df_clean[['L2_Dist']])
model_l2 = sm.OLS(df_clean['y'], X_l2).fit()
# 2. CDSI 单变量（若存在）
if cdsi_col:
    X_cdsi = sm.add_constant(df_clean[[cdsi_col]])
    model_cdsi = sm.OLS(df_clean['y'], X_cdsi).fit()
else:
    model_cdsi = None
# 3. 多变量（所有偏移指标）
X_multi = sm.add_constant(df_clean[shift_cols])
model_multi = sm.OLS(df_clean['y'], X_multi).fit()
# 4. 多变量 + 二次项（L2 和 MMD）
df_clean['L2_sq'] = df_clean['L2_Dist'] ** 2
if 'MMD' in shift_cols:
    df_clean['MMD_sq'] = df_clean['MMD'] ** 2
    quad_vars = shift_cols + ['L2_sq', 'MMD_sq']
else:
    quad_vars = shift_cols + ['L2_sq']
X_quad = sm.add_constant(df_clean[quad_vars])
model_quad = sm.OLS(df_clean['y'], X_quad).fit()
# 5. 交互项（L2 * MMD）
if 'MMD' in shift_cols:
    df_clean['L2_MMD'] = df_clean['L2_Dist'] * df_clean['MMD']
    interact_vars = shift_cols + ['L2_MMD']
    X_interact = sm.add_constant(df_clean[interact_vars])
    model_interact = sm.OLS(df_clean['y'], X_interact).fit()
else:
    model_interact = None

models = {'L2': model_l2}
if model_cdsi: models['CDSI'] = model_cdsi
models['Multi'] = model_multi
models['Quadratic'] = model_quad
if model_interact: models['Interaction'] = model_interact

# ========================== 模型比较 ==========================
comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'R²': [m.rsquared for m in models.values()],
    'Adj_R²': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()],
    'F_pval': [m.f_pvalue for m in models.values()]
})
print("\n===== 模型比较 =====")
print(comparison.to_string(index=False))
comparison.to_csv(f"{OUTPUT_PREFIX}_model_comparison.csv", index=False)

# ========================== 变量重要性（多变量模型） ==========================
importance = pd.DataFrame({
    'Variable': shift_cols,
    'Coefficient': model_multi.params[shift_cols].values,
    'P_value': model_multi.pvalues[shift_cols].values,
    'Std_Error': model_multi.bse[shift_cols].values
}).sort_values('Coefficient', key=abs, ascending=False)
importance.to_csv(f"{OUTPUT_PREFIX}_variable_importance.csv", index=False)
print("\n多变量模型标准化系数（绝对值越大越重要）：")
print(importance)

# ========================== 可视化（与之前一致） ==========================
# 单变量散点图（L2 和 CDSI）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (x_col, x_label, model) in zip(axes,
    [('L2_Dist', 'L2 均值距离 (标准化)', model_l2),
     (cdsi_col, 'CDSI (标准化)', model_cdsi)] if cdsi_col else [('L2_Dist', 'L2 均值距离 (标准化)', model_l2)]):
    ax.scatter(df_merged[x_col], df_merged['y'], c='lightgray', alpha=0.3, label='全量 (含异常)')
    ax.scatter(df_clean[x_col], df_clean['y'], c='blue', label='正常样本', alpha=0.6)
    out = df_merged[df_merged['IsOutlier']]
    if len(out) > 0:
        ax.scatter(out[x_col], out['y'], c='red', marker='x', s=80, label='异常 (IQR)')
    x_range = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100)
    y_range = model.params['const'] + model.params[x_col] * x_range
    ax.plot(x_range, y_range, 'g-', lw=2,
            label=f"线性拟合 (斜率={model.params[x_col]:.3f}, R²={model.rsquared:.3f})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{x_label} 与迁移损失")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_scatter_L2_CDSI.png", dpi=300)
plt.close()

# 分箱趋势（L2 和 CDSI）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_cols = ['L2_Dist'] + ([cdsi_col] if cdsi_col else [])
for ax, x_col in zip(axes, plot_cols):
    x_label = 'L2 距离' if x_col == 'L2_Dist' else 'CDSI'
    df_clean[f"{x_col}_bin"] = pd.qcut(df_clean[x_col], q=8, labels=False)
    bin_stats = df_clean.groupby(f"{x_col}_bin")['y'].agg(['mean', 'std']).reset_index()
    bin_medians = df_clean.groupby(f"{x_col}_bin")[x_col].median()
    ax.errorbar(bin_medians, bin_stats['mean'], yerr=bin_stats['std'],
                fmt='o-', capsize=5, color='darkblue', label='分箱均值')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"平均 {y_label}")
    ax.set_title(f"{x_label} 分箱趋势")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_binned_trend_L2_CDSI.png", dpi=300)
plt.close()

# 残差诊断（三个模型）
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, (name, mod) in enumerate(models.items()):
    if i >= 3: break
    ax = axes[i, 0]
    ax.scatter(mod.fittedvalues, mod.resid, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f'{name}: 残差 vs 拟合')
    ax.set_xlabel('拟合值')
    ax.set_ylabel('残差')
    ax = axes[i, 1]
    sm.qqplot(mod.resid, line='s', ax=ax)
    ax.set_title(f'{name}: Q-Q')
    ax = axes[i, 2]
    ax.hist(mod.resid, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f'{name}: 残差分布')
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_diagnostics_multi.png", dpi=300)
plt.close()

# 多变量模型实际 vs 预测
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(model_multi.fittedvalues, df_clean['y'], alpha=0.6)
min_val = min(model_multi.fittedvalues.min(), df_clean['y'].min())
max_val = max(model_multi.fittedvalues.max(), df_clean['y'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 理想线')
ax.set_xlabel('预测值')
ax.set_ylabel('实际值')
ax.set_title('多变量模型：实际 vs 预测')
ax.legend()
plt.savefig(f"{OUTPUT_PREFIX}_actual_vs_pred_multi.png", dpi=300)
plt.close()

# 源特异性回归（仅用 L2）
if 'Target_Node_Count' not in df_clean.columns:
    df_clean['Target_Node_Count'] = 50
source_models = {}
for src in df_clean['Source'].unique():
    subset = df_clean[df_clean['Source'] == src]
    X = sm.add_constant(subset[['L2_Dist']])
    model = sm.OLS(subset['y'], X).fit()
    source_models[src] = model

source_slopes = {src: model.params['L2_Dist'] for src, model in source_models.items()}
slope_df = pd.DataFrame(list(source_slopes.items()), columns=['Source', 'Slope']).sort_values('Slope')
n_per_group = 5
groups = [slope_df.iloc[i:i+n_per_group]['Source'].tolist() for i in range(0, len(slope_df), n_per_group)]

for g_idx, group_sources in enumerate(groups):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("tab10", n_colors=len(group_sources))
    x_vals = np.linspace(df_clean['L2_Dist'].min(), df_clean['L2_Dist'].max(), 100)
    for i, src in enumerate(group_sources):
        subset = df_clean[df_clean['Source'] == src]
        model = source_models[src]
        ax.scatter(subset['L2_Dist'], subset['y'],
                   s=subset['Target_Node_Count'].apply(lambda x: (x/100)+20),
                   color=colors[i], alpha=0.6, label=src, edgecolors='black', linewidth=0.3)
        pred = model.get_prediction(sm.add_constant(x_vals))
        pred_sum = pred.summary_frame(alpha=0.25)
        ax.plot(x_vals, pred_sum['mean'], color=colors[i], lw=2)
        ax.fill_between(x_vals, pred_sum['mean_ci_lower'], pred_sum['mean_ci_upper'],
                        color=colors[i], alpha=0.15)
    ax.set_xlabel("L2 均值距离 (标准化)")
    ax.set_ylabel(y_label)
    ax.set_title(f"源特异性回归 (组 {g_idx+1}, 75% CI)")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_source_specific_group_{g_idx+1}.png", dpi=300)
    plt.close()

# 实际 vs 预测（源特异性）
fig, ax = plt.subplots(figsize=(8, 8))
all_pred, all_actual = [], []
for src in df_clean['Source'].unique():
    subset = df_clean[df_clean['Source'] == src]
    model = source_models[src]
    pred = model.predict(sm.add_constant(subset[['L2_Dist']]))
    all_pred.extend(pred)
    all_actual.extend(subset['y'])
    ax.scatter(pred, subset['y'], s=50, alpha=0.7, label=src)
min_val = min(min(all_actual), min(all_pred))
max_val = max(max(all_actual), max(all_pred))
pad = (max_val - min_val) * 0.05
ax.plot([min_val-pad, max_val+pad], [min_val-pad, max_val+pad], 'r--', label='1:1')
ax.set_xlim(min_val-pad, max_val+pad)
ax.set_ylim(min_val-pad, max_val+pad)
ax.set_xlabel("预测值 (L2 模型)")
ax.set_ylabel("实际值")
ax.set_title("源特异性模型：实际 vs 预测")
ax.legend(loc='lower right', ncol=2, fontsize=8)
ax.grid(True)
plt.savefig(f"{OUTPUT_PREFIX}_source_specific_actual_vs_pred.png", dpi=300)
plt.close()

print(f"\n分析完成！所有结果已保存至前缀 {OUTPUT_PREFIX}")