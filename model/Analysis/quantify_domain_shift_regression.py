#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域偏移量化回归分析：建立 L2 距离 -> 迁移性能(R²) 的映射，
包含单变量、多变量控制、非线性检验和残差诊断。
自动处理异常值（IQR），并生成基于正常样本的二次拟合与拐点。
新增：可选择对 L2 距离进行标准化（Z-score），使不同模型间的斜率可比。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 中文字体设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set(font='SimHei')

# ---------- 配置 ----------
R2_MATRIX_FILE = r"transfer_results(1)\transfer_matrix_r2_mlp.csv"
DIST_FILE = r"transfer_results(1)\transfer_embedding_distances_mlp.csv"
CITY_STATS_FILE = "city_population_stats.csv"  # 可选
OUTPUT_PREFIX = "transfer_results(1)/regression_quantify"

# ========== 新增：是否标准化 L2 距离 ==========
STANDARDIZE_L2 = True   # 设为 True 则对 L2_Dist 做 Z-score 标准化

# ---------- 1. 加载数据 ----------
df_r2 = pd.read_csv(R2_MATRIX_FILE, index_col=0)
df_dist = pd.read_csv(DIST_FILE)

# 转为长格式
df_long = df_r2.stack().reset_index()
df_long.columns = ["Source", "Target", "R2_cal"]
df_long = df_long.dropna(subset=["R2_cal"])

# 合并距离
df_model = df_long.merge(df_dist, on=["Source", "Target"], how="inner")
print(f"合并后样本数: {len(df_model)}")

# ========== 新增：标准化 L2 距离 ==========
if STANDARDIZE_L2:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_model["L2_Dist_std"] = scaler.fit_transform(df_model[["L2_Dist"]])
    dist_col = "L2_Dist_std"
    print("已对 L2 距离进行 Z-score 标准化，使用列名 'L2_Dist_std'")
else:
    dist_col = "L2_Dist"
    print("使用原始 L2 距离")

# ---------- 2. IQR 异常识别 ----------
vals = df_model["R2_cal"].values
Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_model["IsOutlier"] = (vals < lower) | (vals > upper)
n_outliers = df_model["IsOutlier"].sum()
print(f"异常城市对数: {n_outliers} ({100*n_outliers/len(df_model):.1f}%)")
print(f"正常城市对数: {len(df_model)-n_outliers} ({100*(len(df_model)-n_outliers)/len(df_model):.1f}%)")

# ---------- 3. 剔除异常后的数据 ----------
df_clean = df_model[~df_model["IsOutlier"]].copy()
print(f"用于建模的正常样本数: {len(df_clean)}")

# ---------- 4. 单变量线性回归（全量） ----------
X_all = sm.add_constant(df_model[dist_col])
y_all = df_model["R2_cal"]
model_all = sm.OLS(y_all, X_all).fit()

print("\n" + "="*60)
print(f"【全量数据线性回归（含异常，仅供参考）】使用自变量：{dist_col}")
print("="*60)
print(model_all.summary())

# ---------- 5. 单变量线性回归（剔除异常） ----------
X_clean = sm.add_constant(df_clean[dist_col])
y_clean = df_clean["R2_cal"]
model_clean = sm.OLS(y_clean, X_clean).fit()

print("\n" + "="*60)
print(f"【剔除异常后的线性回归（正常样本 N={len(df_clean)}）】使用自变量：{dist_col}")
print("="*60)
print(model_clean.summary())

# 提取指标
slope_clean = model_clean.params[dist_col]
pval_clean = model_clean.pvalues[dist_col]
r2_clean = model_clean.rsquared
ci_clean = model_clean.conf_int().loc[dist_col].values

print(f"\n量化结论（正常样本）：{dist_col} 每增加 1 个标准差，R² 平均下降 {abs(slope_clean):.4f} (95% CI: [{ci_clean[0]:.4f}, {ci_clean[1]:.4f}])")
print(f"全模型解释力 R² = {r2_clean:.4f}")

# ---------- 6. 二次项检验（剔除异常） ----------
df_clean[f"{dist_col}_sq"] = df_clean[dist_col] ** 2
X_quad = sm.add_constant(df_clean[[dist_col, f"{dist_col}_sq"]])
y_quad = df_clean["R2_cal"]
model_quad = sm.OLS(y_quad, X_quad).fit()

print("\n" + "="*60)
print(f"【剔除异常后的二次项检验】使用自变量：{dist_col}")
print("="*60)
print(model_quad.summary())

# 计算拐点（仅在原始距离时有意义，标准化后拐点单位为标准差，可还原）
b = model_quad.params[dist_col]
c = model_quad.params[f"{dist_col}_sq"]
if c != 0:
    turning_point = -b / (2 * c)
    if STANDARDIZE_L2:
        # 还原到原始尺度（可选）
        original_mean = df_model["L2_Dist"].mean()
        original_std = df_model["L2_Dist"].std()
        turning_point_original = turning_point * original_std + original_mean
        print(f"\n二次拟合拐点（标准化单位）: {dist_col} = {turning_point:.2f}")
        print(f"对应原始 L2 距离: {turning_point_original:.2f}")
    else:
        print(f"\n二次拟合拐点: {dist_col} = {turning_point:.2f} (该点后下降趋势减缓)")
else:
    turning_point = None

# ---------- 7. 多变量控制（若存在城市统计文件） ----------
try:
    df_city_stats = pd.read_csv(CITY_STATS_FILE)
    df_clean = df_clean.merge(df_city_stats, left_on="Target", right_on="City", how="left")
    df_clean = df_clean.rename(columns={"Density_Std": "Target_Density_Std"})
    df_clean = df_clean.merge(df_city_stats, left_on="Source", right_on="City", how="left", suffixes=("_target", "_source"))
    df_clean = df_clean.rename(columns={"Node_Count_source": "Source_Node_Count"})
    df_clean["Target_Density_Std_scaled"] = (df_clean["Target_Density_Std"] - df_clean["Target_Density_Std"].mean()) / df_clean["Target_Density_Std"].std()
    df_clean["Source_Node_Count_scaled"] = (df_clean["Source_Node_Count"] - df_clean["Source_Node_Count"].mean()) / df_clean["Source_Node_Count"].std()
    
    formula = f"R2_cal ~ {dist_col} + Target_Density_Std_scaled + Source_Node_Count_scaled"
    model_multi = smf.ols(formula, data=df_clean).fit()
    print("\n" + "="*60)
    print("【多变量回归控制混杂】")
    print("="*60)
    print(model_multi.summary())
    has_controls = True
except FileNotFoundError:
    print("\n未找到城市统计文件，跳过混杂控制。")
    has_controls = False

# ---------- 8. 保存量化指标 ----------
metrics = {
    "指标": [
        "样本数 (正常)", 
        "Spearman ρ (全量)", 
        "线性斜率", 
        "斜率 95% CI 下限", 
        "斜率 95% CI 上限", 
        "线性 R²", 
        "线性 p值",
        "二次项系数",
        "二次项 p值",
        "二次拟合 R²",
        "二次拟合拐点"
    ],
    "数值": [
        len(df_clean),
        spearmanr(df_model[dist_col], df_model["R2_cal"])[0],
        slope_clean,
        ci_clean[0],
        ci_clean[1],
        r2_clean,
        pval_clean,
        model_quad.params[f"{dist_col}_sq"],
        model_quad.pvalues[f"{dist_col}_sq"],
        model_quad.rsquared,
        turning_point if turning_point is not None else np.nan
    ]
}
suffix = "_std" if STANDARDIZE_L2 else "_raw"
pd.DataFrame(metrics).to_csv(f"{OUTPUT_PREFIX}_metrics_clean{suffix}.csv", index=False)

# ---------- 9. 可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：散点图 + 线性回归
ax = axes[0]
ax.scatter(df_model[dist_col], df_model["R2_cal"], c='lightgray', alpha=0.3, label='全量 (含异常)')
ax.scatter(df_clean[dist_col], df_clean["R2_cal"], c='blue', label='正常样本', alpha=0.6)

outlier = df_model[df_model["IsOutlier"]]
if len(outlier) > 0:
    ax.scatter(outlier[dist_col], outlier["R2_cal"], c='red', marker='x', s=80, label='异常 (IQR)')

x_range = np.linspace(df_clean[dist_col].min(), df_clean[dist_col].max(), 100)
y_linear = model_clean.params["const"] + model_clean.params[dist_col] * x_range
ax.plot(x_range, y_linear, 'g-', linewidth=2, 
        label=f'线性拟合 (斜率={model_clean.params[dist_col]:.3f}, R²={model_clean.rsquared:.3f})')
ax.set_xlabel("L2 均值距离 (标准化)" if STANDARDIZE_L2 else "L2 均值距离")
ax.set_ylabel("校准后 R²")
ax.set_title("域偏移量化映射 (正常样本)")
ax.legend()
ax.grid(True)

# 右图：分箱趋势
ax = axes[1]
df_clean[f"{dist_col}_bin"] = pd.qcut(df_clean[dist_col], q=8, labels=False)
bin_stats = df_clean.groupby(f"{dist_col}_bin")["R2_cal"].agg(['mean', 'std']).reset_index()
bin_medians = df_clean.groupby(f"{dist_col}_bin")[dist_col].median()
ax.errorbar(bin_medians, bin_stats['mean'], yerr=bin_stats['std'], 
            fmt='o-', capsize=5, color='darkblue', label='分箱均值')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.plot(x_range, y_linear, 'r--', label='线性拟合参考')
ax.set_xlabel("L2 距离 (分箱中位数)" + (" (标准化)" if STANDARDIZE_L2 else ""))
ax.set_ylabel("平均校准后 R²")
ax.set_title("正常样本的分箱趋势")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_quantification_plots_clean{suffix}.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 10. 诊断图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(model_clean.fittedvalues, model_clean.resid, alpha=0.6)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel("拟合值")
axes[0].set_ylabel("残差")
axes[0].set_title("残差诊断 (正常样本)")

sm.qqplot(model_clean.resid, line='s', ax=axes[1])
axes[1].set_title("Q-Q 图 (正常样本)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_diagnostics_clean{suffix}.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n量化分析完成！结果已保存至 {OUTPUT_PREFIX}_metrics_clean{suffix}.csv")
print(f"生成图表（前缀 {OUTPUT_PREFIX}）：")
print(f"  - {OUTPUT_PREFIX}_quantification_plots_clean{suffix}.png")
print(f"  - {OUTPUT_PREFIX}_diagnostics_clean{suffix}.png")


# -------------------- 数据准备 --------------------
# 假设 df_clean 已存在（剔除异常值后的数据）
if 'Target_Node_Count' not in df_clean.columns:
    df_clean['Target_Node_Count'] = 50  # 固定大小

# 计算每个源的回归模型
source_models = {}
for src in df_clean['Source'].unique():
    subset = df_clean[df_clean['Source'] == src]
    X = sm.add_constant(subset[['L2_Dist']])
    y = subset['R2_cal']
    model = sm.OLS(y, X).fit()
    source_models[src] = model

# 按斜率分组
source_slopes = {src: model.params['L2_Dist'] for src, model in source_models.items()}
slope_df = pd.DataFrame(list(source_slopes.items()), columns=['Source', 'Slope'])
slope_df = slope_df.sort_values('Slope').reset_index(drop=True)

n_per_group = 5
n_sources = len(slope_df)
groups = []
for i in range(0, n_sources, n_per_group):
    groups.append(slope_df.iloc[i:i+n_per_group]['Source'].tolist())

# -------------------- 图一：源特异性回归（按组绘制，带75% CLI） --------------------
for g_idx, group_sources in enumerate(groups):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    group_data = df_clean[df_clean['Source'].isin(group_sources)]
    x_min_global = group_data['L2_Dist'].min()
    x_max_global = group_data['L2_Dist'].max()
    x_vals_global = np.linspace(x_min_global, x_max_global, 100)
    
    colors = sns.color_palette("tab10", n_colors=len(group_sources))
    
    for i, src in enumerate(group_sources):
        subset = df_clean[df_clean['Source'] == src]
        model = source_models[src]
        
        # 散点
        sizes = subset['Target_Node_Count'].apply(lambda x: (x / 100) + 20)
        ax.scatter(subset['L2_Dist'], subset['R2_cal'], 
                   s=sizes, color=colors[i], alpha=0.6, 
                   label=src, edgecolors='black', linewidth=0.3)
        
        # 预测
        X_pred = sm.add_constant(x_vals_global)
        pred = model.get_prediction(X_pred)
        pred_summary = pred.summary_frame(alpha=0.25)
        mean_pred = pred_summary['mean']
        ci_lower = pred_summary['mean_ci_lower']
        ci_upper = pred_summary['mean_ci_upper']
        
        # 数据范围内实线，外推区域虚线
        x_min_src = subset['L2_Dist'].min()
        x_max_src = subset['L2_Dist'].max()
        mask_in = (x_vals_global >= x_min_src) & (x_vals_global <= x_max_src)
        ax.plot(x_vals_global[mask_in], mean_pred[mask_in], 
                color=colors[i], linewidth=2, linestyle='-')
        mask_out = ~mask_in
        if mask_out.any():
            ax.plot(x_vals_global[mask_out], mean_pred[mask_out], 
                    color=colors[i], linewidth=1.5, linestyle='--', alpha=0.6)
        ax.fill_between(x_vals_global, ci_lower, ci_upper, 
                        color=colors[i], alpha=0.15)
    
    ax.set_xlabel("L2 嵌入均值距离", fontsize=12)
    ax.set_ylabel(r"校准后迁移 $R^2$", fontsize=12)  # 修复R²显示
    ax.set_title(f"源特异性域偏移与迁移性能 (组 {g_idx+1}, 75% CLI)", fontsize=14)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min_global, x_max_global)
    plt.tight_layout()
    plt.savefig(f"Figure_Source_Wise_Group_{g_idx+1}.png", dpi=300)
    plt.close()

# -------------------- 图二：实际 vs 预测 R²（正确限制轴范围） --------------------
fig, ax = plt.subplots(figsize=(8, 8))

all_pred = []
all_actual = []
src_list = df_clean['Source'].unique()
colors = sns.color_palette("tab10", n_colors=len(src_list))

for idx, src in enumerate(src_list):
    subset = df_clean[df_clean['Source'] == src]
    model = source_models[src]
    X_pred = sm.add_constant(subset[['L2_Dist']])
    pred = model.predict(X_pred)
    
    all_pred.extend(pred)
    all_actual.extend(subset['R2_cal'])
    
    ax.scatter(pred, subset['R2_cal'], 
               s=50, color=colors[idx], alpha=0.7, label=src)

# 关键修正：只使用实际R²和预测R²来确定轴范围
min_val = min(np.min(all_actual), np.min(all_pred))
max_val = max(np.max(all_actual), np.max(all_pred))
# 留出5%边距
pad = (max_val - min_val) * 0.05

ax.plot([min_val - pad, max_val + pad], 
        [min_val - pad, max_val + pad], 
        'r--', linewidth=2, label='1:1 完美预测')

ax.set_xlim(min_val - pad, max_val + pad)
ax.set_ylim(min_val - pad, max_val + pad)

ax.set_xlabel(r"预测迁移 $R^2$ (源特异性)", fontsize=12)
ax.set_ylabel(r"实际迁移 $R^2$", fontsize=12)
ax.set_title("实际 vs 预测迁移性能 (按源城市分色)", fontsize=14)
ax.legend(loc='lower right', ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis('equal')  # 保持纵横比一致
plt.tight_layout()
plt.savefig("Figure_Predicted_vs_Actual_SourceSpecific.png", dpi=300)
plt.close()

print("所有图形已保存。")