#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动选择最优域偏移指标的回归分析
因变量：CDS（迁移损失）
自变量：从 domain_shift_*_matrix.csv 中自动提取所有可用指标
支持：单变量筛选 + 逐步回归（AIC向前选择）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')
import os

# -------------------------- 中文字体 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set(font='SimHei')

# ========================== 用户配置 ==========================
CDS_MATRIX_FILE = r"results\transferability\ndvi\CDS_matrix_robust_entropy.csv"
MATRICES_DIR = r"results/domain_shift/aef_plus_ndvi/with_gis"
OUTPUT_PREFIX = r"results\regression\domain_shift_analysis\ndvi\mix"
os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)
TARGET_STRATEGY = 'loss'   # 'loss', 'abs_loss', 'gain'

# ----- 自动选择开关 -----
AUTO_SELECT = True   # True: 自动从所有指标中选择最优模型；False: 使用下方手动指定的 SELECTED_COLS

# 当 AUTO_SELECT=False 时，手动指定指标
SELECTED_COLS = ['L2_Dist', 'Spectral_Dist', 'KL_Div']

# ========================== 加载与合并数据 ==========================
# 1. CDS
df_cds = pd.read_csv(CDS_MATRIX_FILE, index_col=0)
df_cds_long = df_cds.stack().reset_index()
df_cds_long.columns = ["Source", "Target", "CDS_raw"]
df_cds_long = df_cds_long.dropna(subset=["CDS_raw"])
df_cds_long['Source'] = df_cds_long['Source'].astype(str)
df_cds_long['Target'] = df_cds_long['Target'].astype(str)

# 2. 自动发现所有域偏移矩阵文件
import glob
pattern = f"{MATRICES_DIR}/domain_shift_*_matrix.csv"
all_files = glob.glob(pattern)
if not all_files:
    raise FileNotFoundError(f"在 {MATRICES_DIR} 中未找到任何 domain_shift_*_matrix.csv 文件")

# 提取指标名称
available_cols = []
for fpath in all_files:
    fname = os.path.basename(fpath)
    # 解析指标名：domain_shift_{metric}_matrix.csv
    metric = fname.replace("domain_shift_", "").replace("_matrix.csv", "")
    available_cols.append(metric)
print(f"发现可用指标: {available_cols}")

# 3. 加载所有指标矩阵
df_merged = df_cds_long.copy()
for col in available_cols:
    fname = f"{MATRICES_DIR}/domain_shift_{col}_matrix.csv"
    try:
        mat = pd.read_csv(fname, index_col=0)
        mat.index = mat.index.astype(str)
        mat.columns = mat.columns.astype(str)
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", col]
        long['Source'] = long['Source'].astype(str)
        long['Target'] = long['Target'].astype(str)
        df_merged = df_merged.merge(long, on=["Source", "Target"], how="inner")
        print(f"加载 {col} 矩阵，行数: {len(long)}")
    except Exception as e:
        print(f"加载 {col} 失败: {e}")

print(f"合并后总样本数: {len(df_merged)}")

# 4. 定义因变量
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
df_merged = df_merged.dropna(subset=['y'] + available_cols)
print(f"有效样本数（无缺失）: {len(df_merged)}")

# 5. 标准化所有自变量
scaler = StandardScaler()
df_merged[available_cols] = scaler.fit_transform(df_merged[available_cols])

# ========================== IQR 异常剔除 ==========================
vals = df_merged['y'].values
Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_merged['IsOutlier'] = (vals < lower) | (vals > upper)
print(f"异常样本数: {df_merged['IsOutlier'].sum()} ({100*df_merged['IsOutlier'].sum()/len(df_merged):.1f}%)")
df_clean = df_merged[~df_merged['IsOutlier']].copy()
print(f"正常样本数: {len(df_clean)}")

# ========================== 自动模型选择 ==========================
if AUTO_SELECT:
    print("\n===== 自动选择最优模型 =====")
    # 6.1 单变量筛选：对每个指标进行一元回归，按 R² 排序
    # 6.1 单变量筛选：对每个指标进行一元回归，按 R² 排序
    univariate_results = []
    for col in available_cols:
        X = sm.add_constant(df_clean[[col]])
        model = sm.OLS(df_clean['y'], X).fit()
        # 计算 Spearman 相关系数
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
    print("\n单变量回归 R² 排序（前5）：")
    print(df_uni[['Variable', 'R2', 'P_value', 'Spearman_r', 'Spearman_p']].head())
    best_single_var = df_uni.iloc[0]['Variable']
    best_single_r2 = df_uni.iloc[0]['R2']
    print(f"\n最佳单变量: {best_single_var} (R² = {best_single_r2:.4f})")
    df_uni = pd.DataFrame(univariate_results).sort_values('R2', ascending=False)
    print("\n单变量回归 R² 排序（前5）：")
    print(df_uni[['Variable', 'R2', 'P_value']].head())
    best_single_var = df_uni.iloc[0]['Variable']
    best_single_r2 = df_uni.iloc[0]['R2']
    print(f"\n最佳单变量: {best_single_var} (R² = {best_single_r2:.4f})")

    # 6.2 向前逐步回归（基于 AIC）
    def forward_selection(data, response, candidates):
        """向前逐步回归（AIC准则）"""
        selected = []
        remaining = candidates.copy()
        best_aic = np.inf
        while remaining:
            aics = []
            for var in remaining:
                formula = ' + '.join(selected + [var])
                X = sm.add_constant(data[selected + [var]])
                model = sm.OLS(response, X).fit()
                aics.append((var, model.aic))
            # 选择最小 AIC 的变量
            aics.sort(key=lambda x: x[1])
            best_var, new_aic = aics[0]
            if new_aic < best_aic - 0.01:  # 阈值防止过拟合
                selected.append(best_var)
                remaining.remove(best_var)
                best_aic = new_aic
                print(f"加入 {best_var}, AIC = {new_aic:.2f}")
            else:
                break
        return selected

    # 执行向前选择
    selected_vars = forward_selection(df_clean, df_clean['y'], available_cols)
    print(f"\n逐步回归选中的变量: {selected_vars}")
    if not selected_vars:
        # 若没有选中任何变量（可能因AIC阈值），则选最佳单变量作为退路
        selected_vars = [best_single_var]
        print("警告：逐步回归未选中任何变量，将使用最佳单变量。")

    # 6.3 拟合最终多变量模型
    X_multi = sm.add_constant(df_clean[selected_vars])
    model_multi = sm.OLS(df_clean['y'], X_multi).fit()
    print("\n===== 最佳多变量模型摘要 =====")
    print(model_multi.summary())
    multi_r2 = model_multi.rsquared

    # 6.4 比较：最佳单变量 vs 最佳多变量
    comparison = pd.DataFrame({
        'Model': [f'Best Single ({best_single_var})', f'Multivariate ({len(selected_vars)} vars)'],
        'R²': [best_single_r2, multi_r2],
        'Adj_R²': [df_uni[df_uni['Variable']==best_single_var]['Adj_R2'].values[0], model_multi.rsquared_adj],
        'AIC': [df_uni[df_uni['Variable']==best_single_var]['AIC'].values[0], model_multi.aic]
    })
    print("\n模型对比：")
    print(comparison)

    # 保存结果
    comparison.to_csv(f"{OUTPUT_PREFIX}_model_selection_comparison.csv", index=False)
    df_uni.to_csv(f"{OUTPUT_PREFIX}_univariate_ranking.csv", index=False)

    # 设置当前用于后续可视化的模型
    model = model_multi
    used_cols = selected_vars
    best_single = best_single_var

else:
    # 使用手动指定的指标
    used_cols = SELECTED_COLS
    # 检查是否都在 available_cols 中
    missing = [c for c in used_cols if c not in available_cols]
    if missing:
        print(f"警告：手动指定的指标 {missing} 不可用，仅使用可用部分")
        used_cols = [c for c in used_cols if c in available_cols]
    if not used_cols:
        raise ValueError("无有效指标")
    X_multi = sm.add_constant(df_clean[used_cols])
    model = sm.OLS(df_clean['y'], X_multi).fit()
    print(model.summary())

# ========================== 后续输出与可视化（保持不变） ==========================
# 变量重要性
importance = pd.DataFrame({
    'Variable': used_cols,
    'Coefficient': model.params[used_cols],
    'P_value': model.pvalues[used_cols].values,
    'Std_Error': model.bse[used_cols].values
}).sort_values('Coefficient', key=abs, ascending=False)
importance.to_csv(f"{OUTPUT_PREFIX}_variable_importance.csv", index=False)
print("\n变量重要性（标准化系数）：")
print(importance)

# 其余绘图代码与之前完全相同，但需要将 SELECTED_COLS 替换为 used_cols
# 并且部分依赖图、分箱趋势等需基于 used_cols
# 由于篇幅，此处仅示意，实际使用时可将原代码中的 SELECTED_COLS 全部替换为 used_cols
# 或者直接将 used_cols 赋给 SELECTED_COLS 以复用原有逻辑

# 为兼容原有可视化代码，重新定义 SELECTED_COLS = used_cols
SELECTED_COLS = used_cols

# 原有可视化代码（省略，但确保所有引用 SELECTED_COLS 的位置使用新的变量）
# 如果 AUTO_SELECT 为 True，还会额外绘制单变量排名柱状图等

# 额外：单变量排名柱状图（仅当 AUTO_SELECT 时）
if AUTO_SELECT:
    fig, ax = plt.subplots(figsize=(10, 6))
    top10 = df_uni.head(10)
    sns.barplot(data=top10, x='R2', y='Variable', palette='viridis')
    ax.set_xlabel('R²')
    ax.set_title('单变量回归 R² 排名（前10）')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_univariate_ranking.png", dpi=300)
    plt.close()
# ========================== 额外：绘制选定自变量与CDS的散点回归图 ==========================
# 选择最重要的自变量（系数绝对值最大且p<0.05）
if AUTO_SELECT:
    # 从重要性中筛选显著变量，选系数绝对值最大的
    sig_vars = importance[importance['P_value'] < 0.05]
    if not sig_vars.empty:
        main_var = sig_vars.iloc[0]['Variable']
    else:
        main_var = used_cols[0]  # 若无显著，选第一个
else:
    main_var = used_cols[0]  # 手动模式下取第一个

# 绘制散点回归图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_clean[main_var], df_clean['y'], alpha=0.6, label='样本点')
# 单变量回归线
X_single = sm.add_constant(df_clean[[main_var]])
model_single = sm.OLS(df_clean['y'], X_single).fit()
x_range = np.linspace(df_clean[main_var].min(), df_clean[main_var].max(), 100)
y_range = model_single.params['const'] + model_single.params[main_var] * x_range
ax.plot(x_range, y_range, 'r-', linewidth=2,
        label=f"斜率={model_single.params[main_var]:.3f} (p={model_single.pvalues[main_var]:.3f})")
ax.set_xlabel(f"{main_var} (标准化)")
ax.set_ylabel(y_label)
ax.set_title(f"{main_var} 与迁移损失 CDS 的回归关系")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_scatter_regression.png", dpi=300)
plt.close()
print(f"\n分析完成！所有结果已保存至前缀 {OUTPUT_PREFIX}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四变量分层回归（控制人口尺度）
CDS = β0 + β1·Shift_Label + β2·Shift_AEF + β3·Shift_GIS + β4·Shift_Joint
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== 加载数据 ========================
# 1. CDS
df_cds = pd.read_csv(r"results\transferability\ndvi\CDS_matrix_robust_entropy.csv", index_col=0)
df_cds_long = df_cds.stack().reset_index()
df_cds_long.columns = ["Source", "Target", "CDS_raw"]
df_cds_long = df_cds_long.dropna(subset=["CDS_raw"])
df_cds_long['Source'] = df_cds_long['Source'].astype(str).str.strip()
df_cds_long['Target'] = df_cds_long['Target'].astype(str).str.strip()

# 2. 加载四个自变量矩阵
matrices_dir = r"results/domain_shift/aef_plus_ndvi/with_gis"
var_list = ['Label_Shift', 'AEF_L2', 'GIS_L2', 'Weighted_L2']  # 假设您已有这些矩阵var_list = ['GIS_L2']  # 假设您已有这些矩阵



# 如果尚无 Label_Shift，需要计算城市人口密度均值差
# 这里假设您已从 city_cache 中计算并保存了 Label_Shift 矩阵
# 如果没有，可以用以下方式从您的 gdf 计算

df_merged = df_cds_long.copy()
for var in var_list:
    fname = f"{matrices_dir}/domain_shift_{var}_matrix.csv"
    try:
        mat = pd.read_csv(fname, index_col=0)
        mat.index = mat.index.astype(str).str.strip()
        mat.columns = mat.columns.astype(str).str.strip()
        long = mat.stack().reset_index()
        long.columns = ["Source", "Target", var]
        long['Source'] = long['Source'].astype(str).str.strip()
        long['Target'] = long['Target'].astype(str).str.strip()
        df_merged = df_merged.merge(long, on=["Source", "Target"], how="inner")
        print(f"加载 {var}，样本数: {len(long)}")
    except FileNotFoundError:
        print(f"警告: {fname} 不存在，跳过")
        # 如果是 Label_Shift，尝试从原始数据计算
        if var == 'Label_Shift':
            print("尝试从城市数据计算 Label_Shift...")
            # 这里需要您的城市人口密度均值字典，略

# 因变量 (取绝对值，因为您用了 abs_loss)
df_merged['y'] = np.abs(df_merged['CDS_raw'])
y_label = "|CDS| (绝对迁移损失)"

# 剔除缺失
df_merged = df_merged.dropna(subset=['y'] + var_list)
print(f"最终有效样本数: {len(df_merged)}")

# 标准化所有自变量
scaler = StandardScaler()
df_merged[var_list] = scaler.fit_transform(df_merged[var_list])

# ======================== 分层回归（Hierarchical Regression） ========================
models = {}
r2_delta = {}

# 模型1：仅 Label_Shift
X1 = sm.add_constant(df_merged[['Label_Shift']])
models['M1_Label'] = sm.OLS(df_merged['y'], X1).fit()

# 模型2：+ AEF_L2
X2 = sm.add_constant(df_merged[['Label_Shift', 'AEF_L2']])
models['M2_Label+AEF'] = sm.OLS(df_merged['y'], X2).fit()

# 模型3：+ GIS_L2
X3 = sm.add_constant(df_merged[['Label_Shift', 'AEF_L2', 'GIS_L2']])
models['M3_+GIS'] = sm.OLS(df_merged['y'], X3).fit()

# 模型4：+ Weighted_L2
X4 = sm.add_constant(df_merged[['Label_Shift', 'AEF_L2', 'GIS_L2', 'Weighted_L2']])
models['M4_+Joint'] = sm.OLS(df_merged['y'], X4).fit()

# ======================== 输出结果 ========================
print("\n" + "="*70)
print("分层回归结果（因变量：|CDS|）")
print("="*70)

comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'R²': [m.rsquared for m in models.values()],
    'Adj_R²': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()]
})
print("\n模型比较：")
print(comparison)

# ΔR²
print("\nΔR² (增量解释力)：")
print(f"M1→M2: {models['M2_Label+AEF'].rsquared - models['M1_Label'].rsquared:.4f}")
print(f"M2→M3: {models['M3_+GIS'].rsquared - models['M2_Label+AEF'].rsquared:.4f}")
print(f"M3→M4: {models['M4_+Joint'].rsquared - models['M3_+GIS'].rsquared:.4f}")

# 最终模型系数
print("\n最终模型 (M4) 系数详情：")
print(models['M4_+Joint'].summary().tables[1])

# ======================== 可视化：系数柱状图 ========================
fig, ax = plt.subplots(figsize=(8, 5))
coefs = models['M4_+Joint'].params[var_list]
pvals = models['M4_+Joint'].pvalues[var_list]
colors = ['#2E86AB' if p < 0.05 else '#D3D3D3' for p in pvals]
bars = ax.bar(var_list, coefs, color=colors)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('标准化回归系数 (β)')
ax.set_title('控制人口尺度后各偏移指标的独立贡献')
for bar, p in zip(bars, pvals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'p={p:.3f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("hierarchical_regression_coefficients.png", dpi=300)
plt.close()

print("\n分析完成！")
