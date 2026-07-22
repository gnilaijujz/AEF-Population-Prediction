#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分层回归：使用 Label Shift 和 GIS Shift 解释迁移性能（R²），控制人口尺度。
- Label Shift：目标变量（人口密度）的 Wasserstein 距离。
- GIS Shift：输入特征（GIS）的分布差异（例如均值 L2 距离）。
- 控制变量：源/目标城市的人口规模（例如人口总数中位数）。
"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ======================== 配置 ========================
# 数据路径
PERF_CSV = r"results/transfer_results/gis\transfer_matrix_r2_gnn.csv"   # R² 矩阵
GIS_SHIFT_CSV = r"results/domain_shift/gis\domain_shift_L2_Dist_matrix.csv"  # GIS 距离矩阵
# 需要城市数据目录以计算 Label Shift 和人口
AEF_ROOT = r"model_data/aef_root/clean_gis_shapefiles"
POP_CSV = r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv"

OUT_DIR = r"results/domain_shift/gis/with_label"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================== 辅助函数 ========================
def load_city_data(city, aef_root, pop_csv):
    from GNN_transfer_experiments_calibrated import load_city
    from pathlib import Path
    gdf, _, _ = load_city(Path(aef_root), city, Path(pop_csv))
    return gdf

def compute_label_shift_matrix(cities, aef_root, pop_csv):
    """计算城市间人口密度分布的 Wasserstein 距离矩阵"""
    density_dict = {}
    for city in cities:
        gdf = load_city_data(city, aef_root, pop_csv)
        densities = gdf['population_density'].dropna().values
        if len(densities) == 0:
            densities = np.array([0.0])
        density_dict[city] = densities

    n = len(cities)
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for i, src in enumerate(cities):
        for j, tgt in enumerate(cities):
            if i == j:
                mat.loc[src, tgt] = 0.0
            else:
                d_src = density_dict[src]
                d_tgt = density_dict[tgt]
                dist = wasserstein_distance(d_src, d_tgt)
                mat.loc[src, tgt] = dist
    return mat

def get_population_control(cities, aef_root, pop_csv):
    """获取每个城市的人口规模（中位数）"""
    pop_median = {}
    for city in cities:
        gdf = load_city_data(city, aef_root, pop_csv)
        pop_val = gdf['population'].median()
        pop_median[city] = pop_val
    return pd.Series(pop_median)

def filter_outliers_iqr(df, columns, multiplier=1.5):
    """
    对指定列分别计算 IQR，剔除任何一列超出 (Q1 - k*IQR, Q3 + k*IQR) 的样本。
    返回过滤后的 DataFrame 和被剔除的索引。
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask
    return df[mask], df[~mask]

# ======================== 1. 加载数据 ========================
perf_mat = pd.read_csv(PERF_CSV, index_col=0)
gis_shift_mat = pd.read_csv(GIS_SHIFT_CSV, index_col=0)

cities = sorted(set(perf_mat.index) & set(perf_mat.columns) &
                set(gis_shift_mat.index) & set(gis_shift_mat.columns))
print(f"共有 {len(cities)} 个城市: {cities}")

perf_mat = perf_mat.loc[cities, cities]
gis_shift_mat = gis_shift_mat.loc[cities, cities]

# 2. 计算 Label Shift
print("正在计算 Label Shift...")
label_shift_mat = compute_label_shift_matrix(cities, AEF_ROOT, POP_CSV)
label_shift_mat.to_csv(f"{OUT_DIR}/label_shift_matrix.csv")

# 3. 获取控制变量
pop_control = get_population_control(cities, AEF_ROOT, POP_CSV)

# ======================== 4. 构成长格式数据 ========================
rows = []
for src, tgt in itertools.product(cities, repeat=2):
    if src == tgt:
        continue
    r2 = perf_mat.loc[src, tgt]
    gis_shift = gis_shift_mat.loc[src, tgt]
    label_shift = label_shift_mat.loc[src, tgt]
    if not (np.isnan(r2) or np.isnan(gis_shift) or np.isnan(label_shift)):
        rows.append({
            'Source': src,
            'Target': tgt,
            'R2': r2,
            'GIS_Shift': gis_shift,
            'Label_Shift': label_shift,
            'Pop_Source': pop_control[src],
            'Pop_Target': pop_control[tgt]
        })
df = pd.DataFrame(rows)
print(f"原始样本数: {len(df)}")

# ======================== 5. IQR 异常值剔除 ========================
print("\n===== 应用 IQR 准则剔除异常值 =====")
# 对关键变量进行 IQR 过滤（使用 1.5 倍 IQR，也可调整为 2.0）
filter_cols = ['R2', 'GIS_Shift', 'Label_Shift', 'Pop_Source', 'Pop_Target']
df_clean, df_outliers = filter_outliers_iqr(df, filter_cols, multiplier=1.5)
print(f"剔除样本数: {len(df_outliers)}")
print(f"保留样本数: {len(df_clean)}")

# 保存被剔除的样本（供检查）
df_outliers.to_csv(f"{OUT_DIR}/outliers_removed.csv", index=False)
df_clean.to_csv(f"{OUT_DIR}/clean_data.csv", index=False)

# ======================== 6. 回归分析（使用清洗后数据） ========================
df = df_clean.copy()
scaler = StandardScaler()
df[['GIS_Shift_scaled', 'Label_Shift_scaled', 'Pop_Source_scaled', 'Pop_Target_scaled']] = scaler.fit_transform(
    df[['GIS_Shift', 'Label_Shift', 'Pop_Source', 'Pop_Target']]
)

X_cols = ['GIS_Shift_scaled', 'Label_Shift_scaled', 'Pop_Source_scaled', 'Pop_Target_scaled']
y_col = 'R2'

# 整体回归
X = df[X_cols].values
y = df[y_col].values
model_all = LinearRegression()
model_all.fit(X, y)
y_pred = model_all.predict(X)
r2_all = r2_score(y, y_pred)

print("\n===== 整体回归结果 (清洗后) =====")
print(f"R²: {r2_all:.4f}")
print("系数:")
for col, coef in zip(X_cols, model_all.coef_):
    print(f"  {col}: {coef:.4f}")
print(f"截距: {model_all.intercept_:.4f}")

pd.DataFrame({
    '变量': ['截距'] + X_cols,
    '系数': [model_all.intercept_] + list(model_all.coef_)
}).to_csv(f"{OUT_DIR}/overall_regression_coef.csv", index=False)

# 分层回归
source_groups = df.groupby('Source')
results = []
for src, group in source_groups:
    if len(group) < 5:
        continue
    Xg = group[X_cols].values
    yg = group[y_col].values
    model = LinearRegression()
    model.fit(Xg, yg)
    y_pred_g = model.predict(Xg)
    r2_g = r2_score(yg, y_pred_g)
    results.append({
        'Source': src,
        'n': len(group),
        'R2': r2_g,
        'Intercept': model.intercept_,
        'Coef_GIS': model.coef_[0],
        'Coef_Label': model.coef_[1],
        'Coef_PopSrc': model.coef_[2],
        'Coef_PopTgt': model.coef_[3]
    })
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/stratified_regression_results.csv", index=False)
print(f"\n分层回归结果已保存，共 {len(results_df)} 个源城市。")

# ======================== 7. 可视化（使用清洗后数据） ========================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='GIS_Shift', y='R2', hue='Source', alpha=0.6, legend='full')
plt.xlabel('GIS Shift (L2 distance)')
plt.ylabel('Transfer R²')
plt.title('GIS Shift vs Performance (after IQR filtering)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/GIS_shift_vs_R2_clean.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Label_Shift', y='R2', hue='Source', alpha=0.6, legend='full')
plt.xlabel('Label Shift (Wasserstein distance)')
plt.ylabel('Transfer R²')
plt.title('Label Shift vs Performance (after IQR filtering)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Label_shift_vs_R2_clean.png", dpi=300)
plt.close()

# 残差图
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=df['R2']-y_pred, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted R²')
plt.ylabel('Residuals')
plt.axhline(0, linestyle='--', color='gray')
plt.title('Residual Plot (Overall, after IQR)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residual_plot_clean.png", dpi=300)
plt.close()

print(f"\n所有结果已保存至 {OUT_DIR}")