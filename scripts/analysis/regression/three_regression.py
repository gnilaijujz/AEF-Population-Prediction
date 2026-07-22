#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三层偏移分层归因回归（Label尺度 + AEF嵌入 + GIS地理）
完美衔接成果2（域偏移回归）和成果3（归因分析）
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform, jensenshannon
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== 1. 加载三个偏移矩阵 ========================
# 1.1 标签尺度偏移（人口密度均值绝对差）
def compute_label_shift_matrix(city_gdf_dict):
    """计算城市间人口密度均值的绝对差"""
    cities = list(city_gdf_dict.keys())
    densities = {c: gdf['population_density'].mean() for c, gdf in city_gdf_dict.items()}
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for i in cities:
        for j in cities:
            mat.loc[i, j] = abs(densities[i] - densities[j])
    return mat

# 1.2 AEF嵌入偏移（您已有的L2_Dist矩阵）
df_aef_shift = pd.read_csv("domain_shift_matrices/domain_shift_L2_Dist_matrix.csv", index_col=0)

# 1.3 GIS地理偏移（新计算，独立于AEF）
def compute_gis_shift_matrix(city_gdf_dict):
    """
    从每个城市的GeoDataFrame中提取地理指纹（气候、地形、POI结构），
    计算城市间的欧氏距离。
    """
    cities = list(city_gdf_dict.keys())
    fingerprints = {}
    
    for city, gdf in city_gdf_dict.items():
        fp = []
        # 地形/气候：用centroid纬度和经度（作为气候代理）
        centroids = gdf.geometry.centroid
        fp.append(centroids.y.mean())  # 纬度
        fp.append(centroids.y.std())   # 纬度变化
        
        # 景观格局：紧凑度（面积/周长^2）
        gdf['perimeter'] = gdf.geometry.length
        gdf['area'] = gdf.geometry.area
        gdf['compactness'] = gdf['area'] / (gdf['perimeter']**2 + 1e-6)
        fp.append(gdf['compactness'].mean())
        fp.append(gdf['compactness'].std())
        
        # POI密度（如果有poi_count列）
        if 'poi_count' in gdf.columns:
            area_km2 = gdf.geometry.area / 1e6
            poi_density = gdf['poi_count'] / area_km2
            fp.append(poi_density.mean())
            fp.append(poi_density.std())
        
        # 补齐到固定长度（10维）
        if len(fp) < 10:
            fp += [0.0] * (10 - len(fp))
        else:
            fp = fp[:10]
        fingerprints[city] = np.array(fp)
    
    # 标准化并计算欧氏距离
    all_fp = np.array([fingerprints[c] for c in cities])
    all_fp_scaled = StandardScaler().fit_transform(all_fp)
    D_geo = squareform(pdist(all_fp_scaled, metric='euclidean'))
    return pd.DataFrame(D_geo, index=cities, columns=cities)

# ======================== 2. 加载CDS矩阵 ========================
df_cds = pd.read_csv("CDS_matrix.csv", index_col=0)

# 确保所有矩阵的城市顺序一致
common_cities = sorted(set(df_cds.index) & set(df_aef_shift.index))
df_cds = df_cds.loc[common_cities, common_cities]
df_aef_shift = df_aef_shift.loc[common_cities, common_cities]
df_label_shift = compute_label_shift_matrix(city_gdf_dict).loc[common_cities, common_cities]
df_gis_shift = compute_gis_shift_matrix(city_gdf_dict).loc[common_cities, common_cities]

# ======================== 3. 构建长格式回归数据集 ========================
long_data = []
for src in common_cities:
    for tgt in common_cities:
        if src == tgt:
            continue
        long_data.append({
            'Source': src,
            'Target': tgt,
            'CDS': df_cds.loc[src, tgt],
            'Shift_Label': df_label_shift.loc[src, tgt],
            'Shift_AEF': df_aef_shift.loc[src, tgt],
            'Shift_GIS': df_gis_shift.loc[src, tgt]
        })
df_long = pd.DataFrame(long_data)

# 标准化所有自变量（消除量纲，方便比较系数）
from sklearn.preprocessing import StandardScaler
scalers = {}
for col in ['Shift_Label', 'Shift_AEF', 'Shift_GIS']:
    scaler = StandardScaler()
    df_long[col] = scaler.fit_transform(df_long[[col]])
    scalers[col] = scaler

# ======================== 4. 分层回归（核心） ========================
# 模型1：仅标签尺度
X1 = sm.add_constant(df_long[['Shift_Label']])
model1 = sm.OLS(df_long['CDS'], X1).fit()

# 模型2：标签尺度 + AEF偏移
X2 = sm.add_constant(df_long[['Shift_Label', 'Shift_AEF']])
model2 = sm.OLS(df_long['CDS'], X2).fit()

# 模型3：标签尺度 + AEF偏移 + GIS偏移（完整模型）
X3 = sm.add_constant(df_long[['Shift_Label', 'Shift_AEF', 'Shift_GIS']])
model3 = sm.OLS(df_long['CDS'], X3).fit()

# 输出对比
print("="*70)
print("分层回归模型比较（被解释变量：迁移损失 CDS）")
print("="*70)
print(f"模型1 (仅Label尺度): R² = {model1.rsquared:.4f}, AdjR² = {model1.rsquared_adj:.4f}")
print(f"模型2 (+ AEF抽象偏移): R² = {model2.rsquared:.4f}, AdjR² = {model2.rsquared_adj:.4f}, ΔR² = {model2.rsquared - model1.rsquared:.4f}")
print(f"模型3 (+ GIS地理偏移): R² = {model3.rsquared:.4f}, AdjR² = {model3.rsquared_adj:.4f}, ΔR² = {model3.rsquared - model2.rsquared:.4f}")
print("\n模型3 系数详情：")
print(model3.summary().tables[1])

# 保存系数对比图
coef_df = pd.DataFrame({
    'Variable': ['Shift_Label', 'Shift_AEF', 'Shift_GIS'],
    'Coefficient': model3.params[['Shift_Label', 'Shift_AEF', 'Shift_GIS']].values,
    'P_value': model3.pvalues[['Shift_Label', 'Shift_AEF', 'Shift_GIS']].values
})
coef_df['Significant'] = coef_df['P_value'] < 0.05

# ======================== 5. 可视化 ========================
# 图1：标准化系数柱状图（直接回答“哪个因素伤害最大”）
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(coef_df['Variable'], coef_df['Coefficient'], 
              color=['#2E86AB' if p<0.05 else '#D3D3D3' for p in coef_df['P_value']])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('标准化回归系数 (β)')
ax.set_title('迁移损失归因：各偏移维度的独立贡献\n(灰色表示不显著, p>0.05)')
for bar, p in zip(bars, coef_df['P_value']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'p={p:.3f}', ha='center', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("fig_attribution_hierarchical_regression.png", dpi=300)
plt.close()

# 图2：GIS偏移与CDS的散点图（控制其他变量后的偏残差图）
# 计算偏残差：剔除Label和AEF的影响
resid_cds = sm.OLS(df_long['CDS'], sm.add_constant(df_long[['Shift_Label', 'Shift_AEF']])).resid
resid_gis = sm.OLS(df_long['Shift_GIS'], sm.add_constant(df_long[['Shift_Label', 'Shift_AEF']])).resid

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(resid_gis, resid_cds, alpha=0.6)
# 拟合偏回归线
X_partial = sm.add_constant(resid_gis)
model_partial = sm.OLS(resid_cds, X_partial).fit()
x_range = np.linspace(resid_gis.min(), resid_gis.max(), 100)
y_range = model_partial.params['const'] + model_partial.params['x1'] * x_range
ax.plot(x_range, y_range, 'r-', label=f"偏斜率 = {model_partial.params['x1']:.4f}, p={model_partial.pvalues['x1']:.4f}")
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('GIS地理偏移 (控制Label和AEF后)')
ax.set_ylabel('迁移损失CDS (控制Label和AEF后)')
ax.set_title('GIS地理偏移的独立效应（偏残差图）')
ax.legend()
ax.grid(True)
plt.savefig("fig_gis_partial_regression_plot.png", dpi=300)
plt.close()

print("\n归因分析完成！图表已保存。")