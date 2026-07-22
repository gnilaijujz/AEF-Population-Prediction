#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域偏移指标矩阵构建（手动指定GIS特征列）
基于原始 AEF 和 GIS 特征（npoi_water_distance_density），计算独立的距离矩阵及加权组合距离。
"""

import os
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 导入您的数据加载函数
from GNN_transfer_experiments_calibrated import load_city, discover_cities
from GNN_regression import resolve_aef_dir
# -------------------------- 中文字体 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set(font='SimHei')
# ======================== 配置 ========================
AEF_ROOT = r"model_data/aef_root/aef_plus_water_distance"
POP_CSV = r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv"
OUT_DIR = r"results/domain_shift/aef_plus_water_distance/with_gis"
os.makedirs(OUT_DIR, exist_ok=True)

# AEF 特征列（固定）
AEF_COLS = [f'A{i:02d}' for i in range(64)]

# ---- 手动指定 GIS 特征列 ----
GIS_COLS = ['dist_water_m']   # 可添加更多列

# 特征维度
DIM_AEF = len(AEF_COLS)
DIM_GIS = len(GIS_COLS)
print(f"AEF 维度: {DIM_AEF}, GIS 维度: {DIM_GIS}")

# 计算权重（按维度数倒数加权）
if DIM_GIS > 0:
    w_aef = 1 / DIM_AEF
    w_gis = 1 / DIM_GIS
    total = w_aef + w_gis
    w_aef /= total
    w_gis /= total
else:
    w_aef, w_gis = 1.0, 0.0
print(f"AEF 权重: {w_aef:.4f}, GIS 权重: {w_gis:.4f}")

# ======================== 定义距离计算函数（提前定义） ========================
def compute_mean_l2_matrix(feature_dict, group='AEF'):
    """计算均值向量间的 L2 距离"""
    cities = list(feature_dict.keys())
    n = len(cities)
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for i, src in enumerate(cities):
        for j, tgt in enumerate(cities):
            if i == j:
                mat.loc[src, tgt] = 0.0
            else:
                feat_src = feature_dict[src][group]
                feat_tgt = feature_dict[tgt][group]
                if feat_src is None or feat_tgt is None:
                    mat.loc[src, tgt] = np.nan
                else:
                    diff = np.mean(feat_src, axis=0) - np.mean(feat_tgt, axis=0)
                    mat.loc[src, tgt] = np.linalg.norm(diff)
    return mat

def compute_weighted_l2_matrix(feature_dict, w_aef, w_gis):
    """计算加权组合距离"""
    cities = list(feature_dict.keys())
    n = len(cities)
    mat = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for i, src in enumerate(cities):
        for j, tgt in enumerate(cities):
            if i == j:
                mat.loc[src, tgt] = 0.0
            else:
                aef_src = feature_dict[src]['AEF']
                aef_tgt = feature_dict[tgt]['AEF']
                gis_src = feature_dict[src]['GIS']
                gis_tgt = feature_dict[tgt]['GIS']
                if gis_src is None or gis_tgt is None:
                    mat.loc[src, tgt] = np.nan
                else:
                    diff_aef = np.mean(aef_src, axis=0) - np.mean(aef_tgt, axis=0)
                    diff_gis = np.mean(gis_src, axis=0) - np.mean(gis_tgt, axis=0)
                    dist = np.sqrt(w_aef * np.sum(diff_aef**2) + w_gis * np.sum(diff_gis**2))
                    mat.loc[src, tgt] = dist
    return mat
import geopandas as gpd

def load_city_manual(aef_root, city):
    """
    手动加载城市数据：shapefile + CSV（包含 AEF 和 GIS 列）
    返回 gdf 和特征列列表
    """
    city_dir = Path(aef_root) / city
    if not city_dir.exists():
        raise FileNotFoundError(f"目录不存在: {city_dir}")

    # ---------- 读取 shapefile ----------
    shp_files = list(city_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"未找到 shapefile: {city_dir}")
    # 优先选择带 "2020_us_MSA" 的文件（与原有日志一致）
    shp_path = next((f for f in shp_files if "2020_us_MSA" in f.name), shp_files[0])
    gdf = gpd.read_file(shp_path)
    gdf['GEO_ID'] = gdf['cb_2020__3'].astype(str).str.strip()

    # ---------- 读取所有 aef_*.csv 文件 ----------
    csv_files = sorted(city_dir.glob("aef_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 AEF CSV 文件: {city_dir}")

    # 确定需要的列：TRACT_ID + AEF_COLS + GIS_COLS
    needed_cols = ['TRACT_ID'] + AEF_COLS + GIS_COLS
    parts = []
    for f in csv_files:
        df = pd.read_csv(f, dtype={"TRACT_ID": str})
        # 只保留实际存在的列
        exist_cols = [c for c in needed_cols if c in df.columns]
        if exist_cols:
            parts.append(df[exist_cols].drop_duplicates("TRACT_ID"))

    if not parts:
        raise ValueError(f"{city} 的 CSV 中未找到任何 AEF/GIS 列")

    df_all = pd.concat(parts, ignore_index=True).drop_duplicates("TRACT_ID")
    df_all['TRACT_ID'] = df_all['TRACT_ID'].astype(str).str.strip()

    # ---------- 合并到 gdf ----------
    gdf = gdf.merge(df_all, left_on='GEO_ID', right_on='TRACT_ID', how='left')

    # 可用特征列（实际存在于合并后的数据中）
    feature_cols = [c for c in AEF_COLS + GIS_COLS if c in gdf.columns]

    return gdf, feature_cols
# ======================== 1. 加载所有城市数据（自定义） ========================
all_cities = discover_cities(Path(AEF_ROOT))   # 仍可使用 discover_cities 获取城市列表
print(f"发现城市: {all_cities}")

city_cache = {}
for city in all_cities:
    try:
        gdf, feature_cols = load_city_manual(Path(AEF_ROOT), city)
        city_cache[city] = {
            'gdf': gdf,
            'feature_cols': feature_cols
        }
        print(f"已加载 {city}, 特征数量: {len(feature_cols)}")
    except Exception as e:
        print(f"加载 {city} 失败: {e}")

# 过滤掉加载失败的城市
cities = [c for c in all_cities if c in city_cache]
if not cities:
    raise RuntimeError("没有成功加载任何城市，请检查数据路径。")



# ======================== 3. 提取特征向量 ========================
city_features = {}
for city in cities:
    gdf = city_cache[city]['gdf']
    aef_data = gdf[AEF_COLS].to_numpy(dtype=np.float32)

    if DIM_GIS > 0:
        # 检查哪些 GIS 列实际存在
        actual_gis = [c for c in GIS_COLS if c in gdf.columns]
        if actual_gis:
            gis_subset = gdf[actual_gis].dropna()
            if len(gis_subset) > 0:
                gis_data = gis_subset.to_numpy(dtype=np.float32)
                scaler = StandardScaler()
                gis_data = scaler.fit_transform(gis_data)
            else:
                gis_data = None
        else:
            gis_data = None
    else:
        gis_data = None

    city_features[city] = {'AEF': aef_data, 'GIS': gis_data}

# ======================== 4. 计算距离矩阵 ========================
# AEF 距离
aef_l2 = compute_mean_l2_matrix(city_features, 'AEF')
aef_l2.to_csv(f"{OUT_DIR}/domain_shift_AEF_L2_matrix.csv")
print("已保存 AEF_L2 矩阵")

# GIS 距离（仅对有效城市）
gis_valid_cities = [c for c in cities if city_features[c]['GIS'] is not None]
if DIM_GIS > 0 and gis_valid_cities:
    valid_features = {c: city_features[c] for c in gis_valid_cities}
    gis_l2 = compute_mean_l2_matrix(valid_features, 'GIS')
    gis_l2.to_csv(f"{OUT_DIR}/gis_L2_matrix.csv")
    print(f"已保存 GIS_L2 矩阵（基于 {len(gis_valid_cities)} 个城市）")
else:
    print("警告: 无有效 GIS 数据，跳过 GIS_L2 矩阵")

# 加权组合距离（仅对同时具有 AEF 和 GIS 的城市对）
if gis_valid_cities:
    filtered_features = {c: city_features[c] for c in gis_valid_cities}
    weighted_l2 = compute_weighted_l2_matrix(filtered_features, w_aef, w_gis)
    weighted_l2.to_csv(f"{OUT_DIR}/domain_shift_Weighted_L2_matrix.csv")
    print("已保存 Weighted_L2 矩阵")
else:
    print("警告: 无有效 GIS 数据，跳过 Weighted_L2 矩阵")

# ======================== 5. 热力图 ========================
try:
    plot_mats = [('AEF', aef_l2)]
    if 'gis_l2' in locals() and gis_l2 is not None:
        plot_mats.append(('GIS', gis_l2))
    if 'weighted_l2' in locals() and weighted_l2 is not None:
        plot_mats.append(('Weighted', weighted_l2))
    
    fig, axes = plt.subplots(1, len(plot_mats), figsize=(5*len(plot_mats), 4))
    if len(plot_mats) == 1:
        axes = [axes]
    for ax, (name, mat) in zip(axes, plot_mats):
        sns.heatmap(mat, annot=False, cmap='Reds', square=True, ax=ax)
        ax.set_title(f'{name} L2 距离')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_distance_heatmaps.png", dpi=300)
    plt.close()
    print("已保存热力图")
except Exception as e:
    print(f"绘图跳过: {e}")

print(f"\n所有矩阵已保存至 {OUT_DIR}")
