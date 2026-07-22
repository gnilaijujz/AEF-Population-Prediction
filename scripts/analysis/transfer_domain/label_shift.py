#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Label_Shift 矩阵（城市间人口密度均值的绝对差）
用于分层回归中控制人口尺度偏移。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from GNN_transfer_experiments_calibrated import load_city, discover_cities
from GNN_regression import DEFAULT_POP_CSV

# ======================== 配置 ========================
AEF_ROOT = Path(r"model_data/aef_root/aef_plus_poi_diversity")          # 您的 AEF 根目录
POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
OUTPUT_DIR = Path(r"results/domain_shift/aef_plus_poi_diversity/with_gis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================== 1. 发现所有城市 ========================
cities = discover_cities(AEF_ROOT)
print(f"发现 {len(cities)} 个城市: {cities}")

# ======================== 2. 加载每个城市并计算人口密度均值 ========================
city_density_mean = {}
for city in cities:
    try:
        gdf, _, _ = load_city(AEF_ROOT, city, POP_CSV)
        # 计算人口密度均值（所有 tract 的平均）
        mean_density = gdf["population_density"].mean()
        city_density_mean[city] = mean_density
        print(f"{city}: {mean_density:.2f} 人/km²")
    except Exception as e:
        print(f"加载 {city} 失败: {e}")

# ======================== 3. 构建差值矩阵 ========================
cities_sorted = sorted(city_density_mean.keys())
n = len(cities_sorted)
mat = pd.DataFrame(index=cities_sorted, columns=cities_sorted, dtype=float)

for i, src in enumerate(cities_sorted):
    for j, tgt in enumerate(cities_sorted):
        if i == j:
            mat.loc[src, tgt] = 0.0
        else:
            diff = abs(city_density_mean[src] - city_density_mean[tgt])
            mat.loc[src, tgt] = diff

# ======================== 4. 保存 ========================
output_path = OUTPUT_DIR / "domain_shift_Label_Shift_matrix.csv"
mat.to_csv(output_path)
print(f"\nLabel_Shift 矩阵已保存至: {output_path}")

# 可选：打印统计摘要
values = mat.values.flatten()
values = values[~np.isnan(values)]
print(f"\n矩阵统计: 均值={np.mean(values):.2f}, 标准差={np.std(values):.2f}, 最大值={np.max(values):.2f}")