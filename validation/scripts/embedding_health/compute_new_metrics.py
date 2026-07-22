#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新城市嵌入健康度计算（基于预训练模型和数据）
完全复用 old_deep.py 的加载逻辑，只计算纯几何指标。
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== 导入 old_deep.py 中的必要函数 ========================
# 确保以下模块在你的 Python 路径中
from GNN_transfer_experiments_calibrated import (
    load_pretrained_gnn,  discover_cities,
    scan_model_files, prepare_target_data
)
from GNN_regression import DEFAULT_POP_CSV

# ======================== 配置 ========================
# 新城市数据根目录（包含所有城市子文件夹）
NEW_AEF_ROOT = Path(r"validation\data\aef_root\new_15_AEF")
print(f"当前扫描的根目录是: {NEW_AEF_ROOT}")
# 存放预训练模型文件的目录（.pt 文件）
PRETRAINED_DIR = Path(r"validation\data\pretrained_models")
# 人口普查 CSV（用于加载城市数据时获取人口密度，虽然几何指标用不到，但 load_city 需要）
POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
import os
path = r"validation\data\aef_root\new_15_AEF"
print(os.listdir(path))
print(len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]))
# 输出路径
OUTPUT_CSV = Path(r"validation\results\embedding_health_deep_analysis\new_cities_health_metrics.csv")
OUTPUT_DIR = OUTPUT_CSV.parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型超参数（与 old_deep.py 一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 64
DROPOUT = 0.5

# 几何指标参数
K_NEIGHBORS = 5
PCA_COMPONENTS = 5
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_city_flex(root, city, pop_csv=None):
    """
    完全灵活的 load_city 替代函数。
    自动识别 GEOID / TRACT_ID，无需修改原始数据。
    返回: (gdf, feat_cols, edge_index)
    """
    root = Path(root)
    city_path = root / city

    # ---------- 1. 加载 Shapefile ----------
    shp_files = list(city_path.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"未找到 .shp 文件: {city_path}")
    gdf = gpd.read_file(shp_files[0])

    # ---------- 2. 加载 AEF 属性 CSV ----------
    csv_files = list(city_path.glob("aef_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 aef_*.csv 文件: {city_path}")
    df = pd.read_csv(csv_files[0])

    # ---------- 3. 智能识别 ID 列（支持 GEOID / TRACT_ID） ----------
    # 在 gdf 中查找 ID 列
    gdf_id_col = None
    for col in ['GEO_ID', 'TRACT_ID', 'tract_id', 'geoid', 'TRACT']:
        if col in gdf.columns:
            gdf_id_col = col
            break

    # 在 csv 中查找 ID 列
    csv_id_col = None
    for col in ['GEO_ID', 'TRACT_ID', 'tract_id', 'geoid', 'TRACT']:
        if col in df.columns:
            csv_id_col = col
            break

    # 如果都找不到，用行索引强行对齐（极端情况，通常不会发生）
    if gdf_id_col is None or csv_id_col is None:
        print(f"⚠️ 警告: 未找到标准 ID 列，使用行索引对齐 (city={city})")
        gdf['TRACT_ID'] = gdf.index.astype(str)
        df['TRACT_ID'] = df.index.astype(str)
    else:
        # 统一重命名为 'TRACT_ID'（后面的代码只认这个名字）
        gdf = gdf.rename(columns={gdf_id_col: 'TRACT_ID'})
        df = df.rename(columns={csv_id_col: 'TRACT_ID'})
        gdf['TRACT_ID'] = gdf['TRACT_ID'].astype(str).str.strip()
        df['TRACT_ID'] = df['TRACT_ID'].astype(str).str.strip()

    # ---------- 4. 合并属性到地理数据框 ----------
    gdf = gdf.merge(df, on='TRACT_ID', how='left')
    print(f"  ✓ 数据加载成功: {city}, 节点数: {len(gdf)}")

    # ---------- 5. 提取特征列（自动过滤非数值列） ----------
    exclude = ['TRACT_ID', 'geometry', 'OBJECTID', 'FID', 'Shape_Length', 'Shape_Area']
    feat_cols = [
        col for col in gdf.columns 
        if col not in exclude and pd.api.types.is_numeric_dtype(gdf[col])
    ]

    # ---------- 6. 加载或构建边索引（edge_index） ----------
    # 优先加载预存的 .npy 文件（保证与训练时图结构一致）
    edge_file = city_path / "edge_index.npy"
    if edge_file.exists():
        edge_index = np.load(edge_file)
        print(f"  ✓ 加载预存边索引: {edge_index.shape}")
    else:
        # 备选：使用空间邻接（KNN）构建（仅当没有预存文件时）
        print(f"  ⚠️ 未找到 edge_index.npy，使用空间 KNN 构建 (可能与训练不一致)")
        from scipy.spatial import KDTree
        coords = np.array([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).T
        tree = KDTree(coords)
        # k=6 保证连通性
        distances, indices = tree.query(coords, k=6)
        edge_list = []
        for i, neighs in enumerate(indices):
            for j in neighs:
                if i != j:
                    edge_list.append([i, j])
        edge_index = np.array(edge_list).T
        # 去重
        edge_index = np.unique(edge_index, axis=1)
        edge_index = edge_index.astype(np.int64)

    return gdf, feat_cols, edge_index
# ======================== 纯几何指标函数（从 old_deep.py 提取） ========================
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd

def compute_effective_rank(emb):
    emb_std = StandardScaler().fit_transform(emb)
    U, S, Vt = svd(emb_std, full_matrices=False)
    p = S / (S.sum() + 1e-10)
    entropy = -np.sum(p * np.log(p + 1e-10))
    return np.exp(entropy)

def compute_isotropy(emb):
    emb_centered = emb - emb.mean(axis=0)
    cov = np.cov(emb_centered.T)
    cov += 1e-6 * np.eye(cov.shape[0])
    cond_num = np.linalg.cond(cov)
    return 1 / cond_num, cond_num

def compute_pca_variance(emb, n_components=PCA_COMPONENTS):
    emb_std = StandardScaler().fit_transform(emb)
    n_comp = min(n_components, emb_std.shape[1], emb_std.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(emb_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return cumsum[-1] if len(cumsum) > 0 else np.nan

def compute_embedding_diversity(emb, k=K_NEIGHBORS):
    n = emb.shape[0]
    if n < k + 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(emb)
    distances, _ = nn.kneighbors(emb)
    return np.mean(distances[:, 1:])

# ======================== 主流程 ========================
# 1. 发现所有城市
all_cities = discover_cities(NEW_AEF_ROOT)
print(f"发现 {len(all_cities)} 个城市: {all_cities}")

# 2. 扫描模型文件，建立城市到模型路径的映射
model_files = scan_model_files(PRETRAINED_DIR)
print(f"找到 {len(model_files)} 个模型文件")

# 3. 加载城市缓存（避免重复加载）
city_cache = {}
for city in all_cities:
    try:
        gdf, feat_cols, edge_idx = load_city_flex(NEW_AEF_ROOT, city, POP_CSV)
        city_cache[city] = (gdf, feat_cols, edge_idx)
        print(f"  ✓ 已加载城市数据: {city}")
    except Exception as e:
        print(f"  ✗ 加载城市数据失败 {city}: {e}")

# 4. 遍历城市，提取嵌入并计算指标
results = []
for city in all_cities:
    if city not in city_cache:
        print(f"跳过 {city}: 城市数据未加载")
        continue
    if city not in model_files:
        print(f"跳过 {city}: 未找到模型文件 {city}_GraphSAGE_model.pt")
        continue

    try:
        # 加载模型
        model, x_scaler, y_mean, y_std, feat_cols, target_transform = load_pretrained_gnn(
            model_files[city], HIDDEN_CHANNELS, DROPOUT, DEVICE
        )
        # 提取嵌入
        gdf, _, edge_index = city_cache[city]
        data, _ = prepare_target_data(
            gdf, feat_cols, edge_index, x_scaler, y_mean, y_std,
            target_transform, DEVICE, shuffle_features=False
        )
        with torch.no_grad():
            emb = model.forward_emb(data.x, data.edge_index).detach().cpu().numpy()
        print(f"  ✓ 提取嵌入 {city}: 形状 {emb.shape}")

        # 计算几何指标
        n_nodes = emb.shape[0]
        eff_rank = compute_effective_rank(emb)
        isotropy, cond_num = compute_isotropy(emb)
        pca_5 = compute_pca_variance(emb, n_components=PCA_COMPONENTS)
        diversity = compute_embedding_diversity(emb, k=K_NEIGHBORS)

        results.append({
            'city': city,
            'n_nodes': n_nodes,
            'effective_rank': eff_rank,
            'isotropy': isotropy,
            'condition_number': cond_num,
            'pca_5_explained': pca_5,
            'embedding_diversity': diversity,
        })
        print(f"     effective_rank: {eff_rank:.4f}, isotropy: {isotropy:.4f}, diversity: {diversity:.4f}")

    except Exception as e:
        print(f"  ❌ 处理 {city} 时出错: {e}")

# 5. 保存结果
if results:
    df_new = pd.DataFrame(results).set_index('city')
    if OUTPUT_CSV.exists():
        df_existing = pd.read_csv(OUTPUT_CSV, index_col=0)
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined.to_csv(OUTPUT_CSV)
        print(f"✅ 已更新 {OUTPUT_CSV}")
    else:
        df_new.to_csv(OUTPUT_CSV)
        print(f"✅ 已保存至 {OUTPUT_CSV}")
    print("\n新城市特征计算完成！")
    print(df_new)
else:
    print("❌ 未计算出任何城市的结果。")