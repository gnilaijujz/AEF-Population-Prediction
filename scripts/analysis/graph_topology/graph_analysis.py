#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图拓扑偏移细粒度分析
从 GNN 训练时保存的 edge_index 或从 GeoDataFrame 重建图，
计算源-目标对的图拓扑差异，回归到 CDS。
"""

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== 配置 ========================
CDS_MATRIX_FILE = "CDS_matrix.csv"
EDGE_INDEX_DIR = "edge_index_cache"   # 存储每个城市的 edge_index 或 GeoDataFrame
OUTPUT_PREFIX = "topology_shift_analysis"

# ======================== 加载 CDS ========================
df_cds = pd.read_csv(CDS_MATRIX_FILE, index_col=0)
df_long = df_cds.stack().reset_index()
df_long.columns = ["Source", "Target", "CDS_raw"]
df_long = df_long.dropna()

# ======================== 计算图拓扑指标 ========================
# 假设您已从 GNN_transfer_experiments_calibrated.py 中缓存了每个城市的 edge_index
# 此处演示从 GeoDataFrame 重建（若未缓存，需先运行 load_city 获取 gdf）
# 实际使用时，您可以从 city_cache 中获取 gdf 并构建图

def compute_graph_metrics(gdf):
    """从 GeoDataFrame 计算图拓扑指标"""
    # 构建 Queen 邻接图
    from libpysal.weights import Queen
    w = Queen.from_dataframe(gdf)
    G = nx.from_dict_of_lists(w.neighbors)
    
    # 基本指标
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = 2 * n_edges / n_nodes
    clustering = nx.average_clustering(G)
    # 平均最短路径（需连通，取最大连通子图）
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc)
    avg_path_length = nx.average_shortest_path_length(G_largest) if len(G_largest) > 1 else np.nan
    # 模块度（使用 Louvain 社区检测）
    try:
        import community.community_louvain as community_louvain
        partition = community_louvain.best_partition(G)
        mod = community_louvain.modularity(partition, G)
    except:
        mod = np.nan
    # 谱距离（归一化拉普拉斯特征值前K个的Wasserstein距离）
    # 此处略，因为已有 Spectral_Dist 矩阵
    
    return {
        'avg_degree': avg_degree,
        'clustering': clustering,
        'avg_path_length': avg_path_length,
        'modularity': mod,
        'n_nodes': n_nodes,
        'n_edges': n_edges
    }

# 假设您有一个字典 city_gdf {city: gdf}
# 此处用占位，实际需替换
# city_gdf = load_all_cities()
# metrics_dict = {city: compute_graph_metrics(gdf) for city, gdf in city_gdf.items()}

# 为演示，生成模拟数据
np.random.seed(42)
cities = df_long['Source'].unique()
metrics_dict = {}
for city in cities:
    metrics_dict[city] = {
        'avg_degree': np.random.normal(5, 1),
        'clustering': np.random.uniform(0.2, 0.5),
        'avg_path_length': np.random.uniform(3, 6),
        'modularity': np.random.uniform(0.3, 0.7),
        'n_nodes': np.random.randint(100, 500)
    }

# 构建长格式差异数据
diff_data = []
for src in cities:
    for tgt in cities:
        if src == tgt:
            continue
        m_src = metrics_dict[src]
        m_tgt = metrics_dict[tgt]
        row = {'Source': src, 'Target': tgt}
        for key in m_src.keys():
            row[f'Δ_{key}'] = m_src[key] - m_tgt[key]
        # 添加 CDS
        val = df_long[(df_long['Source']==src) & (df_long['Target']==tgt)]['CDS_raw'].values
        if len(val) > 0:
            row['CDS'] = val[0]
            diff_data.append(row)
df_diff = pd.DataFrame(diff_data)

# 自变量列表（拓扑差异指标）
topo_cols = [col for col in df_diff.columns if col.startswith('Δ_')]
# 标准化
scaler = StandardScaler()
df_diff[topo_cols] = scaler.fit_transform(df_diff[topo_cols])

# 多元回归
X = sm.add_constant(df_diff[topo_cols])
y = df_diff['CDS']
model = sm.OLS(y, X).fit()
print(model.summary())

# 变量重要性
imp = pd.DataFrame({
    'Variable': topo_cols,
    'Coefficient': model.params[topo_cols],
    'P_value': model.pvalues[topo_cols]
}).sort_values('Coefficient', key=abs, ascending=False)
print(imp)
imp.to_csv(f"{OUTPUT_PREFIX}_topology_importance.csv", index=False)

# 可视化：显著指标的散点图（如 Δ_avg_degree 与 CDS）
sig_cols = imp[imp['P_value'] < 0.1]['Variable'].tolist()
for col in sig_cols:
    plt.figure(figsize=(6,5))
    plt.scatter(df_diff[col], df_diff['CDS'], alpha=0.6)
    # 添加回归线
    X_single = sm.add_constant(df_diff[[col]])
    model_single = sm.OLS(df_diff['CDS'], X_single).fit()
    x_range = np.linspace(df_diff[col].min(), df_diff[col].max(), 100)
    y_range = model_single.params['const'] + model_single.params[col] * x_range
    plt.plot(x_range, y_range, 'r-')
    plt.xlabel(col)
    plt.ylabel('CDS')
    plt.title(f'{col} 与迁移损失 (斜率={model_single.params[col]:.3f}, p={model_single.pvalues[col]:.3f})')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_PREFIX}_{col}_scatter.png", dpi=300)
    plt.close()