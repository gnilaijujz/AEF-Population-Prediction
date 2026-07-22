#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入健康度深度分析（方案3.2）
计算源域嵌入空间的五个结构特征：
1. 有效秩（Effective Rank）
2. 各向同性（Isotropy / Condition Number）
3. 类间/类内距离比（Inter/Intra Class Distance Ratio）
4. 邻居一致性（Neighbor Label Consistency）
5. 特征方差解释率（PCA Variance Explained）

定位：归因分析第3.6节——"源域可迁移性的本质特征"
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform, cdist
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# 导入您的自定义模块
from GNN_transfer_experiments_calibrated import load_pretrained_gnn, load_city, discover_cities, scan_model_files
from GNN_regression import DEFAULT_POP_CSV
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")
import matplotlib.pyplot as plt

# 方法1：直接指定（Windows 常用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ======================== 导入您的模块 ========================
from GNN_transfer_experiments_calibrated import (
    load_pretrained_gnn, load_city, discover_cities, 
    scan_model_files, prepare_target_data
)
from GNN_regression import DEFAULT_POP_CSV

# ======================== 配置 ========================
AEF_ROOT = Path(r"model_data/aef_root/clean_aef_shapefiles")
POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
PRETRAINED_DIR = Path(r"model_data\pretrained_models\GNN\pretrained_aef")
CDS_MATRIX = Path(r"results\transferability\AEF\CDS_matrix_robust_entropy.csv")
OUTPUT_DIR = Path(r"results\embedding_health_deep_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 64
DROPOUT = 0.5
import numpy as np


all_cities = discover_cities(AEF_ROOT)
print(f"发现 {len(all_cities)} 个城市")

# 加载城市数据
city_cache = {}
for city in all_cities:
    try:
        gdf, feat_cols, edge_idx = load_city(AEF_ROOT, city, POP_CSV)
        city_cache[city] = (gdf, feat_cols, edge_idx)
        print(f"  ✓ 已加载: {city}")
    except Exception as e:
        print(f"  ✗ 加载失败: {city} - {e}")

# 加载预训练模型并提取嵌入
model_files = scan_model_files(PRETRAINED_DIR)
city_embeddings = {}
city_density = {}  # 用于类间/类内距离计算

for city in all_cities:
    if city not in model_files or city not in city_cache:
        print(f"跳过 {city}: 模型或数据缺失")
        continue
    try:
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
        city_embeddings[city] = emb
        city_density[city] = gdf["population_density"].values
        print(f"  ✓ 提取嵌入: {city} (节点数: {emb.shape[0]}, 维度: {emb.shape[1]})")
    except Exception as e:
        print(f"  ✗ 提取失败: {city} - {e}")

print(f"\n成功提取 {len(city_embeddings)} 个城市的嵌入")

# ======================== 2. 计算嵌入健康度指标 ========================
print("\n" + "=" * 70)
print("2. 计算嵌入健康度指标")
print("=" * 70)

def compute_effective_rank(emb, verbose=False):
    """
    有效秩：基于奇异值熵的有效维度
    公式：H = -sum(p_i * log(p_i))，有效秩 = exp(H)
    其中 p_i = σ_i / sum(σ_j)
    越高 → 特征多样性越丰富 → 迁移性越好
    """
    # 标准化嵌入
    emb_std = StandardScaler().fit_transform(emb)
    # SVD
    U, S, Vt = svd(emb_std, full_matrices=False)
    # 归一化奇异值
    p = S / S.sum()
    # 熵
    entropy = -np.sum(p * np.log(p + 1e-10))
    effective_rank = np.exp(entropy)
    return effective_rank

def compute_isotropy(emb):
    """
    嵌入各向同性：协方差矩阵的条件数
    条件数越接近1 → 各向同性越好 → 迁移性越稳定
    """
    emb_centered = emb - emb.mean(axis=0)
    cov = np.cov(emb_centered.T)
    cov += 1e-6 * np.eye(cov.shape[0])
    # 使用 np.linalg.cond 替代 cond
    cond_num = np.linalg.cond(cov)
    isotropy_score = 1 / cond_num
    return isotropy_score, cond_num
# ======================== 1. 加载所有城市数据和模型 ========================
print("=" * 70)
print("1. 加载城市数据和预训练模型")
print("=" * 70)

def compute_inter_intra_ratio(emb, density, n_bins=5):
    """
    类间/类内距离比：按人口密度分层计算
    比值越大 → 类别可分性越强 → 迁移性越好
    """
    if len(density) != emb.shape[0]:
        # 如果长度不匹配，对齐
        min_len = min(len(density), emb.shape[0])
        density = density[:min_len]
        emb = emb[:min_len]
    
    # 按人口密度分位数分层
    try:
        bins = np.percentile(density, np.linspace(0, 100, n_bins + 1))
        labels = np.digitize(density, bins[1:-1])  # 0到n_bins-1
    except:
        # 如果密度值太集中，退化为3层
        bins = np.percentile(density, [0, 33, 67, 100])
        labels = np.digitize(density, bins[1:-1])
    
    # 计算类内距离（同一层内样本的平均距离）
    intra_dists = []
    inter_dists = []
    
    for i in range(n_bins):
        mask_i = (labels == i)
        if mask_i.sum() < 2:
            continue
        emb_i = emb[mask_i]
        # 类内距离
        intra = np.mean(pdist(emb_i))
        intra_dists.append(intra)
        # 类间距离（到其他层的距离）
        for j in range(i + 1, n_bins):
            mask_j = (labels == j)
            if mask_j.sum() < 2:
                continue
            emb_j = emb[mask_j]
            inter = np.mean(cdist(emb_i, emb_j))
            inter_dists.append(inter)
    
    if intra_dists and inter_dists:
        ratio = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-10)
    else:
        ratio = 1.0
    return ratio

def compute_neighbor_consistency(emb, density, k=5):
    """
    邻居一致性：邻居节点的标签（人口密度）相似度
    一致性越高 → GNN聚合越有效 → 迁移性越稳定
    """
    if len(density) != emb.shape[0]:
        min_len = min(len(density), emb.shape[0])
        density = density[:min_len]
        emb = emb[:min_len]
    
    # 构建KNN图
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb)))
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)
    
    # 计算每个节点的邻居标签标准差（越小表示一致性越高）
    consistency_scores = []
    for i in range(len(emb)):
        neighbor_labels = density[indices[i, 1:]]  # 排除自身
        # 用标准差衡量一致性（越小越一致）
        consistency = 1 / (np.std(neighbor_labels) + 1e-10)
        consistency_scores.append(consistency)
    
    return np.mean(consistency_scores)

def compute_pca_variance(emb, n_components=10):
    """
    PCA累计方差解释率：前N个主成分累计方差
    低维结构越清晰 → 迁移特征越容易捕捉
    """
    emb_std = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=min(n_components, emb_std.shape[1], emb_std.shape[0]))
    pca.fit(emb_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return cumsum
def compute_embedding_diversity(emb, k=5):
    """
    计算 AEF 嵌入空间的局部多样性（纯几何指标）
    对每个节点，计算其到 K 个最近邻的平均余弦距离，然后对所有节点取平均。
    数值越高：嵌入流形局部越崎岖，邻居差异大。
    数值越低：嵌入流形局部越平滑，邻居差异小。
    """
    from sklearn.neighbors import NearestNeighbors
    n = emb.shape[0]
    if n < k + 1:
        return np.nan  # 节点太少，无法计算
    
    # 使用余弦距离找 K 个最近邻（包含自身）
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(emb)
    distances, _ = nn.kneighbors(emb)  # distances 形状: (n, k+1)
    
    # 去掉自身（自身距离为 0），只取 k 个邻居的余弦距离
    # 注意：余弦距离 = 1 - 余弦相似度
    neighbor_distances = distances[:, 1:]  # 取第2列到第k+1列
    
    # 对每个节点，计算其到 k 个邻居的平均距离，再对所有节点取平均
    avg_diversity = np.mean(neighbor_distances)
    return avg_diversity
# 计算所有城市的指标
health_results = []
for city, emb in city_embeddings.items():
    density = city_density.get(city)
    if density is None:
        continue
    
    # 有效秩
    eff_rank = compute_effective_rank(emb)
    
    # 各向同性
    isotropy, cond_num = compute_isotropy(emb)
    
    # 类间/类内距离比
    ratio = compute_inter_intra_ratio(emb, density)
    
    # 邻居一致性
    consistency = compute_neighbor_consistency(emb, density, k=5)
    
    # PCA累计方差（取前5个主成分）
    pca_cumsum = compute_pca_variance(emb, n_components=5)
    pca_5 = pca_cumsum[-1] if len(pca_cumsum) >= 5 else pca_cumsum[-1]
    # --- 新增：纯几何版 AEF 嵌入多样性（k=5 保持与之前一致） ---
    diversity = compute_embedding_diversity(emb, k=5)
    
    # --- 保存到结果字典 ---
    health_results.append({
        'city': city,
        'effective_rank': eff_rank,
        'isotropy': isotropy,
        'condition_number': cond_num,
        'inter_intra_ratio': ratio,
        'neighbor_consistency': consistency,        # 原密度版（后续可弃用或保留作对比）
        'embedding_diversity': diversity,           # ✅ 新增的纯 AEF 特征
        'pca_5_explained': pca_5,
        'n_nodes': emb.shape[0],
        'n_dim': emb.shape[1]
    })
    print(f"  ✓ {city}: 有效秩={eff_rank:.2f}, 各向同性={isotropy:.3f}, 嵌入多样性={diversity:.4f}")

df_health = pd.DataFrame(health_results).set_index('city')
print(f"\n成功计算 {len(df_health)} 个城市的健康度指标")

# ======================== 3. 加载迁移性能数据 ========================
print("\n" + "=" * 70)
print("3. 加载迁移性能数据")
print("=" * 70)

df_cds = pd.read_csv(CDS_MATRIX, index_col=0)

# 计算每个源城市的平均CDS（排除自身）
avg_cds = {}
for src in df_cds.index:
    row = df_cds.loc[src]
    vals = row.drop(index=src, errors='ignore').dropna().values
    avg_cds[src] = np.mean(vals) if len(vals) > 0 else np.nan

# 自预测R²
r2_self = {src: df_cds.loc[src, src] for src in df_cds.index if not np.isnan(df_cds.loc[src, src])}

# 合并到健康度数据
df_health['avg_CDS'] = df_health.index.map(lambda c: avg_cds.get(c, np.nan))
df_health['R2_self'] = df_health.index.map(lambda c: r2_self.get(c, np.nan))
df_health = df_health.dropna(subset=['avg_CDS', 'R2_self'])
print(f"有效城市（有迁移性能数据）: {len(df_health)}")

# ======================== 4. 回归与相关性分析 ========================
print("\n" + "=" * 70)
print("4. 回归与相关性分析")
print("=" * 70)

metric_cols = ['effective_rank', 'isotropy', 'inter_intra_ratio', 
               'neighbor_consistency', 'embedding_diversity', 'pca_5_explained','n_nodes']

# 标准化
scaler = StandardScaler()
df_scaled = df_health.copy()
df_scaled[metric_cols] = scaler.fit_transform(df_health[metric_cols])

# 相关性矩阵
corr = df_health[metric_cols + ['avg_CDS', 'R2_self']].corr()
print("\n相关性矩阵（与avg_CDS）:")
print(corr['avg_CDS'].sort_values(ascending=False))

# 单变量回归
univariate_results = []
for col in metric_cols:
    X = sm.add_constant(df_scaled[[col]])
    model = sm.OLS(df_scaled['avg_CDS'], X).fit()
    univariate_results.append({
        'Metric': col,
        'Coefficient': model.params[col],
        'P_value': model.pvalues[col],
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj
    })
df_uni = pd.DataFrame(univariate_results).sort_values('P_value')
print("\n单变量回归结果:")
print(df_uni.to_string(index=False))

# ======================== 5. 可视化 ========================
print("\n" + "=" * 70)
print("5. 生成可视化图表")
print("=" * 70)

# 5.1 有效秩排序柱状图
fig, ax = plt.subplots(figsize=(12, 6))
sorted_df = df_health.sort_values('effective_rank', ascending=False)
colors = ['steelblue' if i < len(sorted_df)//2 else 'lightcoral' for i in range(len(sorted_df))]
bars = ax.bar(sorted_df.index, sorted_df['effective_rank'], color=colors)
ax.set_ylabel('有效秩 (Effective Rank)')
ax.set_xlabel('源城市')
ax.set_title('源城市嵌入空间有效秩排序\n(蓝色=高多样性的"好老师"，红色=低多样性的"差老师")')
ax.axhline(sorted_df['effective_rank'].mean(), color='gray', linestyle='--', label=f'均值={sorted_df["effective_rank"].mean():.2f}')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "effective_rank_ranking.png", dpi=300)
plt.close()

# 5.2 有效秩 vs avg_CDS 散点图
fig, ax = plt.subplots(figsize=(10, 6))
for city in df_health.index:
    ax.scatter(df_health.loc[city, 'effective_rank'], df_health.loc[city, 'avg_CDS'], 
               s=100, alpha=0.7, color='steelblue')
    ax.annotate(city, (df_health.loc[city, 'effective_rank'], df_health.loc[city, 'avg_CDS']),
                fontsize=9, xytext=(5, 5), textcoords='offset points')
# 回归线
X = sm.add_constant(df_health[['effective_rank']])
model = sm.OLS(df_health['avg_CDS'], X).fit()
x_range = np.linspace(df_health['effective_rank'].min(), df_health['effective_rank'].max(), 100)
y_range = model.params['const'] + model.params['effective_rank'] * x_range
ax.plot(x_range, y_range, 'r-', lw=2, 
        label=f"斜率={model.params['effective_rank']:.3f}, R²={model.rsquared:.3f}, p={model.pvalues['effective_rank']:.4f}")
ax.set_xlabel('有效秩 (Effective Rank)')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_title('有效秩 vs 平均迁移性能')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "effective_rank_vs_CDS.png", dpi=300)
plt.close()

# 5.3 各向同性 vs 迁移方差 散点图
df_health['CDS_std'] = df_health.index.map(lambda c: df_cds.loc[c].dropna().std())
df_health = df_health.dropna(subset=['CDS_std'])

fig, ax = plt.subplots(figsize=(10, 6))
for city in df_health.index:
    ax.scatter(df_health.loc[city, 'isotropy'], df_health.loc[city, 'CDS_std'], 
               s=100, alpha=0.7, color='darkgreen')
    ax.annotate(city, (df_health.loc[city, 'isotropy'], df_health.loc[city, 'CDS_std']),
                fontsize=9, xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('各向同性得分 (Isotropy, 1/条件数, 越接近1越好)')
ax.set_ylabel('迁移性能标准差 (CDS_std, 越小越稳定)')
ax.set_title('嵌入各向同性 vs 迁移稳定性')
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "isotropy_vs_stability.png", dpi=300)
plt.close()

# 5.4 综合指标雷达图（"好老师" vs "差老师"）
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 选择前3个和后3个城市
sorted_by_cds = df_health.sort_values('avg_CDS')
good_cities = sorted_by_cds.head(3).index.tolist()
bad_cities = sorted_by_cds.tail(3).index.tolist()

# 标准化指标用于雷达图
radar_metrics = ['effective_rank', 'isotropy', 'inter_intra_ratio', 'neighbor_consistency', 'pca_5_explained']
df_radar = df_health[radar_metrics].copy()
df_radar = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min() + 1e-10)  # 归一化到[0,1]

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

for group, cities, color in [('好老师', good_cities, 'blue'), ('差老师', bad_cities, 'red')]:
    values = df_radar.loc[cities].mean().values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'{group} (n={len(cities)})', color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_ylim(0, 1)
ax.set_title('"好老师" vs "差老师" 嵌入健康度雷达图', fontsize=14)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "good_vs_bad_radar.png", dpi=300)
plt.close()

# 5.5 PCA累计方差图（选择3个代表性城市）
fig, ax = plt.subplots(figsize=(10, 6))
sample_cities = sorted_by_cds.index[[0, len(sorted_by_cds)//2, -1]].tolist()
for city in sample_cities:
    emb = city_embeddings[city]
    cumsum = compute_pca_variance(emb, n_components=10)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'o-', linewidth=2, label=f"{city} (CDS={df_health.loc[city, 'avg_CDS']:.3f})")
ax.set_xlabel('主成分数量')
ax.set_ylabel('累计方差解释率')
ax.set_title('不同迁移性能城市的PCA累计方差对比')
ax.legend()
ax.grid(True)
ax.axhline(0.8, color='gray', linestyle='--', label='80% 阈值')
ax.axhline(0.9, color='gray', linestyle='--', label='90% 阈值')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_cumulative_comparison.png", dpi=300)
plt.close()

# ======================== 6. 保存结果 ========================
print("\n" + "=" * 70)
print("6. 保存结果")
print("=" * 70)

# 保存健康度指标
df_health.to_csv(OUTPUT_DIR / "embedding_health_deep_metrics.csv")

# 保存回归结果
df_uni.to_csv(OUTPUT_DIR / "univariate_health_regression_deep.csv", index=False)

# 生成总结报告
with open(OUTPUT_DIR / "analysis_summary.txt", 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("嵌入健康度深度分析总结报告\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. 相关性分析（与avg_CDS）:\n")
    for metric in metric_cols:
        r = corr.loc[metric, 'avg_CDS']
        f.write(f"   {metric}: r = {r:.4f}\n")
    
    f.write("\n2. 单变量回归结果:\n")
    f.write(df_uni.to_string(index=False))
    
    f.write("\n\n3. 关键发现:\n")
    best_metric = df_uni.iloc[0]['Metric']
    f.write(f"   - 最佳预测指标: {best_metric} (R²={df_uni.iloc[0]['R2']:.3f}, p={df_uni.iloc[0]['P_value']:.4f})\n")
    
    # 好老师 vs 差老师的特征差异
    good_mean = df_health.loc[good_cities, metric_cols].mean()
    bad_mean = df_health.loc[bad_cities, metric_cols].mean()
    f.write(f"\n4. 好老师 vs 差老师 对比:\n")
    for metric in metric_cols:
        diff_pct = (good_mean[metric] - bad_mean[metric]) / bad_mean[metric] * 100
        f.write(f"   {metric}: 好老师 {good_mean[metric]:.3f} vs 差老师 {bad_mean[metric]:.3f} ({diff_pct:+.1f}%)\n")

print(f"\n所有结果已保存至: {OUTPUT_DIR}")
print(f"  - embedding_health_deep_metrics.csv")
print(f"  - univariate_health_regression_deep.csv")
print(f"  - analysis_summary.txt")
print(f"  - 图表: effective_rank_ranking.png, effective_rank_vs_CDS.png, ...")