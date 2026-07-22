#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入健康度分析
从预训练GNN模型中提取每个源城市的嵌入向量（节点级隐藏层表示），
计算嵌入空间的结构特征（紧凑度、各向异性、有效维度等），
并回归到该源城市的平均可迁移性（CDS均值或自预测R²）。
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
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
# ======================== 配置 ========================
AEF_ROOT = Path(r"validation\data\aef_root\new_15_AEF")
POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
PRETRAINED_DIR = Path(r"validation\data\pretrained_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_CHANNELS = 64
DROPOUT = 0.5

OUTPUT_DIR = Path(r"results\embedding_health_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载CDS矩阵（用于获取源城市平均迁移性能）
CDS_MATRIX = Path(r"results\transferability\AEF\CDS_matrix_robust_entropy.csv")

# ======================== 1. 加载所有城市嵌入 ========================
def load_city_embedding(city, model, x_scaler, y_mean, y_std, feat_cols, target_transform, city_cache):
    """
    获取指定城市在GNN第二层输出后的嵌入矩阵 (n_nodes, hidden_dim)
    """
    gdf, _, edge_index = city_cache[city]
    device = next(model.parameters()).device
    
    # 准备数据（复用 prepare_target_data 逻辑，但不打乱）
    from GNN_transfer_experiments_calibrated import prepare_target_data
    data, _ = prepare_target_data(
        gdf, feat_cols, edge_index, x_scaler, y_mean, y_std,
        target_transform, device, shuffle_features=False
    )
    with torch.no_grad():
        emb = model.forward_emb(data.x, data.edge_index).detach().cpu().numpy()
    return emb

# 1.1 发现城市
all_cities = discover_cities(AEF_ROOT)
print(f"发现 {len(all_cities)} 个城市: {all_cities}")

# 1.2 加载城市数据
city_cache = {}
for city in all_cities:
    try:
        gdf, feat_cols, edge_idx = load_city(AEF_ROOT, city, POP_CSV)
        city_cache[city] = (gdf, feat_cols, edge_idx)
        print(f"已加载城市数据: {city}")
    except Exception as e:
        print(f"加载 {city} 失败: {e}")

# 1.3 加载预训练模型并提取嵌入
model_files = scan_model_files(PRETRAINED_DIR)
city_embeddings = {}
city_models_info = {}

for city in all_cities:
    if city not in model_files:
        print(f"跳过 {city}: 无预训练模型")
        continue
    if city not in city_cache:
        print(f"跳过 {city}: 城市数据未加载")
        continue
    try:
        model, x_scaler, y_mean, y_std, feat_cols, target_transform = load_pretrained_gnn(
            model_files[city], HIDDEN_CHANNELS, DROPOUT, DEVICE
        )
        # 提取嵌入
        emb = load_city_embedding(city, model, x_scaler, y_mean, y_std, feat_cols, target_transform, city_cache)
        city_embeddings[city] = emb
        city_models_info[city] = {
            'model': model,
            'x_scaler': x_scaler,
            'y_mean': y_mean,
            'y_std': y_std,
            'feat_cols': feat_cols,
            'target_transform': target_transform
        }
        print(f"提取嵌入: {city} (shape: {emb.shape})")
    except Exception as e:
        print(f"提取 {city} 嵌入失败: {e}")

print(f"成功提取 {len(city_embeddings)} 个城市的嵌入")

# ======================== 2. 计算嵌入健康度指标 ========================
def compute_health_metrics(emb):
    """
    计算嵌入矩阵的结构健康度指标
    emb: (n_samples, n_features)
    返回字典
    """
    n, d = emb.shape
    if n < 5:
        return None
    
    metrics = {}
    
    # 1. 类内紧凑度：平均最近邻距离 (KNN, k=5)
    # 用最近邻距离均值表示样本聚集程度
    if n > 5:
        nn = NearestNeighbors(n_neighbors=min(5, n))
        nn.fit(emb)
        distances, _ = nn.kneighbors(emb)
        # 排除自身 (distances[:,0] = 0)
        avg_knn_dist = np.mean(distances[:, 1:])  # 仅取5个最近邻
    else:
        avg_knn_dist = np.mean(pdist(emb)) if n>1 else 0.0
    metrics['avg_knn_dist'] = avg_knn_dist
    
    # 2. 全局分离度：所有成对距离的95%分位数（越大表示分布越分散）
    if n > 1:
        all_dists = pdist(emb)
        metrics['global_95'] = np.percentile(all_dists, 95)
        metrics['global_mean'] = np.mean(all_dists)
        metrics['global_std'] = np.std(all_dists)
    else:
        metrics['global_95'] = 0.0
        metrics['global_mean'] = 0.0
        metrics['global_std'] = 0.0
    
    # 3. 各向异性：PCA主轴方差比率 (第一主轴 / 第二主轴)
    if n > 1 and d > 1:
        pca = PCA()
        pca.fit(emb)
        var_ratio = pca.explained_variance_ratio_
        metrics['anisotropy'] = var_ratio[0] / (var_ratio[1] + 1e-6) if len(var_ratio) > 1 else 0.0
        # 第一主轴占比
        metrics['pca_first_ratio'] = var_ratio[0]
        # 累积方差解释80%所需维数 (有效维度)
        cumsum = np.cumsum(var_ratio)
        eff_dim = np.argmax(cumsum >= 0.8) + 1
        metrics['eff_dim'] = eff_dim
    else:
        metrics['anisotropy'] = 1.0
        metrics['pca_first_ratio'] = 1.0
        metrics['eff_dim'] = d
    
    # 4. 整体能量：Frobenius范数 (平方和的平方根)
    metrics['frobenius_norm'] = np.linalg.norm(emb, ord='fro')
    
    # 5. 各维度变异系数均值 (CV)
    # 计算每个维度的标准差/均值(绝对值)的平均值，反映各维度贡献均匀性
    std_vals = np.std(emb, axis=0)
    mean_abs = np.mean(np.abs(emb), axis=0)
    cv = std_vals / (mean_abs + 1e-6)  # 避免除零
    metrics['cv_mean'] = np.mean(cv)
    metrics['cv_std'] = np.std(cv)
    
    return metrics

# 计算所有城市的健康度
health_df = pd.DataFrame()
for city, emb in city_embeddings.items():
    metrics = compute_health_metrics(emb)
    if metrics is not None:
        metrics['city'] = city
        health_df = pd.concat([health_df, pd.DataFrame([metrics])], ignore_index=True)

health_df.set_index('city', inplace=True)
print("\n嵌入健康度指标计算完成:")
print(health_df.describe())

# ======================== 3. 获取源城市的平均迁移性能 ========================
# 3.1 加载CDS矩阵并计算每个源城市的平均CDS (越小表示迁移性能越好)
df_cds = pd.read_csv(CDS_MATRIX, index_col=0)
# 取每个源城市对所有目标城市（排除自身）的CDS均值
avg_cds = {}
for src in df_cds.index:
    row = df_cds.loc[src]
    # 排除对角线（自身）和NaN
    vals = row.drop(index=src, errors='ignore').dropna().values
    if len(vals) > 0:
        avg_cds[src] = np.mean(vals)
    else:
        avg_cds[src] = np.nan

# 3.2 获取自预测性能（R²_self，即源->源）
r2_self = {}
for src in df_cds.index:
    val = df_cds.loc[src, src]
    if not np.isnan(val):
        r2_self[src] = val
    else:
        r2_self[src] = np.nan

# 合并到健康度数据框
health_df['avg_CDS'] = health_df.index.map(lambda c: avg_cds.get(c, np.nan))
health_df['R2_self'] = health_df.index.map(lambda c: r2_self.get(c, np.nan))
health_df = health_df.dropna(subset=['avg_CDS', 'R2_self'])
print(f"\n有效城市（有迁移性能数据）: {len(health_df)}")

# ======================== 4. 回归与相关性分析 ========================
# 标准化健康度指标（便于比较系数）
metric_cols = [col for col in health_df.columns if col not in ['city', 'avg_CDS', 'R2_self']]
scaler = StandardScaler()
health_df_scaled = health_df.copy()
health_df_scaled[metric_cols] = scaler.fit_transform(health_df[metric_cols])

# 相关性矩阵
corr = health_df[metric_cols + ['avg_CDS', 'R2_self']].corr()
print("\n相关性矩阵（与avg_CDS和R2_self）:")
print(corr[['avg_CDS', 'R2_self']].sort_values('avg_CDS', ascending=False))

# 单变量回归：每个指标对avg_CDS
univariate_results = []
for col in metric_cols:
    X = sm.add_constant(health_df_scaled[[col]])
    model = sm.OLS(health_df_scaled['avg_CDS'], X).fit()
    univariate_results.append({
        'Metric': col,
        'Coefficient': model.params[col],
        'P_value': model.pvalues[col],
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj
    })
df_uni = pd.DataFrame(univariate_results).sort_values('P_value')
print("\n单变量回归结果（预测 avg_CDS）:")
print(df_uni)

# 多变量回归（全指标）
X_multi = sm.add_constant(health_df_scaled[metric_cols])
model_multi = sm.OLS(health_df_scaled['avg_CDS'], X_multi).fit()
print("\n多变量回归摘要:")
print(model_multi.summary())

# ======================== 5. 可视化 ========================
# 5.1 最佳指标散点图（与avg_CDS）
best_metric = df_uni.iloc[0]['Metric']  # p值最小
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df[best_metric], health_df['avg_CDS'], alpha=0.7, s=80)
# 添加回归线
X = sm.add_constant(health_df[[best_metric]])
model_best = sm.OLS(health_df['avg_CDS'], X).fit()
x_range = np.linspace(health_df[best_metric].min(), health_df[best_metric].max(), 100)
y_range = model_best.params['const'] + model_best.params[best_metric] * x_range
ax.plot(x_range, y_range, 'r-', lw=2, label=f"斜率={model_best.params[best_metric]:.3f}, $R^2$={model_best.rsquared:.3f}")
ax.set_xlabel(best_metric)
ax.set_ylabel('平均 CDS (迁移损失, 越小越好)')
ax.set_title(f'嵌入健康度 vs 平均迁移性能 (最佳指标: {best_metric})')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_health_metric_scatter.png", dpi=300)
plt.close()

# 5.2 城市排序柱状图（按avg_CDS）
fig, ax = plt.subplots(figsize=(12, 6))
sorted_df = health_df.sort_values('avg_CDS')
ax.bar(sorted_df.index, sorted_df['avg_CDS'], color='steelblue')
ax.axhline(0, color='gray', linestyle='--')
ax.set_ylabel('平均 CDS (迁移损失)')
ax.set_xlabel('源城市')
ax.set_title('源城市平均迁移性能排序')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_avg_CDS_ranking.png", dpi=300)
plt.close()

# 5.3 自预测能力 vs 跨域能力 散点图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(health_df['R2_self'], health_df['avg_CDS'], alpha=0.7, s=80)
# 标注城市
for city in health_df.index:
    ax.annotate(city, (health_df.loc[city, 'R2_self'], health_df.loc[city, 'avg_CDS']), fontsize=8)
ax.set_xlabel('自预测 $R^2$ (源->源)')
ax.set_ylabel('平均跨域 CDS (越小越好)')
ax.set_title('自预测能力 vs 跨域迁移性能')
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "self_vs_cross_transfer.png", dpi=300)
plt.close()

# 5.4 箱线图矩阵（各健康度指标分布）
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, col in zip(axes.flatten(), metric_cols[:6]):
    ax.boxplot(health_df[col])
    ax.set_title(col)
    ax.set_xticks([])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "health_metrics_boxplots.png", dpi=300)
plt.close()

# 保存结果
health_df.to_csv(OUTPUT_DIR / "embedding_health_metrics.csv")
df_uni.to_csv(OUTPUT_DIR / "univariate_health_regression.csv", index=False)
with open(OUTPUT_DIR / "multivariate_health_summary.txt", 'w') as f:
    f.write(str(model_multi.summary()))

print(f"\n分析完成！所有结果已保存至 {OUTPUT_DIR}")