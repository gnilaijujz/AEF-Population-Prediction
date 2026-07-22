#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 CDS 矩阵的 Ranking 回归分析
因变量：CDS (综合迁移损失，值越大表示迁移性能越差)
自变量：源域嵌入健康度特征 (effective_rank, isotropy, n_nodes, ...)
方法：留一法（LOOCV）预测排名，Spearman 秩相关评估排序能力
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
# ================== 配置 ==================
CDS_MATRIX_CSV = r"results\transferability\AEF\CDS_matrix_robust_entropy.csv"
OLD_FEAT_CSV = r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"
OUTPUT_DIR = r"validation/results/ranking_analysis_cds"
OUTPUT_PLOT = r"validation/results/ranking_analysis_cds/ranking_scatter_cds.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 1. 读取 CDS 矩阵 ==================
print("1. 读取 CDS 矩阵（旧城市→旧城市）...")
cds_df = pd.read_csv(CDS_MATRIX_CSV, index_col=0)
print(f"CDS 矩阵维度: {cds_df.shape} (源城市: {cds_df.index.tolist()})")

# 计算每个源城市的平均 CDS（排除自身对角线）
avg_cds = []
for src in cds_df.index:
    row = cds_df.loc[src]
    vals = row.drop(index=src, errors='ignore').dropna().values
    if len(vals) > 0:
        avg_cds.append(np.mean(vals))
    else:
        avg_cds.append(np.nan)

city_level = pd.DataFrame({
    'source_city': cds_df.index,
    'avg_CDS': avg_cds
}).dropna()

print(f"有效源城市数量: {len(city_level)}")
print(city_level.head())

# ================== 2. 读取源域嵌入健康度特征 ==================
print("\n2. 读取源域嵌入健康度特征...")
feat_df = pd.read_csv(OLD_FEAT_CSV, index_col=0)
print(f"特征维度: {feat_df.shape}")

# 在合并前，检查 feat_df 中是否有 avg_CDS 列，如果有，先重命名或删除，避免冲突
if 'avg_CDS' in feat_df.columns:
    print("警告：特征数据中包含 'avg_CDS' 列，将被重命名为 'avg_CDS_feat' 以避免冲突。")
    feat_df = feat_df.rename(columns={'avg_CDS': 'avg_CDS_feat'})

# 合并
city_level = city_level.merge(feat_df, left_on='source_city', right_index=True, how='inner')
print(f"合并后样本数: {len(city_level)}")

# 删除任何包含 NaN 的行
city_level = city_level.dropna()
print(f"清理缺失值后样本数: {len(city_level)}")

# 打印列名以便调试
print("合并后的列名:", city_level.columns.tolist())

# ================== 3. 定义特征与目标 ==================
# 目标变量：我们计算出的 avg_CDS（来自 CDS 矩阵）
target = 'avg_CDS'  # 这个列名在合并后应该是存在的

# 检查目标列是否存在
if target not in city_level.columns:
    # 可能被重命名了，尝试查找包含 'avg_CDS' 的列
    potential_cols = [col for col in city_level.columns if 'avg_CDS' in col]
    if potential_cols:
        target = potential_cols[0]
        print(f"目标列被重命名为: {target}")
    else:
        raise KeyError("在合并后的数据中找不到目标列 'avg_CDS'。")

# 将目标列转为数值
city_level[target] = pd.to_numeric(city_level[target], errors='coerce')
city_level = city_level.dropna(subset=[target])

# 计算排名（值越小，排名越靠前，即迁移性能越好）
city_level['rank_true'] = city_level[target].rank(method='dense', ascending=True).astype(int)

# 候选特征（排除目标列和可能的数据泄露特征，如 CDS_std，以及任何与目标相关的统计量）
exclude_features = [target, 'rank_true', 'source_city', 'CDS_std', 'R2_self', 'condition_number', 'n_dim']
features = [col for col in city_level.columns if col not in exclude_features and col != 'source_city']
# 只保留数值型特征
features = [f for f in features if pd.api.types.is_numeric_dtype(city_level[f])]
print(f"\n可用特征: {features}")

# ================== 4. 单变量排名回归（留一法） ==================
print("\n3. 单变量排名回归（LOOCV Spearman）...")
rank_results = []

for feat in features:
    X = city_level[[feat]].values
    y_rank = city_level['rank_true'].values
    
    loo = LeaveOneOut()
    pred_ranks = []
    true_ranks = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y_rank[train_idx]
        y_test = y_rank[test_idx]
        
        # 线性回归预测排名
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        pred_ranks.append(pred)
        true_ranks.append(y_test[0])
    
    pred_ranks = np.array(pred_ranks)
    true_ranks = np.array(true_ranks)
    
    # Spearman 相关（预测排名 vs 真实排名）
    rho, p_rho = spearmanr(pred_ranks, true_ranks)
    tau, p_tau = kendalltau(pred_ranks, true_ranks)
    
    rank_results.append({
        'Feature': feat,
        'Spearman_rho': rho,
        'Spearman_p': p_rho,
        'Kendall_tau': tau,
        'Kendall_p': p_tau
    })

df_rank = pd.DataFrame(rank_results).sort_values('Spearman_rho', ascending=False)
print("\n===== Ranking 回归结果（基于 CDS 矩阵） =====")
print(df_rank.to_string(index=False))

# ================== 5. 最佳特征可视化 ==================
if not df_rank.empty:
    best_feat = df_rank.iloc[0]['Feature']
    best_rho = df_rank.iloc[0]['Spearman_rho']
    best_p = df_rank.iloc[0]['Spearman_p']
    
    print(f"\n最佳排序预测特征: {best_feat} (Spearman ρ = {best_rho:.3f}, p = {best_p:.4f})")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：特征值 vs 真实排名
    ax1.scatter(city_level[best_feat], city_level['rank_true'], color='blue', s=80, alpha=0.7)
    for i, row in city_level.iterrows():
        ax1.annotate(row['source_city'], (row[best_feat], row['rank_true']), 
                     fontsize=8, xytext=(3,3), textcoords='offset points')
    ax1.set_xlabel(best_feat)
    ax1.set_ylabel('True Rank (1=best, lower CDS)')
    ax1.set_title(f'Feature vs True Rank')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 右图：预测排名 vs 真实排名（LOOCV）
    X_best = city_level[[best_feat]].values
    y_rank = city_level['rank_true'].values
    pred_ranks_best = []
    true_ranks_best = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_best):
        X_train, X_test = X_best[train_idx], X_best[test_idx]
        y_train = y_rank[train_idx]
        y_test = y_rank[test_idx]
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        pred_ranks_best.append(pred)
        true_ranks_best.append(y_test[0])
    
    ax2.scatter(true_ranks_best, pred_ranks_best, color='red', s=80, alpha=0.7)
    min_val = min(min(true_ranks_best), min(pred_ranks_best))
    max_val = max(max(true_ranks_best), max(pred_ranks_best))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect')
    ax2.set_xlabel('True Rank')
    ax2.set_ylabel('Predicted Rank')
    ax2.set_title(f'LOO Predictions: {best_feat}\nSpearman ρ = {best_rho:.3f}')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\n图表已保存至 {OUTPUT_PLOT}")
    
    # ================== 6. 保存结果 ==================
    df_rank.to_csv(OUTPUT_DIR + "/ranking_regression_cds_results.csv", index=False)
    city_level.to_csv(OUTPUT_DIR + "/city_level_cds_data.csv", index=False)
    print(f"\n所有结果已保存至 {OUTPUT_DIR}")
    
    # ================== 7. 最终结论 ==================
    print("\n===== 最终结论（基于 CDS 矩阵） =====")
    if best_rho > 0.5 and best_p < 0.05:
        print(f"✅ {best_feat} 具有显著的排序预测能力 (ρ={best_rho:.3f}, p={best_p:.4f})")
        print("→ 该特征可用于源域优先级筛选。")
    elif 0.3 < best_rho <= 0.5 and best_p < 0.05:
        print(f"⚠️ {best_feat} 具有中等排序预测能力 (ρ={best_rho:.3f}, p={best_p:.4f})")
        print("→ 可作为辅助参考，但需谨慎使用。")
    else:
        print(f"❌ 最佳特征 {best_feat} 的排序预测能力较弱或统计不显著 (ρ={best_rho:.3f}, p={best_p:.4f})")
        print("→ 源域特征无法可靠预测跨域迁移的排名。")
else:
    print("没有可用的特征进行分析。")