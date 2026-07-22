#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多特征外部验证（含 IQR 筛选）：对旧城市特征进行离群值剔除后再拟合方程
目的：获得更稳健的旧城市斜率，避免单个离群点主导回归方向
支持排除特定新城市，并为每个特征生成排名对比图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
OLD_CDS_MATRIX = r"results\transferability\AEF\CDS_matrix_robust_entropy.csv"
OLD_FEAT_CSV = r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"
NEW_LOSS_MATRIX = r"validation\results\transferability\transfer_loss_matrix_robust_entropy.csv"
NEW_FEAT_CSV = r"validation\results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"
OUTPUT_DIR = Path(r"validation/results/external_apply_iqr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 要排除的新城市列表（如 ['Syracuse_NY', 'Milwaukee']，留空则排除无）
EXCLUDE_NEW_CITIES = ['Las_Vegas','Syracuse_NY']   # 例如 ['Syracuse_NY', 'Milwaukee']

# IQR 倍数（通常为 1.5）
IQR_MULTIPLIER = 1.5

# 要测试的特征列表（排除数据泄露特征）
FEATURES = ['effective_rank', 'n_nodes', 'isotropy', 'inter_intra_ratio',
            'neighbor_consistency', 'embedding_diversity', 'pca_5_explained']

# ================== 1. 加载旧城市数据 ==================
print("加载旧城市数据...")
cds_old = pd.read_csv(OLD_CDS_MATRIX, index_col=0)
# 计算平均CDS（排除自身）
avg_cds_old = []
for src in cds_old.index:
    row = cds_old.loc[src]
    vals = row.drop(index=src, errors='ignore').dropna().values
    avg_cds_old.append(np.mean(vals) if len(vals) > 0 else np.nan)
old_df = pd.DataFrame({'source_city': cds_old.index, 'avg_CDS': avg_cds_old}).dropna()
feat_old = pd.read_csv(OLD_FEAT_CSV, index_col=0)
old_df = old_df.merge(feat_old[FEATURES], left_on='source_city', right_index=True, how='inner')
print(f"旧城市原始有效样本: {len(old_df)}")

# ================== 2. 加载新城市数据（支持排除特定城市） ==================
print("加载新城市数据...")
loss_new = pd.read_csv(NEW_LOSS_MATRIX, index_col=0)

# 排除指定的新城市
if EXCLUDE_NEW_CITIES:
    print(f"排除新城市: {EXCLUDE_NEW_CITIES}")
    loss_new = loss_new.drop(index=EXCLUDE_NEW_CITIES, errors='ignore')

new_cities = loss_new.index.tolist()
old_targets = [col for col in loss_new.columns if col not in new_cities]
avg_loss_new = loss_new[old_targets].mean(axis=1)
feat_new = pd.read_csv(NEW_FEAT_CSV, index_col=0)
new_df = pd.DataFrame({'source_city': avg_loss_new.index, 'actual_CDS': avg_loss_new.values})
new_df = new_df.merge(feat_new[FEATURES], left_on='source_city', right_index=True, how='inner')
new_df = new_df.dropna()
print(f"新城市有效样本（排除后）: {len(new_df)}")

# ================== 3. 辅助函数：IQR 筛选 ==================
def apply_iqr_filter(df, feature, multiplier=1.5):
    """对 df 中的 feature 列应用 IQR 筛选，返回过滤后的 DataFrame"""
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return filtered

# ================== 4. 对每个特征进行外部验证（旧数据 IQR 筛选）==================
results = []
# 创建子图（散点+回归线）
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

# 创建排名图保存目录
rank_dir = OUTPUT_DIR / "ranking_plots"
rank_dir.mkdir(exist_ok=True)

for idx, feat in enumerate(FEATURES):
    if feat not in old_df.columns or feat not in new_df.columns:
        continue
    
    # ---- A. 对旧城市做 IQR 筛选（剔除离群特征值） ----
    old_filtered = apply_iqr_filter(old_df, feat, IQR_MULTIPLIER)
    if len(old_filtered) < 4:
        print(f"  ⚠️ {feat} IQR 筛选后样本不足 ({len(old_filtered)})，跳过此特征")
        continue
    
    print(f"\n处理特征: {feat}")
    print(f"  旧城市原始样本: {len(old_df)}, IQR 筛选后: {len(old_filtered)} (剔除 {len(old_df)-len(old_filtered)} 个)")

    # ---- B. 用筛选后的旧数据拟合方程 ----
    X_old = old_filtered[[feat]].values
    y_old = old_filtered['avg_CDS'].values
    model = LinearRegression().fit(X_old, y_old)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2_old = model.score(X_old, y_old)

    # ---- C. 在新城市上预测（保留所有新城市样本，不做筛选） ----
    X_new = new_df[[feat]].values
    y_new_true = new_df['actual_CDS'].values
    y_new_pred = intercept + slope * X_new.flatten()

    # ---- D. 误差指标 ----
    rmse = np.sqrt(mean_squared_error(y_new_true, y_new_pred))
    mae = mean_absolute_error(y_new_true, y_new_pred)

    # ---- E. 排名评估 ----
    rank_true = pd.Series(y_new_true).rank(method='dense').values
    rank_pred = pd.Series(y_new_pred).rank(method='dense').values
    rho, p_rho = spearmanr(rank_pred, rank_true)

    # ---- 记录结果 ----
    results.append({
        'Feature': feat,
        'IQR_Removed': len(old_df) - len(old_filtered),
        'Slope': slope,
        'Intercept': intercept,
        'Old_R2': r2_old,
        'New_RMSE': rmse,
        'New_MAE': mae,
        'Spearman_rho': rho,
        'Spearman_p': p_rho,
        'New_n': len(new_df)
    })

    # ---- F. 子图绘制（散点+回归线，图例左上角） ----
    ax = axes[idx]
    ax.scatter(X_new, y_new_true, color='blue', s=60, alpha=0.7, label='Actual')
    ax.scatter(X_new, y_new_pred, color='red', marker='*', s=100, label='Predicted')
    x_min, x_max = X_new.min(), X_new.max()
    x_line = np.linspace(x_min, x_max, 50)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, 'r--', linewidth=1.5, label=f'Old eq (IQR fit)')
    ax.set_xlabel(feat)
    ax.set_ylabel('CDS')
    ax.set_title(f'{feat}\nρ = {rho:.3f} (p={p_rho:.3f})')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.4)

    # ---- G. 生成独立的排名对比图（棒棒糖图） ----
    fig_rank, ax_rank = plt.subplots(figsize=(9, 5))
    cities = new_df['source_city'].values
    y_pos = np.arange(len(cities))
    # 按真实排名排序
    sorted_idx = np.argsort(rank_true)
    cities_sorted = cities[sorted_idx]
    true_sorted = rank_true[sorted_idx]
    pred_sorted = rank_pred[sorted_idx]

    # 绘制连线（黑色虚线）
    for i in range(len(cities_sorted)):
        ax_rank.plot([true_sorted[i], pred_sorted[i]], [i, i], 'k--', alpha=0.6)

    # 实际排名（蓝色圆点）
    ax_rank.scatter(true_sorted, y_pos, color='blue', s=80, label='Actual Rank')
    # 预测排名（红色星号）
    ax_rank.scatter(pred_sorted, y_pos, color='red', marker='*', s=150, label='Predicted Rank')

    # 标注城市名
    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(cities_sorted)
    ax_rank.set_xlabel('Rank (1 = Best)')
    ax_rank.set_title(f'{feat}  Ranking Prediction (IQR filtered old cities)\nSpearman ρ = {rho:.3f}  (p = {p_rho:.3f})')
    ax_rank.invert_yaxis()  # 排名靠前在上方
    ax_rank.legend(loc='upper right')
    ax_rank.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(rank_dir / f'ranking_compare_{feat}_iqr.png', dpi=300)
    plt.close(fig_rank)
    print(f"  排名图已保存: {rank_dir / f'ranking_compare_{feat}_iqr.png'}")

# 隐藏多余的子图
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'all_features_external_apply_iqr.png', dpi=300)
plt.close()

# ================== 5. 汇总结果并输出（含完整参数）==================
df_results = pd.DataFrame(results).sort_values('Spearman_rho', ascending=False)
col_order = ['Feature', 'IQR_Removed', 'Slope', 'Intercept', 'Old_R2', 
             'New_RMSE', 'New_MAE', 'Spearman_rho', 'Spearman_p', 'New_n']
df_results = df_results[col_order]
df_results.to_csv(OUTPUT_DIR / 'external_apply_summary_iqr.csv', index=False)

print("\n" + "="*80)
print("IQR 筛选后外部套用验证汇总（各特征完整参数）")
print("="*80)
print(df_results.to_string(index=False))

# 最佳特征
best = df_results.iloc[0]
print(f"\n最佳泛化特征: {best['Feature']} (Spearman ρ = {best['Spearman_rho']:.4f}, p = {best['Spearman_p']:.4f})")

if best['Spearman_rho'] > 0.5 and best['Spearman_p'] < 0.05:
    print("✅ 该特征具有显著的跨域泛化能力。")
elif best['Spearman_rho'] > 0.3:
    print("⚠️ 该特征具有中等泛化能力，可作为辅助参考。")
else:
    print("❌ 所有特征的跨域泛化能力均较弱，建议采用域自适应策略。")

print(f"\n所有结果已保存至: {OUTPUT_DIR}")
print(f"排名对比图保存在: {rank_dir}")