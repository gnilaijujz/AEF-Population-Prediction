import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# ================== 用户配置：请修改路径 ==================
OLD_FEAT_CSV = r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"  # 源域特征（旧城市）
PAIR_ERROR_CSV = r"validation\data\gnn_tract_transfer_postprocess_outputs\all_pair_tract_predictions_long.csv"           # 请替换
AEF_CLUSTER_CSV = r"validation\data\gnn_tract_transfer_postprocess_outputs\aef_cluster_labels.csv"                # 请替换
CLUSTER_SUMMARY_CSV = r"validation\data\gnn_tract_transfer_postprocess_outputs\cluster_error_summary.csv"                                       # 集群误差汇总
OUTPUT_DIR = r"validation/results/final_evidence"
OUTPUT_FIG = r"validation/results/final_evidence/n_nodes_vs_HGG.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)


# ================== 数据读取 ==================
print("1. 读取数据...")
src_feat = pd.read_csv(OLD_FEAT_CSV, index_col=0)
pair_errors = pd.read_csv(PAIR_ERROR_CSV)
cluster_labels = pd.read_csv(AEF_CLUSTER_CSV)
cluster_summary = pd.read_csv(CLUSTER_SUMMARY_CSV)

# 确保存在必要的列
assert 'geoid' in cluster_labels.columns and 'aef_clu' in cluster_labels.columns, "集群标签缺少 geoid 或 aef_clu"
assert 'target_city' in cluster_summary.columns and 'aef_clu' in cluster_summary.columns and 'mean_true_den' in cluster_summary.columns, "集群汇总缺少必要列"

# ================== 确定困难集群（每个目标域内 mean_true_den 最大的集群） ==================
print("2. 确定每个目标域的困难集群（高建成区）...")
target_hard = cluster_summary.loc[
    cluster_summary.groupby('target_city')['mean_true_den'].idxmax(),
    ['target_city', 'aef_clu']
].rename(columns={'aef_clu': 'hard_cluster'})

# ================== 合并集群标签到预测误差 ==================
pair_with_cluster = pair_errors.merge(
    cluster_labels[['geoid', 'aef_clu']],
    on='geoid',
    how='inner'
)

# 只保留源城市存在于 src_feat 中的行（确保特征可用）
valid_sources = src_feat.index.tolist()
pair_with_cluster = pair_with_cluster[pair_with_cluster['source_city'].isin(valid_sources)]

# ================== 筛选属于困难集群的行并聚合 ==================
print("3. 计算每个源-目标对在困难集群上的平均绝对误差...")
pair_with_cluster = pair_with_cluster.merge(target_hard, on='target_city', how='left')
hard_cluster_errors = pair_with_cluster[
    pair_with_cluster['aef_clu'] == pair_with_cluster['hard_cluster']
]
# 聚合：按 (source_city, target_city) 计算平均 abs_err
agg_errors = hard_cluster_errors.groupby(
    ['source_city', 'target_city']
).agg(
    hc_mean_abs_error=('abs_err', 'mean'),
    hc_n_tracts=('geoid', 'count')
).reset_index()

print(f"   获得 {len(agg_errors)} 个（源, 目标）组合对")

# ================== 合并源域特征 ==================
reg_df = agg_errors.merge(src_feat, left_on='source_city', right_index=True, how='inner')
print(f"   回归数据集大小: {reg_df.shape}")

# ================== 4. 城市级别聚合（每个源城市 -> 平均 HGG） ==================
print("\n4. 城市级别分析...")
city_level = reg_df.groupby('source_city').agg({
    'n_nodes': 'first',                       # 城市规模
    'hc_mean_abs_error': 'mean',              # 平均困难集群误差
    'effective_rank': 'first',
    'CDS_std': 'first'
}).reset_index()

print(f"   城市级别样本数: {len(city_level)}")
print(city_level[['source_city', 'n_nodes', 'hc_mean_abs_error']].head())

# ================== 5. Spearman 相关与置换检验 ==================
print("\n5. Spearman 秩相关与置换检验...")
n_nodes = city_level['n_nodes'].values
hc_error = city_level['hc_mean_abs_error'].values

rho, p_rho = spearmanr(n_nodes, hc_error)
print(f"   Spearman ρ = {rho:.4f}, p = {p_rho:.6f}")

# 置换检验（10000次）
n_perm = 10000
perm_rhos = []
for _ in range(n_perm):
    shuffled = np.random.permutation(n_nodes)
    rho_perm, _ = spearmanr(shuffled, hc_error)
    perm_rhos.append(rho_perm)
p_perm = np.mean(np.array(perm_rhos) <= rho)  # 单侧（负相关）
print(f"   置换检验 p (单侧, 10000次) = {p_perm:.6f}")

# ================== 6. 城市级别线性回归 + LOOCV ==================
print("\n6. 城市级别 LOOCV 线性回归 (n_nodes → hc_error)...")
X_city = n_nodes.reshape(-1, 1)
y_city = hc_error

# 全样本拟合
model_full = LinearRegression().fit(X_city, y_city)
r2_city = model_full.score(X_city, y_city)
print(f"   全样本 R² = {r2_city:.4f}")
print(f"   斜率 = {model_full.coef_[0]:.2f}, 截距 = {model_full.intercept_:.2f}")

# LOOCV
loo = LeaveOneOut()
y_pred_loo = []
for train_idx, test_idx in loo.split(X_city):
    X_train, X_test = X_city[train_idx], X_city[test_idx]
    y_train, y_test = y_city[train_idx], y_city[test_idx]
    model = LinearRegression().fit(X_train, y_train)
    y_pred_loo.append(model.predict(X_test)[0])
y_pred_loo = np.array(y_pred_loo)

rmse_loo = np.sqrt(mean_squared_error(y_city, y_pred_loo))
ss_res = np.sum((y_city - y_pred_loo) ** 2)
ss_tot = np.sum((y_city - np.mean(y_city)) ** 2)
q2_loo = 1 - (ss_res / ss_tot)
print(f"   LOOCV RMSE = {rmse_loo:.4f}")
print(f"   LOOCV Q²   = {q2_loo:.4f}")

# ================== 7. 组合对级别单变量回归（仅 n_nodes）作为对照 ==================
print("\n7. 组合对级别单变量回归 (n_nodes → hc_mean_abs_error)...")
X_pair = reg_df[['n_nodes']].values
y_pair = reg_df['hc_mean_abs_error'].values

model_pair = LinearRegression().fit(X_pair, y_pair)
r2_pair = model_pair.score(X_pair, y_pair)
print(f"   全样本 R² = {r2_pair:.4f}")

# LOOCV 组合对级别
loo_pair = LeaveOneOut()
y_pred_pair_loo = []
for train_idx, test_idx in loo_pair.split(X_pair):
    X_train, X_test = X_pair[train_idx], X_pair[test_idx]
    y_train, y_test = y_pair[train_idx], y_pair[test_idx]
    model = LinearRegression().fit(X_train, y_train)
    y_pred_pair_loo.append(model.predict(X_test)[0])
y_pred_pair_loo = np.array(y_pred_pair_loo)

rmse_pair_loo = np.sqrt(mean_squared_error(y_pair, y_pred_pair_loo))
ss_res_pair = np.sum((y_pair - y_pred_pair_loo) ** 2)
ss_tot_pair = np.sum((y_pair - np.mean(y_pair)) ** 2)
q2_pair_loo = 1 - (ss_res_pair / ss_tot_pair)
print(f"   LOOCV RMSE = {rmse_pair_loo:.4f}")
print(f"   LOOCV Q²   = {q2_pair_loo:.4f}")

# ================== 8. 绘图：城市级别散点图 ==================
print("\n8. 生成散点图...")
fig, ax = plt.subplots(figsize=(8, 6))

# 散点
ax.scatter(n_nodes, hc_error, s=100, color='steelblue', alpha=0.8, edgecolors='black', linewidth=1.2)

# 回归线
x_range = np.linspace(n_nodes.min(), n_nodes.max(), 100)
y_range = model_full.intercept_ + model_full.coef_[0] * x_range
ax.plot(x_range, y_range, 'r-', linewidth=2, label=f'Linear fit (slope={model_full.coef_[0]:.1f})')

# 添加城市标签
for i, city in enumerate(city_level['source_city']):
    ax.annotate(city, (n_nodes[i], hc_error[i]), fontsize=9, xytext=(5,5), textcoords='offset points')

ax.set_xlabel('Number of Nodes (n_nodes)', fontsize=12)
ax.set_ylabel('Average HC Error (HGG)', fontsize=12)
ax.set_title(f'City-level Evidence: n_nodes vs Hard-Cluster Error\n'
             f'Spearman ρ = {rho:.3f} (p = {p_rho:.4f}), '
             f'LOOCV Q² = {q2_loo:.3f}', fontsize=11)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
print(f"   图表已保存至 {OUTPUT_FIG}")

# ================== 9. 保存汇总结果 ==================
print("\n9. 保存数值结果...")
results = {
    'Metric': ['Spearman_rho', 'Spearman_p', 'Permutation_p', 
               'City_R2', 'City_Q2', 'City_RMSE_LOO',
               'Pair_R2', 'Pair_Q2', 'Pair_RMSE_LOO'],
    'Value': [rho, p_rho, p_perm,
              r2_city, q2_loo, rmse_loo,
              r2_pair, q2_pair_loo, rmse_pair_loo]
}
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT_DIR + "/evidence_summary.csv", index=False)
print(f"   汇总保存至 {OUTPUT_DIR}/evidence_summary.csv")

print("\n===== 分析完成 =====")
print("核心结论：")
print(f"  - 城市级别 Spearman ρ = {rho:.4f} (p = {p_rho:.4f}, 置换检验 p = {p_perm:.4f})")
print(f"  - 城市级别 LOOCV Q² = {q2_loo:.4f} (RMSE = {rmse_loo:.2f})")
print(f"  - 组合对级别 LOOCV Q² = {q2_pair_loo:.4f} (RMSE = {rmse_pair_loo:.2f})")