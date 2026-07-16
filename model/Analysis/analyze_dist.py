import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 文件路径
R2_MATRIX_FILE = r"GNN_output\transfer_results_experiments\transfer_matrix_r2_gnn_calibrated.csv"
DIST_FILE = r"GNN_output\transfer_results_experiments\transfer_embedding_distances_gnn.csv"

# 1. 读取数据
df_r2 = pd.read_csv(R2_MATRIX_FILE, index_col=0)
df_long = df_r2.stack().reset_index()
df_long.columns = ['Source', 'Target', 'R2_cal']
df_long = df_long.dropna(subset=['R2_cal'])

df_dist = pd.read_csv(DIST_FILE)
# 合并
df_merged = df_long.merge(df_dist, on=['Source', 'Target'], how='inner')
print(f"合并后有效样本数: {len(df_merged)}")

# 2. 定义要分析的距离/相似度列
metrics = ['L1_Dist', 'L2_Dist', 'Cos_Sim','MMD']

# 3. 计算相关性
results = []
for metric in metrics:
    # Pearson 相关系数
    pearson_r, pearson_p = pearsonr(df_merged[metric], df_merged['R2_cal'])
    # Spearman 秩相关系数
    spearman_r, spearman_p = spearmanr(df_merged[metric], df_merged['R2_cal'])
    
    results.append({
        'Metric': metric,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
    })

# 4. 输出结果
df_results = pd.DataFrame(results)
print("\n=== 各距离度量与校准后 R² 的相关性 ===")
print(df_results.to_string(index=False))

# 保存结果
df_results.to_csv('GNN_output\\attribute\\distance_correlation_results.csv', index=False)
print("\n结果已保存至 distance_correlation_results.csv")