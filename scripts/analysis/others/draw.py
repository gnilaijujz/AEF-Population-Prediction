import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. 读取你的两个CSV（假设列名一致）
df_gnn = pd.read_csv(r'results/transfer_results/aef\transfer_matrix_r2_gnn.csv', index_col=0)
df_ridge = pd.read_csv(r'results/transfer_results/aef\transfer_matrix_r2.csv', index_col=0)
df_perm = pd.read_csv(r'results/transfer_results/aef\transfer_matrix_r2.csv', index_col=0)

# 2. 绘制图1（并排热力图）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(df_gnn, ax=axes[0], cmap='RdBu_r', center=0, vmin=-3, vmax=1, annot=False)
axes[0].set_title('GNN Transfer R²')
sns.heatmap(df_ridge, ax=axes[1], cmap='RdBu_r', center=0, vmin=-3, vmax=1, annot=False)
axes[1].set_title('Ridge Regression Transfer R²')
plt.savefig('Figure1_Heatmap_Compare.png', dpi=300)
plt.show()

# 3. 绘制图3（配对散点图）
flat_gnn = df_gnn.values.flatten()
flat_ridge = df_ridge.values.flatten()
# 过滤掉对角线（源=目标）或保留全部
mask = ~np.isnan(flat_gnn) 
plt.figure(figsize=(6,6))
plt.scatter(flat_gnn[mask], flat_ridge[mask], alpha=0.6)
plt.plot([-5, 1], [-5, 1], 'k--', label='y=x')
plt.xlabel('GNN R²')
plt.ylabel('Ridge R²')
plt.xlim(-3, 1)
plt.ylim(-3, 1)
plt.legend()
plt.title(f'Pearson r = {pearsonr(flat_gnn[mask], flat_ridge[mask])[0]:.3f}')
plt.savefig('Figure3_Scatter_GNN_vs_Ridge.png', dpi=300)
plt.show()

# 4. 绘制图4（箱线图对比打乱）
df_perm_flat = df_perm.values.flatten()
# 构建DataFrame用于boxplot
plot_df = pd.DataFrame({
    'R²': np.concatenate([flat_gnn, flat_ridge, df_perm_flat]),
    'Type': ['GNN']*len(flat_gnn) + ['Ridge']*len(flat_ridge) + ['Permuted']*len(df_perm_flat)
})
plt.figure(figsize=(8,6))
sns.boxplot(data=plot_df, x='Type', y='R²')
plt.axhline(0, color='r', linestyle='--', label='Zero baseline')
plt.legend()
plt.title('Distribution Comparison: GNN vs Ridge vs Permuted')
plt.savefig('Figure4_Boxplot_Permuted_Bias.png', dpi=300)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取校准后的 Ridge 矩阵
df_ridge_cal = pd.read_csv(
    r'results/transfer_results/aef/transfer_matrix_r2.csv',
    index_col=0
)

# 绘制热力图（vmin 截断设为 -3，避免个别极端值拉偏色阶）
plt.figure(figsize=(12, 10))
sns.heatmap(
    df_ridge_cal,
    cmap='RdBu_r',
    center=0,
    vmin=-3,
    vmax=1,
    annot=True,          # 显示具体数字，便于审稿人核查
    fmt='.2f',
    linewidths=0.5
)
plt.title('Ridge Regression Transfer R² (Source-Normalized Calibration)')
plt.savefig('Ridge_Calibrated_Heatmap.png', dpi=300, bbox_inches='tight')