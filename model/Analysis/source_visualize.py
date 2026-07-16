import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 假设 df_clean 已存在，并且包含 'Source', 'Target', 'L2_Dist', 'R2_cal'
# 如果没有目标大小，这里用随机模拟占位（您实际应替换为真实的 Node_Count）
# 从 city_cache 获取目标节点数（假设您已经在之前的代码中加载了）
# 这里假设您有一个字典 city_sizes = {'Baton': 245, 'Bridgeport': 258, ...}

# 1. 准备数据
# 确保每个目标城市有节点数，用于控制散点大小
# 如果您没有这个列，请运行：
df_clean = df_clean.merge(city_stats[['City', 'Node_Count']], left_on='Target', right_on='City', how='left')

# 2. 计算每个 Source 的预测值（基于全局回归线）
# 使用之前拟合的 model_clean (剔除异常后的线性模型)
b = model_clean.params["L2_Dist"]
a = model_clean.params["const"]
df_clean['Predicted_R2'] = a + b * df_clean['L2_Dist']

# 3. 设置绘图样式
sns.set_style("whitegrid")
palette = sns.color_palette("tab10", n_colors=len(df_clean['Source'].unique()))
source_colors = {src: palette[i] for i, src in enumerate(df_clean['Source'].unique())}

# ================== 图 1：按源城市分色的 Segregation vs Transfer R² ==================
fig, ax = plt.subplots(figsize=(10, 7))

# 注意：示例图中 Y 轴是 Transfer R²，X 轴是 Segregation difference（这里对应用 L2_Dist）
for src in df_clean['Source'].unique():
    subset = df_clean[df_clean['Source'] == src]
    # 按大小缩放：这里用目标节点数的对数，便于区分大小城市
    sizes = subset['Target_Node_Count'].apply(lambda x: (x / 100) + 20)  # 调整系数
    ax.scatter(subset['L2_Dist'], subset['R2_cal'], 
               s=sizes, color=source_colors[src], alpha=0.7, 
               label=src, edgecolors='black', linewidth=0.3)

# 添加全局回归线（虚线）
x_range = np.linspace(df_clean['L2_Dist'].min(), df_clean['L2_Dist'].max(), 100)
y_global = a + b * x_range
ax.plot(x_range, y_global, 'k--', linewidth=2, label='全局拟合 (所有源)')

# 重要：如果样本量大，图例可能会重叠，只显示前几个或调整位置
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_xlabel("L2 嵌入均值距离 (域偏移程度)", fontsize=12)
ax.set_ylabel("校准后迁移 R²", fontsize=12)
ax.set_title("源特异性域偏移与迁移性能 (散点大小 = 目标城市规模)", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_Source_Wise_Scatter.png", dpi=300)
plt.show()

# ================== 图 2：实际迁移 R² vs 预测迁移 R² (带 1:1 线) ==================
fig, ax = plt.subplots(figsize=(8, 8))

# 按源分色
for src in df_clean['Source'].unique():
    subset = df_clean[df_clean['Source'] == src]
    ax.scatter(subset['Predicted_R2'], subset['R2_cal'], 
               s=50, color=source_colors[src], alpha=0.7, label=src)

# 绘制 1:1 完美预测线（y=x）
min_val = min(df_clean['Predicted_R2'].min(), df_clean['R2_cal'].min())
max_val = max(df_clean['Predicted_R2'].max(), df_clean['R2_cal'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 完美预测')

# 计算整体预测误差（R² 或 RMSE）
rmse = np.sqrt(np.mean((df_clean['R2_cal'] - df_clean['Predicted_R2'])**2))
ax.text(0.05, 0.95, f'全局 RMSE = {rmse:.3f}', transform=ax.transAxes, 
        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white'))

ax.set_xlabel("预测迁移 R² (基于全局线性映射)", fontsize=12)
ax.set_ylabel("实际迁移 R²", fontsize=12)
ax.set_title("实际 vs 预测迁移性能 (按源城市分色)", fontsize=14)
ax.legend(loc='lower right', ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig("Figure_Predicted_vs_Actual.png", dpi=300)
plt.show()