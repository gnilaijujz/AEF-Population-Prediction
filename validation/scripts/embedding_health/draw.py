import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')  # 可选，忽略警告信息

# ---------- 强制指定中文字体（Windows 通用） ----------
# 方法1：直接指定黑体（Windows 标配）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方法2（保险）：如果上述方法仍失效，手动查找可用中文字体
# 取消下面代码的注释，可以自动查找并使用系统里的第一个中文字体
# font_list = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft YaHei' in f.name]
# if font_list:
#     plt.rcParams['font.sans-serif'] = [font_list[0]]
# plt.rcParams['axes.unicode_minus'] = False
# 加载数据
df = pd.read_csv(r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv", index_col=0)
# 按avg_CDS排序（好→差）
city_order = df.sort_values('avg_CDS', ascending=True).index.tolist()
df = df.loc[city_order]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ---------- Fig4-A: 散点图矩阵 ----------
sns.pairplot(df[['n_nodes', 'effective_rank', 'isotropy', 'neighbor_consistency', 'inter_intra_ratio', 'pca_5_explained', 'avg_CDS']])
plt.savefig(r"paper_figures\figure4\Pairplot.png", dpi=300)
plt.close()

# ---------- Fig4-B: 按节点数分箱的箱线图 ----------
df['size_group'] = pd.cut(df['n_nodes'], bins=3, labels=['小型(<200)', '中型(200-1000)', '大型(>1000)'])
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='size_group', y='avg_CDS', data=df, ax=ax)
ax.set_xlabel('城市规模 (节点数)')
ax.set_ylabel('平均迁移损失 (avg_CDS)')
ax.set_title('城市规模与迁移性能的关系')
plt.tight_layout()
plt.savefig(r"paper_figures\figure4\Size_Boxplot.png", dpi=300)
plt.close()

# ---------- Fig4-C: 邻居一致性 vs avg_CDS ----------
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df['neighbor_consistency'], df['avg_CDS'], s=80)
for city in df.index:
    ax.annotate(city, (df.loc[city, 'neighbor_consistency'], df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=8)
slope, intercept, r, p, _ = linregress(df['neighbor_consistency'], df['avg_CDS'])
x_vals = np.linspace(df['neighbor_consistency'].min(), df['neighbor_consistency'].max(), 50)
ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'R²={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('邻居一致性')
ax.set_ylabel('avg_CDS')
ax.set_title('邻居一致性 vs 迁移损失')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(r"paper_figures\figure4\NeighborConsistency_vs_CDS.png", dpi=300)
plt.close()

# ---------- Fig4-D: 有效秩 vs avg_CDS ----------
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df['effective_rank'], df['avg_CDS'], s=80)
slope, intercept, r, p, _ = linregress(df['effective_rank'], df['avg_CDS'])
x_vals = np.linspace(df['effective_rank'].min(), df['effective_rank'].max(), 50)
ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'R²={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('有效秩')
ax.set_ylabel('avg_CDS')
ax.set_title('有效秩 vs 迁移损失')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(r"paper_figures\figure4\EffectiveRank_vs_CDS.png", dpi=300)
plt.close()

# ---------- Fig4-E: 节点数 vs 邻居一致性 ----------
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df['n_nodes'], df['neighbor_consistency'], s=80)
slope, intercept, r, p, _ = linregress(df['n_nodes'], df['neighbor_consistency'])
x_vals = np.linspace(df['n_nodes'].min(), df['n_nodes'].max(), 50)
ax.plot(x_vals, intercept + slope*x_vals, 'r--', label=f'R²={r**2:.3f}, p={p:.4f}')
ax.set_xlabel('节点数 (城市规模)')
ax.set_ylabel('邻居一致性')
ax.set_title('城市规模与邻居一致性的关系')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(r"paper_figures\figure4\Size_vs_Consistency.png", dpi=300)
plt.close()

# ---------- Fig4-F: 三维视角（节点数 vs avg_CDS，颜色=邻居一致性）----------
fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(df['n_nodes'], df['avg_CDS'], c=df['neighbor_consistency'], cmap='viridis', s=80)
for city in df.index:
    ax.annotate(city, (df.loc[city, 'n_nodes'], df.loc[city, 'avg_CDS']),
                xytext=(5,5), textcoords='offset points', fontsize=8)
ax.set_xlabel('节点数')
ax.set_ylabel('avg_CDS')
ax.set_title('城市规模 vs 迁移损失 (颜色表示邻居一致性)')
cbar = plt.colorbar(sc)
cbar.set_label('邻居一致性')
plt.tight_layout()
plt.savefig(r"paper_figures\figure4\Size_vs_CDS_colored_by_Consistency.png", dpi=300)
plt.close()