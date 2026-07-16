
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# -------------------------- 配置 --------------------------
# 请根据实际文件名修改
R2_MATRIX_FILE = r"GNN_output/transfer_results_experiments/transfer_matrix_r2_null_calibrated.csv"   # 校准后零模型 R² 矩阵
EMBEDDING_DIST_FILE = r"GNN_output/transfer_results_experiments/transfer_embedding_distances_gnn.csv"  # 可选，格式见下文

OUTPUT_PREFIX = "null_calibrated"

# -------------------------- 中文字体设置 --------------------------
# 设置中文字体（根据系统字体调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set(font='SimHei')  # 若系统中无 SimHei，可改为 'Microsoft YaHei'


# -------------------------- 1. 读取数据 --------------------------
df_r2 = pd.read_csv(R2_MATRIX_FILE, index_col=0)
df_r2.index = df_r2.index.astype(str)
df_r2.columns = df_r2.columns.astype(str)

# 获取所有数值（展平，忽略 NaN）
vals = df_r2.values.flatten()
vals = vals[~np.isnan(vals)]
n_total = len(vals)
print(f"总样本数（城市对）: {n_total}")

# -------------------------- 2. IQR 异常识别 --------------------------
Q1 = np.percentile(vals, 25)
Q3 = np.percentile(vals, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_mask = (df_r2 < lower) | (df_r2 > upper)
outlier_mask = outlier_mask.fillna(False)

n_outliers = outlier_mask.sum().sum()
n_normal = n_total - n_outliers
print(f"异常城市对数: {n_outliers} ({100*n_outliers/n_total:.1f}%)")
print(f"正常城市对数: {n_normal} ({100*n_normal/n_total:.1f}%)")

outlier_pairs = []
for src in df_r2.index:
    for tgt in df_r2.columns:
        if outlier_mask.loc[src, tgt]:
            outlier_pairs.append((src, tgt, df_r2.loc[src, tgt]))
outlier_df = pd.DataFrame(outlier_pairs, columns=["Source", "Target", "R2_cal"])
outlier_df.to_csv(f"GNN_output/1.3/{OUTPUT_PREFIX}_outlier_pairs.csv", index=False)
print(f"异常城市对列表已保存至 GNN_output/1.3/{OUTPUT_PREFIX}_outlier_pairs.csv")

# -------------------------- 3. 统计摘要（全量与剔除异常） --------------------------
def summarize(vals_array, label):
    return {
        "样本数": len(vals_array),
        "均值": np.mean(vals_array),
        "中位数": np.median(vals_array),
        "标准差": np.std(vals_array),
        "最小值": np.min(vals_array),
        "最大值": np.max(vals_array),
    }

normal_vals = vals[~((vals < lower) | (vals > upper))]
stats_all = summarize(vals, "全量")
stats_normal = summarize(normal_vals, "正常")

summary_df = pd.DataFrame([stats_all, stats_normal], index=["全量", "剔除异常"])
summary_df.to_csv(f"GNN_output/1.3/{OUTPUT_PREFIX}_summary_stats.csv")
print("\n统计摘要:")
print(summary_df)

# -------------------------- 4. 长格式转换（为回归准备） --------------------------
df_long = df_r2.stack().reset_index()
df_long.columns = ["Source", "Target", "R2_cal"]
df_long["IsOutlier"] = df_long.apply(
    lambda row: outlier_mask.loc[row["Source"], row["Target"]], axis=1
)
df_long = df_long.dropna(subset=["R2_cal"])

# -------------------------- 5. 合并嵌入差异（若提供） --------------------------
has_embedding = False
try:
    df_emb = pd.read_csv(EMBEDDING_DIST_FILE)
    df_emb = df_emb.rename(columns={"source": "Source", "target": "Target", "distance": "EmbeddingDistance"})
    df_long = df_long.merge(df_emb, on=["Source", "Target"], how="left")
    has_embedding = True
except FileNotFoundError:
    print(f"未找到嵌入距离文件 {EMBEDDING_DIST_FILE}，跳过回归分析。")

if has_embedding and "EmbeddingDistance" in df_long.columns:
    df_reg = df_long.dropna(subset=["EmbeddingDistance", "R2_cal"])
    reg_all = linregress(df_reg["EmbeddingDistance"], df_reg["R2_cal"])
    df_reg_normal = df_reg[~df_reg["IsOutlier"]]
    reg_normal = linregress(df_reg_normal["EmbeddingDistance"], df_reg_normal["R2_cal"])
    
    print("\n=== 回归分析（嵌入差异 vs 校准后 R²） ===")
    print(f"全量数据: R² = {reg_all.rvalue**2:.4f}, p = {reg_all.pvalue:.4e}, 斜率 = {reg_all.slope:.4f}")
    print(f"剔除异常: R² = {reg_normal.rvalue**2:.4f}, p = {reg_normal.pvalue:.4e}, 斜率 = {reg_normal.slope:.4f}")
    
    reg_results = pd.DataFrame({
        "数据集": ["全量", "剔除异常"],
        "R2": [reg_all.rvalue**2, reg_normal.rvalue**2],
        "p值": [reg_all.pvalue, reg_normal.pvalue],
        "斜率": [reg_all.slope, reg_normal.slope],
        "截距": [reg_all.intercept, reg_normal.intercept],
    })
    reg_results.to_csv(f"GNN_output/1.3/{OUTPUT_PREFIX}_regression_results.csv", index=False)

# -------------------------- 6. 图表生成 --------------------------
# 6.1 箱线图：全量 vs 剔除异常后
fig, ax = plt.subplots(figsize=(8, 5))
data_to_plot = [vals, normal_vals]
bp = ax.boxplot(data_to_plot, patch_artist=True)  # 移除 labels 参数
ax.set_xticklabels(["全量", "剔除异常"])           # 用 set_xticklabels 替代
ax.set_ylabel("校准后 $R^2$ (零模型)")
ax.set_title("IQR 异常剔除前后对比")
plt.savefig(f"{OUTPUT_PREFIX}_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# 6.2 热图：原始矩阵 vs 剔除异常后的矩阵
df_r2_masked = df_r2.copy()
df_r2_masked[outlier_mask] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(df_r2, ax=axes[0], cmap="RdBu_r", center=0, annot=False, cbar=True,
            square=True, xticklabels=True, yticklabels=True)
axes[0].set_title("原始校准后 R² (全量)")

sns.heatmap(df_r2_masked, ax=axes[1], cmap="RdBu_r", center=0, annot=False, cbar=True,
            square=True, xticklabels=True, yticklabels=True)
axes[1].set_title("剔除异常值后 (白色为异常)")
plt.tight_layout()
plt.savefig(f"GNN_output/1.3/{OUTPUT_PREFIX}_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# 6.3 散点图（如果有嵌入距离）
if has_embedding and "EmbeddingDistance" in df_long.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    normal_pts = df_long[~df_long["IsOutlier"]]
    outlier_pts = df_long[df_long["IsOutlier"]]
    ax.scatter(normal_pts["EmbeddingDistance"], normal_pts["R2_cal"],
               c="blue", label="正常", alpha=0.6, s=30)
    ax.scatter(outlier_pts["EmbeddingDistance"], outlier_pts["R2_cal"],
               c="red", label="异常", alpha=0.8, s=50, marker="x")
    ax.set_xlabel("嵌入差异 (MMD / CKA 等)")
    ax.set_ylabel("校准后 R² (零模型)")
    ax.legend()
    if has_embedding and "EmbeddingDistance" in df_long.columns and len(df_reg_normal) > 1:
        x_range = np.linspace(df_reg_normal["EmbeddingDistance"].min(),
                              df_reg_normal["EmbeddingDistance"].max(), 100)
        y_pred = reg_normal.intercept + reg_normal.slope * x_range
        ax.plot(x_range, y_pred, color="green", linestyle="--",
                label=f"拟合线 (剔除异常, R²={reg_normal.rvalue**2:.3f})")
        ax.legend()
    ax.set_title("嵌入差异 vs 迁移性能 (校准后零模型)")
    plt.tight_layout()
    plt.savefig(f"GNN_output/1.3/{OUTPUT_PREFIX}_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

# 6.4 异常城市对列表表格
if len(outlier_pairs) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(outlier_pairs)*0.4)))
    ax.axis('tight')
    ax.axis('off')
    table_data = outlier_df[["Source", "Target", "R2_cal"]].values
    col_labels = ["源城市", "目标城市", "校准后 R²"]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center', colColours=["#f5f5f5"]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.title("异常城市对列表 (IQR 法)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"GNN_output/1.3/{OUTPUT_PREFIX}_outlier_table.png", dpi=300, bbox_inches="tight")
    plt.close()

print(f"\n所有图表已保存，前缀为 {OUTPUT_PREFIX}")