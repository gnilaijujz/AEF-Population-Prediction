import os
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# ========================
# 1. 用户配置区
# ========================

# 相对脚本所在目录定位
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "transferability_results")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(SCRIPT_DIR, "transfer_results_log", "transfer_metrics_gnn.csv")

# 设置字体：中文宋体，英文Times New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'SimHei', 'Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 指标方向定义
INDICATOR_DIRECTION = {
    'rmse_raw': -1,
    'mae_raw': -1,
    'r2_raw': 1,
    'mape_raw': -1,
    'r_raw': 1
}

# ========================
# 2-7. 数据读取与计算（保持不变）
# ========================
df = pd.read_csv(DATA_PATH)
required_cols = ['source', 'target'] + list(INDICATOR_DIRECTION.keys())
assert all(col in df.columns for col in required_cols), "数据列缺失"

def robust_normalize(series):
    z_scored = zscore(series, nan_policy='omit')
    min_val = z_scored.min()
    max_val = z_scored.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (z_scored - min_val) / (max_val - min_val)

norm_df = pd.DataFrame(index=df.index)
for col, direction in INDICATOR_DIRECTION.items():
    norm_vals = robust_normalize(df[col])
    if direction == -1:
        norm_vals = 1 - norm_vals
    norm_df[col] = norm_vals

print("正向化后的数据（前5行，值越大越好）：")
print(norm_df.head())

def entropy_weight(data):
    data = data.copy()
    data[data == 0] = 1e-10
    p = data.div(data.sum(axis=0), axis=1)
    n = len(data)
    k = 1 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    w = d / d.sum()
    return w

weights = entropy_weight(norm_df)
print("\n===== 熵权法计算得到的指标权重（稳健标准化） =====")
for col, w in weights.items():
    print(f"{col}: {w:.4f}")
print(f"权重总和: {weights.sum():.4f}")

df['Score'] = 0.0
for col in weights.index:
    df['Score'] += norm_df[col] * weights[col]

print("\n综合得分计算完成。")
print(df[['source', 'target', 'Score']].head())

cities = sorted(df['source'].unique())
m = len(cities)
print(f"共 {m} 个城市: {cities}")

score_C = pd.DataFrame(index=cities, columns=cities, dtype=float)
for i, j in itertools.product(cities, repeat=2):
    row = df[(df['source'] == i) & (df['target'] == j)]
    if len(row) == 1:
        score_C.loc[i, j] = row.iloc[0]['Score']
    else:
        score_C.loc[i, j] = np.nan

score_S = pd.Series(index=cities, dtype=float)
score_T = pd.Series(index=cities, dtype=float)
for city in cities:
    val = score_C.loc[city, city]
    score_S[city] = val
    score_T[city] = val

CDS = pd.DataFrame(index=cities, columns=cities, dtype=float)
CDA = pd.DataFrame(index=cities, columns=cities, dtype=float)
CDT = pd.DataFrame(index=cities, columns=cities, dtype=float)

for i in cities:
    for j in cities:
        s_i = score_S[i]
        t_j = score_T[j]
        c_ij = score_C.loc[i, j]
        
        CDT.loc[i, j] = s_i - t_j
        
        if i == j:
            CDS.loc[i, j] = 0.0
            CDA.loc[i, j] = 0.0
        else:
            if not np.isnan(c_ij):
                CDS.loc[i, j] = s_i - c_ij
                CDA.loc[i, j] = t_j - c_ij
            else:
                CDS.loc[i, j] = np.nan
                CDA.loc[i, j] = np.nan

def calc_stats(matrix, name):
    values = []
    for i in matrix.index:
        for j in matrix.columns:
            if i != j and not np.isnan(matrix.loc[i, j]):
                values.append(matrix.loc[i, j])
    if values:
        arr = np.array(values)
        stats = {
            '平均值': np.mean(arr),
            '标准差': np.std(arr),
            '最小值': np.min(arr),
            '最大值': np.max(arr),
            '中位数': np.median(arr)
        }
        print(f"\n--- {name} 统计量（排除对角线） ---")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")
        return arr
    else:
        print(f"\n{name} 无非对角线有效值。")
        return None

cds_vals = calc_stats(CDS, "CDS")
cda_vals = calc_stats(CDA, "CDA")
cdt_vals = calc_stats(CDT, "CDT")

# ========================
# 8. 可视化（修正后的版本）
# ========================
def save_heatmap(matrix, title, cbar_label, cmap, fname):
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # 绘制热力图
    heatmap = sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, center=0,
                          square=True, cbar_kws={'label': cbar_label}, ax=ax,
                          annot_kws={'size': 10, 'fontproperties': 'Times New Roman'},
                          linewidths=0.5, linecolor='white')
    
    # 设置标题（英文用Times New Roman）
    ax.set_title(title, fontsize=15, fontfamily='Times New Roman')
    
    
    # 修正：使用 labelfontfamily 而不是 fontfamily
    ax.tick_params(labelsize=12, labelfontfamily='Times New Roman')
    
    # 分别设置x轴和y轴刻度标签的字体
    x_labels = ax.get_xticklabels()
    y_labels = ax.get_yticklabels()
    for label in x_labels:
        label.set_fontfamily('Times New Roman')
    for label in y_labels:
        label.set_fontfamily('Times New Roman')
    
    # ====== 调整右侧图例（colorbar）的字体和大小 ======
    cbar = heatmap.collections[0].colorbar
    # 设置图例标题的字体和大小
    cbar.set_label(cbar_label, fontsize=13, fontfamily='Times New Roman', weight='bold')
    # 设置图例刻度标签的字体和大小
    cbar.ax.tick_params(labelsize=11, labelfontfamily='Times New Roman')
    
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {out}")

def save_hist(vals, title, color, median_color, fname):
    if vals is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(vals, kde=True, bins=10, color=color, ax=ax)
    ax.axvline(np.mean(vals), color='red', linestyle='--', label=f'Mean={np.mean(vals):.2f}')
    ax.axvline(np.median(vals), color=median_color, linestyle='-.', label=f'Median={np.median(vals):.2f}')
    ax.legend(fontsize=10, prop={'family': 'Times New Roman'})
    ax.set_title(title, fontsize=13, fontfamily='Times New Roman')
    ax.set_xlabel('Value', fontsize=11, fontfamily='Times New Roman')
    ax.set_ylabel('Frequency', fontsize=11, fontfamily='Times New Roman')
    ax.tick_params(labelsize=10, labelfontfamily='Times New Roman')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {out}")

try:
    # ---- 三张热力图(各自单独一张) ----
    save_heatmap(CDS, 'CDS (S - C)', 'CDS', 'RdBu_r', 'CDS_heatmap.png')
    save_heatmap(CDA, 'CDA (T - C)', 'CDA', 'RdBu_r', 'CDA_heatmap.png')
    save_heatmap(CDT, 'CDT (S - T) Baseline', 'CDT', 'coolwarm', 'CDT_heatmap.png')

    # ---- 三张分布直方图(各自单独一张) ----
    save_hist(cds_vals, 'CDS Distribution', 'blue', 'green', 'CDS_distribution.png')
    save_hist(cda_vals, 'CDA Distribution', 'orange', 'green', 'CDA_distribution.png')
    save_hist(cdt_vals, 'CDT Distribution (Baseline)', 'green', 'blue', 'CDT_distribution.png')

    # ---- 额外:箱线图对比(单独一张) ----
    data_to_plot = []
    labels = []
    for arr, name in zip([cds_vals, cda_vals, cdt_vals], ['CDS', 'CDA', 'CDT']):
        if arr is not None:
            data_to_plot.append(arr)
            labels.append(name)
    if data_to_plot:
        fig2, ax_box = plt.subplots(figsize=(8, 5))
        bp = ax_box.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
            patch.set_facecolor(color)
        ax_box.set_ylabel('Transferability Value', fontsize=11, fontfamily='Times New Roman')
        ax_box.set_title('CDS / CDA / CDT Boxplot Comparison', fontsize=13, fontfamily='Times New Roman')
        ax_box.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_box.tick_params(labelsize=10, labelfontfamily='Times New Roman')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'transferability_boxplot.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"已保存: {out}")

except Exception as e:
    print(f"绘图出错: {e}")

# ========================
# 9. 保存结果
# ========================
weights_df = pd.DataFrame({'指标': weights.index, '熵权法权重': weights.values})
weights_df.to_csv(os.path.join(OUT_DIR, "entropy_weights.csv"), index=False)
print("\n权重已保存至 entropy_weights.csv")

full_result = pd.concat([df, norm_df.add_suffix('_norm')], axis=1)
full_result.to_csv(os.path.join(OUT_DIR, "full_scores.csv"), index=False, encoding='utf-8-sig')

CDS.to_csv(os.path.join(OUT_DIR, "CDS_matrix.csv"))
CDA.to_csv(os.path.join(OUT_DIR, "CDA_matrix.csv"))
CDT.to_csv(os.path.join(OUT_DIR, "CDT_matrix.csv"))
print("矩阵已保存：CDS_matrix.csv, CDA_matrix.csv, CDT_matrix.csv")