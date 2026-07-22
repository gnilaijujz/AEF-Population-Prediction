import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import os
# ========================
# 1. 用户配置区
# ========================
DATA_PATH = r"results/transfer_results/aef_plus_nighttime_lights/transfer_metrics_gnn.csv"   # 请替换为实际路径
OUTPUT_PATH = r"results/transferability/nighttime_lights"   # 输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)   # 确保输出目录存在
from scipy.stats import zscore   # 用于稳健标准化


# 指标方向定义：-1 负向（越小越好），1 正向（越大越好）
INDICATOR_DIRECTION = {
    'rmse_raw': -1,
    'mae_raw': -1,
    'r2_raw': 1,
    'mape_raw': -1,
    'r_raw': 1
}

# ========================
# 2. 数据读取与稳健正向化
# ========================
df = pd.read_csv(DATA_PATH)
required_cols = ['source', 'target'] + list(INDICATOR_DIRECTION.keys())
assert all(col in df.columns for col in required_cols), "数据列缺失"

def robust_normalize(series):
    """
    稳健标准化：先 Z-score 去量纲，再将结果线性映射到 [0,1]。
    此方法对异常值不敏感，且不会产生大量 0 或 1，适合熵权法。
    """
    # 1. Z-score 标准化
    z_scored = zscore(series, nan_policy='omit')
    # 2. 映射到 [0,1]
    min_val = z_scored.min()
    max_val = z_scored.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (z_scored - min_val) / (max_val - min_val)

# 正向化处理：所有指标统一为“值越大越好”
norm_df = pd.DataFrame(index=df.index)
for col, direction in INDICATOR_DIRECTION.items():
    # 1. 稳健标准化（无方向）
    norm_vals = robust_normalize(df[col])
    # 2. 若为负向指标，取补数使其正向
    if direction == -1:
        norm_vals = 1 - norm_vals
    norm_df[col] = norm_vals

print("正向化后的数据（前5行，值越大越好）：")
print(norm_df.head())

# ========================
# 3. 熵权法计算权重
# ========================
def entropy_weight(data):
    """
    输入：DataFrame，每列为指标，行为样本，数据已正向化（非负）
    输出：各指标权重（Series）
    """
    data = data.copy()
    # 为避免 log(0)，对 0 值替换为一个极小正数
    data[data == 0] = 1e-10
    
    # 计算比重矩阵 p_ij = x_ij / sum(x_j)
    p = data.div(data.sum(axis=0), axis=1)
    
    # 计算信息熵
    n = len(data)
    k = 1 / np.log(n)
    e = -k * (p * np.log(p)).sum(axis=0)
    
    # 差异系数
    d = 1 - e
    # 权重
    w = d / d.sum()
    return w

weights = entropy_weight(norm_df)
print("\n===== 熵权法计算得到的指标权重（稳健标准化） =====")
for col, w in weights.items():
    print(f"{col}: {w:.4f}")
print(f"权重总和: {weights.sum():.4f}")

# ========================
# 4. 计算综合得分 Score
# ========================
df['Score'] = 0.0
for col in weights.index:
    df['Score'] += norm_df[col] * weights[col]

print("\n综合得分计算完成。")
print(df[['source', 'target', 'Score']].head())

# ========================
# 5. 构建城市列表及得分矩阵
# ========================
cities = sorted(df['source'].unique())
m = len(cities)
print(f"共 {m} 个城市: {cities}")

# 构建 Score C 矩阵（源 i -> 目标 j）
score_C = pd.DataFrame(index=cities, columns=cities, dtype=float)
for i, j in itertools.product(cities, repeat=2):
    row = df[(df['source'] == i) & (df['target'] == j)]
    if len(row) == 1:
        score_C.loc[i, j] = row.iloc[0]['Score']
    else:
        score_C.loc[i, j] = np.nan

# 提取 Score S 和 Score T（自测值）
score_S = pd.Series(index=cities, dtype=float)
score_T = pd.Series(index=cities, dtype=float)
for city in cities:
    val = score_C.loc[city, city]
    score_S[city] = val
    score_T[city] = val   # 数值相同，保留变量名以示区分

# ========================
# 6. 计算可迁移性指标 CDS, CDA, CDT
# ========================
CDS = pd.DataFrame(index=cities, columns=cities, dtype=float)  # S - C
CDA = pd.DataFrame(index=cities, columns=cities, dtype=float)  # T - C
CDT = pd.DataFrame(index=cities, columns=cities, dtype=float)  # S - T (基线)

for i in cities:
    for j in cities:
        s_i = score_S[i]
        t_j = score_T[j]
        c_ij = score_C.loc[i, j]
        
        CDT.loc[i, j] = s_i - t_j   # 基线差异（与 C 无关）
        
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

# ========================
# 7. 统计量计算（排除对角线）
# ========================
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
# 9. 保存结果
# ========================
# 保存权重
weights_df = pd.DataFrame({'指标': weights.index, '熵权法权重': weights.values})
weights_df.to_csv("{OUTPUT_PATH}\\entropy_weights_robust.csv".format(OUTPUT_PATH=OUTPUT_PATH), index=False)
print("\n权重已保存至 entropy_weights_robust.csv")

# 保存完整明细（含原始数据、正向化数据、Score）
full_result = pd.concat([df, norm_df.add_suffix('_norm')], axis=1)
full_result.to_csv("{OUTPUT_PATH}\\full_scores_robust.csv".format(OUTPUT_PATH=OUTPUT_PATH), index=False, encoding='utf-8-sig')

# 保存可迁移性矩阵
# 保存可迁移性矩阵（保留行索引）
CDS.to_csv(f"{OUTPUT_PATH}/CDS_matrix_robust_entropy.csv")  # 默认 index=True
CDA.to_csv(f"{OUTPUT_PATH}/CDA_matrix_robust_entropy.csv")
CDT.to_csv(f"{OUTPUT_PATH}/CDT_matrix_robust_entropy.csv")
