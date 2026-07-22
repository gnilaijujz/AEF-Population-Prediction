import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import os
# ========================
# 1. 用户配置区
# ========================
DATA_PATH = r"validation\results\transfer_results\mean\transfer_metrics_gnn.csv"   # 请替换为实际路径
OUTPUT_PATH = r"validation\results\transferability"   # 输出目录
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
# 保留 error 列为空的有效行（NaN 或 ''）
df = df[df['error'].isna() | (df['error'] == '')].reset_index(drop=True)
df = df.drop(columns=['error'], errors='ignore')

required_cols = ['source', 'target'] + list(INDICATOR_DIRECTION.keys())
assert all(col in df.columns for col in required_cols), "数据列缺失"

def robust_normalize(series):
    series = pd.to_numeric(series, errors='coerce')
    if series.isna().all():
        return pd.Series([0.5] * len(series), index=series.index)
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)

norm_df = pd.DataFrame(index=df.index)
for col, direction in INDICATOR_DIRECTION.items():
    norm_vals = robust_normalize(df[col])
    if direction == -1:
        norm_vals = 1 - norm_vals
    norm_df[col] = norm_vals

print("正向化后的数据（前5行）：")
print(norm_df.head())
print("\n每列缺失数量：")
print(norm_df.isna().sum())

# ========================
# 3. 熵权法计算权重
# ========================
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
print("\n===== 熵权法计算得到的指标权重 =====")
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
# 5. 构建迁移损失矩阵
# ========================
sources = sorted(df['source'].unique())
targets = sorted(df['target'].unique())

print(f"\n源城市（{len(sources)}个）：{sources}")
print(f"目标城市（{len(targets)}个）：{targets}")

score_matrix = pd.DataFrame(index=sources, columns=targets, dtype=float)
for _, row in df.iterrows():
    score_matrix.loc[row['source'], row['target']] = row['Score']

# 源自测得分
self_scores = {src: score_matrix.loc[src, src] if src in score_matrix.columns else np.nan for src in sources}

loss_matrix = pd.DataFrame(index=sources, columns=targets, dtype=float)
for i in sources:
    for j in targets:
        if i == j:
            loss_matrix.loc[i, j] = 0.0
        else:
            s_i = self_scores.get(i, np.nan)
            c_ij = score_matrix.loc[i, j]
            if not np.isnan(s_i) and not np.isnan(c_ij):
                loss_matrix.loc[i, j] = s_i - c_ij
            else:
                loss_matrix.loc[i, j] = np.nan

# ========================
# 6. 统计分析（排除自测）
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
        print(f"\n--- {name} 统计量（排除自测） ---")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")
        return arr
    else:
        print(f"\n{name} 无非自测有效值。")
        return None

loss_vals = calc_stats(loss_matrix, "迁移损失 (S - C)")

# ========================
# 7. 保存结果
# ========================
weights_df = pd.DataFrame({'指标': weights.index, '熵权法权重': weights.values})
weights_df.to_csv(f"{OUTPUT_PATH}/entropy_weights_robust.csv", index=False)

full_result = pd.concat([df, norm_df.add_suffix('_norm')], axis=1)
full_result.to_csv(f"{OUTPUT_PATH}/full_scores_robust.csv", index=False, encoding='utf-8-sig')

loss_matrix.to_csv(f"{OUTPUT_PATH}/transfer_loss_matrix_robust_entropy.csv")
print(f"\n损失矩阵已保存至 {OUTPUT_PATH}/transfer_loss_matrix_robust_entropy.csv")

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
