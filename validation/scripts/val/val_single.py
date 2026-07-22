import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

# ================== Configuration ==================
OLD_HEALTH_CSV = r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"  # 旧城市健康度
NEW_HEALTH_CSV = r"validation\results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"  # 新城市健康度
OUTPUT_ROOT = Path("validation/results/validation_single")   # 统一输出根目录

FEATURE = 'inter_intra_ratio'   # 可选择 'effective_rank', 'n_nodes', 'isotropy' 等
OLD_TARGET_COL = 'avg_CDS'
NEW_TARGET_COL = 'avg_CDS_old'
IQR_MULTIPLIER = 100000        # IQR 异常值阈值

# ================== 创建输出目录 ==================
output_dir = OUTPUT_ROOT / FEATURE
output_dir.mkdir(parents=True, exist_ok=True)
print(f"所有结果将保存至: {output_dir}")

# ================== 数据读取 ==================
old_df = pd.read_csv(OLD_HEALTH_CSV, index_col=0)
new_df = pd.read_csv(NEW_HEALTH_CSV, index_col=0)

# 检查列存在
for col in [FEATURE, OLD_TARGET_COL]:
    if col not in old_df.columns:
        raise ValueError(f"Old data missing {col}")
for col in [FEATURE, NEW_TARGET_COL]:
    if col not in new_df.columns:
        raise ValueError(f"New data missing {col}")

old_df = old_df.dropna(subset=[FEATURE, OLD_TARGET_COL])
new_df = new_df.dropna(subset=[FEATURE, NEW_TARGET_COL])

print(f"Old cities (original): {len(old_df)}")
print(f"New cities: {len(new_df)}")

# ================== IQR 异常值筛选（仅针对旧城市训练集） ==================
Q1 = old_df[FEATURE].quantile(0.25)
Q3 = old_df[FEATURE].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - IQR_MULTIPLIER * IQR
upper_bound = Q3 + IQR_MULTIPLIER * IQR

old_df_filtered = old_df[(old_df[FEATURE] >= lower_bound) & (old_df[FEATURE] <= upper_bound)]
print(f"Old cities after IQR filtering: {len(old_df_filtered)} (removed {len(old_df)-len(old_df_filtered)})")

if len(old_df_filtered) < 5:
    print("Warning: too few samples after filtering, using original data.")
    old_df_filtered = old_df

# ================== 回归拟合（使用筛选后的旧城市数据） ==================
X_old = old_df_filtered[[FEATURE]]
y_old = old_df_filtered[OLD_TARGET_COL]

X_old_sm = sm.add_constant(X_old)
model_old = sm.OLS(y_old, X_old_sm).fit()
print("\n===== Regression on Old Cities (after IQR) =====")
print(model_old.summary())

intercept = model_old.params['const']
slope = model_old.params[FEATURE]
print(f"\nRegression equation: {OLD_TARGET_COL} = {slope:.4f} * {FEATURE} + {intercept:.4f}")

# ================== 预测新城市 ==================
X_new = new_df[[FEATURE]]
X_new_sm = sm.add_constant(X_new)
y_new_pred = model_old.predict(X_new_sm)
y_new_true = new_df[NEW_TARGET_COL]

rmse = np.sqrt(mean_squared_error(y_new_true, y_new_pred))
mae = mean_absolute_error(y_new_true, y_new_pred)
corr = np.corrcoef(y_new_true, y_new_pred)[0, 1]

print("\n===== Prediction on New Cities =====")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"Pearson correlation: {corr:.4f}")

# 预测对比表
comparison = pd.DataFrame({
    'city': new_df.index,
    'actual_CDS': y_new_true,
    'predicted_CDS': y_new_pred,
    'residual': y_new_true - y_new_pred
})
print("\nPrediction vs Actual (first 5):")
print(comparison.head())

# ================== 保存回归摘要 ==================
with open(output_dir / "regression_summary.txt", "w") as f:
    f.write(model_old.summary().as_text())
    f.write(f"\n\nRegression equation: {OLD_TARGET_COL} = {slope:.6f} * {FEATURE} + {intercept:.6f}\n")
    f.write(f"RMSE on new cities: {rmse:.4f}\n")
    f.write(f"MAE on new cities:  {mae:.4f}\n")
    f.write(f"Pearson correlation: {corr:.4f}\n")

# ================== 保存预测结果 ==================
comparison.to_csv(output_dir / "prediction_results.csv", index=False)

# ================== 保存实验配置 ==================
with open(output_dir / "config.txt", "w") as f:
    f.write(f"FEATURE: {FEATURE}\n")
    f.write(f"OLD_TARGET_COL: {OLD_TARGET_COL}\n")
    f.write(f"NEW_TARGET_COL: {NEW_TARGET_COL}\n")
    f.write(f"IQR_MULTIPLIER: {IQR_MULTIPLIER}\n")
    f.write(f"Old cities (original): {len(old_df)}\n")
    f.write(f"Old cities (after IQR): {len(old_df_filtered)}\n")
    f.write(f"New cities: {len(new_df)}\n")

# ================== 绘图 ==================
fig, ax = plt.subplots(figsize=(10, 6))

# 旧城市散点
ax.scatter(X_old, y_old, color='blue', s=60, alpha=0.7, label='Old cities (actual)')

# 回归线
x_min = min(X_old.min().iloc[0], X_new.min().iloc[0])
x_max = max(X_old.max().iloc[0], X_new.max().iloc[0])
x_range = np.linspace(x_min, x_max, 100)
y_range = intercept + slope * x_range
ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression line (old cities)')

# 新城市实际值
ax.scatter(X_new, y_new_true, color='green', s=80, marker='o', label='New cities (actual)')

# 新城市预测值
ax.scatter(X_new, y_new_pred, color='orange', s=150, marker='*', label='New cities (predicted)')

# 连接预测与实际
for i in range(len(X_new)):
    x_val = X_new.iloc[i, 0]
    y_pred = y_new_pred.iloc[i]
    y_true = y_new_true.iloc[i]
    ax.plot([x_val, x_val], [y_pred, y_true], 'k--', alpha=0.4, linewidth=1)

ax.set_xlabel(FEATURE)
ax.set_ylabel('Average Transfer Loss (avg_CDS)')
ax.set_title(f'Cross-domain Validation: Old-city Regression on New Cities\nFeature = {FEATURE}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(output_dir / f"validation_plot_{FEATURE}.png", dpi=300)
print(f"\nFigure saved to {output_dir / f'validation_plot_{FEATURE}.png'}")
plt.close()  # 不显示，节省资源

print(f"\nAll results saved to: {output_dir}")