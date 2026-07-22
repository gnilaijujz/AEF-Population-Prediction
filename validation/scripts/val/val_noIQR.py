import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
OLD_HEALTH_CSV = r"results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"  # 旧城市健康度
NEW_HEALTH_CSV = r"validation\results\embedding_health_deep_analysis\embedding_health_deep_metrics.csv"  # 新城市健康度
OUTPUT_ROOT = Path("validation/results/validation_noIQR")   # 独立输出目录

OLD_TARGET_COL = 'avg_CDS'
NEW_TARGET_COL = 'avg_CDS_old'

# ================== 数据读取 ==================
old_df = pd.read_csv(OLD_HEALTH_CSV, index_col=0)
new_df = pd.read_csv(NEW_HEALTH_CSV, index_col=0)

if OLD_TARGET_COL not in old_df.columns:
    raise ValueError(f"旧数据缺少目标列 '{OLD_TARGET_COL}'")
if NEW_TARGET_COL not in new_df.columns:
    raise ValueError(f"新数据缺少目标列 '{NEW_TARGET_COL}'")

old_df = old_df.dropna(subset=[OLD_TARGET_COL])
new_df = new_df.dropna(subset=[NEW_TARGET_COL])

# 候选特征：两数据共有、数值型、且在旧数据中非恒定
candidate_features = []
for col in old_df.columns:
    if col == OLD_TARGET_COL:
        continue
    if col not in new_df.columns:
        continue
    if not (pd.api.types.is_numeric_dtype(old_df[col]) and pd.api.types.is_numeric_dtype(new_df[col])):
        continue
    if old_df[col].std() < 1e-10:
        print(f"跳过常数特征: {col}")
        continue
    candidate_features.append(col)

print(f"候选特征（共 {len(candidate_features)} 个）: {candidate_features}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
summary_list = []

for feature in candidate_features:
    print(f"\n{'='*60}")
    print(f"处理特征: {feature}")

    old_feat = old_df[[feature, OLD_TARGET_COL]].dropna()
    new_feat = new_df[[feature, NEW_TARGET_COL]].dropna()

    if len(old_feat) < 5 or len(new_feat) < 1:
        print(f"  跳过：样本数不足 (旧:{len(old_feat)}, 新:{len(new_feat)})")
        continue

    # ---- 回归（使用全部旧数据，不筛选） ----
    X_old = old_feat[[feature]]
    y_old = old_feat[OLD_TARGET_COL]
    X_old_sm = sm.add_constant(X_old)

    try:
        model = sm.OLS(y_old, X_old_sm).fit()
    except Exception as e:
        print(f"  回归拟合失败: {e}")
        continue

    if 'const' not in model.params.index:
        print("  回归结果中无截距项，跳过")
        continue

    intercept = model.params['const']
    slope = model.params[feature]
    r2 = model.rsquared
    p_value = model.pvalues[feature]

    # ---- 预测新城市 ----
    X_new = new_feat[[feature]]
    X_new_sm = sm.add_constant(X_new)
    try:
        y_pred = model.predict(X_new_sm)
    except Exception as e:
        print(f"  预测失败: {e}")
        continue
    y_true = new_feat[NEW_TARGET_COL]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    print(f"  旧 R²={r2:.4f}, p={p_value:.4f} | 新 RMSE={rmse:.4f}, r={corr:.4f}")

    # ---- 保存结果 ----
    feature_dir = OUTPUT_ROOT / feature
    feature_dir.mkdir(parents=True, exist_ok=True)

    with open(feature_dir / "regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f"\n\nRegression equation: {OLD_TARGET_COL} = {slope:.6f} * {feature} + {intercept:.6f}\n")
        f.write(f"RMSE on new cities: {rmse:.4f}\n")
        f.write(f"MAE on new cities:  {mae:.4f}\n")
        f.write(f"Pearson correlation: {corr:.4f}\n")

    comparison = pd.DataFrame({
        'city': new_feat.index,
        'actual_CDS': y_true,
        'predicted_CDS': y_pred,
        'residual': y_true - y_pred
    })
    comparison.to_csv(feature_dir / "prediction_results.csv", index=False)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_old, y_old, color='blue', s=60, alpha=0.7, label='Old cities (actual)')
    x_min = min(X_old.min().iloc[0], X_new.min().iloc[0])
    x_max = max(X_old.max().iloc[0], X_new.max().iloc[0])
    x_range = np.linspace(x_min, x_max, 100)
    y_range = intercept + slope * x_range
    ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression line')
    ax.scatter(X_new, y_true, color='green', s=80, marker='o', label='New cities (actual)')
    ax.scatter(X_new, y_pred, color='orange', s=150, marker='*', label='New cities (predicted)')
    for i in range(len(X_new)):
        x_val = X_new.iloc[i, 0]
        ax.plot([x_val, x_val], [y_pred.iloc[i], y_true.iloc[i]], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel('Average Transfer Loss')
    ax.set_title(f'Cross-domain Validation: Feature = {feature}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(feature_dir / f"validation_plot_{feature}.png", dpi=300)
    plt.close()

    summary_list.append({
        'Feature': feature,
        'Old_R2': r2,
        'Old_p_value': p_value,
        'Slope': slope,
        'Intercept': intercept,
        'New_RMSE': rmse,
        'New_MAE': mae,
        'New_Pearson_r': corr,
        'Old_samples': len(old_feat),
        'New_samples': len(new_feat)
    })

# ================== 汇总表 ==================
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('New_Pearson_r', ascending=False)
    summary_df.to_csv(OUTPUT_ROOT / "summary_all_features.csv", index=False)
    print(f"\n{'='*60}")
    print("汇总结果（按预测相关系数降序）：")
    print(summary_df[['Feature', 'Old_R2', 'Old_p_value', 'New_RMSE', 'New_Pearson_r']].to_string(index=False))
    print(f"\n汇总表保存至: {OUTPUT_ROOT / 'summary_all_features.csv'}")
else:
    print("没有成功处理任何特征。")

print(f"\n所有结果已保存至根目录: {OUTPUT_ROOT}")