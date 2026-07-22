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
OUTPUT_ROOT = Path("validation/results/validation_exclude_cities")   # 输出目录

OLD_TARGET_COL = 'avg_CDS'
NEW_TARGET_COL = 'avg_CDS_old'

# 要从新城市中排除的城市（索引名）
EXCLUDE_NEW_CITIES = ['Syracuse_NY', 'Milwaukee']

# ================== 数据读取 ==================
old_df = pd.read_csv(OLD_HEALTH_CSV, index_col=0)
new_df = pd.read_csv(NEW_HEALTH_CSV, index_col=0)

old_df = old_df.dropna(subset=[OLD_TARGET_COL])
new_df = new_df.dropna(subset=[NEW_TARGET_COL])

# 从新数据中排除指定城市
new_df_excluded = new_df[~new_df.index.isin(EXCLUDE_NEW_CITIES)]
print(f"原始新城市: {len(new_df)} 个，排除 {EXCLUDE_NEW_CITIES} 后剩余 {len(new_df_excluded)} 个")

# 候选特征
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

# 辅助函数：格式化数值，处理 NaN
def fmt_num(v, fmt="{:.4f}"):
    return fmt.format(v) if not pd.isna(v) else "NaN"

for feature in candidate_features:
    print(f"\n{'='*60}")
    print(f"处理特征: {feature}")

    old_feat = old_df[[feature, OLD_TARGET_COL]].dropna()
    new_feat = new_df_excluded[[feature, NEW_TARGET_COL]].dropna()

    if len(old_feat) < 5 or len(new_feat) < 1:
        print(f"  跳过：样本数不足 (旧:{len(old_feat)}, 新:{len(new_feat)})")
        continue

    # ---- 回归（使用全部旧数据） ----
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

    # ---- 预测新城市（排除后） ----
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

    # ---- 外部验证指标 ----
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    q2_ext = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # 校准曲线：实际 = a + b * 预测（仅当样本数≥3且预测值有变异时计算）
    if len(y_pred) >= 3 and np.std(y_pred) > 1e-6:
        X_cal = sm.add_constant(y_pred)
        try:
            cal_model = sm.OLS(y_true, X_cal).fit()
            calib_slope = cal_model.params[y_pred.name] if hasattr(y_pred, 'name') else cal_model.params.iloc[1]
            calib_intercept = cal_model.params['const']
        except:
            calib_slope = np.nan
            calib_intercept = np.nan
    else:
        calib_slope = np.nan
        calib_intercept = np.nan

    gen_gap = r2 - q2_ext if not np.isnan(q2_ext) else np.nan
    bias = np.mean(y_pred - y_true)

    print(f"  旧 R²={r2:.4f}, p={p_value:.4f} | 新 RMSE={rmse:.4f}, r={corr:.4f}")
    print(f"  新 Q²={q2_ext:.4f}, 校准斜率={calib_slope:.3f}, 偏差={bias:.4f}, 泛化差距={gen_gap:.4f}")

    # ---- 保存结果（使用 utf-8 编码） ----
    feature_dir = OUTPUT_ROOT / feature
    feature_dir.mkdir(parents=True, exist_ok=True)

    with open(feature_dir / "regression_summary.txt", "w", encoding='utf-8') as f:
        f.write(model.summary().as_text())
        f.write(f"\n\nRegression equation: {OLD_TARGET_COL} = {slope:.6f} * {feature} + {intercept:.6f}\n")
        f.write(f"Excluded new cities: {EXCLUDE_NEW_CITIES}\n")
        f.write(f"RMSE on remaining new cities: {rmse:.4f}\n")
        f.write(f"MAE on remaining new cities:  {mae:.4f}\n")
        f.write(f"Pearson correlation: {corr:.4f}\n")
        f.write(f"Q^2 (external R^2): {fmt_num(q2_ext)}\n")
        f.write(f"Calibration slope: {fmt_num(calib_slope)} (ideal=1)\n")
        f.write(f"Calibration intercept: {fmt_num(calib_intercept)} (ideal=0)\n")
        f.write(f"Bias: {fmt_num(bias)}\n")
        f.write(f"Generalization gap (Train R^2 - Q^2): {fmt_num(gen_gap)}\n")

    comparison = pd.DataFrame({
        'city': new_feat.index,
        'actual_CDS': y_true,
        'predicted_CDS': y_pred,
        'residual': y_true - y_pred
    })
    comparison.to_csv(feature_dir / "prediction_results.csv", index=False, encoding='utf-8-sig')

    # ---- 绘图（添加统计信息框） ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_old, y_old, color='blue', s=60, alpha=0.7, label='Old cities (actual)')
    x_min = min(X_old.min().iloc[0], X_new.min().iloc[0])
    x_max = max(X_old.max().iloc[0], X_new.max().iloc[0])
    x_range = np.linspace(x_min, x_max, 100)
    y_range = intercept + slope * x_range
    ax.plot(x_range, y_range, 'r-', linewidth=2, label='Regression line')
    ax.scatter(X_new, y_true, color='green', s=80, marker='o', label='Remaining new cities (actual)')
    ax.scatter(X_new, y_pred, color='orange', s=150, marker='*', label='Remaining new cities (predicted)')
    for i in range(len(X_new)):
        x_val = X_new.iloc[i, 0]
        ax.plot([x_val, x_val], [y_pred.iloc[i], y_true.iloc[i]], 'k--', alpha=0.4, linewidth=1)

    ax.set_xlabel(feature)
    ax.set_ylabel('Average Transfer Loss')
    ax.set_title(f'Cross-domain Validation (Excluded new cities: {EXCLUDE_NEW_CITIES})\nFeature = {feature}')
    ax.legend()
    ax.grid(True)

    # 统计信息文本框（使用纯文本避免特殊字符问题）
    stats_text = (
        f"Train R^2 = {fmt_num(r2, fmt='{:.3f}')}\n"
        f"p = {fmt_num(p_value, fmt='{:.3f}')}\n"
        f"Q^2 (ext) = {fmt_num(q2_ext, fmt='{:.3f}')}\n"
        f"Pearson r = {fmt_num(corr, fmt='{:.3f}')}\n"
        f"RMSE = {fmt_num(rmse, fmt='{:.3f}')}\n"
        f"Calib slope = {fmt_num(calib_slope, fmt='{:.3f}')} (ideal=1)\n"
        f"Bias = {fmt_num(bias, fmt='{:.3f}')}\n"
        f"Gen Gap = {fmt_num(gen_gap, fmt='{:.3f}')} (越小越好)"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
        'New_Q2': q2_ext,
        'Calib_Slope': calib_slope,
        'Calib_Intercept': calib_intercept,
        'Bias': bias,
        'Gen_Gap': gen_gap,
        'Old_samples': len(old_feat),
        'Remaining_new_samples': len(new_feat)
    })

# ================== 汇总表（按外部 Q^2 排序） ==================
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('New_Q2', ascending=False, na_position='last')
    summary_df.to_csv(OUTPUT_ROOT / "summary_all_features.csv", index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print("汇总结果（按外部 Q^2 降序）：")
    print(summary_df[['Feature', 'Old_R2', 'Old_p_value', 'New_Q2', 'New_Pearson_r', 'New_RMSE', 'Gen_Gap']].to_string(index=False))
    print(f"\n汇总表保存至: {OUTPUT_ROOT / 'summary_all_features.csv'}")
else:
    print("没有成功处理任何特征。")

print(f"\n所有结果已保存至根目录: {OUTPUT_ROOT}")