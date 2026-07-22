import pandas as pd
import numpy as np
from pathlib import Path

# 指向无IQR结果根目录
ROOT_DIR = Path("validation/results/validation")

# 收集所有城市的残差
city_residuals = {}  # {city: [残差列表]}

for feature_dir in ROOT_DIR.iterdir():
    if not feature_dir.is_dir():
        continue
    csv_path = feature_dir / "prediction_results.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    # 确保有 city 和 residual 列
    for _, row in df.iterrows():
        city = row['city']
        residual = row['residual']  # 实际 - 预测
        if city not in city_residuals:
            city_residuals[city] = []
        city_residuals[city].append(abs(residual))  # 存储绝对残差

# 计算每个城市的平均绝对残差
avg_abs_residual = {city: np.mean(residuals) for city, residuals in city_residuals.items()}

# 排序
sorted_cities = sorted(avg_abs_residual.items(), key=lambda x: x[1], reverse=True)

# 输出结果
print("预测差异最大的新城市（基于所有特征的平均绝对残差）：")
print("城市\t平均绝对残差")
for city, mar in sorted_cities[:5]:
    print(f"{city}\t{mar:.4f}")

# 同时保存为 CSV
result_df = pd.DataFrame(sorted_cities, columns=['city', 'mean_abs_residual'])
result_df.to_csv(ROOT_DIR / "worst_cities_by_residual.csv", index=False)
print(f"\n结果已保存至 {ROOT_DIR / 'worst_cities_by_residual.csv'}")