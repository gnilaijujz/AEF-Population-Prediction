import pandas as pd
from sklearn.linear_model import LinearRegression
from utils_health import get_health_df

# 1. 加载现有15个城市的原始数据
df = get_health_df('AEF')  # 替换成你的配置名

# 2. 提取原始尺度的 X 和 y
X = df[['n_nodes']].values  # 注意是双括号，保持二维
y = df['avg_CDS'].values

# 3. 用原始数据拟合线性回归
model = LinearRegression()
model.fit(X, y)

# 4. 打印回归函数（这就是你要的数学表达式！）
slope = model.coef_[0]
intercept = model.intercept_
print(f"当前回归函数（原始尺度）：")
print(f"avg_CDS = ({slope:.8f}) × (n_nodes) + ({intercept:.6f})")
print(f"\n其中 n_nodes 为该城市的节点总数（非标准化数值）")