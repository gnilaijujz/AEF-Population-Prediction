import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取你刚跑出来的外部验证结果（包含实际排名和预测排名）
# 请修改为你的实际路径
CSV_PATH = r"validation\results\external_apply_all_features\external_apply_summary.csv"
# 如果你对多个特征都保存了结果，这里直接读取即可
df = pd.read_csv(CSV_PATH)

# 展示效果最好的 n_nodes 和效果反转的 isotropy
features_to_plot = ['n_nodes', 'isotropy']

for feat in features_to_plot:
    # 提取该特征对应的预测值（假设你的csv中包含 predicted_CDS_n_nodes 这样的列名）
    # 如果csv中只有 predicted_CDS，则直接使用
    if f'predicted_CDS' in df.columns:
        # 直接计算预测排名
        df['rank_true'] = df['actual_CDS'].rank(method='dense').astype(int)
        df['rank_pred'] = df['predicted_CDS'].rank(method='dense').astype(int)
    else:
        # 如果是旧格式，可能需要手动指定特征
        # 这里模拟你的数据结构
        pass
    
    # 按真实排名排序，便于观察
    df_sorted = df.sort_values('rank_true').reset_index(drop=True)
    cities = df_sorted['source_city']
    true_rank = df_sorted['rank_true']
    pred_rank = df_sorted['rank_pred']

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(cities))

    # 1. 绘制棒棒糖的线（连接预测和实际）
    for i in range(len(cities)):
        ax.plot([true_rank[i], pred_rank[i]], [y_pos[i], y_pos[i]], 
                color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 2. 绘制实际排名（蓝色圆点）
    ax.scatter(true_rank, y_pos, color='steelblue', s=120, label='Actual Rank', zorder=5, edgecolors='black')

    # 3. 绘制预测排名（红色星号）
    ax.scatter(pred_rank, y_pos, color='coral', marker='*', s=200, label='Predicted Rank', zorder=5)

    # 4. 装饰：显示偏差值（在右侧标注高估/低估）
    for i in range(len(cities)):
        diff = pred_rank[i] - true_rank[i]
        # 只标注偏差绝对值大于0.5的
        if abs(diff) > 0.5:
            offset_text = f'+{diff:.0f}' if diff > 0 else f'{diff:.0f}'
            ax.annotate(offset_text, xy=(max(true_rank[i], pred_rank[i]) + 0.2, y_pos[i]), 
                        fontsize=8, color='red' if diff > 0 else 'green', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities)
    ax.set_xlabel('Rank (1 = Best, Lowest CDS Loss)', fontsize=12)
    ax.set_title(f'Ranking Prediction Performance: {feat}\n(Left is Better)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.invert_yaxis()  # 让排名靠前的城市位于顶部

    plt.tight_layout()
    plt.savefig(f'ranking_lollipop_{feat}.png', dpi=300)
    plt.show()