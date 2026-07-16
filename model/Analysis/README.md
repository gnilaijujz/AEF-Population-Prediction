# Analysis — 迁移矩阵分析与归因

读取 `*_transfer.py` 产出的**迁移矩阵**(`transfer_matrix_r2_*_calibrated.csv`)与**域距离**(`transfer_embedding_distances_*.csv`),做异常剔除、相关性、可视化,并回归量化"**域距离 → 迁移性能**"。

## 脚本

| 脚本 | 作用 | 主要输出 |
|------|------|---------|
| `quantify_domain_shift_regression.py` | 核心:OLS / 二次 / 分源回归,量化 L2 域距离对校准后 R² 的边际效应 | `regression_quantify_metrics_*.csv`、量化/诊断图 |
| `analyze_dist.py` | 计算 4 种距离(L1/L2/Cos/MMD)与校准 R² 的 Pearson & Spearman 相关 | `distance_correlation_results.csv` |
| `post_drop.py` | 对 **null 模型**校准 R² 矩阵做 IQR 异常剔除、汇总、热图/散点 | `null_calibrated_*`(在 `GNN_output/1.3/`) |
| `post_drop_sec.py` | `post_drop` 的推广版,一次跑 `null` 与 `gnn` 两个矩阵 | `transfer_analysis_<mode>_*` |
| `post_process_attribute.py` | 域距离↔校准 R² 的归因(相关 + IQR 前后 + 诊断图) | `attribution_*`(在 `GNN_output/attribute/`) |
| `post_visualize.py` | 真实 GNN 迁移的出版级归因图(聚焦 L2 vs 校准 R²) | `attribute_final_*` |
| `source_visualize.py` | 分源城散点 + 实测-预测对比图 | `Figure_Source_Wise_*.png` |

## 归因结论(摘自《实验方案记录》1.1–1.8)

- **1.1 解码器退化**:Ridge 迁移比 GNN 更差 → 图结构信息对迁移有正向作用,复杂解码器不是负迁移主因。
- **1.2 零模型**:打乱特征后仍出现大负值 → 模型存在**源域均值锚定的输出偏置**。
- **1.3–1.4 均值校准 + IQR**:校准并剔除异常后,真实 GNN 迁移均值由 -0.355 回升到 **+0.286**(剔除 16 对,8.2%);异常高度集中在目标域为 Oklahoma/Jackson_MS/Montgomery、源域为 Duluth/losangeles 的城市对。
- **1.5–1.6 域偏移定量**:源-目标嵌入 L2 距离每增 1 单位,校准 R² 平均下降 **0.019**(p<0.001,R²=0.109;Spearman ρ=-0.334)。此斜率 β₀=-0.0189 作为后续引入 GIS 辅助特征缓解域偏移的量化参照。

## ⚠️ 注意:路径为硬编码

分析脚本里的输入/输出路径是**硬编码相对路径**,且彼此不完全对齐:

- 训练/迁移默认写到 `GNN_output/output_try_log`、`GNN_output/transfer_results_try_log`;
- 但分析脚本读的是 `GNN_output/transfer_results_experiments/`、`transfer_results(1)/`,并写入需要**预先创建**的 `GNN_output/1.3/`、`GNN_output/attribute/`。
- `source_visualize.py` **不能独立运行**:它依赖在同一 Python 会话中先跑 `quantify_domain_shift_regression.py` 后留下的 `df_clean` / `model_clean` / `city_stats` 变量。

复现前请对照各脚本顶部的路径常量,按实际输出目录调整。
