# scripts/figures/paper_figure4_and_cross_config

## Purpose

Paper/report figure-generation scripts.

## Contents

- Subdirectories: 0
- Files: 12
- Key files: `4_1.py`, `cross_2.py`, `cross_3.py`, `cross_4.py`, `cross_config_analysis.py`, `dim1_predictive_modeling.py`, `dim2_moderation_effect.py`, `dim3_structure_decomposition.py`, `dim4_robustness_bootstrap.py`, `dim5_anomaly_mining.py`, `domain_shift.py`, `utils_health.py`

## Code

- `4_1.py`: 成果4：嵌入空间特征与可迁移性关系 生成所有指定图表（稳健版，兼容所有异常）
- `cross_2.py`: Cross-Configuration Domain Shift Distance Similarity Analysis Compare L2 distance matrices across different configurations.
- `cross_3.py`: 域偏移距离贡献分解（针对Building配置） 分析AEF和GIS特征在加权欧氏距离中的各自贡献。
- `cross_4.py`: Cross-configuration Self-prediction vs Cross-domain Transferability with regression lines for each configuration.
- `cross_config_analysis.py`: 跨配置城市排名一致性分析 Compare source city ranking (by avg_CDS) across different feature configurations.
- `dim1_predictive_modeling.py`: 成果4 综合分析与可视化（修正自预测得分、中文字体、$R^2$ 渲染） 输出所有图表至 paper_figures/figure4_embedding_health/figure4_panels/
- `dim2_moderation_effect.py`: 维度二：调节效应——邻居一致性是否缓冲了域偏移（L2距离）的负面影响？
- `dim3_structure_decomposition.py`: 维度三：结构分解——将“邻居一致性”拆解为全局平滑度、局部方差、空间自相关 分别回归，找出最关键的子成分 注意：此脚本需要每个城市的图结构和标签（人口密度），需从原始gdf计算。 这里我们用模拟数据演示，实际使用时需替换为真实计算。 为了可运行，我们采用已有指标作为近似代替（实际应用中请替换）。
- `dim4_robustness_bootstrap.py`: 维度四：稳健性检验——通过Bootstrap重采样评估好老师排名的稳定性
- `dim5_anomaly_mining.py`: 维度五：异常挖掘——识别实际迁移性能与预测值偏差较大的城市，进行地理归因
- `domain_shift.py`: 为每个域偏移指标矩阵和 CDSI 矩阵单独绘制热力图。 要求： - 输入目录包含 domain_shift_*.csv 文件（由 domain_shifting.py 生成） - 输出图片保存至同一目录下的 plots/ 子目录 - 每个指标一张图，标题为英文
- `utils_health.py`: 成果4 数据加载与通用工具（修正版：自预测得分从 full_scores_robust.csv 读取熵权法 Score）

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
