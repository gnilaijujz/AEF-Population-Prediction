# scripts/analysis/others

## Purpose

Analysis, domain-shift, embedding-health, interpretability, and post-processing scripts.

## Contents

- Subdirectories: 0
- Files: 9
- Key files: `analyze_dist.py`, `diagnostic.py`, `draw.py`, `post_drop.py`, `post_drop_sec.py`, `post_visualize.py`, `quantify_CDS_domain.py`, `quantify_with_transferability.py`, `source_visualize.py`

## Code

- `analyze_dist.py`: Python script in this workflow.
- `diagnostic.py`: diagnostic.py  Diagnostic experiments to identify the source of negative transfer R² values: 1. Global standardization vs. per-MSA standardization. 2. Ridge regression (no graph, n
- `draw.py`: Python script in this workflow.
- `post_drop.py`: Python script in this workflow.
- `post_drop_sec.py`: 后处理脚本：对校准后的 R² 矩阵进行 IQR 异常剔除， 并执行“嵌入差异 vs 迁移性”回归分析，生成论文图表。 可同时处理零模型 (null) 和真实迁移 (gnn)。
- `post_visualize.py`: 最终归因可视化脚本：针对 L2 距离与校准后 R² 的显著相关性， 生成论文级图表，并突出异常城市对。
- `quantify_CDS_domain.py`: 域偏移与迁移性能回归分析（读取矩阵文件） 依赖：CDS_matrix.csv（由 transfer_ability.py 生成）       以及 domain_shift_matrix_entropy.py 生成的所有矩阵文件。
- `quantify_with_transferability.py`: 域偏移量化回归分析（因变量：CDS） 建立 L2 距离 -> CDS（迁移损失）的映射， 包含单变量、多变量控制、非线性检验和残差诊断。 自动处理异常值（IQR），并生成基于正常样本的二次拟合与拐点。
- `source_visualize.py`: Python script in this workflow.

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
