# validation/ranking_analysis

## Purpose

External validation workflow directory.

## Contents

- Subdirectories: 0
- Files: 4
- Key files: `plot_ranking.py`, `ranking_outer.py`, `ranking_outer_iqr.py`, `ranking_regression.py`

## Code

- `plot_ranking.py`: Python script in this workflow.
- `ranking_outer.py`: 多特征外部验证（支持排除特定新城市） 对每个纯拓扑特征，从旧城市拟合方程 → 应用于新城市预测排名 评估各特征的跨域泛化能力，并输出每个特征的完整参数 同时为每个特征生成排名对比图（棒棒糖图）
- `ranking_outer_iqr.py`: 多特征外部验证（含 IQR 筛选）：对旧城市特征进行离群值剔除后再拟合方程 目的：获得更稳健的旧城市斜率，避免单个离群点主导回归方向 支持排除特定新城市，并为每个特征生成排名对比图
- `ranking_regression.py`: 基于 CDS 矩阵的 Ranking 回归分析 因变量：CDS (综合迁移损失，值越大表示迁移性能越差) 自变量：源域嵌入健康度特征 (effective_rank, isotropy, n_nodes, ...) 方法：留一法（LOOCV）预测排名，Spearman 秩相关评估排序能力

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
