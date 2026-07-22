# scripts/analysis/regression

## Purpose

Analysis, domain-shift, embedding-health, interpretability, and post-processing scripts.

## Contents

- Subdirectories: 0
- Files: 6
- Key files: `GNN_gis_regression.py`, `GNN_transfer_experiments_calibrated.py`, `RE_AN_GIS.py`, `regression_analysis.py`, `regression_vis.py`, `three_regression.py`

## Code

- `GNN_gis_regression.py`: Python script in this workflow.
- `GNN_transfer_experiments_calibrated.py`: GNN_transfer_experiments.py (with mean calibration) 支持 --calibrate-mean 选项，用于诊断均值偏移。
- `RE_AN_GIS.py`: 分层回归：使用 Label Shift 和 GIS Shift 解释迁移性能（R²），控制人口尺度。 - Label Shift：目标变量（人口密度）的 Wasserstein 距离。 - GIS Shift：输入特征（GIS）的分布差异（例如均值 L2 距离）。 - 控制变量：源/目标城市的人口规模（例如人口总数中位数）。
- `regression_analysis.py`: 自动选择最优域偏移指标的回归分析 因变量：CDS（迁移损失） 自变量：从 domain_shift_*_matrix.csv 中自动提取所有可用指标 支持：单变量筛选 + 逐步回归（AIC向前选择）
- `regression_vis.py`: 优化版回归分析（论文级可视化） - 自动选择最优域偏移指标（单变量筛选 + 向前逐步回归） - 分层回归（四变量：Label, AEF, slpoe, Joint） 所有图表：白底、英文、标注 R²、系数、p 值
- `three_regression.py`: 三层偏移分层归因回归（Label尺度 + AEF嵌入 + GIS地理） 完美衔接成果2（域偏移回归）和成果3（归因分析）

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
