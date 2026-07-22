# scripts/figures

## Purpose

Paper/report figure-generation scripts.

## Contents

- Subdirectories: 1
- Files: 1
- Key subdirectories: `paper_figure4_and_cross_config/`
- Key files: `plot_domain_shift_figures.py`

## Code

- `plot_domain_shift_figures.py`: 高水平论文风格可视化（修正版） 直接从原始矩阵文件读取数据，不依赖中间缓存。 输出：paperfigure/ 目录下的独立图表（英文标题、无子图）

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
