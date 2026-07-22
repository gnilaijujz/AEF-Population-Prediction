# scripts/analysis/graph_topology

## Purpose

Analysis, domain-shift, embedding-health, interpretability, and post-processing scripts.

## Contents

- Subdirectories: 0
- Files: 1
- Key files: `graph_analysis.py`

## Code

- `graph_analysis.py`: 图拓扑偏移细粒度分析 从 GNN 训练时保存的 edge_index 或从 GeoDataFrame 重建图， 计算源-目标对的图拓扑差异，回归到 CDS。

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
