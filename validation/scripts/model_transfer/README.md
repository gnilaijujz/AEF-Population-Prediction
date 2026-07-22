# validation/scripts/model_transfer

## Purpose

Validation-stage script snapshots.

## Contents

- Subdirectories: 0
- Files: 8
- Key files: `GNN_gis_regression.py`, `GNN_regression.py`, `GNN_transfer_GIS.py`, `GNN_transfer_experiments_calibrated.py`, `MLP_regression.py`, `MLP_transfer.py`, `embedding_health_deep_analysis.py`, `old_deep.py`

## Code

- `GNN_gis_regression.py`: Python script in this workflow.
- `GNN_regression.py`: Python script in this workflow.
- `GNN_transfer_GIS.py`: GNN_transfer.py  Load pre-trained GraphSAGE models (saved by GNN_regression.py) and evaluate them on all target cities. No training is performed.  Now automatically reads the targe
- `GNN_transfer_experiments_calibrated.py`: GNN_transfer_experiments.py (with mean calibration) 支持 --calibrate-mean 选项，用于诊断均值偏移。
- `MLP_regression.py`: Python script in this workflow.
- `MLP_transfer.py`: MLP_transfer_experiments_calibrated.py 支持 MLP 模型的迁移学习诊断，包含均值校准选项。
- `embedding_health_deep_analysis.py`: Extra figures: 1. Custom correlation heatmap (exclude NYC/Duluth ONLY for neighbor_consistency) 2. Boxplot of city size (natural breaks) vs. avg_CDS
- `old_deep.py`: 嵌入健康度深度分析（方案3.2） 计算源域嵌入空间的五个结构特征： 1. 有效秩（Effective Rank） 2. 各向同性（Isotropy / Condition Number） 3. 类间/类内距离比（Inter/Intra Class Distance Ratio） 4. 邻居一致性（Neighbor Label Consistency） 5. 

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
