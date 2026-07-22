# validation/scripts/transfer_domain

## Purpose

Validation-stage script snapshots.

## Contents

- Subdirectories: 0
- Files: 8
- Key files: `GNN_gis_regression.py`, `GNN_regression.py`, `GNN_transfer_experiments_calibrated.py`, `domain_shift_onlyGIS.py`, `domain_shift_with_gis.py`, `domain_shifting.py`, `label_shift.py`, `transfer_ability.py`

## Code

- `GNN_gis_regression.py`: Python script in this workflow.
- `GNN_regression.py`: Python script in this workflow.
- `GNN_transfer_experiments_calibrated.py`: GNN_transfer_experiments.py (with mean calibration) 支持 --calibrate-mean 选项，用于诊断均值偏移。
- `domain_shift_onlyGIS.py`: 从 GIS 特征计算得到的各距离矩阵，使用熵权法合成综合域偏移指数 (CDSI)。 假设距离矩阵文件名为 gis_{metric}_matrix.csv， 并放置在指定目录中。
- `domain_shift_with_gis.py`: 域偏移指标矩阵构建（手动指定GIS特征列） 基于原始 AEF 和 GIS 特征（npoi_water_distance_density），计算独立的距离矩阵及加权组合距离。
- `domain_shifting.py`: 构建域偏移指标矩阵（含熵权法 CDSI） 从 transfer_embedding_distances_gnn.csv 提取各偏移指标， 计算每对城市之间的偏移量，并通过熵权法合成综合域偏移指数 (CDSI)。 输出：各指标矩阵 CSV 及 CDSI 矩阵 CSV。
- `label_shift.py`: 生成 Label_Shift 矩阵（城市间人口密度均值的绝对差） 用于分层回归中控制人口尺度偏移。
- `transfer_ability.py`: Python script in this workflow.

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
