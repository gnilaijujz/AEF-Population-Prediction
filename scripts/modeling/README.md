# scripts/modeling

## Purpose

Main model training, regression, and cross-city transfer experiment scripts.

## Contents

- Subdirectories: 0
- Files: 6
- Key files: `GNN_gis_regression.py`, `GNN_regression.py`, `GNN_transfer_GIS.py`, `GNN_transfer_experiments_calibrated.py`, `MLP_regression.py`, `MLP_transfer.py`

## Code

- `GNN_gis_regression.py`: Python script in this workflow.
- `GNN_regression.py`: Python script in this workflow.
- `GNN_transfer_GIS.py`: GNN_transfer.py  Load pre-trained GraphSAGE models (saved by GNN_regression.py) and evaluate them on all target cities. No training is performed.  Now automatically reads the targe
- `GNN_transfer_experiments_calibrated.py`: GNN_transfer_experiments.py (with mean calibration) 支持 --calibrate-mean 选项，用于诊断均值偏移。
- `MLP_regression.py`: Python script in this workflow.
- `MLP_transfer.py`: MLP_transfer_experiments_calibrated.py 支持 MLP 模型的迁移学习诊断，包含均值校准选项。

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
