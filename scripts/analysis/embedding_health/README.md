# scripts/analysis/embedding_health

## Purpose

Analysis, domain-shift, embedding-health, interpretability, and post-processing scripts.

## Contents

- Subdirectories: 0
- Files: 7
- Key files: `GNN_regression.py`, `GNN_transfer_experiments_calibrated.py`, `draw.py`, `embedding_health_analysis.py`, `embedding_health_deep_analysis.py`, `old_deep.py`, `old_embedding.py`

## Code

- `GNN_regression.py`: Python script in this workflow.
- `GNN_transfer_experiments_calibrated.py`: GNN_transfer_experiments.py (with mean calibration) 支持 --calibrate-mean 选项，用于诊断均值偏移。
- `draw.py`: Python script in this workflow.
- `embedding_health_analysis.py`: Embedding Health Analysis and Visualization Based on deep health metrics (embedding_health_deep_metrics.csv). Generates all key figures for "Which source cities are good teachers?"
- `embedding_health_deep_analysis.py`: Extra figures: 1. Custom correlation heatmap (exclude NYC/Duluth ONLY for neighbor_consistency) 2. Boxplot of city size (natural breaks) vs. avg_CDS
- `old_deep.py`: 嵌入健康度深度分析（方案3.2） 计算源域嵌入空间的五个结构特征： 1. 有效秩（Effective Rank） 2. 各向同性（Isotropy / Condition Number） 3. 类间/类内距离比（Inter/Intra Class Distance Ratio） 4. 邻居一致性（Neighbor Label Consistency） 5. 
- `old_embedding.py`: 嵌入健康度分析 从预训练GNN模型中提取每个源城市的嵌入向量（节点级隐藏层表示）， 计算嵌入空间的结构特征（紧凑度、各向异性、有效维度等）， 并回归到该源城市的平均可迁移性（CDS均值或自预测R²）。

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
