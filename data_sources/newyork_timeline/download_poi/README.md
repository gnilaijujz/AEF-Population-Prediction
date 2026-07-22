# data_sources/newyork_timeline/download_poi

## Purpose

New York timeline data and download/post-processing scripts retained for future time-dimension expansion.

## Contents

- Subdirectories: 0
- Files: 2
- Key files: `poi_features_newyork_2021.csv`, `poi_newyork.py`

## Code

- `poi_newyork.py`: 纽约 MSA 单年份 POI 密度（ohsome）—— 改写自 中期前/poi_density_15msa.py 只跑纽约那 4941 个 tract（几何+陆地面积来自 data_gis_newyork 的 shapefile）。 方法：POST /elements/count/groupBy/boundary → 每 tract 计数 ÷ 陆地面积(AL

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
