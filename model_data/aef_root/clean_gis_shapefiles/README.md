# model_data/aef_root/clean_gis_shapefiles

## Purpose

Model-ready derived shapefile/CSV inputs.

## Contents

- Subdirectories: 15
- Files: 1
- Key subdirectories: `Albany_S/`, `Atlanta/`, `Baton/`, `Bridgeport/`, `Chicago/`, `Duluth/`, `Fort_wayne/`, `Hartford/`, `Jackson_MS/`, `Knoxville/`, `Lansing_east/`, `Modesto/`, ... (3 more)
- Key files: `split_gis_to_msa.py`

## Code

- `split_gis_to_msa.py`: 把标准化后的 9 个 GIS 特征按 MSA 拆分,输出到 processed_gis/<MSA>/, 结构与 processed_aef/<MSA>/ 一致(每个 MSA = 清理过的 shapefile + 一个特征 csv)。  对应关系:   - 每个 MSA 的 tract 列表以 processed_aef/<MSA>/ 里【清理过的 shape

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
