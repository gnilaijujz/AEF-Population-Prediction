# data_sources/newyork_timeline/download_aef

## Purpose

New York timeline data and download/post-processing scripts retained for future time-dimension expansion.

## Contents

- Subdirectories: 0
- Files: 9
- Key files: `aef_export_gee.js`, `aef_newyork_2017.csv`, `aef_newyork_2018.csv`, `aef_newyork_2019.csv`, `aef_newyork_2020.csv`, `aef_newyork_2021.csv`, `aef_postprocess.py`, `merge_aef_2020.py`, `数据来源说明.md`

## Code

- `aef_export_gee.js`: =====================================================================
- `aef_postprocess.py`: 把 GEE 导出的 aef_newyork_YYYY.csv 处理成与 2020 相同结构的一套文件：   - 列改成 TRACT_ID, MSA_NAME, A00..A63   - 按 2020 的 b0..b5 归属切成 6 个文件   - 复制 2020 的 shapefile 附属文件   - 输出到 data_aef_newyork_YYYY/ 
- `merge_aef_2020.py`: Python script in this workflow.

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
