# data_sources/newyork_timeline/download_road

## Purpose

New York timeline data and download/post-processing scripts retained for future time-dimension expansion.

## Contents

- Subdirectories: 2
- Files: 8
- Key subdirectories: `roads_2017/`, `roads_2018/`
- Key files: `road_process.py`, `urls_2017.txt`, `urls_2018.txt`, `urls_2019.txt`, `urls_2020.txt`, `urls_2021.txt`, `urls_all.txt`, `数据来源说明.md`

## Code

- `road_process.py`: road_density_local 一步到位处理脚本 · New York MSA · 时间维度 ================================================================ 做什么（全自动）：   1. 读 2020 纽约 tract 矢量 → 反推涉及的全部 county FIPS   2. 逐年(2

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
