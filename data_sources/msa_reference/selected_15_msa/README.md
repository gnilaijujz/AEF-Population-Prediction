# data_sources/msa_reference/selected_15_msa

## Purpose

MSA boundary, tract, and population reference data.

## Contents

- Subdirectories: 0
- Files: 5
- Key files: `msa_population_2020.csv`, `msa_population_2020_100.csv`, `select_msa.py`, `selected_15_msa_over100.csv`, `tracts_in_15msa_over100.csv`

## Code

- `select_msa.py`: 从 ACS 2020 tract 人口 (B01003) + list1_2020.xls 计算各 MSA 的总人口与 tract 数量。 仅依赖 xlrd + 标准库。输出 msa_population_2020.csv（全部 MSA，按人口降序）。

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
