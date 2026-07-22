# data_sources/raw_gis_features/socioeconomic/population_structure/selected_15_msa

## Purpose

Downloaded and assembled traditional GIS feature sources used to build model inputs.

## Contents

- Subdirectories: 0
- Files: 2
- Key files: `make_pop_features_15msa.py`, `pop_features_15msa.csv`

## Code

- `make_pop_features_15msa.py`: 从 pop_features.csv (ACS DP05) 计算人口结构特征，只保留15个MSA的tract。 未成年人口占比 = DP05_0019E / DP05_0001E 老年人口占比   = DP05_0024E / DP05_0001E 性别比(男/女×100) = DP05_0002E / DP05_0003E * 100 输出：pop_fea

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
