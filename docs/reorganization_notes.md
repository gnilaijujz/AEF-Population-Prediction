# Reorganization Notes

Date: 2026-07-23

This note records the current repository organization after the second cleanup. The repository root is the working root; all local paths in code and documentation should be relative to it.

## Current Top-Level Structure

| Path | Purpose |
| --- | --- |
| `data_sources/` | Source data and curated input data. This includes raw GIS feature downloads, processed AEF/GIS inputs, New York timeline data, validation new-MSA data, and MSA reference data. |
| `model_data/` | Model-ready derived shapefiles/CSV inputs and pretrained GNN/MLP model artifacts. |
| `scripts/` | Main project scripts for modeling, analysis, and figure generation. |
| `results/` | Main experiment outputs and legacy analysis output folders. |
| `validation/` | External validation scripts, validation data, tract-level analysis, ranking analysis, and validation outputs. |
| `paper_figures/` | Consolidated figure outputs used in reports or papers. |
| `docs/` | Documentation and report materials. |
| `archive/` | Legacy snapshots retained for reference rather than primary reruns. |

## Data Source Rename Map

| Old Path | New Path | Notes |
| --- | --- | --- |
| `data_source/` | `data_sources/` | Pluralized because it contains multiple input data groups. |
| `data_source/data_AEF/` | `data_sources/processed_aef/` | Cleaned AEF input data actually used by the project. |
| `data_source/data_GIS/` | `data_sources/processed_gis/` | Cleaned GIS input data actually used by the project. |
| `data_source/data_newyork_timeline/` | `data_sources/newyork_timeline/` | New York time-dimension data for future expansion. |
| `data_source/data_validation_newMSA/` | `data_sources/validation_new_msa/` | New 15-MSA validation input data. |
| `data_source/MSA/` | `data_sources/msa_reference/` | MSA boundary, tract, and population reference data. |
| `data_source/传统GIS特征数据/` | `data_sources/raw_gis_features/` | Downloaded traditional GIS feature sources. |
| `15MSA/` | `selected_15_msa/` | Selected 15-MSA subsets in source-data folders. |
| `15msa/` | `selected_15_msa_features/` | Combined feature tables for the selected 15 MSA set. |
| `原始数据/` | `raw/` | Raw downloaded files under individual feature folders. |
| `各MSA/` | `by_msa/` | Per-MSA files under feature folders. |

## Other Rename Map

| Old Path | New Path | Notes |
| --- | --- | --- |
| `data/` | `model_data/` | Clarifies that this is model-ready derived data, not source downloads. |
| `data/AEF_root/` | `model_data/aef_root/` | Main derived AEF/GIS shapefile tree. |
| `data/pretrain_model/` | `model_data/pretrained_models/` | Pretrained model artifacts. |
| `result/` | `results/` | Main output tree. |
| Legacy misspelled transferability output folder | `results/transferability/` | Fixed historical spelling and normalized result naming. |
| `Validation/` | `validation/` | Lowercase project directory naming. |
| `Validation/code/` | `validation/scripts/` | Validation-stage script snapshots. |
| `Validation/result/` | `validation/results/` | Validation outputs. |
| `Validation/ranking/` | `validation/ranking_analysis/` | Validation ranking scripts. |
| `Validation/tract_code/` | `validation/tract_level_analysis/` | Tract-level validation scripts. |
| `paper_figures/Figure4/` | `paper_figures/figure4/` | Lowercase figure folder. |
| Root PPT/DOCX report files | `docs/report_materials/` | Keeps project root focused on code/data structure. |

## Structural Fixes

`data_sources/processed_aef/Chicago/` and `data_sources/processed_aef/Duluth/` were found nested under `data_sources/processed_aef/Bridgeport/`. They were moved back to the same city-directory level as the other processed AEF inputs.

## Path Cleanup

Local workstation paths were replaced with project-root-relative paths. Examples:

| Old Pattern | New Pattern |
| --- | --- |
| `previous ACS population absolute path` | `data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/...` |
| `previous GIS absolute path` | `data_sources/processed_gis` |
| `legacy AEF model shapefile folder` | `model_data/aef_root/clean_aef_shapefiles` |
| `previous GNN output folder` | `results/legacy_analysis/...` or `results/transfer_results/...` depending on the script. |
| `previous MLP output folder` | `model_data/pretrained_models/mlp` or `results/transfer_results/...` depending on the script. |

Google Earth Engine `projects/.../assets/...` strings remain in export scripts because they are remote GEE asset identifiers, not local paths.

## Current Model/Result Naming

The model-ready data folders now use readable configuration names:

| Category | Current Paths |
| --- | --- |
| Base model inputs | `model_data/aef_root/clean_aef_shapefiles/`, `model_data/aef_root/clean_gis_shapefiles/` |
| AEF plus feature inputs | `model_data/aef_root/aef_plus_building/`, `aef_plus_gis9/`, `aef_plus_impervious/`, `aef_plus_ndvi/`, `aef_plus_nighttime_lights/`, `aef_plus_poi/`, `aef_plus_poi_diversity/`, `aef_plus_roads/`, `aef_plus_slope/`, `aef_plus_water_distance/` |
| Pretrained GNN models | `model_data/pretrained_models/GNN/pretrained_aef/`, `pretrained_gis/`, `selftrain_gnn_plus_gis9/`, `selftrain_gnn_plus_water_distance/`, `selftrain_gnn_plus_nighttime_lights/`, `selftrain_gnn_plus_roads/` |
| Transfer outputs | `results/transfer_results/aef/`, `aef_plus_building/`, `aef_plus_gis9/`, `aef_plus_nighttime_lights/`, `aef_plus_roads/`, `aef_plus_water_distance/`, `gis/` |
| Transferability outputs | `results/transferability/AEF/`, `GIS/`, `aef_plus_building/`, `aef_plus_gis9/`, `nighttime_lights/`, `roads/`, `water_distance/` |
| Domain-shift outputs | `results/domain_shift/aef/`, `gis/`, `aef_plus_building/`, `aef_plus_gis9/`, `aef_plus_nighttime_lights/`, `aef_plus_roads/`, `aef_plus_water_distance/` |

## Duplicate Script Groups

The following duplicate script snapshots still exist. They were kept because they may reflect different experiment stages or validation snapshots.

| Duplicate Group | Copies |
| --- | --- |
| GNN GIS regression | `scripts/modeling/GNN_gis_regression.py`, `scripts/analysis/regression/GNN_gis_regression.py`, `scripts/analysis/transfer_domain/GNN_gis_regression.py`, `validation/scripts/model_transfer/GNN_gis_regression.py`, `validation/scripts/transfer_domain/GNN_gis_regression.py` |
| GNN regression | `scripts/analysis/embedding_health/GNN_regression.py`, `scripts/analysis/transfer_domain/GNN_regression.py`, `validation/scripts/embedding_health/GNN_regression.py`, `validation/scripts/transfer_domain/GNN_regression.py` |
| GNN transfer calibrated | `scripts/analysis/embedding_health/GNN_transfer_experiments_calibrated.py`, `scripts/analysis/transfer_domain/GNN_transfer_experiments_calibrated.py`, `validation/scripts/embedding_health/GNN_transfer_experiments_calibrated.py`, `validation/scripts/transfer_domain/GNN_transfer_experiments_calibrated.py` |
| Domain shift scripts | `scripts/analysis/transfer_domain/domain_shifting.py`, `domain_shift_onlyGIS.py`, `domain_shift_with_gis.py` and matching copies under `validation/scripts/transfer_domain/` |
| MLP scripts | `scripts/modeling/MLP_regression.py`, `scripts/modeling/MLP_transfer.py` and matching copies under `validation/scripts/model_transfer/` |
| Figure helper | `scripts/figures/paper_figure4_and_cross_config/utils_health.py`, `validation/scripts/original_regression/utils_health.py` |

## Recommended Next Cleanup

1. Convert shared model code into importable modules so duplicate GNN/MLP files can be retired.
2. Add command-line arguments to more one-off analysis scripts instead of relying on top-level constants.
3. Create a `requirements.txt`, `environment.yml`, or `pyproject.toml` for Python dependencies.
4. Add smoke tests that verify imports and required input roots.
