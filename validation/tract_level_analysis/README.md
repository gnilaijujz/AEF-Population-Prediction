# validation/tract_level_analysis

## Purpose

External validation workflow directory.

## Contents

- Subdirectories: 0
- Files: 2
- Key files: `postprocess_gnn_tract_predictions.py`, `regression.py`

## Code

- `postprocess_gnn_tract_predictions.py`: Post-process GNN source-target tract prediction shapefiles.  This script turns many pair-level prediction shapefiles, such as NewYork_to_Chicago_gnn.shp, into target-city diagnosti
- `regression.py`: Python script in this workflow.

## Path Notes

- Local project paths are relative to the repository root.
- Generated outputs should stay under `results/` or `validation/results/`; reusable inputs should stay under `data_sources/` or `model_data/`.
