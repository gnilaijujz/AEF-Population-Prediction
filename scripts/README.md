# Script Organization

The scripts are organized by use case rather than as a Python package. Many files are one-off research scripts with top-level constants, so run them from the project root and verify paths before execution.

## Folders

| Folder | Contents |
| --- | --- |
| `modeling/` | Main GNN/MLP regression and transfer experiment scripts. |
| `analysis/embedding_health/` | Embedding-health metric computation, diagnostics, and older embedding analysis variants. |
| `analysis/transfer_domain/` | Domain-shift and transferability score scripts. |
| `analysis/regression/` | Regression analysis linking transfer performance, domain shift, and GIS/AEF metrics. |
| `analysis/graph_topology/` | Graph topology shift diagnostics. |
| `analysis/interpretability/` | Attribution and residual interpretation scripts. |
| `analysis/others/` | Older exploratory scripts and one-off post-processing code. |
| `figures/` | Figure generation scripts for paper/report outputs. |

## Primary Entry Points

Use these first when rerunning experiments:

```powershell
python scripts/modeling/GNN_regression.py --help
python scripts/modeling/GNN_transfer_experiments_calibrated.py --help
python scripts/modeling/MLP_regression.py --help
python scripts/modeling/MLP_transfer.py --help
```

Some scripts do not provide `--help` because they were written as direct notebooks/scripts with constants at the top. For those, inspect the first 30-60 lines and update the path constants before running.

## Import Notes

Model transfer scripts use same-folder imports such as `import GNN_regression as gnn` and `import MLP_regression as mlp`. Run them directly from their own folder or add that folder to `PYTHONPATH` if importing from elsewhere.

Example:

```powershell
Set-Location scripts/modeling
python GNN_transfer_experiments_calibrated.py --help
```

When a script reads `data_sources/...`, `model_data/...`, `results/...`, or `validation/...`, run it from the repository root unless the script has been updated to resolve paths relative to `__file__`.

