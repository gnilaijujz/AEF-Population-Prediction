# GNN tract transfer post-processing outputs

Input pair shapefiles are aggregated by target city.

- `target_mean_error_shp/`: one shapefile per target city for GIS visualization.
- `target_mean_error_csv/`: same attributes as CSV.
- `moran_summary.csv`: spatial autocorrelation of error variables.
- `aef_cluster_labels.csv`: AEF KMeans cluster label for each tract.
- `cluster_error_summary.csv`: which AEF clusters fail most.
- `figures/`: quick diagnostic figures in PNG/SVG/PDF/TIFF.

Self pairs included: False

Key fields in target shapefiles:

- `true_den`: true population density.
- `pred_mean`: mean predicted density across source cities.
- `res_mean`: mean residual, prediction minus truth. Negative means underestimation.
- `abs_mean`: mean absolute error across source cities.
- `under_rt`: fraction of source cities that underestimated the tract.
- `src_count`: number of source cities used.
- `aef_clu`: AEF KMeans cluster label.
