"""
Post-process GNN source-target tract prediction shapefiles.

This script turns many pair-level prediction shapefiles, such as
NewYork_to_Chicago_gnn.shp, into target-city diagnostics:

1. One aggregated shapefile per target city:
   mean prediction, mean residual, mean absolute error, underestimation rate.
2. Moran's I summary for spatial clustering of errors.
3. AEF KMeans cluster labels and cluster-level error summaries.
4. Simple publication-friendly diagnostic figures.

Default inputs are set for the current 3S project. Outputs default to the
current Codex workspace so this script can run without writing outside the project workspace.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Some local conda/pyproj installs cannot discover proj.db when a script is
# launched outside the activated shell. Set it before importing geopandas.
_ENV_PREFIX = Path(sys.prefix)
_PROJ_CANDIDATES = [
    _ENV_PREFIX / "Library" / "share" / "proj",
    _ENV_PREFIX / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj",
]
for _proj_path in _PROJ_CANDIDATES:
    if (_proj_path / "proj.db").exists():
        os.environ["PROJ_LIB"] = str(_proj_path)
        os.environ["PROJ_DATA"] = str(_proj_path)
        try:
            import pyproj

            pyproj.datadir.set_data_dir(str(_proj_path))
        except Exception:
            pass
        break

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_PRED_DIR = Path(r"validation/data/tract_predictions_gnn")
DEFAULT_AEF_DIR = Path(r"model_data/aef_root/clean_aef_shapefiles")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "gnn_tract_transfer_postprocess_outputs"

CITY_ORDER = [
    "Albany_S",
    "Atlanta",
    "Baton",
    "Bridgeport",
    "Chicago",
    "Duluth",
    "Fort_wayne",
    "Hartford",
    "Jackson_MS",
    "Knoxville",
    "Lansing_east",
    "Modesto",
    "Montgomery",
    "Oklahoma",
    "newyork",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--aef-dir", type=Path, default=DEFAULT_AEF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--include-self", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_geoid(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d{11})$", text)
    return match.group(1) if match else text


def city_sort_key(city: str) -> Tuple[int, str]:
    return (CITY_ORDER.index(city), city) if city in CITY_ORDER else (999, city)


def find_geoid_column(df: pd.DataFrame) -> str:
    candidates = [
        "TRACT_ID",
        "GEOID",
        "GEOID20",
        "GEO_ID",
        "geoid",
        "tract_id",
        "cb_2020__3",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        lowered = col.lower()
        if "geoid" in lowered or "tract" in lowered:
            return col

    raise KeyError(f"Cannot find a tract GEOID column. Columns: {list(df.columns)}")


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise KeyError(f"Cannot find {label} column. Columns: {list(df.columns)}")


def parse_pair_path(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if stem.endswith("_gnn"):
        stem = stem[: -len("_gnn")]
    if "_to_" not in stem:
        raise ValueError(f"Cannot parse source-target pair from {path.name}")
    source, target = stem.split("_to_", 1)
    return source, target


def list_prediction_shps(pred_dir: Path) -> List[Path]:
    paths = sorted(pred_dir.glob("*_to_*_gnn.shp"))
    if not paths:
        raise FileNotFoundError(f"No *_to_*_gnn.shp files found in {pred_dir}")
    return paths


def read_pair_table(path: Path) -> Tuple[str, str, pd.DataFrame, gpd.GeoDataFrame]:
    source, target = parse_pair_path(path)
    gdf = gpd.read_file(path)
    geoid_col = find_geoid_column(gdf)
    true_col = find_first_column(gdf, ["true_den", "true_dens", "den_true"], "true density")
    pred_col = find_first_column(gdf, ["pred_den", "pred_dens", "den_pred"], "predicted density")
    resid_col = find_first_column(gdf, ["resid", "residual"], "residual")
    abs_col = find_first_column(gdf, ["abs_err", "abs_error", "abs_er"], "absolute error")

    table = pd.DataFrame(
        {
            "target_city": target,
            "source_city": source,
            "geoid": gdf[geoid_col].map(clean_geoid),
            "true_den": pd.to_numeric(gdf[true_col], errors="coerce"),
            "pred_den": pd.to_numeric(gdf[pred_col], errors="coerce"),
            "resid": pd.to_numeric(gdf[resid_col], errors="coerce"),
            "abs_err": pd.to_numeric(gdf[abs_col], errors="coerce"),
        }
    )
    geom = gdf[[geoid_col, "geometry"]].copy()
    geom["geoid"] = geom[geoid_col].map(clean_geoid)
    geom = geom[["geoid", "geometry"]].drop_duplicates("geoid")
    return source, target, table, geom


def aggregate_target_errors(
    shp_paths: List[Path], include_self: bool
) -> Tuple[Dict[str, gpd.GeoDataFrame], pd.DataFrame]:
    rows_by_target: Dict[str, List[pd.DataFrame]] = {}
    geom_by_target: Dict[str, gpd.GeoDataFrame] = {}

    for path in shp_paths:
        source, target, table, geom = read_pair_table(path)
        if source == target and not include_self:
            continue
        rows_by_target.setdefault(target, []).append(table)
        geom_by_target.setdefault(target, geom)

    target_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    all_pair_rows = []

    for target, parts in rows_by_target.items():
        pair_rows = pd.concat(parts, ignore_index=True)
        pair_rows = pair_rows.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["true_den", "pred_den", "resid", "abs_err"]
        )
        all_pair_rows.append(pair_rows)

        grouped = (
            pair_rows.groupby("geoid", as_index=False)
            .agg(
                true_den=("true_den", "mean"),
                pred_mean=("pred_den", "mean"),
                pred_std=("pred_den", "std"),
                res_mean=("resid", "mean"),
                res_std=("resid", "std"),
                abs_mean=("abs_err", "mean"),
                abs_med=("abs_err", "median"),
                under_rt=("resid", lambda s: float(np.mean(np.asarray(s) < 0))),
                src_count=("source_city", "nunique"),
            )
            .fillna({"pred_std": 0.0, "res_std": 0.0})
        )
        target_gdf = geom_by_target[target].merge(grouped, on="geoid", how="inner")
        target_gdf["target"] = target
        target_gdfs[target] = target_gdf

    if not all_pair_rows:
        raise RuntimeError("No pair rows were available after filtering.")
    return target_gdfs, pd.concat(all_pair_rows, ignore_index=True)


def load_aef_features(aef_dir: Path) -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in aef_dir.iterdir() if p.is_dir()], key=lambda p: city_sort_key(p.name)):
        city = city_dir.name
        for csv_path in sorted(city_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            feature_cols = [col for col in df.columns if re.fullmatch(r"A\d{2}", str(col))]
            if len(feature_cols) != 64:
                continue
            geoid_col = find_geoid_column(df)
            part = df[[geoid_col] + feature_cols].copy()
            part["geoid"] = part[geoid_col].map(clean_geoid)
            part["city"] = city
            rows.append(part[["city", "geoid"] + feature_cols])

    if not rows:
        raise FileNotFoundError(f"No AEF csv files with A00-A63 found in {aef_dir}")

    aef = pd.concat(rows, ignore_index=True)
    aef = aef.drop_duplicates(["city", "geoid"])
    return aef


def add_aef_clusters(
    target_gdfs: Dict[str, gpd.GeoDataFrame],
    aef_dir: Path,
    n_clusters: int,
    out_dir: Path,
) -> Dict[str, gpd.GeoDataFrame]:
    aef = load_aef_features(aef_dir)
    feature_cols = [f"A{i:02d}" for i in range(64)]
    x = aef[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(x_scaled)
    aef_labels = aef[["city", "geoid"]].copy()
    aef_labels["aef_clu"] = labels.astype(int)
    aef_labels.to_csv(out_dir / "aef_cluster_labels.csv", index=False, encoding="utf-8-sig")

    updated = {}
    for city, gdf in target_gdfs.items():
        labels_city = aef_labels[aef_labels["city"].eq(city)][["geoid", "aef_clu"]]
        updated[city] = gdf.merge(labels_city, on="geoid", how="left")
    return updated


def calculate_moran(gdf: gpd.GeoDataFrame, value_col: str) -> Dict[str, float]:
    values = pd.to_numeric(gdf[value_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3 or np.nanstd(values[finite]) == 0:
        return {"moran_i": np.nan, "p_sim": np.nan, "z_sim": np.nan, "n": int(finite.sum())}

    try:
        import esda
        import libpysal

        work = gdf.loc[finite].copy()
        w = libpysal.weights.Queen.from_dataframe(work, use_index=False)
        w.transform = "r"
        moran = esda.Moran(values[finite], w, permutations=999)
        return {
            "moran_i": float(moran.I),
            "p_sim": float(moran.p_sim),
            "z_sim": float(moran.z_sim),
            "n": int(finite.sum()),
        }
    except Exception as exc:
        return {
            "moran_i": np.nan,
            "p_sim": np.nan,
            "z_sim": np.nan,
            "n": int(finite.sum()),
            "error": str(exc),
        }


def save_target_outputs(target_gdfs: Dict[str, gpd.GeoDataFrame], out_dir: Path) -> pd.DataFrame:
    shp_dir = ensure_dir(out_dir / "target_mean_error_shp")
    csv_dir = ensure_dir(out_dir / "target_mean_error_csv")
    moran_rows = []

    for city, gdf in sorted(target_gdfs.items(), key=lambda item: city_sort_key(item[0])):
        csv_path = csv_dir / f"{city}_mean_error.csv"
        shp_path = shp_dir / f"{city}_mean_error.shp"
        gdf.drop(columns="geometry").to_csv(csv_path, index=False, encoding="utf-8-sig")
        gdf.to_file(shp_path, encoding="utf-8")

        for value_col in ["abs_mean", "res_mean", "under_rt"]:
            row = {"target_city": city, "variable": value_col}
            row.update(calculate_moran(gdf, value_col))
            moran_rows.append(row)

    moran_df = pd.DataFrame(moran_rows)
    moran_df.to_csv(out_dir / "moran_summary.csv", index=False, encoding="utf-8-sig")
    return moran_df


def summarize_clusters(target_gdfs: Dict[str, gpd.GeoDataFrame], out_dir: Path) -> pd.DataFrame:
    parts = []
    for city, gdf in target_gdfs.items():
        if "aef_clu" not in gdf.columns:
            continue
        df = gdf.drop(columns="geometry").copy()
        df["target_city"] = city
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    all_targets = pd.concat(parts, ignore_index=True)
    summary = (
        all_targets.dropna(subset=["aef_clu"])
        .groupby(["target_city", "aef_clu"], as_index=False)
        .agg(
            n=("geoid", "count"),
            mean_true_den=("true_den", "mean"),
            mean_pred_den=("pred_mean", "mean"),
            mean_abs_error=("abs_mean", "mean"),
            median_abs_error=("abs_med", "median"),
            mean_residual=("res_mean", "mean"),
            under_rate=("under_rt", "mean"),
        )
    )
    summary["aef_clu"] = summary["aef_clu"].astype(int)
    summary.to_csv(out_dir / "cluster_error_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure_all_formats(fig: plt.Figure, out_base: Path) -> None:
    for suffix in [".png", ".svg", ".pdf", ".tiff"]:
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight")


def plot_diagnostics(moran_df: pd.DataFrame, cluster_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    set_plot_style()

    if not moran_df.empty:
        abs_moran = moran_df[moran_df["variable"].eq("abs_mean")].copy()
        abs_moran = abs_moran.sort_values("moran_i", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = np.where(abs_moran["moran_i"] >= 0, "#d94e45", "#3b73b9")
        ax.barh(abs_moran["target_city"], abs_moran["moran_i"], color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Moran's I of mean absolute error")
        ax.set_ylabel("Target city")
        ax.set_title("Spatial clustering of transfer error")
        save_figure_all_formats(fig, fig_dir / "Fig_Target_Moran_Error")
        plt.close(fig)

    if not cluster_df.empty:
        global_cluster = (
            cluster_df.groupby("aef_clu", as_index=False)
            .agg(mean_abs_error=("mean_abs_error", "mean"), mean_residual=("mean_residual", "mean"), n=("n", "sum"))
            .sort_values("mean_abs_error", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(
            global_cluster["aef_clu"].astype(str),
            global_cluster["mean_abs_error"],
            color="#9aa4b2",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("AEF KMeans cluster")
        ax.set_ylabel("Mean absolute transfer error")
        ax.set_title("Transfer error by AEF cluster")
        save_figure_all_formats(fig, fig_dir / "Fig_AEF_Cluster_Error")
        plt.close(fig)


def save_readme(out_dir: Path, include_self: bool) -> None:
    text = f"""# GNN tract transfer post-processing outputs

Input pair shapefiles are aggregated by target city.

- `target_mean_error_shp/`: one shapefile per target city for GIS visualization.
- `target_mean_error_csv/`: same attributes as CSV.
- `moran_summary.csv`: spatial autocorrelation of error variables.
- `aef_cluster_labels.csv`: AEF KMeans cluster label for each tract.
- `cluster_error_summary.csv`: which AEF clusters fail most.
- `figures/`: quick diagnostic figures in PNG/SVG/PDF/TIFF.

Self pairs included: {include_self}

Key fields in target shapefiles:

- `true_den`: true population density.
- `pred_mean`: mean predicted density across source cities.
- `res_mean`: mean residual, prediction minus truth. Negative means underestimation.
- `abs_mean`: mean absolute error across source cities.
- `under_rt`: fraction of source cities that underestimated the tract.
- `src_count`: number of source cities used.
- `aef_clu`: AEF KMeans cluster label.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    shp_paths = list_prediction_shps(args.pred_dir)
    print(f"Prediction shapefiles found: {len(shp_paths)}")
    print(f"Include self pairs: {args.include_self}")

    target_gdfs, pair_rows = aggregate_target_errors(shp_paths, args.include_self)
    pair_rows.to_csv(out_dir / "all_pair_tract_predictions_long.csv", index=False, encoding="utf-8-sig")
    print(f"Targets aggregated: {len(target_gdfs)}")

    target_gdfs = add_aef_clusters(target_gdfs, args.aef_dir, args.n_clusters, out_dir)
    moran_df = save_target_outputs(target_gdfs, out_dir)
    cluster_df = summarize_clusters(target_gdfs, out_dir)
    plot_diagnostics(moran_df, cluster_df, out_dir)
    save_readme(out_dir, args.include_self)

    print("\nDone.")
    print(f"Output directory: {out_dir}")
    print(f"Target SHPs: {out_dir / 'target_mean_error_shp'}")
    print(f"Moran summary: {out_dir / 'moran_summary.csv'}")
    print(f"Cluster summary: {out_dir / 'cluster_error_summary.csv'}")


if __name__ == "__main__":
    main()
