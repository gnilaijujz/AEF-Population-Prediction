#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN_transfer.py

Load pre-trained GraphSAGE models (saved by GNN_regression.py) and evaluate
them on all target cities. No training is performed.

Now automatically reads the target_transform used during training from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import GNN_gis_regression as gnn


# ---------------------------- 配置 ---------------------------------
DEFAULT_gis_ROOT = Path(r"data_sources/processed_gis")
DEFAULT_POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
DEFAULT_PRETRAINED_DIR = Path("model_data/pretrained_models/GNN/pretrained_gis")
DEFAULT_OUT_DIR = Path("results/transfer_results/gis")



# ---------------------------- 辅助函数 -----------------------------
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def discover_cities(gis_root: Path) -> List[str]:
    cities = []
    for city_dir in sorted(p for p in gis_root.iterdir() if p.is_dir()):
        has_gis = any(city_dir.glob("gis_*_b*_2020.csv"))
        has_shp = any(city_dir.glob("*.shp"))
        if has_gis and has_shp:
            cities.append(city_dir.name)
    if not cities:
        raise ValueError(f"No city folders with both gis CSV files and shapefile found under {gis_root}")
    return cities


def load_city(gis_root: Path, city: str, pop_csv: Path):
    gis_dir = gnn.resolve_gis_dir(str(gis_root), city, None)
    shp_path = gnn.resolve_shp_path(gis_dir, None)
    gdf, feature_cols = gnn.load_and_merge_data(gis_dir, pop_csv, shp_path)
    edge_index = gnn.build_edge_index(gdf)
    return gdf, feature_cols, edge_index


class GraphSAGERegressor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)

def scan_model_files(pretrained_dir: Path) -> Dict[str, Path]:
    model_files = {}
    suffix = "_GraphSAGE_model.pt"
    for pt_path in pretrained_dir.glob("*_model.pt"):
        name = pt_path.name
        if name.endswith(suffix):
            city = name[:-len(suffix)]
        else:
            city = name.split("_")[0]
        model_files[city] = pt_path
    return model_files


def load_pretrained_model(
    city: str,
    model_path: Path,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
    allow_missing_scaler: bool = False,
    target_gdf: Optional[gpd.GeoDataFrame] = None,
    target_feature_cols: Optional[List[str]] = None,
):
    """
    加载单个模型。返回:
        model, x_scaler, y_mean, y_std, feature_cols, target_transform, metrics
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    required_keys = ['model_state_dict', 'feature_cols', 'target_mean', 'target_std']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint for {city} missing key '{key}'. Please retrain with scaler saved.")

    in_channels = len(checkpoint['feature_cols'])
    model = GraphSAGERegressor(in_channels, hidden_channels, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_mean = checkpoint['target_mean']
    y_std = checkpoint['target_std']
    feature_cols = checkpoint['feature_cols']
    metrics = checkpoint.get('metrics', {})
    # 读取变换模式，若不存在则默认为 'raw'（兼容旧模型）
    target_transform = checkpoint.get('target_transform', 'raw')

    # 处理 scaler
    x_scaler = checkpoint.get('scaler')
    if x_scaler is None:
        if allow_missing_scaler and target_gdf is not None and target_feature_cols is not None:
            print(f"⚠️  Warning: {city} model missing scaler. Fitting a new scaler from target city data.")
            x_raw = target_gdf[feature_cols].to_numpy(dtype=np.float32)
            x_scaler = StandardScaler()
            x_scaler.fit(x_raw)
        else:
            raise ValueError(
                f"Model for {city} has no 'scaler'. Please retrain with scaler saved, "
                "or use --allow_missing_scaler to force fitting from target data (not recommended)."
            )
    return model, x_scaler, y_mean, y_std, feature_cols, target_transform, metrics


def prepare_target_data(gdf, feature_cols, edge_index, x_scaler, y_mean, y_std, target_transform, device):
    missing = [col for col in feature_cols if col not in gdf.columns]
    if missing:
        raise ValueError(f"Target city missing features: {missing}")
    x_raw = gdf[feature_cols].to_numpy(dtype=np.float32)
    x = torch.tensor(x_scaler.transform(x_raw), dtype=torch.float32)
    y_density = gdf["population_density"].to_numpy(dtype=np.float32)
    # 应用与训练相同的变换
    y_model = gnn.transform_target(y_density, target_transform)
    y = torch.tensor((y_model - y_mean) / y_std, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    return data, y_density


def save_pair_outputs(out_dir: Path, source: str, target: str, gdf, y_true, y_pred):
    pair_dir = out_dir / "pair_predictions" / f"{safe_name(source)}__to__{safe_name(target)}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_name(source)}__to__{safe_name(target)}_tract_pred"

    train_mask = torch.zeros(len(gdf), dtype=torch.bool)
    val_mask = torch.zeros(len(gdf), dtype=torch.bool)
    test_mask = torch.ones(len(gdf), dtype=torch.bool)
    shp_path = gnn.save_prediction_shapefile(gdf, train_mask, val_mask, test_mask, y_true, y_pred, pair_dir, stem)

    csv_cols = [col for col in ["GEO_ID", "NAME", "MSA_NAME", "gis_SOURCE", "population", "tract_area_sqkm"] if col in gdf.columns]
    output = gdf[csv_cols].copy()
    output["source_city"] = source
    output["target_city"] = target
    output["split"] = "test"
    output["population_density_true"] = y_true
    output["population_density_pred"] = y_pred
    output["density_residual"] = output["population_density_pred"] - output["population_density_true"]
    output["abs_error"] = output["density_residual"].abs()
    csv_path = pair_dir / f"{safe_name(source)}__to__{safe_name(target)}_predictions.csv"
    output.to_csv(csv_path, index=False)
    return csv_path, shp_path


# ---------------------------- 主流程 -----------------------------
def run_transfer(args: argparse.Namespace) -> None:
    gis_root = Path(args.gis_root)
    pop_csv = Path(args.pop_csv)
    pretrained_dir = Path(args.pretrained_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cities = discover_cities(gis_root)
    source_cities = args.source_cities or args.cities or all_cities
    target_cities = args.target_cities or args.cities or all_cities

    available_models = scan_model_files(pretrained_dir)
    print(f"Found model files: {list(available_models.keys())}")

    valid_sources = [src for src in source_cities if src in available_models]
    missing_sources = [src for src in source_cities if src not in available_models]
    if missing_sources:
        print(f"Warning: No model file found for sources: {missing_sources}. They will be skipped.")
    if not valid_sources:
        print("No valid source models. Exiting.")
        return

    print(f"Source cities with models: {valid_sources}")
    print(f"Target cities: {target_cities}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    city_cache = {}
    city_errors = {}

    def get_city(city: str):
        if city in city_errors:
            return None
        if city not in city_cache:
            print(f"\n--- Load city: {city} ---")
            try:
                city_cache[city] = load_city(gis_root, city, pop_csv)
            except Exception as exc:
                city_errors[city] = str(exc)
                print(f"Skip city {city}: {exc}")
                return None
        return city_cache[city]

    # 加载源模型
    source_models = {}
    for src in valid_sources:
        try:
            model, x_scaler, y_mean, y_std, feat_cols, target_transform, _ = load_pretrained_model(
                src,
                available_models[src],
                args.hidden_channels,
                args.dropout,
                device,
                allow_missing_scaler=args.allow_missing_scaler,
            )
            source_models[src] = {
                'model': model,
                'x_scaler': x_scaler,
                'y_mean': y_mean,
                'y_std': y_std,
                'feature_cols': feat_cols,
                'target_transform': target_transform,   # 存储变换模式
            }
        except Exception as e:
            print(f"Failed to load model for {src}: {e}")
            city_errors[src] = str(e)

    if not source_models:
        print("No source models loaded. Exiting.")
        return

    # 指标存储
    metrics_rows = []
    matrix_r2 = pd.DataFrame(index=source_models.keys(), columns=target_cities, dtype=float)
    matrix_rmse = pd.DataFrame(index=source_models.keys(), columns=target_cities, dtype=float)
    matrix_mae = pd.DataFrame(index=source_models.keys(), columns=target_cities, dtype=float)

    for source, src_bundle in source_models.items():
        print(f"\n=== Source: {source} ===")
        model = src_bundle['model']
        x_scaler = src_bundle['x_scaler']
        y_mean = src_bundle['y_mean']
        y_std = src_bundle['y_std']
        src_feature_cols = src_bundle['feature_cols']
        target_transform = src_bundle['target_transform']   # 模型保存的变换

        # 如果用户命令行指定了--target-transform，我们可以选择警告或覆盖
        # 这里我们强制使用模型保存的变换，但允许用户用--target-transform覆盖（需谨慎）
        # 设计：若命令行指定且与模型不一致，发出警告并采用模型保存的
        if args.target_transform and args.target_transform != target_transform:
            print(f"⚠️  Command-line target_transform ({args.target_transform}) differs from model's ({target_transform}). Using model's transform.")
        # 实际使用的变换为 target_transform

        if x_scaler is None and args.allow_missing_scaler:
            print(f"⚠️  scaler for {source} is missing. Will fit a new scaler on each target city (results may be unreliable).")

        for target in target_cities:
            print(f"\nTransfer: {source} -> {target}")
            target_loaded = get_city(target)
            if target_loaded is None:
                metrics_rows.append({"source_city": source, "target_city": target, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_percent": np.nan, "error": city_errors.get(target, "target load failed")})
                print(f"{source} -> {target}: skipped because target city failed to load.")
                continue
            target_gdf, target_feature_cols, target_edge_index = target_loaded
            if target_feature_cols != src_feature_cols:
                err = f"Feature columns differ between {source} and {target}."
                metrics_rows.append({"source_city": source, "target_city": target, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_percent": np.nan, "error": err})
                print(f"{source} -> {target}: skipped. {err}")
                continue

            if x_scaler is None and args.allow_missing_scaler:
                print(f"Fitting scaler on target city {target} (inconsistent with training)")
                x_raw = target_gdf[src_feature_cols].to_numpy(dtype=np.float32)
                temp_scaler = StandardScaler()
                temp_scaler.fit(x_raw)
                current_scaler = temp_scaler
            elif x_scaler is None:
                err = "Scaler is missing and --allow_missing_scaler not set."
                metrics_rows.append({"source_city": source, "target_city": target, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_percent": np.nan, "error": err})
                print(err)
                continue
            else:
                current_scaler = x_scaler

            try:
                target_data, y_true = prepare_target_data(
                    target_gdf, src_feature_cols, target_edge_index,
                    current_scaler, y_mean, y_std, target_transform, device
                )
            except Exception as e:
                err = f"Failed to prepare target data: {e}"
                metrics_rows.append({"source_city": source, "target_city": target, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape_percent": np.nan, "error": err})
                print(err)
                continue

            y_pred = gnn.predict_density(model, target_data, y_mean, y_std, target_transform)
            metric_mask = np.ones(len(target_gdf), dtype=bool)
            # 原代码：
            # metrics = gnn.compute_metrics(y_true[metric_mask], y_pred[metric_mask])

            # 新代码：
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            if not np.all(valid):
                print(f"Warning: {np.sum(~valid)} invalid values found, excluding them.")
            metrics = gnn.compute_metrics(y_true[valid], y_pred[valid])

            metrics_rows.append({"source_city": source, "target_city": target, **metrics, "error": ""})
            matrix_r2.loc[source, target] = metrics["r2"]
            matrix_rmse.loc[source, target] = metrics["rmse"]
            matrix_mae.loc[source, target] = metrics["mae"]
            print(f"{source} -> {target}: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")

            if args.save_pair_outputs:
                csv_path, shp_path = save_pair_outputs(out_dir, source, target, target_gdf, y_true, y_pred)
                print(f"Saved pair CSV: {csv_path}")
                print(f"Saved pair SHP: {shp_path}")

    # 保存结果
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "transfer_metrics_long.csv", index=False)
    matrix_r2.to_csv(out_dir / "transfer_matrix_r2.csv")
    matrix_rmse.to_csv(out_dir / "transfer_matrix_rmse.csv")
    matrix_mae.to_csv(out_dir / "transfer_matrix_mae.csv")
    if city_errors:
        pd.DataFrame([{"city": city, "error": error} for city, error in city_errors.items()]).to_csv(out_dir / "transfer_city_errors.csv", index=False)

    with (out_dir / "transfer_run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nAll transfer results saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer evaluation using pre-trained GraphSAGE models.")
    parser.add_argument("--gis-root", default=str(DEFAULT_gis_ROOT))
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV))
    parser.add_argument("--pretrained-dir", default=str(DEFAULT_PRETRAINED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--source-cities", nargs="+", default=None)
    parser.add_argument("--target-cities", nargs="+", default=None)
    parser.add_argument("--save-pair-outputs", action="store_true")
    parser.add_argument("--target-transform", choices=["log1p", "raw"], default="log1p",
                        help="Override the transform used by the model (not recommended). If not set, uses model's saved transform.")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--allow-missing-scaler", action="store_true", help="Allow loading models without scaler (fit from target data, risky).")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_transfer(args)
