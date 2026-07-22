from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from scipy.stats import pearsonr

def compute_metrics_with_r(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # 转换为 numpy 数组并确保浮点数
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # 检查并过滤 inf/nan
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        print(f"Warning: {np.sum(~mask)} samples have inf/nan. Removing them.")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, 
                    "mape_percent": np.nan, "pearson_r": np.nan}
    
    # 对预测值进行裁剪，防止极端值（人口密度不应超过合理范围）
    y_pred = np.clip(y_pred, 0, 1e6)  # 假设最大人口密度 100万/km²
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = 0.0
    else:
        r, _ = pearsonr(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2),
            "mape_percent": float(mape), "pearson_r": float(r)}

# CHANGED: pyproj must know where proj.db is before geopandas/libpysal are imported.
def configure_proj_data() -> None:
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj",
        Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "share" / "proj",
        Path.home() / ".conda" / "envs" / "3s" / "Library" / "share" / "proj",
        Path.home() / ".conda" / "envs" / "3s" / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj",
    ]
    for candidate in candidates:
        if candidate and (candidate / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(candidate)
            os.environ["PROJ_DATA"] = str(candidate)
            try:
                import pyproj
                pyproj.datadir.set_data_dir(str(candidate))
            except Exception:
                pass
            return

configure_proj_data()

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from libpysal.weights import Queen
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

DEFAULT_aef_ROOT = Path(r"model_data/aef_root/clean_aef_shapefiles")
DEFAULT_POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
DEFAULT_SHP = None
DEFAULT_OUT_DIR = Path(r"model_data\pretrained_models")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_aef_dir(aef_root: str | None, city: str | None, aef_dir: str | None) -> Path:
    if aef_dir:
        resolved = Path(aef_dir)
    else:
        if not city:
            raise ValueError("Either --aef-dir or --city must be provided.")
        resolved = Path(aef_root or DEFAULT_aef_ROOT) / city
    if not resolved.exists():
        raise FileNotFoundError(f"aef directory does not exist: {resolved}")
    return resolved


def resolve_shp_path(aef_dir: Path, shp: str | None) -> Path:
    if shp:
        resolved = Path(shp)
        if not resolved.exists():
            raise FileNotFoundError(f"Shapefile does not exist: {resolved}")
        return resolved

    shp_files = sorted(aef_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in city aef directory: {aef_dir}")

    msa_shps = [p for p in shp_files if "track" in p.stem.lower() or "tract" in p.stem.lower()]
    resolved = msa_shps[0] if msa_shps else shp_files[0]
    print(f"Shapefile auto-detected: {resolved}")
    return resolved


def detect_shp_geoid_col(gdf: gpd.GeoDataFrame) -> str:
    preferred = ["GEO_ID", "GEOID", "TRACT_ID", "cb_2020__3", "ACSDT5Y202"]
    for col in preferred:
        if col in gdf.columns:
            sample = gdf[col].dropna().astype(str).head(20)
            if sample.str.startswith("1400000US").any():
                return col

    for col in gdf.columns:
        if col == "geometry":
            continue
        sample = gdf[col].dropna().astype(str).head(50)
        if sample.str.match(r"^1400000US\d{11}$").any():
            return col

    raise ValueError(
        "Could not find a full ACS tract GEOID column in the shapefile. "
        "Expected values like '1400000US36005009200'."
    )


def load_aef_features(aef_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [f"A{i:02d}" for i in range(64)]
    aef_files = sorted(aef_dir.glob("aef_*_b*_2020.csv"))
    if not aef_files:
        raise ValueError(f"No aef files matching aef_*_b*_2020.csv found in {aef_dir}")

    parts = []
    for aef_file in aef_files:
        aef_part = pd.read_csv(aef_file, dtype={"TRACT_ID": str})
        if "TRACT_ID" not in aef_part.columns:
            raise ValueError(f"{aef_file} must contain a TRACT_ID column.")
        missing_features = [col for col in feature_cols if col not in aef_part.columns]
        if missing_features:
            raise ValueError(f"{aef_file} is missing feature columns: {missing_features}")

        keep_cols = ["TRACT_ID", *feature_cols]
        if "MSA_NAME" in aef_part.columns:
            keep_cols.insert(1, "MSA_NAME")
        aef_part = aef_part[keep_cols].drop_duplicates("TRACT_ID")
        aef_part["aef_SOURCE"] = aef_file.name
        parts.append(aef_part)

    aef = pd.concat(parts, ignore_index=True)
    dup_count = int(aef["TRACT_ID"].duplicated().sum())
    if dup_count:
        print(f"Warning: {dup_count:,} duplicated TRACT_ID rows across aef files; keeping the first occurrence.")
        aef = aef.drop_duplicates("TRACT_ID")

    print(f"aef directory: {aef_dir}")
    print(f"aef files loaded: {len(aef_files)} ({', '.join(p.name for p in aef_files)})")
    print(f"aef rows after row-concat: {len(aef):,}")
    print(f"aef feature dimensions: {len(feature_cols)}")
    return aef, feature_cols


def load_and_merge_data(
    aef_dir: Path,
    pop_csv: Path,
    shp_path: Path,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    aef, feature_cols = load_aef_features(aef_dir)

    pop = pd.read_csv(pop_csv, dtype=str)
    required_pop_cols = ["GEO_ID", "B01003_001E"]
    missing_pop_cols = [col for col in required_pop_cols if col not in pop.columns]
    if missing_pop_cols:
        raise ValueError(f"Population CSV is missing columns: {missing_pop_cols}")

    pop = pop[pop["GEO_ID"] != "Geography"].copy()
    pop["population"] = pd.to_numeric(pop["B01003_001E"], errors="coerce")
    pop = pop.dropna(subset=["GEO_ID", "population"])
    pop = pop[["GEO_ID", "NAME", "population"]].drop_duplicates("GEO_ID")

    gdf = gpd.read_file(shp_path)
    geoid_col = detect_shp_geoid_col(gdf)
    gdf = gdf.rename(columns={geoid_col: "GEO_ID"})
    gdf["GEO_ID"] = gdf["GEO_ID"].astype(str)

    if "cb_2020_11" in gdf.columns:
        gdf["tract_area_sqkm"] = pd.to_numeric(gdf["cb_2020_11"], errors="coerce") / 1_000_000.0
    else:
        area_gdf = gdf.to_crs(epsg=5070)
        gdf["tract_area_sqkm"] = area_gdf.geometry.area / 1_000_000.0
    gdf.loc[gdf["tract_area_sqkm"] <= 0, "tract_area_sqkm"] = np.nan

    aef = aef.rename(columns={"TRACT_ID": "GEO_ID"})
    aef["GEO_ID"] = aef["GEO_ID"].astype(str)
    aef = aef[[col for col in ["GEO_ID", "MSA_NAME", "aef_SOURCE", *feature_cols] if col in aef.columns]].drop_duplicates("GEO_ID")

    merged = gdf.merge(aef, on="GEO_ID", how="inner").merge(pop, on="GEO_ID", how="inner")
    merged = merged.dropna(subset=feature_cols + ["population", "tract_area_sqkm", "geometry"]).copy()
    merged["population_density"] = merged["population"] / merged["tract_area_sqkm"]
    merged["population_density"] = merged["population_density"].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["population_density"]).reset_index(drop=True)
    merged = merged.dropna(subset=["population_density"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("Merged dataset is empty. Check GEOID formats across aef, ACS, and shapefile.")

    print(f"aef rows: {len(aef):,}")
    print(f"Population rows: {len(pop):,}")
    print(f"Shapefile tracts: {len(gdf):,}")
    print(f"Merged training tracts: {len(merged):,}")
    print(f"Shapefile GEOID column used: {geoid_col}")
    print("Target: population_density = population / tract_area_sqkm (people per sq km)")

    return merged, feature_cols


def edge_set_to_tensor(edge_set: Iterable[Tuple[int, int]], label: str) -> torch.Tensor:
    edges = sorted(set(edge_set))
    if not edges:
        raise ValueError(f"{label} produced no edges. Check tract geometries.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"{label} edges: {edge_index.size(1):,} directed edges")
    return edge_index


def build_queen_edges(gdf: gpd.GeoDataFrame) -> set[Tuple[int, int]]:
    graph_gdf = gdf.reset_index(drop=True)
    try:
        weights = Queen.from_dataframe(graph_gdf, use_index=True)
    except TypeError:
        weights = Queen.from_dataframe(graph_gdf)

    edge_set = set()
    for src, neighbors in weights.neighbors.items():
        src_i = int(src)
        for dst in neighbors:
            dst_i = int(dst)
            if src_i == dst_i:
                continue
            edge_set.add((src_i, dst_i))
            edge_set.add((dst_i, src_i))
    return edge_set


def build_edge_index(gdf: gpd.GeoDataFrame) -> torch.Tensor:
    return edge_set_to_tensor(build_queen_edges(gdf), "Queen")


def make_masks(num_nodes: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


class GraphSAGERegressor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
    def forward_emb(self, x, edge_index):
        """返回第二层卷积后的嵌入（即图卷积后的特征）"""
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 不使用 dropout, 保留特征
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x  # shape: [num_nodes, hidden_channels]


def inverse_standardize(y_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_scaled * std + mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_percent": mape}

def transform_target(y_density: np.ndarray, mode: str) -> np.ndarray:
    """将原始人口密度转换为模型训练使用的目标值（raw 或 log1p）。"""
    if mode == "raw":
        return y_density.astype(np.float32)
    if mode == "log1p":
        return np.log1p(np.clip(y_density, 0, None)).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {mode}")

def inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y_model.astype(np.float32)
    if mode == "log1p":
        # 防止 exp 溢出：输入裁剪到 [-700, 700] 是安全的
        y_clipped = np.clip(y_model, -700, 700)
        return np.expm1(y_clipped).clip(min=0).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {mode}")

# MODIFIED: evaluate now accepts target_transform and applies inverse transform before metrics
def evaluate(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    y_mean: float,
    y_std: float,
    target_transform: str,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(data.x, data.edge_index)
        mse_scaled = F.mse_loss(pred_scaled[mask], data.y[mask]).item()
        # 先将标准化值逆标准化为模型空间（变换后的空间）
        pred_model = inverse_standardize(pred_scaled[mask].detach().cpu().numpy(), y_mean, y_std)
        true_model = inverse_standardize(data.y[mask].detach().cpu().numpy(), y_mean, y_std)
        # 再逆变换回原始密度
        y_pred = inverse_target(pred_model, target_transform)
        y_true = inverse_target(true_model, target_transform)
        metrics = compute_metrics(y_true, y_pred)
        metrics["mse_scaled"] = mse_scaled
        return metrics


def save_prediction_shapefile(
    gdf: gpd.GeoDataFrame,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    stem: str,
) -> Path:
    shp = gdf[["GEO_ID", "geometry"]].copy()
    if "NAME" in gdf.columns:
        shp["name"] = gdf["NAME"].astype(str)
    shp["split"] = "unused"
    shp.loc[train_mask.numpy(), "split"] = "train"
    shp.loc[val_mask.numpy(), "split"] = "val"
    shp.loc[test_mask.numpy(), "split"] = "test"
    shp["pop"] = gdf["population"].to_numpy(dtype=float)
    shp["area_km2"] = gdf["tract_area_sqkm"].to_numpy(dtype=float)
    shp["den_true"] = y_true.astype(float)
    shp["den_pred"] = y_pred.astype(float)
    shp["den_res"] = shp["den_pred"] - shp["den_true"]
    shp["abs_err"] = np.abs(shp["den_res"])

    shp_path = out_dir / f"{stem}.shp"
    shp.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    return shp_path


def train_single_city(city: str, args: argparse.Namespace, out_dir: Path) -> None:
    # 训练单个城市并保存模型（包含 scaler 和变换模式）。
    set_seed(args.seed)
    aef_dir = resolve_aef_dir(args.aef_root, city, args.aef_dir)
    shp_path = resolve_shp_path(aef_dir, args.shp)
    gdf, feature_cols = load_and_merge_data(aef_dir, Path(args.pop_csv), shp_path)
    edge_index = build_edge_index(gdf)

    x_raw = gdf[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()# 减均值除标准差
    x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float32)

    # ---- 目标变换 ----
    y_raw = gdf["population_density"].to_numpy(dtype=np.float32)
    y_transformed = transform_target(y_raw, args.target_transform)   # 应用变换

    train_mask, val_mask, test_mask = make_masks(len(gdf), args.train_ratio, args.val_ratio, args.seed)

    # 在变换后的空间上计算均值和标准差（仅用训练集）
    y_mean = float(y_transformed[train_mask.numpy()].mean())
    y_std = float(y_transformed[train_mask.numpy()].std())
    if y_std <= 0:
        raise ValueError(f"{city}: Training target (transformed) has zero standard deviation.")
    # 标准化
    y = torch.tensor((y_transformed - y_mean) / y_std, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = GraphSAGERegressor(x.size(1), args.hidden_channels, args.dropout).to(device)
    print(f"\n=== Training {city} using GraphSAGE with NeighborLoader ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = torch.where(train_mask)[0]
    loader_train = NeighborLoader(
        data.cpu(),
        num_neighbors=[10, 5],
        batch_size=args.batch_size,
        input_nodes=train_idx.cpu(),
        shuffle=True,
        num_workers=0,
    )

    data = data.to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience
    min_delta = getattr(args, 'early_stop_delta', 1e-4)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in loader_train:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)

            loss = F.mse_loss(
                pred[:batch.batch_size],
                batch.y[:batch.batch_size]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)

        if epoch % args.eval_every == 0 or epoch == 1:
            # 评估时传入变换模式
            train_metrics = evaluate(model, data, data.train_mask, y_mean, y_std, args.target_transform)
            val_metrics = evaluate(model, data, data.val_mask, y_mean, y_std, args.target_transform)

            row = {
                "epoch": epoch,
                "train_loss_scaled_mse": train_metrics["mse_scaled"],
                "val_loss_scaled_mse": val_metrics["mse_scaled"],
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "mse_scaled"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "mse_scaled"},
            }
            history.append(row)
            print(
                f"{city} Epoch {epoch:04d} | train loss={train_metrics['mse_scaled']:.6f}, val loss={val_metrics['mse_scaled']:.6f} | "
                f"train RMSE={train_metrics['rmse']:.2f}, R2={train_metrics['r2']:.3f} | "
                f"val RMSE={val_metrics['rmse']:.2f}, R2={val_metrics['r2']:.3f}"
            )

            if val_metrics["mse_scaled"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["mse_scaled"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.patience
                print(f"  ✓ Improvement: val loss {val_metrics['mse_scaled']:.6f}")
            else:
                patience_left -= args.eval_every
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best model from epoch {best_epoch} with val loss {best_val_loss:.6f}")

    # 最终评估
    train_metrics = evaluate(model, data, data.train_mask, y_mean, y_std, args.target_transform)
    val_metrics = evaluate(model, data, data.val_mask, y_mean, y_std, args.target_transform)
    test_metrics = evaluate(model, data, data.test_mask, y_mean, y_std, args.target_transform)

    print(f"\n{city} Final metrics on original population-density scale (people per sq km)")
    print(json.dumps({"train": train_metrics, "val": val_metrics, "test": test_metrics}, indent=2))

    # 预测全图（先标准化->逆标准化->逆变换）
    model.eval()
    with torch.no_grad():
        pred_scaled_all = model(data.x, data.edge_index).detach().cpu().numpy()
    pred_model = inverse_standardize(pred_scaled_all, y_mean, y_std)
    pred_density = inverse_target(pred_model, args.target_transform)

    # 保存 CSV
    csv_cols = [col for col in ["GEO_ID", "NAME", "MSA_NAME", "aef_SOURCE", "population", "tract_area_sqkm"] if col in gdf.columns]
    output = gdf[csv_cols].copy()
    output["split"] = "unused"
    output.loc[train_mask.numpy(), "split"] = "train"
    output.loc[val_mask.numpy(), "split"] = "val"
    output.loc[test_mask.numpy(), "split"] = "test"
    output["population_density_true"] = y_raw
    output["population_density_pred"] = pred_density
    output["density_residual"] = output["population_density_pred"] - output["population_density_true"]
    output["abs_error"] = output["density_residual"].abs()
    output_stem = f"{city}_GraphSAGE"
    output_path = out_dir / f"{output_stem}_predictions.csv"
    output.to_csv(output_path, index=False)

    # 保存 shapefile
    shp_path = save_prediction_shapefile(
        gdf=gdf,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_true=y_raw,
        y_pred=pred_density,
        out_dir=out_dir,
        stem=f"{output_stem}_tract_pred",
    )

    # 保存历史
    history_path = out_dir / f"{output_stem}_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # 保存模型（包含 scaler, transform mode 等）
    model_path = out_dir / f"{output_stem}_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "target_mean": y_mean,
            "target_std": y_std,
            "target_transform": args.target_transform,   # 保存变换模式
            "scaler": scaler,
            "args": vars(args),
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        },
        model_path,
    )

    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        df_hist = pd.DataFrame(history)
        if not df_hist.empty:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax1 = axes[0]
            ax1.plot(df_hist["epoch"], df_hist["train_loss_scaled_mse"], label="Train Loss", color="blue")
            ax1.plot(df_hist["epoch"], df_hist["val_loss_scaled_mse"], label="Val Loss", color="red", linestyle="--")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Scaled MSE")
            ax1.set_title("Training & Validation Loss")
            ax1.legend()
            ax1.grid(True)

            ax2 = axes[1]
            ax2.plot(df_hist["epoch"], df_hist["train_rmse"], label="Train RMSE", color="green")
            ax2.plot(df_hist["epoch"], df_hist["val_rmse"], label="Val RMSE", color="orange", linestyle="--")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("RMSE (people/km²)")
            ax2.set_title("RMSE")
            ax2.legend()
            ax2.grid(True)

            ax3 = axes[2]
            ax3.plot(df_hist["epoch"], df_hist["train_r2"], label="Train R²", color="green")
            ax3.plot(df_hist["epoch"], df_hist["val_r2"], label="Val R²", color="red", linestyle="--")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("R²")
            ax3.set_title("R² Score")
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            plot_path = out_dir / f"{output_stem}_loss_curves.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved loss curves: {plot_path}")
    except ImportError:
        print("matplotlib not installed; skipping loss curve plot.")
    except Exception as e:
        print(f"Failed to plot loss curves: {e}")

    print(f"\n{city} saved predictions: {output_path}")
    print(f"{city} saved prediction shapefile: {shp_path}")
    print(f"{city} saved training history: {history_path}")
    print(f"{city} saved model checkpoint: {model_path}")
    print(gdf["population_density"].describe())


def discover_cities(aef_root: Path) -> List[str]:
    cities = []
    for city_dir in sorted(p for p in aef_root.iterdir() if p.is_dir()):
        has_aef = any(city_dir.glob("aef_*_b*_2020.csv"))
        has_shp = any(city_dir.glob("*.shp"))
        if has_aef and has_shp:
            cities.append(city_dir.name)
    return cities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphSAGE regression for tract population-density prediction.")
    parser.add_argument("--aef-root", default=str(DEFAULT_aef_ROOT), help="Root directory containing one subfolder per city.")
    parser.add_argument("--city", default=None, help="Single city to train (if --cities and --all not given).")
    parser.add_argument("--cities", nargs="+", default=None, help="List of cities to train (overrides --city).")
    parser.add_argument("--all", action="store_true", help="Train on all available cities (overrides --city and --cities).")
    parser.add_argument("--aef-dir", default=None, help="Optional direct aef city directory; overrides --aef-root/--city.")
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV), help="ACS B01003 population CSV.")
    parser.add_argument("--shp", default=DEFAULT_SHP, help="Optional tract shapefile; if omitted, auto-detect *.shp in the city aef folder.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for NeighborLoader.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4, help="Min improvement in val loss to reset patience.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--target-transform", choices=["raw", "log1p"], default="log1p",
                    help="Transform applied to target before standardization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        cities = discover_cities(Path(args.aef_root))
        if not cities:
            raise ValueError("No cities found with --all.")
    elif args.cities:
        cities = args.cities
    elif args.city:
        cities = [args.city]
    else:
        raise ValueError("Must specify --city, --cities, or --all.")

    print(f"Cities to train: {cities}")
    for city in cities:
        try:
            train_single_city(city, args, out_dir)
        except Exception as e:
            print(f"Error training {city}: {e}")
            continue


if __name__ == "__main__":
    main()