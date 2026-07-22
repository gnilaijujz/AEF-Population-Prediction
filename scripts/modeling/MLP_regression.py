from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- 配置 pyproj（与 GNN 版本相同） ----------
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

# ---------- 导入 ----------
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from libpysal.weights import Queen
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------- 默认路径 ----------
DEFAULT_GIS_ROOT = Path(r"data_sources/processed_gis")
DEFAULT_POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
DEFAULT_SHP = None
DEFAULT_OUT_DIR = Path(r"model_data/pretrained_models/mlp")

# ---------- 工具函数（复用 GNN 版本） ----------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_gis_dir(gis_root: str | None, city: str | None, gis_dir: str | None) -> Path:
    if gis_dir:
        resolved = Path(gis_dir)
    else:
        if not city:
            raise ValueError("Either --gis-dir or --city must be provided.")
        resolved = Path(gis_root or DEFAULT_GIS_ROOT) / city
    if not resolved.exists():
        raise FileNotFoundError(f"GIS directory does not exist: {resolved}")
    return resolved

def resolve_shp_path(gis_dir: Path, shp: str | None) -> Path:
    if shp:
        resolved = Path(shp)
        if not resolved.exists():
            raise FileNotFoundError(f"Shapefile does not exist: {resolved}")
        return resolved

    shp_files = sorted(gis_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in city GIS directory: {gis_dir}")

    msa_shps = [p for p in shp_files if "track" in p.stem.lower() or "tract" in p.stem.lower()]
    resolved = msa_shps[0] if msa_shps else shp_files[0]
    print(f"Shapefile auto-detected: {resolved}")
    return resolved

def detect_shp_geoid_col(gdf: gpd.GeoDataFrame) -> str:
    preferred = ["GEO_ID", "GEOID", "TRACT_ID", "cb_2020__3", "ACSDT5Y202","cb_2020_3","tract_ID"]
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

def load_gis_features(gis_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    # 不再预定义 feature_cols，改为自动检测
    gis_files = sorted(gis_dir.glob("gis_*_b*_2020.csv"))
    if not gis_files:
        raise ValueError(f"No GIS files matching gis_*_b*_2020.csv found in {gis_dir}")

    parts = []
    for gis_file in gis_files:
        gis_part = pd.read_csv(gis_file)
        # 统一列名大写，避免大小写问题（同时解决 tract_ID 问题）
        gis_part.columns = [col.upper() for col in gis_part.columns]
        if "TRACT_ID" not in gis_part.columns:
            raise ValueError(f"{gis_file} must contain a TRACT_ID column.")
        gis_part["TRACT_ID"] = gis_part["TRACT_ID"].astype(str)

        # 保留需要列，但不预先指定特征列，后面统一提取
        # 如果存在 MSA_NAME，保留
        if "MSA_NAME" in gis_part.columns:
            keep_cols = ["TRACT_ID", "MSA_NAME"]
        else:
            keep_cols = ["TRACT_ID"]
        # 添加一个来源标记列
        gis_part["GIS_SOURCE"] = gis_file.name
        # 保留所有列（除了可能存在的无用列，但我们先保留全部，后续统一特征提取）
        # 为简化，我们不对列做过滤，直接保留全部
        parts.append(gis_part)

    # 合并所有文件（外连接，不同文件可能有不同列）
    gis = pd.concat(parts, ignore_index=True)
    # 去重
    gis = gis.drop_duplicates("TRACT_ID")

    # 自动识别特征列：排除已知的标识列，剩下的数值列作为特征
    exclude_cols = {"TRACT_ID", "MSA_NAME", "GIS_SOURCE", "ACSDT5Y2_2", "ACSDT5Y2_3", "ACSDT5Y2_4"} # 这些不是特征
    # 所有列名
    all_cols = set(gis.columns)
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    # 进一步过滤掉非数值列（如字符串列），确保特征都是数值
    # 这里简单检查数据类型，保留数值类型列
    numeric_cols = gis[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    # 如果特征列数不足，给出警告
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric feature columns found in GIS data. Columns: {list(all_cols)}")
    # 最终特征列
    feature_cols = numeric_cols

    # 过滤出这些特征列，并确保包含 TRACT_ID 等标识列
    keep_final = ["TRACT_ID"] + (["MSA_NAME"] if "MSA_NAME" in gis.columns else []) + feature_cols
    # 如果 GIS_SOURCE 存在，保留
    if "GIS_SOURCE" in gis.columns:
        keep_final.append("GIS_SOURCE")
    gis = gis[keep_final]

    print(f"GIS directory: {gis_dir}")
    print(f"GIS files loaded: {len(gis_files)} ({', '.join(p.name for p in gis_files)})")
    print(f"GIS rows after merging: {len(gis):,}")
    print(f"Auto-detected feature columns ({len(feature_cols)}): {feature_cols}")

    return gis, feature_cols

def load_and_merge_data(
    gis_dir: Path,
    pop_csv: Path,
    shp_path: Path,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    gis, feature_cols = load_gis_features(gis_dir)

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

    gis = gis.rename(columns={"TRACT_ID": "GEO_ID"})
    gis["GEO_ID"] = gis["GEO_ID"].astype(str)
    gis = gis[[col for col in ["GEO_ID", "MSA_NAME", "GIS_SOURCE", *feature_cols] if col in gis.columns]].drop_duplicates("GEO_ID")

    merged = gdf.merge(gis, on="GEO_ID", how="inner").merge(pop, on="GEO_ID", how="inner")
    merged = merged.dropna(subset=feature_cols + ["population", "tract_area_sqkm", "geometry"]).copy()
    merged["population_density"] = merged["population"] / merged["tract_area_sqkm"]
    merged = merged.dropna(subset=["population_density"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError("Merged dataset is empty. Check GEOID formats across GIS, ACS, and shapefile.")

    print(f"GIS rows: {len(gis):,}")
    print(f"Population rows: {len(pop):,}")
    print(f"Shapefile tracts: {len(gdf):,}")
    print(f"Merged training tracts: {len(merged):,}")
    print(f"Shapefile GEOID column used: {geoid_col}")
    print("Target: population_density = population / tract_area_sqkm (people per sq km)")

    return merged, feature_cols

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

# ---------- 目标变换（复用） ----------
def transform_target(y_density: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y_density.astype(np.float32)
    if mode == "log1p":
        return np.log1p(np.clip(y_density, 0, None)).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {mode}")

def inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y_model.astype(np.float32)
    if mode == "log1p":
        return np.expm1(y_model).clip(min=0).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {mode}")

def inverse_standardize(y_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_scaled * std + mean

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_percent": mape}

# ---------- MLP 模型 ----------
class MLPRegressor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc3(x).squeeze(-1)

# ---------- 评估函数（无图） ----------
def evaluate_mlp(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    y_mean: float,
    y_std: float,
    target_transform: str,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x)   # 全量预测
        mse_scaled = F.mse_loss(pred_scaled[mask], y[mask]).item()
        pred_model = inverse_standardize(pred_scaled[mask].detach().cpu().numpy(), y_mean, y_std)
        true_model = inverse_standardize(y[mask].detach().cpu().numpy(), y_mean, y_std)
        y_pred = inverse_target(pred_model, target_transform)
        y_true = inverse_target(true_model, target_transform)
        metrics = compute_metrics(y_true, y_pred)
        metrics["mse_scaled"] = mse_scaled
        return metrics

# ---------- 保存 Shapefile（复用） ----------
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

# ---------- 训练单个城市（MLP 版本） ----------
def train_single_city(city: str, args: argparse.Namespace, out_dir: Path) -> None:
    set_seed(args.seed)
    gis_dir = resolve_gis_dir(args.gis_root, city, args.gis_dir)
    shp_path = resolve_shp_path(gis_dir, args.shp)
    gdf, feature_cols = load_and_merge_data(gis_dir, Path(args.pop_csv), shp_path)

    # 特征标准化
    x_raw = gdf[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_tensor = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float32)

    # 目标变换与标准化
    y_raw = gdf["population_density"].to_numpy(dtype=np.float32)
    y_transformed = transform_target(y_raw, args.target_transform)

    train_mask, val_mask, test_mask = make_masks(len(gdf), args.train_ratio, args.val_ratio, args.seed)

    y_mean = float(y_transformed[train_mask.numpy()].mean())
    y_std = float(y_transformed[train_mask.numpy()].std())
    if y_std <= 0:
        raise ValueError(f"{city}: Training target (transformed) has zero standard deviation.")
    y_tensor = torch.tensor((y_transformed - y_mean) / y_std, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    model = MLPRegressor(x_tensor.size(1), args.hidden_channels, args.dropout).to(device)
    print(f"\n=== Training {city} using MLP ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 构建训练 DataLoader
    train_idx = torch.where(train_mask)[0]
    train_dataset = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

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
        for batch_x, batch_y in loader_train:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if epoch % args.eval_every == 0 or epoch == 1:
            train_metrics = evaluate_mlp(model, x_tensor, y_tensor, train_mask, y_mean, y_std, args.target_transform)
            val_metrics = evaluate_mlp(model, x_tensor, y_tensor, val_mask, y_mean, y_std, args.target_transform)

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
    train_metrics = evaluate_mlp(model, x_tensor, y_tensor, train_mask, y_mean, y_std, args.target_transform)
    val_metrics = evaluate_mlp(model, x_tensor, y_tensor, val_mask, y_mean, y_std, args.target_transform)
    test_metrics = evaluate_mlp(model, x_tensor, y_tensor, test_mask, y_mean, y_std, args.target_transform)

    print(f"\n{city} Final metrics on original population-density scale (people per sq km)")
    print(json.dumps({"train": train_metrics, "val": val_metrics, "test": test_metrics}, indent=2))

    # 全量预测
    model.eval()
    with torch.no_grad():
        pred_scaled_all = model(x_tensor).detach().cpu().numpy()
    pred_model = inverse_standardize(pred_scaled_all, y_mean, y_std)
    pred_density = inverse_target(pred_model, args.target_transform)

    # 保存 CSV
    csv_cols = [col for col in ["GEO_ID", "NAME", "MSA_NAME", "GIS_SOURCE", "population", "tract_area_sqkm"] if col in gdf.columns]
    output = gdf[csv_cols].copy()
    output["split"] = "unused"
    output.loc[train_mask.numpy(), "split"] = "train"
    output.loc[val_mask.numpy(), "split"] = "val"
    output.loc[test_mask.numpy(), "split"] = "test"
    output["population_density_true"] = y_raw
    output["population_density_pred"] = pred_density
    output["density_residual"] = output["population_density_pred"] - output["population_density_true"]
    output["abs_error"] = output["density_residual"].abs()
    output_stem = f"{city}_MLP"
    output_path = out_dir / f"{output_stem}_predictions.csv"
    output.to_csv(output_path, index=False)

    # 保存 Shapefile
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

    # 保存训练历史
    history_path = out_dir / f"{output_stem}_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # 保存模型及元数据
    model_path = out_dir / f"{output_stem}_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "target_mean": y_mean,
            "target_std": y_std,
            "target_transform": args.target_transform,
            "scaler": scaler,
            "args": vars(args),
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        },
        model_path,
    )

    # 绘制损失曲线（可选）
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

# ---------- 城市发现（复用） ----------
def discover_cities(gis_root: Path) -> List[str]:
    cities = []
    for city_dir in sorted(p for p in gis_root.iterdir() if p.is_dir()):
        has_gis = any(city_dir.glob("gis_*_b*_2020.csv"))
        has_shp = any(city_dir.glob("*.shp"))
        if has_gis and has_shp:
            cities.append(city_dir.name)
    return cities

# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP regression for tract population-density prediction.")
    parser.add_argument("--gis-root", default=str(DEFAULT_GIS_ROOT), help="Root directory containing one subfolder per city.")
    parser.add_argument("--city", default=None, help="Single city to train (if --cities and --all not given).")
    parser.add_argument("--cities", nargs="+", default=None, help="List of cities to train (overrides --city).")
    parser.add_argument("--all", action="store_true", help="Train on all available cities (overrides --city and --cities).")
    parser.add_argument("--gis-dir", default=None, help="Optional direct GIS city directory; overrides --gis-root/--city.")
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV), help="ACS B01003 population CSV.")
    parser.add_argument("--shp", default=DEFAULT_SHP, help="Optional tract shapefile; if omitted, auto-detect *.shp in the city GIS folder.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for DataLoader.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
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

# ---------- 主函数 ----------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        cities = discover_cities(Path(args.gis_root))
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