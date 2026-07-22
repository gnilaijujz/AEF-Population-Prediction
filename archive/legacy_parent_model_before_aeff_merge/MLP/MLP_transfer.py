#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLP_transfer_experiments_calibrated.py
支持 MLP 模型的迁移学习诊断，包含均值校准选项。
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# 导入 MLP 训练脚本中的通用函数（需确保 MLP_regression.py 在同目录或可导入）
import MLP_regression as mlp

# ---------- 默认路径 ----------
# 相对脚本/仓库根定位:任意机器 clone 后无需改路径(仍可用 --xxx 覆盖)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]                      # .../model/MLP -> 仓库根
DEFAULT_AEF_ROOT = REPO_ROOT / "data_AEF"
DEFAULT_MLP_INPUT_ROOT = SCRIPT_DIR / "MLP_input"
DEFAULT_INPUT_FEATURE = "poi_trans"
DEFAULT_POP_CSV = REPO_ROOT / "MSA" / "raw_msa_data" / "ACSDT5Y2020.pop" / "ACSDT5Y2020.B01003-Data.csv"  # 人口标签;可用 --pop-csv 覆盖
DEFAULT_PRETRAINED_DIR = SCRIPT_DIR / "selftrain"      # 与 MLP_regression 的 --out-dir 一致
DEFAULT_OUT_DIR = SCRIPT_DIR / "transfer_results_log_mlp"

# ---------- 工具函数（部分从 GNN_transfer 复制） ----------
def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")

def discover_cities(aef_root: Path) -> List[str]:
    cities = []
    for city_dir in sorted(p for p in aef_root.iterdir() if p.is_dir()):
        has_aef = any(city_dir.glob("aef_*_b*_2020.csv"))
        has_shp = any(city_dir.glob("*.shp"))
        if has_aef and has_shp:
            cities.append(city_dir.name)
    return cities

def discover_mlp_input_cities(input_root: Path, input_feature: str) -> List[str]:
    return mlp.discover_mlp_input_cities(input_root, input_feature)

def load_city(args: argparse.Namespace, city: str):
    """加载单个城市数据（复用 MLP_regression 的加载逻辑）"""
    if args.input_mode == "mlp_input":
        # MODIFIED: use the same prepared 65-dim CSVs as MLP_regression.py.
        input_csv = mlp.resolve_mlp_input_csv(args.input_root, city, args.input_feature, None)
        gdf, feature_cols = mlp.load_mlp_input_csv(input_csv)
    else:
        aef_dir = mlp.resolve_aef_dir(str(args.aef_root), city, None)
        shp_path = mlp.resolve_shp_path(aef_dir, None)
        gdf, feature_cols = mlp.load_and_merge_data(aef_dir, Path(args.pop_csv), shp_path)
    return gdf, feature_cols   # 无 edge_index

# ---------- 指标计算（含皮尔逊相关系数） ----------
def compute_metrics_with_r(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from scipy.stats import pearsonr
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0)
    r, _ = pearsonr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_percent": mape, "pearson_r": r}

# ---------- 嵌入距离度量（从 GNN_transfer 复制） ----------
def mmd_rbf(X, Y, gamma=0.5):
    from sklearn.metrics.pairwise import rbf_kernel
    X = np.asarray(X)
    Y = np.asarray(Y)
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    n = X.shape[0]
    m = Y.shape[0]
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return float(mmd)

def cka_linear(X, Y):
    def centering(K):
        n = K.shape[0]
        unit = np.ones((n, n)) / n
        return K - unit @ K - K @ unit + unit @ K @ unit
    n = X.shape[0]
    K = X @ X.T
    L = Y @ Y.T
    Kc = centering(K)
    Lc = centering(L)
    hsic_xy = np.sum(Kc * Lc) / ((n - 1) ** 2)
    hsic_xx = np.sum(Kc * Kc) / ((n - 1) ** 2)
    hsic_yy = np.sum(Lc * Lc) / ((n - 1) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return hsic_xy / denom

# ---------- 定义 MLP 模型（与 MLP_regression 一致，但增加 forward_emb） ----------
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

    def forward_emb(self, x: torch.Tensor) -> torch.Tensor:
        """返回第二个隐藏层的输出（嵌入）"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)          # 不经过激活？为了与训练时一致，也可以加激活
        x = F.relu(x)            # 通常隐藏层后接 ReLU
        return x

# ---------- Ridge 回归训练与预测（复制自 GNN_transfer） ----------
def train_ridge_on_source(
    source_gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    target_transform: str,
    alpha: float = 1.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Ridge, StandardScaler, float, float]:
    train_mask, _, _ = mlp.make_masks(len(source_gdf), train_ratio, val_ratio, seed)
    idx_train = train_mask.numpy()
    x_raw = source_gdf[feature_cols].to_numpy(dtype=np.float32)[idx_train]
    y_raw = source_gdf["population_density"].to_numpy(dtype=np.float32)[idx_train]

    y_trans = mlp.transform_target(y_raw, target_transform)
    y_mean = float(y_trans.mean())
    y_std = float(y_trans.std())
    if y_std <= 0:
        raise ValueError("Training target has zero standard deviation.")
    y_scaled = (y_trans - y_mean) / y_std

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    ridge = Ridge(alpha=alpha, random_state=seed)
    ridge.fit(x_scaled, y_scaled)
    return ridge, scaler, y_mean, y_std

def predict_ridge_on_target(
    ridge: Ridge,
    scaler: StandardScaler,
    target_gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    y_mean: float,
    y_std: float,
    target_transform: str,
) -> np.ndarray:
    x_raw = target_gdf[feature_cols].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x_raw)
    pred_scaled = ridge.predict(x_scaled)
    pred_model = mlp.inverse_standardize(pred_scaled, y_mean, y_std)
    return mlp.inverse_target(pred_model, target_transform)

# ---------- 加载预训练 MLP 模型 ----------
def scan_model_files(pretrained_dir: Path, input_feature: str | None = None) -> Dict[str, Path]:
    model_files = {}
    suffix = "_MLP_model.pt"
    feature_suffix = f"_{safe_name(input_feature)}_MLP_model.pt" if input_feature else None
    for pt_path in pretrained_dir.glob("*_model.pt"):
        name = pt_path.name
        if feature_suffix and name.endswith(feature_suffix):
            city = name[:-len(feature_suffix)]
        elif feature_suffix:
            continue
        elif name.endswith(suffix):
            city = name[:-len(suffix)]
        else:
            city = name.split("_")[0]
        model_files[city] = pt_path
    return model_files

def load_pretrained_mlp(
    model_path: Path,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
) -> Tuple[torch.nn.Module, StandardScaler, float, float, List[str], str]:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    in_channels = len(checkpoint['feature_cols'])
    model = MLPRegressor(in_channels, hidden_channels, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_mean = checkpoint['target_mean']
    y_std = checkpoint['target_std']
    feature_cols = checkpoint['feature_cols']
    target_transform = checkpoint.get('target_transform', 'raw')

    x_scaler = checkpoint.get('scaler')
    if x_scaler is None:
        raise ValueError("Scaler missing in checkpoint. Please retrain with scaler saved.")
    return model, x_scaler, y_mean, y_std, feature_cols, target_transform

# ---------- 准备目标数据（无图结构） ----------
def prepare_target_data(
    gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    x_scaler: StandardScaler,
    y_mean: float,
    y_std: float,
    target_transform: str,
    device: torch.device,
    shuffle_features: bool = False,
    shuffle_seed: int = 123,
):
    x_raw = gdf[feature_cols].to_numpy(dtype=np.float32)
    if shuffle_features:
        np.random.seed(shuffle_seed)
        perm = np.random.permutation(len(x_raw))
        x_raw = x_raw[perm, :]

    x = torch.tensor(x_scaler.transform(x_raw), dtype=torch.float32)
    y_density = gdf["population_density"].to_numpy(dtype=np.float32)
    y_model = mlp.transform_target(y_density, target_transform)
    y = torch.tensor((y_model - y_mean) / y_std, dtype=torch.float32)
    return x.to(device), y.to(device), y_density

# ---------- 执行一组迁移实验 ----------
def run_single_transfer(
    mode: str,
    source_models: Dict,          # 对于 MLP 模式，包含模型及参数
    source_cities: List[str],     # 对于 Ridge，用作源城市列表
    target_cities: List[str],
    city_cache: Dict,
    city_errors: Dict,
    args: argparse.Namespace,
    out_dir: Path,
    use_ridge: bool,
    use_null: bool,
    calibrate_mean: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    执行迁移预测，返回 (r2_raw, rmse_raw, mae_raw, r_raw,
                             r2_cal, rmse_cal, mae_cal, r_cal)
    """
    print(f"\n=== Running mode: {mode} (ridge={use_ridge}, null={use_null}, calibrate={calibrate_mean}) ===")
    metrics_rows = []
    embed_dist_list = []
    actual_sources = source_cities if use_ridge else list(source_models.keys())

    # 初始化矩阵
    matrix_r2_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_rmse_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_mae_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_r_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    if calibrate_mean:
        matrix_r2_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_rmse_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_mae_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_r_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    else:
        matrix_r2_cal = matrix_rmse_cal = matrix_mae_cal = matrix_r_cal = None

    if use_ridge:
        # Ridge 模式：为每个源训练 Ridge，然后预测所有目标
        source_ridges = {}
        for src in source_cities:
            src_data = city_cache.get(src)
            if src_data is None:
                city_errors[src] = city_errors.get(src, "City data not loaded")
                continue
            src_gdf, src_feature_cols = src_data
            try:
                ridge, scaler, y_mean, y_std = train_ridge_on_source(
                    src_gdf,
                    src_feature_cols,
                    args.target_transform,
                    alpha=args.ridge_alpha,
                    train_ratio=args.train_ratio,
                    seed=args.seed,
                )
                source_ridges[src] = {
                    'ridge': ridge,
                    'scaler': scaler,
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'feature_cols': src_feature_cols,
                    'target_transform': args.target_transform,
                }
            except Exception as e:
                print(f"Failed to train Ridge for {src}: {e}")
                city_errors[src] = str(e)

        for src, src_bundle in source_ridges.items():
            print(f"\nSource (Ridge): {src}")
            for tgt in target_cities:
                print(f"  -> {tgt}")
                tgt_data = city_cache.get(tgt)
                if tgt_data is None:
                    metrics_rows.append({"source": src, "target": tgt, "error": city_errors.get(tgt, "target load failed")})
                    continue
                tgt_gdf, tgt_feature_cols = tgt_data
                if tgt_feature_cols != src_bundle['feature_cols']:
                    err = "Feature mismatch"
                    metrics_rows.append({"source": src, "target": tgt, "error": err})
                    continue
                y_pred = predict_ridge_on_target(
                    src_bundle['ridge'],
                    src_bundle['scaler'],
                    tgt_gdf,
                    src_bundle['feature_cols'],
                    src_bundle['y_mean'],
                    src_bundle['y_std'],
                    src_bundle['target_transform'],
                )
                y_true = tgt_gdf["population_density"].to_numpy(dtype=np.float32)
                metrics_raw = compute_metrics_with_r(y_true, y_pred)
                row = {"source": src, "target": tgt,
                       "rmse_raw": metrics_raw["rmse"], "mae_raw": metrics_raw["mae"],
                       "r2_raw": metrics_raw["r2"], "mape_raw": metrics_raw["mape_percent"],
                       "r_raw": metrics_raw["pearson_r"]}
                if calibrate_mean:
                    y_pred_mean = np.mean(y_pred)
                    y_true_mean = np.mean(y_true)
                    y_pred_cal = y_pred - y_pred_mean + y_true_mean
                    metrics_cal = compute_metrics_with_r(y_true, y_pred_cal)
                    row.update({"rmse_cal": metrics_cal["rmse"], "mae_cal": metrics_cal["mae"],
                                "r2_cal": metrics_cal["r2"], "mape_cal": metrics_cal["mape_percent"],
                                "r_cal": metrics_cal["pearson_r"]})
                    matrix_r2_cal.loc[src, tgt] = metrics_cal["r2"]
                    matrix_rmse_cal.loc[src, tgt] = metrics_cal["rmse"]
                    matrix_mae_cal.loc[src, tgt] = metrics_cal["mae"]
                    matrix_r_cal.loc[src, tgt] = metrics_cal["pearson_r"]
                metrics_rows.append(row)
                matrix_r2_raw.loc[src, tgt] = metrics_raw["r2"]
                matrix_rmse_raw.loc[src, tgt] = metrics_raw["rmse"]
                matrix_mae_raw.loc[src, tgt] = metrics_raw["mae"]
                matrix_r_raw.loc[src, tgt] = metrics_raw["pearson_r"]
                print(f"    R2_raw={metrics_raw['r2']:.3f}, RMSE_raw={metrics_raw['rmse']:.2f}" +
                      (f", R2_cal={metrics_cal['r2']:.3f}" if calibrate_mean else ""))

    else:
        # MLP 模式（可能打乱）
        for src, src_bundle in source_models.items():
            print(f"\nSource (MLP): {src}")
            model = src_bundle['model']
            x_scaler = src_bundle['x_scaler']
            y_mean = src_bundle['y_mean']
            y_std = src_bundle['y_std']
            feat_cols = src_bundle['feature_cols']
            target_transform = src_bundle['target_transform']

            if args.target_transform and args.target_transform != target_transform:
                print(f"Using model's transform: {target_transform}")

            # 提取源域嵌入（使用所有样本）
            src_gdf, src_feat_cols = city_cache[src]
            src_x, _, _ = prepare_target_data(
                src_gdf,
                feat_cols,
                x_scaler,
                y_mean,
                y_std,
                target_transform,
                next(model.parameters()).device,
                shuffle_features=False
            )
            with torch.no_grad():
                src_emb = model.forward_emb(src_x).detach().cpu().numpy()
            print(f"  Source embedding shape: {src_emb.shape}")

            for tgt in target_cities:
                print(f"  -> {tgt}")
                tgt_data = city_cache.get(tgt)
                if tgt_data is None:
                    metrics_rows.append({"source": src, "target": tgt, "error": city_errors.get(tgt, "target load failed")})
                    continue
                tgt_gdf, tgt_feature_cols = tgt_data
                if tgt_feature_cols != feat_cols:
                    err = "Feature mismatch"
                    metrics_rows.append({"source": src, "target": tgt, "error": err})
                    continue

                device = next(model.parameters()).device

                # 提取目标域原始嵌入（用于距离计算，不打乱）
                tgt_x_orig, _, _ = prepare_target_data(
                    tgt_gdf,
                    feat_cols,
                    x_scaler,
                    y_mean,
                    y_std,
                    target_transform,
                    device,
                    shuffle_features=False
                )
                with torch.no_grad():
                    tgt_emb_orig = model.forward_emb(tgt_x_orig).detach().cpu().numpy()

                # 计算分布距离
                l1_dist = np.mean(np.abs(np.mean(src_emb, axis=0) - np.mean(tgt_emb_orig, axis=0)))
                l2_dist = np.linalg.norm(np.mean(src_emb, axis=0) - np.mean(tgt_emb_orig, axis=0))
                cos_dist = np.mean(cdist(src_emb, tgt_emb_orig, metric='cosine'))
                cos_sim = 1 - np.nan_to_num(cos_dist, nan=1.0)
                if np.isnan(cos_sim):
                    cos_sim = 0.0
                mmd_val = mmd_rbf(src_emb, tgt_emb_orig)
                # cka_val = cka_linear(src_emb, tgt_emb_orig)

                embed_dist_list.append({
                    "Source": src,
                    "Target": tgt,
                    "L1_Dist": l1_dist,
                    "L2_Dist": l2_dist,
                    "Cos_Sim": cos_sim,
                    "MMD": mmd_val,
                    # "CKA_Sim": cka_val
                })

                # 准备预测数据（可能打乱）
                tgt_x, _, y_true = prepare_target_data(
                    tgt_gdf,
                    feat_cols,
                    x_scaler,
                    y_mean,
                    y_std,
                    target_transform,
                    device,
                    shuffle_features=use_null,
                    shuffle_seed=args.null_seed,
                )
                model.eval()
                with torch.no_grad():
                    pred_scaled = model(tgt_x).detach().cpu().numpy()
                pred_model = mlp.inverse_standardize(pred_scaled, y_mean, y_std)
                y_pred = mlp.inverse_target(pred_model, target_transform)

                metrics_raw = compute_metrics_with_r(y_true, y_pred)
                row = {"source": src, "target": tgt,
                       "rmse_raw": metrics_raw["rmse"], "mae_raw": metrics_raw["mae"],
                       "r2_raw": metrics_raw["r2"], "mape_raw": metrics_raw["mape_percent"],
                       "r_raw": metrics_raw["pearson_r"]}
                if calibrate_mean:
                    y_pred_mean = np.mean(y_pred)
                    y_true_mean = np.mean(y_true)
                    y_pred_cal = y_pred - y_pred_mean + y_true_mean
                    metrics_cal = compute_metrics_with_r(y_true, y_pred_cal)
                    row.update({"rmse_cal": metrics_cal["rmse"], "mae_cal": metrics_cal["mae"],
                                "r2_cal": metrics_cal["r2"], "mape_cal": metrics_cal["mape_percent"],
                                "r_cal": metrics_cal["pearson_r"]})
                    matrix_r2_cal.loc[src, tgt] = metrics_cal["r2"]
                    matrix_rmse_cal.loc[src, tgt] = metrics_cal["rmse"]
                    matrix_mae_cal.loc[src, tgt] = metrics_cal["mae"]
                    matrix_r_cal.loc[src, tgt] = metrics_cal["pearson_r"]
                metrics_rows.append(row)
                matrix_r2_raw.loc[src, tgt] = metrics_raw["r2"]
                matrix_rmse_raw.loc[src, tgt] = metrics_raw["rmse"]
                matrix_mae_raw.loc[src, tgt] = metrics_raw["mae"]
                matrix_r_raw.loc[src, tgt] = metrics_raw["pearson_r"]
                print(f"    R2_raw={metrics_raw['r2']:.3f}, RMSE_raw={metrics_raw['rmse']:.2f}" +
                      (f", R2_cal={metrics_cal['r2']:.3f}" if calibrate_mean else ""))

    # 保存结果
    suffix = f"_{mode}"
    pd.DataFrame(metrics_rows).to_csv(out_dir / f"transfer_metrics{suffix}.csv", index=False)
    matrix_r2_raw.to_csv(out_dir / f"transfer_matrix_r2{suffix}.csv")
    matrix_rmse_raw.to_csv(out_dir / f"transfer_matrix_rmse{suffix}.csv")
    matrix_mae_raw.to_csv(out_dir / f"transfer_matrix_mae{suffix}.csv")
    matrix_r_raw.to_csv(out_dir / f"transfer_matrix_r{suffix}.csv")

    if calibrate_mean:
        matrix_r2_cal.to_csv(out_dir / f"transfer_matrix_r2{suffix}_calibrated.csv")
        matrix_rmse_cal.to_csv(out_dir / f"transfer_matrix_rmse{suffix}_calibrated.csv")
        matrix_mae_cal.to_csv(out_dir / f"transfer_matrix_mae{suffix}_calibrated.csv")
        matrix_r_cal.to_csv(out_dir / f"transfer_matrix_r{suffix}_calibrated.csv")

    if embed_dist_list:
        df_emb = pd.DataFrame(embed_dist_list)
        df_emb.to_csv(out_dir / f"transfer_embedding_distances{suffix}.csv", index=False)
        print(f"  Saved embedding distances to transfer_embedding_distances{suffix}.csv")

    return (matrix_r2_raw, matrix_rmse_raw, matrix_mae_raw, matrix_r_raw,
            matrix_r2_cal, matrix_rmse_cal, matrix_mae_cal, matrix_r_cal)

# ---------- 主实验流程 ----------
def run_experiments(args: argparse.Namespace) -> None:
    pretrained_dir = Path(args.pretrained_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "mlp_input":
        all_cities = discover_mlp_input_cities(Path(args.input_root), args.input_feature)
    else:
        all_cities = discover_cities(Path(args.aef_root))
    source_cities = args.source_cities or args.cities or all_cities
    target_cities = args.target_cities or args.cities or all_cities
    print(f"Discovered input cities: {len(all_cities)}")
    print(f"Source cities: {len(source_cities)}")
    print(f"Target cities: {len(target_cities)}")

    # 加载城市数据
    city_cache = {}
    city_errors = {}
    for city in set(source_cities + target_cities):
        try:
            city_cache[city] = load_city(args, city)
            print(f"Loaded city: {city}")
        except Exception as e:
            city_errors[city] = str(e)
            print(f"Error loading {city}: {e}")

    # 加载预训练 MLP 模型
    mlp_models = {}
    available_models = scan_model_files(
        pretrained_dir,
        args.input_feature if args.input_mode == "mlp_input" else None,
    )
    print(f"Pretrained model dir: {pretrained_dir}")
    print(f"Available source models: {len(available_models)}")
    for src in source_cities:
        if src in available_models:
            try:
                model, x_scaler, y_mean, y_std, feat_cols, target_transform = load_pretrained_mlp(
                    available_models[src],
                    args.hidden_channels,
                    args.dropout,
                    torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"),
                )
                mlp_models[src] = {
                    'model': model,
                    'x_scaler': x_scaler,
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'feature_cols': feat_cols,
                    'target_transform': target_transform,
                }
                print(f"Loaded MLP model for {src}")
            except Exception as e:
                print(f"Failed to load MLP model for {src}: {e}")
                city_errors[src] = f"MLP load error: {e}"

    results = {}

    run_mlp_baseline = bool(mlp_models) and not args.null_model
    run_ridge = args.ridge
    run_null = args.null_model

    if mlp_models and not args.ridge:
        run_mlp_baseline = True

    if run_mlp_baseline:
        mlp_r2_raw, mlp_rmse_raw, mlp_mae_raw, mlp_r2_cal, _, _,_,_ = run_single_transfer(
            "mlp", mlp_models, source_cities, target_cities,
            city_cache, city_errors, args, out_dir,
            use_ridge=False, use_null=False,
            calibrate_mean=args.calibrate_mean
        )
        results["mlp"] = {"r2_raw": mlp_r2_raw, "rmse_raw": mlp_rmse_raw, "mae_raw": mlp_mae_raw,
                          "r2_cal": mlp_r2_cal}

    if run_ridge:
        ridge_r2_raw, ridge_rmse_raw, ridge_mae_raw, ridge_r2_cal, _, _,_,_ = run_single_transfer(
            "ridge", None, source_cities, target_cities,
            city_cache, city_errors, args, out_dir,
            use_ridge=True, use_null=False,
            calibrate_mean=args.calibrate_mean
        )
        results["ridge"] = {"r2_raw": ridge_r2_raw, "rmse_raw": ridge_rmse_raw, "mae_raw": ridge_mae_raw,
                            "r2_cal": ridge_r2_cal}

    if run_null:
        if mlp_models:
            null_r2_raw, null_rmse_raw, null_mae_raw, null_r2_cal, _, _,_,_ = run_single_transfer(
                "null", mlp_models, source_cities, target_cities,
                city_cache, city_errors, args, out_dir,
                use_ridge=False, use_null=True,
                calibrate_mean=args.calibrate_mean
            )
            results["null"] = {"r2_raw": null_r2_raw, "rmse_raw": null_rmse_raw, "mae_raw": null_mae_raw,
                               "r2_cal": null_r2_cal}
        else:
            print("Skipping null model because no MLP models are available.")

    if run_null and "null" in results and args.calibrate_mean:
        null_r2_cal = results["null"]["r2_cal"]
        if null_r2_cal is not None and not null_r2_cal.empty:
            vals = null_r2_cal.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                mean_r2 = np.mean(vals)
                std_r2 = np.std(vals)
                print(f"\n=== Null model CALIBRATED R2 summary: mean={mean_r2:.4f}, std={std_r2:.4f} ===")
                if np.abs(mean_r2) < 0.05:
                    print("Calibrated null R2 near zero, confirming mean bias is the dominant issue.")
                else:
                    print("Calibrated null R2 still significantly deviates from zero, indicating other biases.")

    with open(out_dir / "experiment_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nAll results saved to: {out_dir}")

# ---------- 命令行参数 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP transferability experiments with mean calibration.")
    parser.add_argument("--input-mode", choices=["mlp_input", "aef"], default="mlp_input",
                        help="mlp_input reads prepared 65-dim CSVs; aef rebuilds data from AEF/pop/shp.")
    parser.add_argument("--input-root", default=str(DEFAULT_MLP_INPUT_ROOT),
                        help="Root directory of prepared MLP_input CSVs.")
    parser.add_argument("--input-feature", default=DEFAULT_INPUT_FEATURE,
                        help="GIS feature name used in files like city__AEF64_plus_feature.csv.")
    parser.add_argument("--aef-root", default=str(DEFAULT_AEF_ROOT))
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV))
    parser.add_argument("--pretrained-dir", default=str(DEFAULT_PRETRAINED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--source-cities", nargs="+", default=None)
    parser.add_argument("--target-cities", nargs="+", default=None)

    parser.add_argument("--ridge", action="store_true", help="Run Ridge regression baseline")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--null-model", action="store_true", help="Run feature shuffling null model")
    parser.add_argument("--null-seed", type=int, default=123)

    parser.add_argument("--hidden-channels", type=int, default=64, help="MLP hidden channels (must match trained model)")
    parser.add_argument("--dropout", type=float, default=0.5, help="MLP dropout (must match trained model)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--target-transform", choices=["log1p", "raw"], default="log1p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--calibrate-mean", action="store_true",
                        help="Calibrate predictions by shifting mean to match target mean (diagnostic)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
