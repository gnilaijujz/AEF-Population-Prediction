#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN_transfer_experiments.py (with mean calibration)
支持 --calibrate-mean 选项，用于诊断均值偏移。
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from scipy.spatial.distance import cdist
import os
import GNN_regression as gnn


DEFAULT_AEF_ROOT = Path(r"model_data/aef_root/clean_aef_shapefiles")
DEFAULT_POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
DEFAULT_PRETRAINED_DIR = Path(r"validation\data\pretrained_models")
DEFAULT_OUT_DIR = Path(r"validation\results\transfer_results\mean")
GIS_COLS = ['']   # 您要添加的 GIS 特征列
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)


from scipy.linalg import eigh, fractional_matrix_power
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.stats import wasserstein_distance

def compute_coral(source_emb, target_emb):
    """计算 CORAL 距离：协方差矩阵差的 Frobenius 范数 / (4*d^2)"""
    d = source_emb.shape[1]
    cov_s = np.cov(source_emb, rowvar=False) + 1e-6 * np.eye(d)
    cov_t = np.cov(target_emb, rowvar=False) + 1e-6 * np.eye(d)
    diff = cov_s - cov_t
    coral = np.linalg.norm(diff, ord='fro') ** 2 / (4 * d * d)
    return float(coral)

def compute_kl_gaussian(source_emb, target_emb):
    """
    计算两个多元高斯分布之间的 KL 散度：KL(N_s || N_t)
    假设嵌入服从高斯分布，使用样本均值和协方差（正则化）。
    """
    d = source_emb.shape[1]
    mu_s = np.mean(source_emb, axis=0)
    mu_t = np.mean(target_emb, axis=0)
    cov_s = np.cov(source_emb, rowvar=False) + 1e-6 * np.eye(d)
    cov_t = np.cov(target_emb, rowvar=False) + 1e-6 * np.eye(d)

    # 计算逆协方差（用伪逆防止奇异）
    inv_cov_t = np.linalg.pinv(cov_t)
    # 迹项
    trace_term = np.trace(inv_cov_t @ cov_s)
    # 二次项
    mean_diff = mu_t - mu_s
    quad_term = mean_diff @ inv_cov_t @ mean_diff
    # log det 项
    sign, logdet_s = np.linalg.slogdet(cov_s)
    sign, logdet_t = np.linalg.slogdet(cov_t)
    log_det_term = logdet_t - logdet_s

    kl = 0.5 * (trace_term + quad_term - d + log_det_term)
    return float(max(kl, 0))  # 防止数值误差

def compute_spectral_distance(edge_index_s, edge_index_t, num_nodes_s, num_nodes_t, k=None):
    """
    计算两个图之间的图谱距离（基于归一化拉普拉斯特征值的 Wasserstein 距离）。
    若 k 为 None，则取 min(num_nodes) 个特征值。
    """
    def get_laplacian_eigenvals(edge_index, num_nodes):
        # 构建邻接矩阵（无向）
        adj = np.zeros((num_nodes, num_nodes))
        for i, j in edge_index.t().numpy():
            adj[i, j] = 1
        # 对称化
        adj = (adj + adj.T) > 0
        adj = adj.astype(float)
        degree = np.sum(adj, axis=1)
        # 归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-12))
        L = np.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
        # 计算特征值（使用 eigh 或 eigsh）
        try:
            vals = eigh(L, eigvals_only=True)
        except:
            vals = np.linalg.eigvalsh(L)
        return np.sort(vals)

    vals_s = get_laplacian_eigenvals(edge_index_s, num_nodes_s)
    vals_t = get_laplacian_eigenvals(edge_index_t, num_nodes_t)
    if k is not None:
        vals_s = vals_s[:k]
        vals_t = vals_t[:k]
    # 使用 1D Wasserstein 距离（即排序后 L1 距离）
    dist = wasserstein_distance(vals_s, vals_t)
    return float(dist)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def discover_cities(aef_root: Path) -> List[str]:
    cities = []
    for city_dir in sorted(p for p in aef_root.iterdir() if p.is_dir()):
        has_aef = any(city_dir.glob("aef_*.csv"))
        has_shp = any(city_dir.glob("*.shp"))
        if has_aef and has_shp:
            cities.append(city_dir.name)
    return cities


def load_city(aef_root: Path, city: str, pop_csv: Path):
    aef_dir = gnn.resolve_aef_dir(str(aef_root), city, None)
    shp_path = gnn.resolve_shp_path(aef_dir, None)
    gdf, feature_cols = gnn.load_and_merge_data(aef_dir, pop_csv, shp_path)
    edge_index = gnn.build_edge_index(gdf)
    return gdf, feature_cols, edge_index


from sklearn.metrics.pairwise import rbf_kernel

def mmd_rbf(X, Y, gamma=0.5):
    """计算两个样本集之间的 Maximum Mean Discrepancy (MMD)，使用高斯核"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    n = X.shape[0]
    m = Y.shape[0]
    # 使用无偏估计：mean(K_XX) + mean(K_YY) - 2*mean(K_XY)
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return float(mmd)
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

def cka_linear(X, Y):
    """
    计算两个特征矩阵 X (n_samples, d1) 和 Y (n_samples, d2) 之间的线性 CKA。
    返回值为 [0, 1] 之间的相似度，1 表示表示结构完全相同。
    """
    def centering(K):
        """对核矩阵 K 进行中心化处理"""
        n = K.shape[0]
        unit = np.ones((n, n)) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    n = X.shape[0]
    # 计算线性核矩阵
    K = X @ X.T
    L = Y @ Y.T
    # 中心化核矩阵
    Kc = centering(K)
    Lc = centering(L)
    # 计算 HSIC
    hsic_xy = np.sum(Kc * Lc) / ((n - 1) ** 2)
    hsic_xx = np.sum(Kc * Kc) / ((n - 1) ** 2)
    hsic_yy = np.sum(Lc * Lc) / ((n - 1) ** 2)
    # 计算 CKA
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return hsic_xy / denom

def cosine_similarity_matrix(X, Y):
    """
    计算两个特征矩阵之间的平均余弦相似度。
    使用 scipy.spatial.distance.cdist 计算余弦距离，然后转换为相似度[reference:6]。
    """
    # cdist 计算的是余弦距离 (1 - 余弦相似度)[reference:7]
    cosine_distances = cdist(X, Y, metric='cosine')
    # 取所有向量对相似度的平均值
    return 1 - np.mean(cosine_distances)


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
    def forward_emb(self, x, edge_index):
        """返回第二层卷积后的嵌入（即图卷积后的特征）"""
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 不使用 dropout（保留特征）
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x  # shape: [num_nodes, hidden_channels]


def train_ridge_on_source(
    source_gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    target_transform: str,
    alpha: float = 1.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Ridge, StandardScaler, float, float]:
    train_mask, _, _ = gnn.make_masks(len(source_gdf), train_ratio, val_ratio, seed)
    idx_train = train_mask.numpy()
    x_raw = source_gdf[feature_cols].to_numpy(dtype=np.float32)[idx_train]
    y_raw = source_gdf["population_density"].to_numpy(dtype=np.float32)[idx_train]

    y_trans = gnn.transform_target(y_raw, target_transform)
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
    x_scaled = scaler.transform(x_raw)# 使用pt的scaler，对target数据进行标准化
    pred_scaled = ridge.predict(x_scaled)
    pred_model = gnn.inverse_standardize(pred_scaled, y_mean, y_std)
    return gnn.inverse_target(pred_model, target_transform)


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


def load_pretrained_gnn(
    model_path: Path,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
    allow_missing_scaler: bool = False,
) -> Tuple[torch.nn.Module, StandardScaler, float, float, List[str], str]:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    in_channels = len(checkpoint['feature_cols'])
    model = GraphSAGERegressor(in_channels, hidden_channels, dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_mean = checkpoint['target_mean']
    y_std = checkpoint['target_std']
    feature_cols = checkpoint['feature_cols']
    target_transform = checkpoint.get('target_transform', 'raw')

    x_scaler = checkpoint.get('scaler')
    if x_scaler is None:
        if allow_missing_scaler:
            warnings.warn("Scaler missing; using identity scaler (risky).")
            x_scaler = StandardScaler()
            # 无法拟合，只能使用空scaler，后续会报错，因此这里直接抛出异常更安全
            raise ValueError("Scaler missing in checkpoint. Please retrain with scaler saved or use --allow-missing-scaler and provide target data.")
        else:
            raise ValueError("No scaler in checkpoint.")
    return model, x_scaler, y_mean, y_std, feature_cols, target_transform


def prepare_target_data(
    gdf: gpd.GeoDataFrame,
    feature_cols: List[str],
    edge_index: torch.Tensor,
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
        y_density = y_density[perm]          # 同步打乱
        y_model = y_model[perm]              # 同步打乱

    x = torch.tensor(x_scaler.transform(x_raw), dtype=torch.float32)
    y_density = gdf["population_density"].to_numpy(dtype=np.float32)
    y_model = gnn.transform_target(y_density, target_transform)
    y = torch.tensor((y_model - y_mean) / y_std, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    return data, y_density


def run_single_transfer(
    mode: str,
    source_bundles: Dict,
    source_cities: List[str],
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
    执行一组迁移预测，返回 (r2_raw, rmse_raw, mae_raw, r_raw,
                             r2_cal, rmse_cal, mae_cal, r_cal)
    若 calibrate_mean=False，则校准矩阵为 None。
    同时，如果 mode 为 GNN（非Ridge），会计算并保存源-目标嵌入 MMD 距离。
    """
    print(f"\n=== Running mode: {mode} (ridge={use_ridge}, null={use_null}, calibrate={calibrate_mean}) ===")
    metrics_rows = []
    embed_dist_list = []   # 存储嵌入距离
    actual_sources = source_cities if use_ridge else list(source_bundles.keys())

    # 初始化矩阵（原始和校准）
    matrix_r2_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_rmse_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_mae_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    matrix_r_raw = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    if calibrate_mean:
        matrix_r_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_r2_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_rmse_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
        matrix_mae_cal = pd.DataFrame(index=actual_sources, columns=target_cities, dtype=float)
    else:
        matrix_r2_cal = matrix_rmse_cal = matrix_mae_cal = matrix_r_cal = None

    # 如果是 Ridge，动态训练每个源（不提取嵌入）
    if use_ridge:
        source_ridges = {}
        for src in source_cities:
            src_data = city_cache.get(src)
            if src_data is None:
                city_errors[src] = city_errors.get(src, "City data not loaded")
                continue
            src_gdf, src_feature_cols, _ = src_data
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
                tgt_gdf, tgt_feature_cols, _ = tgt_data
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
                # 计算指标
                metrics_raw = gnn.compute_metrics_with_r(y_true, y_pred)
                row = {"source": src, "target": tgt,
                       "rmse_raw": metrics_raw["rmse"], "mae_raw": metrics_raw["mae"],
                       "r2_raw": metrics_raw["r2"], "mape_raw": metrics_raw["mape_percent"],
                       "r_raw": metrics_raw["pearson_r"]}
                if calibrate_mean:
                    y_pred_mean = np.mean(y_pred)
                    y_true_mean = np.mean(y_true)
                    y_pred_cal = y_pred - y_pred_mean + y_true_mean
                    metrics_cal = gnn.compute_metrics_with_r(y_true, y_pred_cal)
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
        # GNN 模式（可能打乱）
        for src, src_bundle in source_bundles.items():
            print(f"\nSource (GNN): {src}")
            model = src_bundle['model']
            x_scaler = src_bundle['x_scaler']
            y_mean = src_bundle['y_mean']
            y_std = src_bundle['y_std']
            feat_cols = src_bundle['feature_cols']
            target_transform = src_bundle['target_transform']

            if args.target_transform and args.target_transform != target_transform:
                print(f"⚠️  Using model's transform: {target_transform}")

            # ---- 尝试提取源域嵌入（仅当源数据可用时） ----
            src_data = city_cache.get(src)   # 可能为 None
            if src_data is not None:
                src_gdf, src_feat_cols, src_edge_index = src_data
                # 构建源域数据（使用模型的标准化参数）
                src_data_obj, _ = prepare_target_data(
                    src_gdf,
                    feat_cols,
                    src_edge_index,
                    x_scaler,
                    y_mean,
                    y_std,
                    target_transform,
                    next(model.parameters()).device,
                    shuffle_features=False
                )
                with torch.no_grad():
                    src_emb = model.forward_emb(src_data_obj.x, src_data_obj.edge_index).detach().cpu().numpy()
                print(f"  Source embedding shape: {src_emb.shape}")
            else:
                src_emb = None
                src_edge_index = None
                print(f"  Source city {src} data not available; skipping embedding distance calculation.")

            for tgt in target_cities:
                print(f"  -> {tgt}")
                tgt_data = city_cache.get(tgt)
                if tgt_data is None:
                    metrics_rows.append({"source": src, "target": tgt, "error": city_errors.get(tgt, "target load failed")})
                    continue
                tgt_gdf, tgt_feature_cols, tgt_edge_index = tgt_data
                # 检查目标数据是否缺少源模型所需的特征列
                missing = set(feat_cols) - set(tgt_gdf.columns)
                if missing:
                    err = f"Missing features: {missing}"
                    metrics_rows.append({"source": src, "target": tgt, "error": err})
                    continue
                # 若列都存在，则忽略顺序差异，直接使用 feat_cols 顺序提取数据
                device = next(model.parameters()).device

                # ---- 提取目标域原始嵌入（用于距离计算，不打乱） ----
                data_orig, _ = prepare_target_data(
                    tgt_gdf,
                    feat_cols,
                    tgt_edge_index,
                    x_scaler,
                    y_mean,
                    y_std,
                    target_transform,
                    device,
                    shuffle_features=False
                )
                with torch.no_grad():
                    tgt_emb_orig = model.forward_emb(data_orig.x, data_orig.edge_index).detach().cpu().numpy()
                # 计算 MMD 距离
                dist = mmd_rbf(src_emb, tgt_emb_orig)

                # 1. L1 距离 (均值差)
                l1_dist = np.mean(np.abs(np.mean(src_emb, axis=0) - np.mean(tgt_emb_orig, axis=0)))

                # 2. L2 距离 (均值差)
                l2_dist = np.linalg.norm(np.mean(src_emb, axis=0) - np.mean(tgt_emb_orig, axis=0))

                # 3. 平均余弦相似度 (值越大表示越相似)
                cos_dist = np.mean(cdist(src_emb, tgt_emb_orig, metric='cosine'))
                cos_sim = 1 - np.nan_to_num(cos_dist, nan=1.0)  # 若 cos_dist 为 NaN
                # 更合理：若出现 NaN，可能是零向量，则相似度设为 0（完全不相似）
                if np.isnan(cos_sim):
                    cos_sim = 0.0

                # 4. 线性 CKA (值越大表示表示结构越相似)[reference:8]
                #cka_sim = cka_linear(src_emb, tgt_emb_orig)
                # 保存所有距离
                # ---- 计算额外的分布差异 ----
                # CORAL 距离（协方差差）
                coral_dist = compute_coral(src_emb, tgt_emb_orig)

                # KL 散度（高斯假设）
                kl_div = compute_kl_gaussian(src_emb, tgt_emb_orig)

                # 图谱距离（使用源和目标图的 edge_index）
                spectral_dist = compute_spectral_distance(
                    src_edge_index, tgt_edge_index,
                    len(src_gdf), len(tgt_gdf),
                    k=50  # 取前50个特征值，加速计算
                )

                # ---- 平均度差（修正） ----
                num_edges_s = src_edge_index.size(1) // 2
                num_edges_t = tgt_edge_index.size(1) // 2
                degree_s = (2 * num_edges_s) / len(src_gdf)
                degree_t = (2 * num_edges_t) / len(tgt_gdf)
                degree_diff = abs(degree_s - degree_t)

                # 平均聚类系数差（可选，这里略，可借助 networkx）

                embed_dist_list.append({
                    "Source": src,
                    "Target": tgt,
                    "L1_Dist": l1_dist,          # 均值 L1
                    "L2_Dist": l2_dist,          # 均值 L2
                    "Cos_Sim": cos_sim,          # 余弦相似度（越高越好）
                    "MMD": dist,                 # 核 MMD（越低越好）
                    "CORAL": coral_dist,         # 协方差差
                    "KL_Div": kl_div,            # 高斯 KL 散度
                    "Spectral_Dist": spectral_dist,  # 图谱距离
                    "Degree_Diff": degree_diff,      # 平均度差（可选）
                })
                # ---- 准备预测数据（可能打乱） ----
                data, y_true = prepare_target_data(
                    tgt_gdf,
                    feat_cols,
                    tgt_edge_index,
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
                    pred_scaled = model(data.x, data.edge_index).detach().cpu().numpy()
                pred_model = gnn.inverse_standardize(pred_scaled, y_mean, y_std)
                y_pred = gnn.inverse_target(pred_model, target_transform)

                # 指标计算
                metrics_raw = gnn.compute_metrics_with_r(y_true, y_pred)
                row = {"source": src, "target": tgt,
                       "rmse_raw": metrics_raw["rmse"], "mae_raw": metrics_raw["mae"],
                       "r2_raw": metrics_raw["r2"], "mape_raw": metrics_raw["mape_percent"],
                       "r_raw": metrics_raw["pearson_r"]}
                if calibrate_mean:
                    y_pred_mean = np.mean(y_pred)
                    y_true_mean = np.mean(y_true)
                    y_pred_cal = y_pred - y_pred_mean + y_true_mean
                    metrics_cal = gnn.compute_metrics_with_r(y_true, y_pred_cal)
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

    # 保存原始指标
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

    # ---- 保存嵌入距离 ----
    if len(embed_dist_list) > 0:
        df_emb = pd.DataFrame(embed_dist_list)
        df_emb.to_csv(out_dir / f"transfer_embedding_distances{suffix}.csv", index=False)
        print(f"  Saved embedding distances to transfer_embedding_distances{suffix}.csv")

    return (matrix_r2_raw, matrix_rmse_raw, matrix_mae_raw, matrix_r_raw,
            matrix_r2_cal, matrix_rmse_cal, matrix_mae_cal, matrix_r_cal)

def run_experiments(args: argparse.Namespace) -> None:
    aef_root = Path(args.aef_root)
    pop_csv = Path(args.pop_csv)
    pretrained_dir = Path(args.pretrained_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 先扫描模型文件，获取所有可用的城市名（从模型文件名中解析）
    available_models = scan_model_files(pretrained_dir)
    model_cities = list(available_models.keys())

    # 2. 发现数据目录中的城市（可能不全）
    all_cities = discover_cities(aef_root)
    source_cities = args.source_cities or args.cities or all_cities
    target_cities = args.target_cities or args.cities or all_cities

    # 3. 合并需要加载的城市：用户指定的源/目标 + 模型城市（确保模型城市数据可用）
    cities_to_load = set(source_cities) | set(target_cities) | set(model_cities)

    # 4. 加载所有城市数据
    city_cache = {}
    city_errors = {}
    for city in cities_to_load:
        try:
            city_cache[city] = load_city(aef_root, city, pop_csv)
            print(f"Loaded city: {city}")
        except Exception as e:
            city_errors[city] = str(e)
            print(f"Error loading {city}: {e}")

    # 5. 加载 GNN 模型（仅保留数据加载成功的城市）
    gnn_models = {}
    for src in model_cities:
        # 不再要求 src in city_cache，直接尝试加载模型
        if src in available_models:
            try:
                model, x_scaler, y_mean, y_std, feat_cols, target_transform = load_pretrained_gnn(
                    available_models[src],
                    args.hidden_channels,
                    args.dropout,
                    torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"),
                    allow_missing_scaler=args.allow_missing_scaler,
                )
                gnn_models[src] = {
                    'model': model,
                    'x_scaler': x_scaler,
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'feature_cols': feat_cols,
                    'target_transform': target_transform,
                }
                print(f"Loaded GNN model for {src}")
            except Exception as e:
                print(f"Failed to load GNN model for {src}: {e}")
                city_errors[src] = f"GNN load error: {e}"


    results = {}

    # 决定运行哪些实验
    run_gnn_baseline = bool(gnn_models) and not args.null_model  # 若只跑 null，依然需要 GNN 基线
    run_ridge = args.ridge
    run_null = args.null_model

    # 如果有 GNN 模型，且不是仅运行 Ridge（Ridge 独立），我们总是运行 GNN 基线以便对比
    if gnn_models and not args.ridge:
        run_gnn_baseline = True

    # 执行 GNN 基线（非打乱）
    if run_gnn_baseline:
        gnn_r2_raw, gnn_rmse_raw, gnn_mae_raw, gnn_r2_cal, _, _,_,_ = run_single_transfer(
            "gnn", gnn_models, source_cities, target_cities,
            city_cache, city_errors, args, out_dir,
            use_ridge=False, use_null=False,
            calibrate_mean=args.calibrate_mean
        )
        results["gnn"] = {"r2_raw": gnn_r2_raw, "rmse_raw": gnn_rmse_raw, "mae_raw": gnn_mae_raw,
                          "r2_cal": gnn_r2_cal}

    # 执行 Ridge（如果启用）
    if run_ridge:
        ridge_r2_raw, ridge_rmse_raw, ridge_mae_raw, ridge_r2_cal, _, _,_,_ = run_single_transfer(
            "ridge", None, source_cities, target_cities,
            city_cache, city_errors, args, out_dir,
            use_ridge=True, use_null=False,
            calibrate_mean=args.calibrate_mean
        )
        results["ridge"] = {"r2_raw": ridge_r2_raw, "rmse_raw": ridge_rmse_raw, "mae_raw": ridge_mae_raw,
                            "r2_cal": ridge_r2_cal}

    # 执行 Null 模型（如果启用）
    if run_null:
        if gnn_models:
            null_r2_raw, null_rmse_raw, null_mae_raw, null_r2_cal, _, _,_,_ = run_single_transfer(
                "null", gnn_models, source_cities, target_cities,
                city_cache, city_errors, args, out_dir,
                use_ridge=False, use_null=True,
                calibrate_mean=args.calibrate_mean
            )
            results["null"] = {"r2_raw": null_r2_raw, "rmse_raw": null_rmse_raw, "mae_raw": null_mae_raw,
                               "r2_cal": null_r2_cal}
        else:
            print("Skipping null model because no GNN models are available.")

    # 输出校准后零模型的摘要
    if run_null and "null" in results and args.calibrate_mean:
        null_r2_cal = results["null"]["r2_cal"]
        if null_r2_cal is not None and not null_r2_cal.empty:
            vals = null_r2_cal.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                mean_r2 = np.mean(vals)
                std_r2 = np.std(vals)
                print(f"\n=== Null model CALIBRATED R² summary: mean={mean_r2:.4f}, std={std_r2:.4f} ===")
                if np.abs(mean_r2) < 0.05:
                    print("✓ Calibrated null R² near zero, confirming mean bias is the dominant issue.")
                else:
                    print("⚠️  Calibrated null R² still significantly deviates from zero, indicating other biases.")

    # 保存运行参数
    with open(out_dir / "experiment_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nAll results saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiments for transferability analysis with mean calibration.")
    parser.add_argument("--aef-root", default=str(DEFAULT_AEF_ROOT))
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV))
    parser.add_argument("--pretrained-dir", default=str(DEFAULT_PRETRAINED_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--source-cities", nargs="+", default=None)
    parser.add_argument("--target-cities", nargs="+", default=None)

    parser.add_argument("--ridge", action="store_true", help="Run Ridge regression baseline (实验1.1)")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--null-model", action="store_true", help="Run spatial shuffling null model (实验1.2)")
    parser.add_argument("--null-seed", type=int, default=123)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--allow-missing-scaler", action="store_true")
    parser.add_argument("--target-transform", choices=["log1p", "raw"], default="log1p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # 新增校准选项
    parser.add_argument("--calibrate-mean", action="store_true",
                        help="Calibrate predictions by shifting mean to match target mean (diagnostic)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)