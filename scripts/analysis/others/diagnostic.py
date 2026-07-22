#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnostic.py

Diagnostic experiments to identify the source of negative transfer R² values:
1. Global standardization vs. per-MSA standardization.
2. Ridge regression (no graph, no nonlinearity).
3. Null model: shuffle target features to break feature-label relationship.

Each experiment can be run separately via command-line arguments.
Results are saved as CSV matrices with mean and std (if repeats > 1).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch_geometric.data import Data  # 修复：导入 Data

# 导入现有模块中的函数
import GNN_regression as gnn
from GNN_transfer import load_city, prepare_target_data, predict_density, GraphSAGERegressor

# 配置
DEFAULT_AEF_ROOT = Path(r"data_sources/processed_aef")
DEFAULT_POP_CSV = Path(r"data_sources/msa_reference/raw_msa_data/ACSDT5Y2020.pop/ACSDT5Y2020.B01003-Data.csv")
DEFAULT_OUT_DIR = Path("results/diagnostics")


def get_city_data(city: str, aef_root: Path, pop_csv: Path, cache: dict):
    """加载城市数据并缓存。返回 (gdf, feature_cols, edge_index, x_raw, y_density)"""
    if city in cache:
        return cache[city]
    gdf, feature_cols, edge_index = load_city(aef_root, city, pop_csv)
    x_raw = gdf[feature_cols].to_numpy(dtype=np.float32)
    y_density = gdf["population_density"].to_numpy(dtype=np.float32)
    result = (gdf, feature_cols, edge_index, x_raw, y_density)
    cache[city] = result
    return result


def build_global_scaler(cities, aef_root, pop_csv):
    """合并所有城市的特征，拟合全局 StandardScaler。"""
    all_x = []
    cache = {}
    for city in cities:
        _, _, _, x_raw, _ = get_city_data(city, aef_root, pop_csv, cache)
        all_x.append(x_raw)
    all_x = np.vstack(all_x)
    scaler = StandardScaler()
    scaler.fit(all_x)
    return scaler


def run_transfer_experiment(args, global_scaler=None):
    """
    执行迁移实验，返回 R² 矩阵。
    根据 args 中的开关选择实验类型：
        - args.global_std: 使用全局标准化（否则使用源城市的 per-MSA scaler）
        - args.ridge: 使用 Ridge 回归（否则使用 GNN）
        - args.shuffle: 在预测前打乱目标特征（列打乱）
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    city_cache = {}

    # 获取城市列表
    aef_root = Path(args.aef_root)
    pop_csv = Path(args.pop_csv)
    source_cities = args.source_cities or args.cities or gnn.discover_cities(aef_root)
    target_cities = args.target_cities or args.cities or gnn.discover_cities(aef_root)

    # 加载所有城市数据
    all_city_data = {}
    for city in set(source_cities + target_cities):
        gdf, feat_cols, edge_idx, x_raw, y_density = get_city_data(city, aef_root, pop_csv, city_cache)
        all_city_data[city] = {
            'gdf': gdf,
            'feature_cols': feat_cols,
            'edge_index': edge_idx,
            'x_raw': x_raw,
            'y_density': y_density,
        }

    r2_matrix = pd.DataFrame(index=source_cities, columns=target_cities, dtype=float)

    for src in source_cities:
        src_data = all_city_data[src]
        src_feat = src_data['feature_cols']
        src_x = src_data['x_raw']
        src_y = src_data['y_density']

        # ----- 准备模型（Ridge 或 GNN） -----
        if args.ridge:
            # Ridge 回归，使用源城市全部数据训练
            model = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
            model.fit(src_x, src_y)

            def predictor_ridge(target_x):
                return model.predict(target_x)
        else:
            # 加载预训练的 GNN 模型
            pretrained_dir = Path(args.pretrained_dir)
            model_files = list(pretrained_dir.glob(f"{src}*_model.pt"))
            if not model_files:
                print(f"Warning: No model file found for {src}, skipping.")
                continue
            model_path = model_files[0]
            checkpoint = torch.load(model_path, map_location='cpu')
            y_mean = checkpoint['target_mean']
            y_std = checkpoint['target_std']
            feature_cols = checkpoint['feature_cols']
            if feature_cols != src_feat:
                print(f"Warning: Feature columns mismatch for {src}. Skipping.")
                continue

            # 获取 scaler
            if args.global_std:
                if global_scaler is None:
                    raise ValueError("Global scaler not provided for global_std experiment.")
                scaler = global_scaler
            else:
                scaler = checkpoint.get('scaler')
                if scaler is None:
                    print(f"Warning: Model for {src} has no scaler. Skipping.")
                    continue

            in_channels = len(feature_cols)
            model = GraphSAGERegressor(in_channels, args.hidden_channels, args.dropout).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            def predictor_gnn(target_x, target_edge):
                # 标准化特征
                x_norm = scaler.transform(target_x)
                x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)
                edge_idx = target_edge.to(device)
                data = Data(x=x_tensor, edge_index=edge_idx)  # 使用 Data
                with torch.no_grad():
                    pred_scaled = model(data.x, data.edge_index).cpu().numpy()
                pred_density = gnn.inverse_standardize(pred_scaled, y_mean, y_std)
                return pred_density

        # ----- 对每个目标城市预测 -----
        for tgt in target_cities:
            tgt_data = all_city_data[tgt]
            tgt_x = tgt_data['x_raw']
            tgt_y = tgt_data['y_density']
            tgt_edge = tgt_data['edge_index']

            # 特征处理（打乱或保持）
            if args.shuffle:
                # 列打乱：随机置换特征列
                cols = tgt_x.shape[1]
                perm_cols = np.random.permutation(cols)
                tgt_x_processed = tgt_x[:, perm_cols]
            else:
                tgt_x_processed = tgt_x

            # 预测
            if args.ridge:
                y_pred = predictor_ridge(tgt_x_processed)
            else:
                y_pred = predictor_gnn(tgt_x_processed, tgt_edge)

            # 计算 R²
            r2 = r2_score(tgt_y, y_pred)
            r2_matrix.loc[src, tgt] = r2

    return r2_matrix


def main():
    parser = argparse.ArgumentParser(description="Diagnostic experiments for transfer R².")
    parser.add_argument("--aef-root", default=str(DEFAULT_AEF_ROOT))
    parser.add_argument("--pop-csv", default=str(DEFAULT_POP_CSV))
    parser.add_argument("--pretrained-dir", default="outputs", help="Directory with pre-trained GNN models.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--source-cities", nargs="+", default=None, help="Sources to evaluate.")
    parser.add_argument("--target-cities", nargs="+", default=None, help="Targets to evaluate.")
    parser.add_argument("--cities", nargs="+", default=None, help="Use same list for sources and targets.")
    # 实验开关
    parser.add_argument("--global-std", action="store_true", help="Use global standardization instead of per-MSA.")
    parser.add_argument("--ridge", action="store_true", help="Use Ridge regression instead of GNN.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Regularization strength for Ridge.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle target features (column-wise) before prediction.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats for confidence intervals.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="GNN hidden channels (must match training).")
    parser.add_argument("--dropout", type=float, default=0.5, help="GNN dropout (must match training).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # 确定城市列表
    aef_root = Path(args.aef_root)
    all_cities = gnn.discover_cities(aef_root)
    if args.cities:
        source_cities = target_cities = args.cities
    else:
        source_cities = args.source_cities or all_cities
        target_cities = args.target_cities or all_cities

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 如果是全局标准化，预计算全局 scaler
    global_scaler = None
    if args.global_std:
        print("Building global scaler from all cities...")
        global_scaler = build_global_scaler(source_cities + target_cities, aef_root, Path(args.pop_csv))
        with open(out_dir / "global_scaler.pkl", "wb") as f:
            pickle.dump(global_scaler, f)

    # 重复实验
    r2_matrices = []
    for rep in range(args.repeats):
        if args.repeats > 1:
            current_seed = args.seed + rep
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
        else:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        print(f"\n=== Repeat {rep+1}/{args.repeats} ===")
        r2_matrix = run_transfer_experiment(args, global_scaler)
        r2_matrices.append(r2_matrix)

    # 汇总结果
    if args.repeats == 1:
        r2_matrices[0].to_csv(out_dir / "r2_matrix.csv")
        print(f"Saved R² matrix to {out_dir / 'r2_matrix.csv'}")
    else:
        all_sources = r2_matrices[0].index
        all_targets = r2_matrices[0].columns
        mean_df = pd.DataFrame(index=all_sources, columns=all_targets, dtype=float)
        std_df = pd.DataFrame(index=all_sources, columns=all_targets, dtype=float)
        for src in all_sources:
            for tgt in all_targets:
                vals = [m.loc[src, tgt] for m in r2_matrices if not np.isnan(m.loc[src, tgt])]
                if vals:
                    mean_df.loc[src, tgt] = np.mean(vals)
                    std_df.loc[src, tgt] = np.std(vals) if len(vals) > 1 else 0.0
        mean_df.to_csv(out_dir / "r2_mean.csv")
        std_df.to_csv(out_dir / "r2_std.csv")
        print(f"Saved mean and std R² matrices to {out_dir}")

    with open(out_dir / "diagnostic_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()