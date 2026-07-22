#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""成果4 数据加载与通用工具（修正版：自预测得分从 full_scores_robust.csv 读取熵权法 Score）"""

import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

BASE_RESULT = r"results"

def get_config_paths(config_name):
    """根据配置名称返回对应文件路径"""
    transfer_dir_map = {
        'AEF': 'transferability/AEF',
        'Buildings': 'transferability/aef_plus_building',
        'GIS': 'transferability/GIS',
        'Impervious': 'transferability/impervious',
        'Road': 'transferability/roads',
        'Slope': 'transferability/slope',
        'Waterdis': 'transferability/water_distance',
        'POI_Diversity': 'transferability/poi_diversity',
    }
    domain_dir_map = {
        'AEF': 'domain_shift/aef',
        'Buildings': 'domain_shift/aef_plus_building/with_gis',
        'GIS': 'domain_shift/gis',
        'Impervious': 'domain_shift/aef_plus_impervious/with_gis',
        'Road': 'domain_shift/aef_plus_roads/with_gis',
        'Slope': 'domain_shift/aef_plus_slope/with_gis',
        'Waterdis': 'domain_shift/aef_plus_water_distance/with_gis',
        'POI_Diversity': 'domain_shift/aef_plus_poi_diversity/with_gis',
    }

    transfer_dir = os.path.join(BASE_RESULT, transfer_dir_map.get(config_name, 'transferability/AEF'))
    domain_dir = os.path.join(BASE_RESULT, domain_dir_map.get(config_name, 'domain_shift/aef'))

    cds_file = os.path.join(transfer_dir, "CDS_matrix_robust_entropy.csv")
    full_scores_file = os.path.join(transfer_dir, "full_scores_robust.csv")  # 包含熵权法综合得分

    if config_name == 'Buildings':
        l2_file = os.path.join(domain_dir, "domain_shift_Weighted_L2_matrix.csv")
        if not os.path.exists(l2_file):
            l2_file = os.path.join(domain_dir, "domain_shift_L2_Dist_matrix.csv")
    else:
        l2_file = os.path.join(domain_dir, "domain_shift_L2_Dist_matrix.csv")

    label_file = os.path.join(domain_dir, "domain_shift_Label_Shift_matrix.csv")
    if not os.path.exists(label_file):
        label_file = os.path.join(BASE_RESULT, "domain_shift/gis/domain_shift_Label_Shift_matrix.csv")

    return {
        'cds_path': cds_file,
        'full_scores_path': full_scores_file,
        'l2_path': l2_file,
        'label_path': label_file if os.path.exists(label_file) else None,
    }

def get_city_order(config_name='AEF'):
    paths = get_config_paths(config_name)
    cds_mat = pd.read_csv(paths['cds_path'], index_col=0)
    avg_cds = {}
    for city in cds_mat.index:
        vals = cds_mat.loc[city].drop(index=city, errors='ignore').dropna().values
        avg_cds[city] = np.mean(vals) if len(vals) > 0 else np.nan
    return pd.Series(avg_cds).sort_values().index.tolist()

def get_health_df(config_name='AEF'):
    """
    返回健康度指标 DataFrame，并依据当前配置计算：
      - avg_CDS: 跨域平均迁移损失
      - CDS_std: 跨域迁移损失标准差
      - Self_Score: 源域自预测综合得分（熵权法 Score，越大越好）
      - R2_self: (兼容旧名称，指向 Self_Score)
    """
    # 加载嵌入健康度指标
    health_path = os.path.join(BASE_RESULT, "embedding_health_deep_analysis", "embedding_health_deep_metrics.csv")
    health_df = pd.read_csv(health_path, index_col=0)

    # 加载跨域性能
    paths = get_config_paths(config_name)
    cds_mat = pd.read_csv(paths['cds_path'], index_col=0)

    avg_cds, cds_std = {}, {}
    for city in cds_mat.index:
        vals = cds_mat.loc[city].drop(index=city, errors='ignore').dropna().values
        avg_cds[city] = np.mean(vals) if len(vals) > 0 else np.nan
        cds_std[city] = np.std(vals) if len(vals) > 0 else np.nan

    # 读取自预测综合得分 (full_scores_robust.csv)
    scores_path = paths['full_scores_path']
    self_scores = {}
    if os.path.exists(scores_path):
        df_scores = pd.read_csv(scores_path)
        self_rows = df_scores[df_scores['source'] == df_scores['target']]
        for _, row in self_rows.iterrows():
            city = row['source']
            if city in health_df.index:
                self_scores[city] = row.get('Score', np.nan)
        if not self_scores:
            print(f"警告: 在 {scores_path} 中未找到 source==target 的 Score 数据")
    else:
        print(f"警告: 未找到完整得分文件 {scores_path}")

    health_df['avg_CDS'] = health_df.index.map(lambda x: avg_cds.get(x, np.nan))
    health_df['CDS_std'] = health_df.index.map(lambda x: cds_std.get(x, np.nan))
    health_df['Self_Score'] = health_df.index.map(lambda x: self_scores.get(x, np.nan))
    health_df['R2_self'] = health_df['Self_Score']  # 兼容旧名称

    return health_df.dropna(subset=['avg_CDS', 'Self_Score'])

def safe_linregress(x, y):
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return linregress(x, y)

def load_matrix(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df
