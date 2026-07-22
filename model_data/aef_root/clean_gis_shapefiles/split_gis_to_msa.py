# -*- coding: utf-8 -*-
"""
把标准化后的 9 个 GIS 特征按 MSA 拆分,输出到 processed_gis/<MSA>/,
结构与 processed_aef/<MSA>/ 一致(每个 MSA = 清理过的 shapefile + 一个特征 csv)。

对应关系:
  - 每个 MSA 的 tract 列表以 processed_aef/<MSA>/ 里【清理过的 shapefile】字段 cb_2020__3 为准
  - 特征来自 selected_9_features_15msa_standardized.csv 的 cb_2020_3(同为 1400000US 码)
  - 输出 csv 的键列命名为 TRACT_ID(与 AEF csv 一致,便于 AEF+GIS 融合)
以 shapefile 左连接特征:保证每个 shp tract 都有一行,缺失特征处为空。
"""
import os
import glob
import shutil
import pandas as pd
import geopandas as gpd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # model_data/aef_root/clean_gis_shapefiles/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_SOURCES_ROOT = os.path.join(PROJECT_ROOT, "data_sources")
AEF_ROOT = os.path.join(DATA_SOURCES_ROOT, "processed_aef")
STD_CSV = os.path.join(DATA_SOURCES_ROOT, "raw_gis_features", "selected_15_msa_features", "selected_9_features_15msa_standardized.csv")

SHP_KEY = "cb_2020__3"     # 清理过 shapefile 里的 tract 键(1400000US 码)
STD_KEY = "cb_2020_3"      # 标准化特征表的 tract 键
OUT_KEY = "TRACT_ID"       # 输出统一键名(对齐 AEF csv)

# 读标准化特征
std = pd.read_csv(STD_CSV, encoding="utf-8-sig", dtype={STD_KEY: str})
std.columns = [c.lstrip("﻿").strip() for c in std.columns]
feat_cols = [c for c in std.columns if c != STD_KEY]
std[STD_KEY] = std[STD_KEY].astype(str).str.strip()

# 找到 15 个 MSA 的 shapefile(含嵌套的 Chicago/Duluth)
shps = sorted(glob.glob(os.path.join(AEF_ROOT, "**", "2020*MSA_in_track_*.shp"), recursive=True))
print(f"找到 {len(shps)} 个 MSA shapefile\n")

summary = []
for shp in shps:
    msa = os.path.basename(os.path.dirname(shp))          # MSA 名 = shapefile 所在文件夹名
    base = shp[:-4]                                        # 去掉 .shp 的基名
    dst_dir = os.path.join(SCRIPT_DIR, msa)
    os.makedirs(dst_dir, exist_ok=True)

    # 1) 取 shp 的 tract 列表(仅属性,不载入几何,快)
    attrs = gpd.read_file(shp, ignore_geometry=True)
    if SHP_KEY not in attrs.columns:
        raise SystemExit(f"[{msa}] shapefile 缺少键字段 {SHP_KEY};现有字段: {list(attrs.columns)}")
    tracts = (attrs[[SHP_KEY]].rename(columns={SHP_KEY: OUT_KEY}).astype(str))
    tracts[OUT_KEY] = tracts[OUT_KEY].str.strip()
    tracts = tracts.drop_duplicates(subset=OUT_KEY)
    n_shp = len(tracts)

    # 2) 内连接特征(只保留既在 shp、又有特征的 tract)
    merged = tracts.merge(std, left_on=OUT_KEY, right_on=STD_KEY, how="inner")
    merged = merged[[OUT_KEY] + feat_cols]                # 只留键 + 9 特征
    keep = set(merged[OUT_KEY])

    # 3) 写 shapefile:先清掉目标目录旧的同名组件
    dst_base = os.path.join(dst_dir, os.path.basename(base))
    for old in glob.glob(dst_base + ".*"):
        os.remove(old)
    if len(keep) == n_shp:
        # 全部 tract 都有特征 -> 原样复制(保留 .qix/.qmd/.shp.xml 等全部边角文件)
        for comp in glob.glob(base + ".*"):
            shutil.copy2(comp, os.path.join(dst_dir, os.path.basename(comp)))
        shp_note = "复制"
    else:
        # 有无特征的 tract -> 按 keep 过滤几何后重写,使 shp 与 csv 一致
        gdf = gpd.read_file(shp)
        gdf = gdf[gdf[SHP_KEY].astype(str).str.strip().isin(keep)].copy()
        gdf.to_file(dst_base + ".shp", encoding="utf-8")
        for ext in (".qmd", ".shp.xml"):        # 附带描述性元数据(如有)
            if os.path.exists(base + ext):
                shutil.copy2(base + ext, dst_base + ext)
        shp_note = f"裁剪 {n_shp}->{len(gdf)}"

    # 4) 写特征 csv
    out_csv = os.path.join(dst_dir, f"gis_{msa}_2020.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary.append((msa, n_shp, len(keep)))
    print(f"  {msa:14s}  shp_tracts={n_shp:5d}  kept={len(keep):5d}  shp:{shp_note:12s} -> {os.path.relpath(out_csv, PROJECT_ROOT)}")

print("\n汇总:")
total_t = sum(s[1] for s in summary)
total_m = sum(s[2] for s in summary)
print(f"  15 个 MSA,shp tract 合计 {total_t},匹配到特征 {total_m}")

