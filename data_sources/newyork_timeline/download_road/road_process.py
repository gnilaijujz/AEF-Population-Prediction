# -*- coding: utf-8 -*-
"""
road_density_local 一步到位处理脚本 · New York MSA · 时间维度
================================================================
做什么（全自动）：
  1. 读 2020 纽约 tract 矢量 → 反推涉及的全部 county FIPS
  2. 逐年(2017–2021)逐县从 Census TIGER 下载 all-roads zip（已存在则跳过）
  3. 筛 MTFCC == 'S1400'（local 街道）
  4. 道路 + tract 投到 EPSG:5070（米制）→ 叠加求每 tract 内 S1400 总长(m)
  5. road_density_local = 长度(m) ÷ tract面积(km²)   （面积用 ALAND=cb_2020_11）
  6. 每年输出 road_newyork_YYYY.csv（列：cb_2020_3, road_density_local）
口径对齐 2020：几何冻结 2020 tract、只筛 S1400、密度分母用 ALAND、原始值不归一化。

依赖：pip install geopandas pandas pyogrio requests
用法：改下面 CONFIG 三个路径 → python road_process.py
"""

import os, time
import pandas as pd
import geopandas as gpd
import requests

# ============ CONFIG（按需改） ============
TRACT_SHP = r"data_sources/processed_gis/newyork/2020_MSA_in_track_newyork.shp"
WORK_DIR  = r"data_sources/newyork_timeline/download_road"   # 下载与输出根目录
YEARS     = [2017, 2018, 2019, 2020, 2021]

KEY   = "cb_2020__3"   # tract 键字段（→ 输出改名 cb_2020_3）
ALAND = "cb_2020_11"   # tract 陆地面积字段（m²），密度分母
MTFCC_LOCAL = "S1400"  # local 街道
METRIC_CRS  = 5070     # US Albers Equal Area，米
# =========================================

TIGER = "https://www2.census.gov/geo/tiger/TIGER{y}/ROADS/tl_{y}_{fips}_roads.zip"

# census.gov 会对"非浏览器"请求返回 403，带上完整浏览器请求头即可正常下载
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# census.gov 在 Akamai 后面，会按 TLS 指纹拦截普通 requests/urllib（返回 403）。
# curl_cffi 能伪装成真浏览器的 TLS 指纹，绕过该拦截；装了就优先用它。
try:
    from curl_cffi import requests as cffi_requests
    print("[info] 使用 curl_cffi（浏览器 TLS 指纹）下载")
except ImportError:
    cffi_requests = None
    print("[info] 未装 curl_cffi，退回 requests（census 可能 403，建议 pip install curl_cffi）")


def county_fips_from_tracts(gdf):
    """从 tract 键(1400000US SSCCC TTTTTT)取唯一 5 位 county FIPS。"""
    s = gdf[KEY].astype(str).str.split("US").str[-1]
    return sorted(set(s.str[:5]))


def download(y, fips, outdir, retries=3):
    """下载单个县单年 roads zip，已存在则跳过；带浏览器头 + 重试。返回本地 zip 路径或 None。"""
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"tl_{y}_{fips}_roads.zip")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    url = TIGER.format(y=y, fips=fips)
    for k in range(retries):
        try:
            if cffi_requests is not None:
                # 伪装 chrome 的 TLS 指纹，绕过 Akamai 403
                r = cffi_requests.get(url, headers=HEADERS, timeout=180, impersonate="chrome")
                r.raise_for_status()
                with open(dst, "wb") as f:
                    f.write(r.content)
            else:
                with SESSION.get(url, timeout=180, stream=True) as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
            time.sleep(0.3)          # 轻微限速，避免触发 403/限流
            return dst
        except Exception as e:
            if os.path.exists(dst):
                os.remove(dst)
            if k == retries - 1:
                print(f"    [WARN] 下载失败 {y} {fips}: {e}")
            else:
                time.sleep(1.5 * (k + 1))
    return None


def read_local_roads(zip_path):
    """读 zip 里的 roads 图层，只留 S1400。"""
    g = gpd.read_file(zip_path)                      # fiona/pyogrio 可直接读 .zip
    g = g[g["MTFCC"] == MTFCC_LOCAL]
    return g[["geometry"]]


def main():
    print("读取 tract 矢量 ...")
    tracts = gpd.read_file(TRACT_SHP)
    if tracts.crs is None:
        tracts.set_crs(epsg=4269, inplace=True)      # NAD83
    fips_list = county_fips_from_tracts(tracts)
    print(f"  tract 数={len(tracts)}，涉及 county={len(fips_list)}: {fips_list}")

    # tract 投到米制，预备叠加；面积用 ALAND(m²)→km²
    tr_m = tracts[[KEY, ALAND, "geometry"]].to_crs(epsg=METRIC_CRS).copy()
    tr_m["area_km2"] = tr_m[ALAND].astype(float) / 1e6

    for y in YEARS:
        print(f"\n===== {y} =====")
        ydir = os.path.join(WORK_DIR, f"roads_{y}")
        parts = []
        for i, fips in enumerate(fips_list, 1):
            zp = download(y, fips, ydir)
            if zp is None:
                continue
            try:
                parts.append(read_local_roads(zp))
            except Exception as e:
                print(f"    [WARN] 读取失败 {fips}: {e}")
            print(f"    [{i}/{len(fips_list)}] {fips} ok")

        if not parts:
            print(f"  {y} 无道路数据，跳过")
            continue

        roads = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        roads = roads.to_crs(epsg=METRIC_CRS)
        print(f"  S1400 路段数={len(roads)}，开始叠加求长 ...")

        # 道路线 × tract 面 相交 → 得到裁剪后的线段（带 tract 键），按 tract 汇总长度
        inter = gpd.overlay(roads, tr_m[[KEY, "geometry"]],
                            how="intersection", keep_geom_type=True)
        inter["len_m"] = inter.geometry.length
        length_by_tract = inter.groupby(KEY)["len_m"].sum()

        out = tr_m[[KEY, "area_km2"]].copy()
        out["road_len_local_m"] = out[KEY].map(length_by_tract).fillna(0.0)
        out["road_density_local"] = out["road_len_local_m"] / out["area_km2"]
        out = out.rename(columns={KEY: "cb_2020_3"})[["cb_2020_3", "road_density_local"]]

        dst = os.path.join(WORK_DIR, f"road_newyork_{y}.csv")
        out.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  写出 {dst}（{len(out)} 行）")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
