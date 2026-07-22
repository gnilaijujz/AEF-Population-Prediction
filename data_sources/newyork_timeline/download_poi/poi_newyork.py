# -*- coding: utf-8 -*-
"""
纽约 MSA 单年份 POI 密度（ohsome）—— 改写自 中期前/poi_density_15msa.py
只跑纽约那 4941 个 tract（几何+陆地面积来自 data_gis_newyork 的 shapefile）。
方法：POST /elements/count/groupBy/boundary → 每 tract 计数 ÷ 陆地面积(ALAND) = 个/km²
输出：poi_features_newyork_<YEAR>.csv（列：cb_2020_3 + 7类密度 + poi_total_all + poi_diversity）

用法（每台机器/每个终端跑一个年份）：
    python poi_newyork.py 2017
    python poi_newyork.py 2018
    ...
可选：把同一年拆给多台机器同时跑（加“总份数 本机第几份”，从0数）：
    python poi_newyork.py 2017 3 0     # A机
    python poi_newyork.py 2017 3 1     # B机
    python poi_newyork.py 2017 3 2     # C机
断点续跑：中断后重跑同一条命令即可，已完成的 tract 自动跳过。
依赖：pyshp（pip install pyshp）；建议再装 shapely（pip install shapely）简化几何。
"""
import shapefile, json, csv, os, sys, time, math, urllib.request, urllib.parse

try:
    from shapely.geometry import shape, mapping
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False
    print("提示: 未装 shapely，几何不简化(复杂海岸 tract 可能仍 500)。装它: pip install shapely")

# ---------------- 参数 ----------------
YEAR   = sys.argv[1] if len(sys.argv) > 1 else "2017"
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1      # 同一年拆几份(默认1=不拆)
SHARD  = int(sys.argv[3]) if len(sys.argv) > 3 else 0      # 本机第几份(0起)

SIMPLIFY_TOL = 0.0003     # 几何简化容差(度,约33m)，几乎不影响 POI 计数
BATCH = 10                # 每批 tract 数
SLEEP = 1.2               # 每次请求间隔(秒)——多机/多终端并发时别调太小，礼貌限速
RETRY = 6                 # 失败重试次数(指数退避)
TEST_LIMIT = None         # 先测试可设 200；正式跑设 None

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)


def find_shp():
    """在几个常见位置自动查找 2020_MSA_in_track_newyork.shp，兼容不同机器的目录布局。"""
    name = "2020_MSA_in_track_newyork"
    candidates = [
        os.path.join(BASE, "data_gis_newyork", name),   # 脚本同级下的 data_gis_newyork（previous workstation layout）
        os.path.join(ROOT, "data_gis_newyork", name),   # 脚本上一级下的 data_gis_newyork（download_poi 布局）
        os.path.join(BASE, "data_aef_newyork", name),   # 也可用 aef 文件夹里那份（内容相同）
        os.path.join(ROOT, "data_aef_newyork", name),
        os.path.join(BASE, name),                        # 直接放在脚本同目录
    ]
    for c in candidates:
        if os.path.exists(c + ".shp"):
            return c
    raise FileNotFoundError(
        "找不到 2020_MSA_in_track_newyork.shp。请确认 data_gis_newyork 文件夹"
        "（含 .shp/.shx/.dbf/.prj）放在脚本同级或上一级目录。已查找:\n  " +
        "\n  ".join(c + ".shp" for c in candidates))


SHP  = find_shp()
URL  = "https://api.ohsome.org/v1/elements/count/groupBy/boundary"
TIME = f"{YEAR}-01-01"
OUT  = os.path.join(BASE, f"poi_features_newyork_{YEAR}.csv"
                    if NSHARD == 1 else f"poi_features_newyork_{YEAR}_s{SHARD}.csv")

CATS = {
    "poi_edu":     "(amenity in (school, college, university, kindergarten)) and geometry:point",
    "poi_comm":    "(shop=* or amenity=marketplace) and geometry:point",
    "poi_med":     "(amenity in (hospital, clinic, pharmacy, doctors)) and geometry:point",
    "poi_trans":   "(public_transport=* or railway=station or amenity=bus_station) and geometry:point",
    "poi_pub":     "(amenity in (police, fire_station, townhall, post_office, library)) and geometry:point",
    "poi_food":    "(amenity in (restaurant, cafe, fast_food, bar, pub, food_court)) and geometry:point",
    "poi_finance": "(amenity in (bank, atm, bureau_de_change)) and geometry:point",
}
ALL_FILTER = "(amenity=* or shop=* or office=* or leisure=* or tourism=*) and geometry:point"


def post_ohsome(features, filt):
    fc = {"type": "FeatureCollection", "features": features}
    body = urllib.parse.urlencode({"bpolys": json.dumps(fc), "filter": filt,
                                   "time": TIME, "format": "json"}).encode()
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(URL, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            out = {}
            for g in data.get("groupByResult", []):
                out[str(g["groupByObject"])] = g["result"][0]["value"]
            return out
        except Exception as e:
            wait = min(60, 5 * (2 ** attempt))
            print(f"    请求失败(第{attempt+1}/{RETRY}次): {e} → {wait}s后重试")
            time.sleep(wait)
    return None


print(f"年份={YEAR}  time={TIME}  分片={SHARD}/{NSHARD}  输出={os.path.basename(OUT)}")
print(f"使用 shapefile: {SHP}.shp")

# 断点续跑：已完成的 tract
done = set()
if os.path.exists(OUT):
    for row in csv.DictReader(open(OUT, encoding="utf-8-sig")):
        done.add(row["cb_2020_3"])
    print(f"已完成 {len(done)}，将跳过")

# 从 shapefile 取全部纽约 tract 的几何 + 陆地面积（4941 个）；可选按分片切分
r = shapefile.Reader(SHP)
flds = [f[0] for f in r.fields[1:]]
i_id, i_land = flds.index("cb_2020__3"), flds.index("cb_2020_11")

all_recs = list(r.iterShapeRecords())
total = len(all_recs)
chunk = -(-total // NSHARD)
LO, HI = SHARD * chunk, min(SHARD * chunk + chunk, total)
print(f"纽约 tract 总数 {total}；本片负责序号 [{LO}, {HI})")

tracts = []
for idx, sr in enumerate(all_recs):
    if idx < LO or idx >= HI:
        continue
    tid = str(sr.record[i_id])
    if tid in done:
        continue
    try:
        aland = float(sr.record[i_land])
    except Exception:
        aland = 0
    geom = sr.shape.__geo_interface__
    if HAVE_SHAPELY:
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        g = g.simplify(SIMPLIFY_TOL, preserve_topology=True)
        geom = mapping(g)
    tracts.append((tid, aland, geom))
    if TEST_LIMIT and len(tracts) >= TEST_LIMIT:
        break
print("待处理 tract:", len(tracts))

write_header = not os.path.exists(OUT)
fout = open(OUT, "a", newline="", encoding="utf-8-sig")
w = csv.writer(fout)
if write_header:
    w.writerow(["cb_2020_3"] + list(CATS.keys()) + ["poi_total_all", "poi_diversity"])

failed = 0
for b in range(0, len(tracts), BATCH):
    batch = tracts[b:b + BATCH]
    feats = [{"type": "Feature", "id": t[0], "properties": {"id": t[0]}, "geometry": t[2]} for t in batch]
    counts = {t[0]: {} for t in batch}
    batch_ok = True
    for cat, filt in CATS.items():
        res = post_ohsome(feats, filt)
        if res is None:
            batch_ok = False; break
        for t in batch:
            counts[t[0]][cat] = res.get(t[0], 0)
        time.sleep(SLEEP)
    res_all = post_ohsome(feats, ALL_FILTER) if batch_ok else None
    if batch_ok and res_all is None:
        batch_ok = False
    if not batch_ok:
        failed += len(batch)
        print(f"  ⚠ 本批失败已跳过(重跑会补): {b}~{min(b+BATCH,len(tracts))}")
        time.sleep(10)
        continue
    time.sleep(SLEEP)
    for tid, aland, _ in batch:
        km2 = aland / 1e6 if aland else 0
        row = [tid]; cat_counts = []
        for cat in CATS:
            c = counts[tid].get(cat, 0); cat_counts.append(c)
            row.append(round(c / km2, 6) if km2 > 0 else "")
        call = res_all.get(tid, 0)
        row.append(round(call / km2, 6) if km2 > 0 else "")
        tot = sum(cat_counts)
        H = -sum((c/tot) * math.log(c/tot) for c in cat_counts if c > 0) if tot > 0 else 0.0
        row.append(round(H, 6))
        w.writerow(row)
    fout.flush()
    print(f"  进度 {min(b+BATCH, len(tracts))}/{len(tracts)}")

fout.close()
print(f"完成 -> {OUT}   本轮跳过 {failed} 个(重跑即可补齐)")
