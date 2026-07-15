# -*- coding: utf-8 -*-
"""
只对 tracts_in_15msa_over100.csv 里的 15 个 MSA 的 tract 计算 POI 密度(2020)。
几何来自 2020_MSA_in_track.shp（按 cb_2020_3 匹配），POI 用 ohsome。
方法：POST /elements/count/groupBy/boundary → 每tract计数 ÷ 陆地面积(ALAND) = 个/km²
输出：poi_features_15msa.csv
依赖：pyshp + 标准库；断点续跑、失败重试、礼貌限速
"""
import shapefile, json, csv, os, time, math, urllib.request, urllib.parse
try:
    from shapely.geometry import shape, mapping   # 需要: pip install shapely
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False
    print("提示: 未装 shapely，几何不简化(复杂海岸tract可能仍500)。装它: pip install shapely")
SIMPLIFY_TOL = 0.0003   # 几何简化容差(度,约33m)，只为减少海岸线等超多顶点，几乎不影响POI计数

BASE = os.path.dirname(os.path.abspath(__file__))
SHP    = os.path.join(BASE, "数据准备", "行政区划", "pop", "2020_MSA_in_track")
TARGET = os.path.join(BASE, "tracts_in_15msa_over100.csv")   # 目标 tract 名单
OUT    = os.path.join(BASE, "poi_features_15msa.csv")
URL  = "https://api.ohsome.org/v1/elements/count/groupBy/boundary"
TIME = "2020-01-01"
BATCH = 10          # 每批tract数(再调小→降低单次POST体积,避开复杂几何导致的500)
SLEEP = 1.2         # 每次请求间隔(秒)
RETRY = 6           # 失败重试次数(配指数退避,扛住短暂网络波动)
TEST_LIMIT = None      # 先测试可设 200；正式跑设 None

# 多机分工：两台机器都设 NUM_WORKERS=2；A机 WORKER_ID=0，B机 WORKER_ID=1（单机就填1/0）
NUM_WORKERS = 2
WORKER_ID   = 1
if NUM_WORKERS > 1:
    OUT = os.path.join(BASE, f"poi_features_15msa_w{WORKER_ID}.csv")

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
            wait = min(60, 5 * (2 ** attempt))     # 指数退避 5,10,20,40,60,60...
            print(f"    请求失败(第{attempt+1}/{RETRY}次): {e} → {wait}s后重试")
            time.sleep(wait)
    return None                                     # 彻底失败→返回None，不崩溃

# 1) 目标 tract 名单
target = set()
for row in csv.DictReader(open(TARGET, encoding="utf-8-sig")):
    target.add(row["cb_2020_3"].strip())
print("目标 tract:", len(target))

# 2) 断点续跑
done = set()
if os.path.exists(OUT):
    for row in csv.DictReader(open(OUT, encoding="utf-8-sig")):
        done.add(row["cb_2020_3"])
    print(f"已完成 {len(done)}，将跳过")

# 3) 从 shapefile 取这些 tract 的几何 + 陆地面积；多机=范围切分
r = shapefile.Reader(SHP)
flds = [f[0] for f in r.fields[1:]]
i_id, i_land = flds.index("cb_2020__3"), flds.index("cb_2020_11")

# 目标tract在shapefile里的总数(用于范围切分,两机按同一顺序对齐)
total = sum(1 for rec in r.records() if str(rec[i_id]) in target)
chunk = -(-total // NUM_WORKERS)            # 向上取整
LO = WORKER_ID * chunk
HI = min(LO + chunk, total)
if NUM_WORKERS > 1:
    print(f"目标(有几何) {total}；本机(worker {WORKER_ID}) 负责序号 [{LO}, {HI})")

tracts = []
idx = 0
for sr in r.iterShapeRecords():
    tid = str(sr.record[i_id])
    if tid not in target:
        continue
    i = idx; idx += 1
    if i < LO or i >= HI:        # 不在本机负责的范围
        continue
    if tid in done:             # 断点续跑
        continue
    try:
        aland = float(sr.record[i_land])
    except Exception:
        aland = 0
    geom = sr.shape.__geo_interface__
    if HAVE_SHAPELY:                      # 简化几何:抽稀顶点,大幅缩小payload,避免500
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)              # 修复自相交等无效几何
        g = g.simplify(SIMPLIFY_TOL, preserve_topology=True)
        geom = mapping(g)
    tracts.append((tid, aland, geom))
    if TEST_LIMIT and len(tracts) >= TEST_LIMIT:
        break
print("待处理 tract:", len(tracts))

# 4) 分批跑
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
        if res is None:                 # 该请求彻底失败
            batch_ok = False; break
        for t in batch:
            counts[t[0]][cat] = res.get(t[0], 0)
        time.sleep(SLEEP)
    res_all = post_ohsome(feats, ALL_FILTER) if batch_ok else None
    if batch_ok and res_all is None:
        batch_ok = False
    if not batch_ok:                    # 本批失败：跳过,不写,重跑时靠续跑补
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
