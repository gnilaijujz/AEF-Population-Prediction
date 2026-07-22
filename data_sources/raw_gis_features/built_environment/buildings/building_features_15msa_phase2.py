# -*- coding: utf-8 -*-
"""
建筑特征 第二轮：把 w0/w2/w1 之后剩余未完成的 tract 平均分给三台机器补齐。
- 已完成(w0/w2/w1 + 本机 r 文件)全部跳过,不重跑。
- 剩余按固定顺序均分为 3 份,本机只做 NEW_WORKER 那份,写 r{NEW_WORKER}.csv。
- 保留稳健逻辑：超时→二分、瞬时错误→退避重试、按需几何简化、断点续跑。
最后合并：merge 所有 w*/r* → building_features_15msa.csv
"""
import shapefile, json, csv, os, time, glob, urllib.request, urllib.parse
try:
    from shapely.geometry import shape, mapping
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False

BASE = os.path.dirname(os.path.abspath(__file__))
_SHP_CANDS = [os.path.join(BASE, "census_tracts", "2020_MSA_in_track"),
              os.path.join(BASE, "2020_MSA_in_track"),
              os.path.join(BASE, "数据准备", "行政区划", "pop", "2020_MSA_in_track")]
SHP = next((p for p in _SHP_CANDS if os.path.exists(p + ".shp")), _SHP_CANDS[0])
TARGET = os.path.join(BASE, "tracts_in_15msa_over100.csv")
URL_COUNT = "https://api.ohsome.org/v1/elements/count/groupBy/boundary"
URL_AREA  = "https://api.ohsome.org/v1/elements/area/groupBy/boundary"
FILTER = "building=* and geometry:polygon"
TIME = "2020-01-01"
BATCH = 8; SLEEP = 0.4; RETRY = 3; TIMEOUT = 90
SIMPLIFY_TOL = 0.0001; VERTEX_THRESHOLD = 1000

# ===== 三台各设一个：machine0→0, machine2→1, machine1→2 (随意,只要3台各不同) =====
NEW_WORKER = 0
N_NEW = 3
# 第一轮的三个结果文件(用于确定"已完成",本身不改)
PRE_FILES = ["building_features_15msa_w0.csv",
             "building_features_15msa_w2.csv",
             "building_features_15msa_w1.csv"]
OUT = os.path.join(BASE, f"building_features_15msa_r{NEW_WORKER}.csv")

def nverts(c):
    if not c: return 0
    if isinstance(c[0], (int, float)): return 1
    return sum(nverts(x) for x in c)

def post_ohsome(features, url, filt=FILTER):
    fc = {"type": "FeatureCollection", "features": features}
    body = urllib.parse.urlencode({"bpolys": json.dumps(fc), "filter": filt,
                                   "time": TIME, "format": "json"}).encode()
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(url, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.load(r)
            res = {str(g["groupByObject"]): g["result"][0]["value"]
                   for g in data.get("groupByResult", [])}
            time.sleep(SLEEP); return res
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return None
            wait = min(20, 3 * (2 ** attempt))
            print(f"    请求失败(第{attempt+1}/{RETRY}次): {e} → {wait}s后重试")
            time.sleep(wait)
    return None

# 目标名单
target = set(row["cb_2020_3"].strip()
             for row in csv.DictReader(open(TARGET, encoding="utf-8-sig")))

# 第一轮已完成(稳定,用于切分)
pre_done = set()
for f in PRE_FILES:
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            pre_done.add(row["cb_2020_3"])
print(f"第一轮已完成: {len(pre_done)}")

# 本机 r 文件已完成(断点续跑)
own_done = set()
if os.path.exists(OUT):
    for row in csv.DictReader(open(OUT, encoding="utf-8-sig")):
        own_done.add(row["cb_2020_3"])
    print(f"本机(r{NEW_WORKER})已完成: {len(own_done)}")

# tract 几何 + 陆地面积(按shapefile顺序)
r = shapefile.Reader(SHP)
flds = [f[0] for f in r.fields[1:]]
i_id, i_land = flds.index("cb_2020__3"), flds.index("cb_2020_11")
order = [(str(rec[i_id]), rec) for rec in r.records()]  # 占位,顺序对齐
# 剩余(固定顺序) = 目标 - 第一轮已完成
remaining_ids = [str(rec[i_id]) for rec in r.records()
                 if str(rec[i_id]) in target and str(rec[i_id]) not in pre_done]
print(f"剩余总数: {len(remaining_ids)}")

# 均分 3 份,取本机那份
chunk = -(-len(remaining_ids) // N_NEW)
LO, HI = NEW_WORKER * chunk, min((NEW_WORKER + 1) * chunk, len(remaining_ids))
my_ids = set(remaining_ids[LO:HI]) - own_done
print(f"本机(r{NEW_WORKER}) 负责 {remaining_ids and f'[{LO},{HI})'} → 待处理 {len(my_ids)}")

# 取这些 tract 的几何
tracts = []
for sr in r.iterShapeRecords():
    tid = str(sr.record[i_id])
    if tid not in my_ids:
        continue
    try: aland = float(sr.record[i_land])
    except Exception: aland = 0
    geom = sr.shape.__geo_interface__
    if HAVE_SHAPELY and nverts(geom.get("coordinates", [])) > VERTEX_THRESHOLD:
        g = shape(geom)
        if not g.is_valid: g = g.buffer(0)
        geom = mapping(g.simplify(SIMPLIFY_TOL, preserve_topology=True))
    tracts.append((tid, aland, geom))

# 输出
write_header = not os.path.exists(OUT)
fout = open(OUT, "a", newline="", encoding="utf-8-sig"); w = csv.writer(fout)
if write_header:
    w.writerow(["cb_2020_3", "building_count", "building_area_m2",
                "building_density", "coverage_ratio"])

def query_batch(batch):
    feats = [{"type": "Feature", "id": t[0], "properties": {"id": t[0]}, "geometry": t[2]} for t in batch]
    cnt = post_ohsome(feats, URL_COUNT)
    area = post_ohsome(feats, URL_AREA) if cnt is not None else None
    if cnt is not None and area is not None:
        return {t[0]: (cnt.get(t[0], 0), area.get(t[0], 0.0)) for t in batch}
    if len(batch) == 1:
        return {batch[0][0]: None}
    mid = len(batch) // 2
    print(f"    批({len(batch)})失败→二分")
    res = query_batch(batch[:mid]); res.update(query_batch(batch[mid:])); return res

failed = wrote = 0
for b in range(0, len(tracts), BATCH):
    batch = tracts[b:b + BATCH]
    res = query_batch(batch)
    for tid, aland, _ in batch:
        v = res.get(tid)
        if v is None: failed += 1; continue
        c, a = v; km2 = aland / 1e6 if aland else 0
        dens = round(c / km2, 6) if km2 > 0 else ""
        cov  = round(a / aland, 6) if aland > 0 else ""
        w.writerow([tid, c, round(a, 3), dens, cov]); wrote += 1
    fout.flush()
    print(f"  进度 {min(b+BATCH, len(tracts))}/{len(tracts)}  已写{wrote} 跳过{failed}")
fout.close()
print(f"完成 -> {OUT}  写入{wrote} 跳过{failed}(重跑可补)")
