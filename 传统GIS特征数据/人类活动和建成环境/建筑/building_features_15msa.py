# -*- coding: utf-8 -*-
"""
对 tracts_in_15msa_over100.csv 里 15个MSA 的 tract 计算建筑特征(2020, OSM via ohsome)。
  building_count   = tract内建筑数(building=* 面)
  building_area_m2 = tract内建筑总占地面积(m²)
  building_density = building_count / tract陆地面积(km²)   个/km²
  coverage_ratio   = building_area_m2 / tract陆地面积(m²)  0~1
几何来自 2020_MSA_in_track.shp(按cb_2020_3匹配)。
输出：building_features_15msa.csv
依赖：pyshp + shapely(pip install pyshp shapely) + 标准库；断点续跑/重试/失败跳过
"""
import shapefile, json, csv, os, time, urllib.request, urllib.parse
try:
    from shapely.geometry import shape, mapping
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False
    print("提示: 未装 shapely，几何不简化(复杂tract可能500)。装它: pip install shapely")

BASE = os.path.dirname(os.path.abspath(__file__))
# 自动找 shapefile：优先嵌套目录，其次脚本同目录(远程机把3个文件放.py旁边即可)
_SHP_CANDS = [os.path.join(BASE, "数据准备", "行政区划", "pop", "2020_MSA_in_track"),
              os.path.join(BASE, "2020_MSA_in_track")]
SHP = next((p for p in _SHP_CANDS if os.path.exists(p + ".shp")), _SHP_CANDS[0])
TARGET = os.path.join(BASE, "tracts_in_15msa_over100.csv")
OUT    = os.path.join(BASE, "building_features_15msa.csv")
URL_COUNT = "https://api.ohsome.org/v1/elements/count/groupBy/boundary"
URL_AREA  = "https://api.ohsome.org/v1/elements/area/groupBy/boundary"
FILTER = "building=* and geometry:polygon"
TIME = "2020-01-01"
BATCH = 10
SLEEP = 1.2
RETRY = 6
SIMPLIFY_TOL = 0.0001     # ≈11m,仅对超复杂tract轻度抽稀
VERTEX_THRESHOLD = 1000   # 顶点数超过此值才简化;绝大多数tract保持原样(精确)
TEST_LIMIT = None          # 先测试;正式跑设 None

# 多机分工: 两台都设NUM_WORKERS=2; A机WORKER_ID=0, B机WORKER_ID=1 (单机填1/0)
NUM_WORKERS = 2
WORKER_ID   = 0
if NUM_WORKERS > 1:
    OUT = os.path.join(BASE, f"building_features_15msa_w{WORKER_ID}.csv")

def nverts(coords):
    """统计geojson几何的顶点数"""
    if not coords:
        return 0
    if isinstance(coords[0], (int, float)):
        return 1
    return sum(nverts(c) for c in coords)

def post_ohsome(features, url, filt=FILTER):
    fc = {"type": "FeatureCollection", "features": features}
    body = urllib.parse.urlencode({"bpolys": json.dumps(fc), "filter": filt,
                                   "time": TIME, "format": "json"}).encode()
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(url, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            return {str(g["groupByObject"]): g["result"][0]["value"]
                    for g in data.get("groupByResult", [])}
        except Exception as e:
            wait = min(60, 5 * (2 ** attempt))
            print(f"    请求失败(第{attempt+1}/{RETRY}次): {e} → {wait}s后重试")
            time.sleep(wait)
    return None

# 目标名单
target = set()
for row in csv.DictReader(open(TARGET, encoding="utf-8-sig")):
    target.add(row["cb_2020_3"].strip())
print("目标 tract:", len(target))

# 断点续跑
done = set()
if os.path.exists(OUT):
    for row in csv.DictReader(open(OUT, encoding="utf-8-sig")):
        done.add(row["cb_2020_3"])
    print(f"已完成 {len(done)}，将跳过")

# tract 几何 + 陆地面积
r = shapefile.Reader(SHP)
flds = [f[0] for f in r.fields[1:]]
i_id, i_land = flds.index("cb_2020__3"), flds.index("cb_2020_11")

total = sum(1 for rec in r.records() if str(rec[i_id]) in target)
chunk = -(-total // NUM_WORKERS)
LO, HI = WORKER_ID * chunk, min((WORKER_ID + 1) * chunk, total)
if NUM_WORKERS > 1:
    print(f"本机(worker {WORKER_ID}) 负责序号 [{LO}, {HI})")

tracts = []; idx = 0
for sr in r.iterShapeRecords():
    tid = str(sr.record[i_id])
    if tid not in target:
        continue
    i = idx; idx += 1
    if i < LO or i >= HI or tid in done:
        continue
    try:
        aland = float(sr.record[i_land])
    except Exception:
        aland = 0
    geom = sr.shape.__geo_interface__
    # 只对顶点超标的复杂tract(海岸线等)轻度简化;其余原样发送、精确
    if HAVE_SHAPELY and nverts(geom.get("coordinates", [])) > VERTEX_THRESHOLD:
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        geom = mapping(g.simplify(SIMPLIFY_TOL, preserve_topology=True))
    tracts.append((tid, aland, geom))
    if TEST_LIMIT and len(tracts) >= TEST_LIMIT:
        break
print("待处理 tract:", len(tracts))

# 分批跑
write_header = not os.path.exists(OUT)
fout = open(OUT, "a", newline="", encoding="utf-8-sig")
w = csv.writer(fout)
if write_header:
    w.writerow(["cb_2020_3", "building_count", "building_area_m2",
                "building_density", "coverage_ratio"])

failed = 0
for b in range(0, len(tracts), BATCH):
    batch = tracts[b:b + BATCH]
    feats = [{"type": "Feature", "id": t[0], "properties": {"id": t[0]}, "geometry": t[2]} for t in batch]
    cnt = post_ohsome(feats, URL_COUNT)
    time.sleep(SLEEP)
    area = post_ohsome(feats, URL_AREA) if cnt is not None else None
    if cnt is None or area is None:
        failed += len(batch)
        print(f"  ⚠ 本批失败已跳过(重跑会补): {b}~{min(b+BATCH,len(tracts))}")
        time.sleep(10); continue
    time.sleep(SLEEP)
    for tid, aland, _ in batch:
        c = cnt.get(tid, 0); a = area.get(tid, 0.0)
        km2 = aland / 1e6 if aland else 0
        dens = round(c / km2, 6) if km2 > 0 else ""
        cov  = round(a / aland, 6) if aland > 0 else ""
        w.writerow([tid, c, round(a, 3), dens, cov])
    fout.flush()
    print(f"  进度 {min(b+BATCH, len(tracts))}/{len(tracts)}")

fout.close()
print(f"完成 -> {OUT}   本轮跳过 {failed} 个(重跑即可补齐)")
