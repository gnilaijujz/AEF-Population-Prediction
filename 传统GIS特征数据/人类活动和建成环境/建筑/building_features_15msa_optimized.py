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
_SHP_CANDS = [os.path.join(BASE, "census_tracts", "2020_MSA_in_track"),
              os.path.join(BASE, "2020_MSA_in_track")]
SHP = next((p for p in _SHP_CANDS if os.path.exists(p + ".shp")), _SHP_CANDS[0])
TARGET = os.path.join(BASE, "tracts_in_15msa_over100.csv")
OUT    = os.path.join(BASE, "building_features_15msa.csv")
URL_COUNT = "https://api.ohsome.org/v1/elements/count/groupBy/boundary"
URL_AREA  = "https://api.ohsome.org/v1/elements/area/groupBy/boundary"
FILTER = "building=* and geometry:polygon"
TIME = "2020-01-01"
BATCH = 8            # 初始批;失败自动二分,不必太小
SLEEP = 0.4          # 每次成功请求后的间隔(秒),调小提速
RETRY = 3            # 非超时错误(500/SSL)的重试次数;超时则立即二分
TIMEOUT = 90         # 单请求读超时(秒);超时=批太重->立即二分,而非死等到300s
SIMPLIFY_TOL = 0.0001     # ≈11m,仅对超复杂tract轻度抽稀
VERTEX_THRESHOLD = 1000   # 顶点数超过此值才简化;绝大多数tract保持原样(精确)
TEST_LIMIT = None          # 先测试;正式跑设 None

# 多机分工:
# worker 0: 前四分之一
# worker 2: 第二个四分之一
# worker 1: 后半段
# 输出保持 w0 / w2 / w1 命名

WORKER_ID = 2

WORKER_RANGES = {
    0: (0.00, 0.25),   # 前1/4
    2: (0.25, 0.50),   # 原前半段的后1/2
    1: (0.50, 1.00),   # 后半段
}

OUT = os.path.join(BASE, f"building_features_15msa_w{WORKER_ID}.csv")

def nverts(coords):
    """统计geojson几何的顶点数"""
    if not coords:
        return 0
    if isinstance(coords[0], (int, float)):
        return 1
    return sum(nverts(c) for c in coords)

def post_ohsome(features, url, filt=FILTER):
    """返回结果dict；超时立即返回None(触发上层二分)；500/SSL等瞬时错误退避重试"""
    fc = {"type": "FeatureCollection", "features": features}
    body = urllib.parse.urlencode({"bpolys": json.dumps(fc), "filter": filt,
                                   "time": TIME, "format": "json"}).encode()
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(url, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.load(r)
            result = {str(g["groupByObject"]): g["result"][0]["value"]
                      for g in data.get("groupByResult", [])}
            time.sleep(SLEEP)          # 成功后礼貌间隔
            return result
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return None            # 超时=批太重,立即返回让上层二分(不死等)
            wait = min(20, 3 * (2 ** attempt))   # 500/SSL等:退避重试
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

ratio_lo, ratio_hi = WORKER_RANGES[WORKER_ID]
LO = int(total * ratio_lo)
HI = int(total * ratio_hi)

print(f"本机(worker {WORKER_ID}) 负责序号 [{LO}, {HI}) / 总数 {total}")

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

def query_batch(batch):
    """对一批tract取 count+area；失败(尤其超时)则二分递归,直到单个;单个仍失败=>None"""
    feats = [{"type": "Feature", "id": t[0], "properties": {"id": t[0]}, "geometry": t[2]} for t in batch]
    cnt = post_ohsome(feats, URL_COUNT)
    area = post_ohsome(feats, URL_AREA) if cnt is not None else None
    if cnt is not None and area is not None:
        return {t[0]: (cnt.get(t[0], 0), area.get(t[0], 0.0)) for t in batch}
    if len(batch) == 1:
        return {batch[0][0]: None}          # 单个仍失败,放弃(极少)
    mid = len(batch) // 2
    print(f"    批({len(batch)})失败→二分")
    res = query_batch(batch[:mid])
    res.update(query_batch(batch[mid:]))
    return res

failed = 0; wrote = 0
for b in range(0, len(tracts), BATCH):
    batch = tracts[b:b + BATCH]
    res = query_batch(batch)
    for tid, aland, _ in batch:
        v = res.get(tid)
        if v is None:
            failed += 1; continue
        c, a = v
        km2 = aland / 1e6 if aland else 0
        dens = round(c / km2, 6) if km2 > 0 else ""
        cov  = round(a / aland, 6) if aland > 0 else ""
        w.writerow([tid, c, round(a, 3), dens, cov]); wrote += 1
    fout.flush()
    print(f"  进度 {min(b+BATCH, len(tracts))}/{len(tracts)}  已写{wrote} 跳过{failed}")

fout.close()
print(f"完成 -> {OUT}   写入{wrote} 跳过{failed}(重跑即可补齐)")
