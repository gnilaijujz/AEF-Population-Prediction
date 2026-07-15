# -*- coding: utf-8 -*-
"""
按 tracts_in_15msa_over100.csv 提取 15个MSA 的 tract，关联 dem/ndvi/tcc/water 四类特征，
输出一张总表。只保留四类特征都齐全的 tract(剔除名单里无特征的21个)，不重不漏。
输出：features_15msa_merged.csv
"""
import csv, os
BASE = os.path.dirname(os.path.abspath(__file__))

def load(fn):
    rd = csv.reader(open(os.path.join(BASE, fn), encoding="utf-8-sig"))
    h = next(rd)
    d = {}
    for r in rd:
        if r and r[0]:
            d[r[0]] = r[1:]
    return h[1:], d

# 目标名单(含MSA元信息)
tgt_rows = []
seen = set()
rd = csv.DictReader(open(os.path.join(BASE, "tracts_in_15msa_over100.csv"), encoding="utf-8-sig"))
for row in rd:
    k = row["cb_2020_3"].strip()
    if k in seen:
        continue                      # 去重(名单本身无重复,保险)
    seen.add(k)
    tgt_rows.append((k, row["cbsa_code"], row["cbsa_title"], row["group"]))

# 四类特征
dem_h, dem = load("dem_features_tract.csv")
ndvi_h, ndvi = load("ndvi_features_tract.csv")
tcc_h, tcc = load("tcc_features_tract.csv")
wat_h, wat = load("water_dist_features_tract_combine.csv")

header = ["cb_2020_3", "cbsa_code", "cbsa_title", "group"] + dem_h + ndvi_h + tcc_h + wat_h

out_rows = []
dropped = []
empty_cells = 0
for k, code, title, group in tgt_rows:
    if not (k in dem and k in ndvi and k in tcc and k in wat):
        dropped.append(k)             # 四类里缺任一 -> 剔除
        continue
    vals = dem[k] + ndvi[k] + tcc[k] + wat[k]
    empty_cells += sum(1 for v in vals if not str(v).strip())
    out_rows.append([k, code, title, group] + vals)

OUT = os.path.join(BASE, "features_15msa_merged.csv")
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(out_rows)

print("表头:", header)
print(f"目标名单: {len(tgt_rows)}")
print(f"输出行数: {len(out_rows)}  (剔除无特征的 {len(dropped)} 个)")
print(f"key唯一: {len(set(r[0] for r in out_rows))}  重复: {len(out_rows)-len(set(r[0] for r in out_rows))}")
print(f"内部空单元格数(个别水体tract等,非21个那批): {empty_cells}")
# 分MSA计数
from collections import Counter
cnt = Counter(r[2] for r in out_rows)
print("各MSA tract数:")
for t, n in sorted(cnt.items()):
    print(f"  {t[:40]:<42}{n}")
print("输出:", OUT)
