# -*- coding: utf-8 -*-
"""
从 pop_features.csv (ACS DP05) 计算人口结构特征，只保留15个MSA的tract。
未成年人口占比 = DP05_0019E / DP05_0001E
老年人口占比   = DP05_0024E / DP05_0001E
性别比(男/女×100) = DP05_0002E / DP05_0003E * 100
输出：pop_features_15msa.csv  列: cb_2020_3, minor_ratio, elderly_ratio, sex_ratio
"""
import csv, os
BASE = os.path.dirname(os.path.abspath(__file__))
POP    = os.path.join(BASE, "pop_features.csv")
TARGET = os.path.join(BASE, "..", "tracts_in_15msa_over100.csv")   # 上一级目录
OUT    = os.path.join(BASE, "pop_features_15msa.csv")

# 15个MSA的目标tract
target = set()
for row in csv.DictReader(open(TARGET, encoding="utf-8-sig")):
    target.add(row["cb_2020_3"].strip())
print("目标 tract:", len(target))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

rd = csv.reader(open(POP, encoding="utf-8-sig"))
h = next(rd)
i_geo = h.index("GEO_ID")
i_tot = h.index("DP05_0001E")   # 总人口
i_u18 = h.index("DP05_0019E")   # 未成年
i_o65 = h.index("DP05_0024E")   # 老年
i_m   = h.index("DP05_0002E")   # 男
i_f   = h.index("DP05_0003E")   # 女

out = []
found = set()
for r in rd:
    gid = r[i_geo].strip()
    if gid not in target:
        continue
    found.add(gid)
    tot, u18, o65, m, f = num(r[i_tot]), num(r[i_u18]), num(r[i_o65]), num(r[i_m]), num(r[i_f])
    minor   = round(u18 / tot, 6) if (tot and u18 is not None and tot > 0) else ""
    elderly = round(o65 / tot, 6) if (tot and o65 is not None and tot > 0) else ""
    sexr    = round(m / f * 100, 4) if (f and m is not None and f > 0) else ""
    out.append([gid, minor, elderly, sexr])

out.sort(key=lambda x: x[0])
with open(OUT, "w", newline="", encoding="utf-8-sig") as fo:
    w = csv.writer(fo)
    w.writerow(["cb_2020_3", "minor_ratio", "elderly_ratio", "sex_ratio"])
    w.writerows(out)

# 核查
print("输出行数:", len(out), " key唯一:", len(set(x[0] for x in out)))
print("目标里在pop表中未找到:", len(target - found))
for name, ci in [("minor_ratio",1),("elderly_ratio",2),("sex_ratio",3)]:
    vals=[x[ci] for x in out if x[ci] != ""]
    empty=sum(1 for x in out if x[ci]=="")
    if vals:
        print(f"  {name:<14} 空{empty}  范围{min(vals):.4f}~{max(vals):.4f}  均值{sum(vals)/len(vals):.4f}")
print("输出:", OUT)
