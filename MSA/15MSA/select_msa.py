# -*- coding: utf-8 -*-
"""
从 ACS 2020 tract 人口 (B01003) + list1_2020.xls 计算各 MSA 的总人口与 tract 数量。
仅依赖 xlrd + 标准库。输出 msa_population_2020.csv（全部 MSA，按人口降序）。
"""
import csv, xlrd, os

BASE = os.path.dirname(os.path.abspath(__file__))
ACS  = os.path.join(BASE, "ACSDT5Y2020.pop", "ACSDT5Y2020.B01003-Data.csv")
LIST1= os.path.join(BASE, "list1_2020.xls")
OUT  = os.path.join(BASE, "msa_population_2020.csv")

# 1) ACS tract 人口 → 按 5 位县 FIPS 聚合（人口和、tract 计数）
county_pop = {}    # {'01001': 人口}
county_tracts = {} # {'01001': tract数}
with open(ACS, encoding="utf-8-sig") as f:
    r = csv.reader(f)
    next(r); next(r)  # 跳过表头两行
    for row in r:
        if not row or not row[0].startswith("1400000US"):
            continue
        g = row[0]
        fips5 = g[9:14]          # 州(2)+县(3)
        try:
            pop = int(row[2])    # B01003_001E
        except (ValueError, IndexError):
            continue
        county_pop[fips5] = county_pop.get(fips5, 0) + pop
        county_tracts[fips5] = county_tracts.get(fips5, 0) + 1

# 2) list1 → 每个 CBSA 的成分县，仅保留 Metropolitan Statistical Area
wb = xlrd.open_workbook(LIST1)
sh = wb.sheet_by_index(0)
msa = {}  # {cbsa_code: {'title':..,'pop':0,'tracts':0,'counties':[], 'miss':[]}}
for i in range(3, sh.nrows):
    row = sh.row_values(i)
    cbsa = str(row[0]).strip()
    if not cbsa:
        continue
    typ = str(row[4]).strip()
    if typ != "Metropolitan Statistical Area":
        continue
    title = str(row[3]).strip()
    st = str(row[9]).strip()
    co = str(row[10]).strip()
    if not (st and co):
        continue
    fips5 = st.zfill(2) + co.zfill(3)
    m = msa.setdefault(cbsa, {"title": title, "pop": 0, "tracts": 0,
                              "counties": [], "miss": []})
    m["counties"].append(fips5)
    if fips5 in county_pop:
        m["pop"] += county_pop[fips5]
        m["tracts"] += county_tracts.get(fips5, 0)
    else:
        m["miss"].append(fips5)

# 3) 排序输出
rows = []
for cbsa, m in msa.items():
    rows.append((cbsa, m["title"], m["pop"], m["tracts"],
                 len(m["counties"]), len(m["miss"]),
                 ";".join(m["miss"])))
rows.sort(key=lambda x: x[2], reverse=True)

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rank", "cbsa_code", "cbsa_title", "population_2020",
                "n_tracts", "n_counties", "n_missing_counties", "missing_fips"])
    for i, r in enumerate(rows, 1):
        w.writerow([i] + list(r))

# 4) 摘要
n = len(rows)
print(f"MSA 总数: {n}")
print(f"人口范围: {rows[0][2]:,} ({rows[0][1]})  ~  {rows[-1][2]:,} ({rows[-1][1]})")
mid = n // 2
print(f"中位 MSA (rank {mid+1}): {rows[mid][1]}  人口 {rows[mid][2]:,}  tracts {rows[mid][3]}")
miss_total = sum(1 for r in rows if r[5] > 0)
print(f"存在县人口未匹配的 MSA 数: {miss_total}（通常为个别县FIPS差异，可忽略或核查）")
print(f"输出: {OUT}")
