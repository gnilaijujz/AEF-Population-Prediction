# -*- coding: utf-8 -*-
"""合并所有 building_features_15msa_w*.csv 和 _r*.csv → building_features_15msa.csv"""
import csv, glob, os
BASE = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(BASE, "building_features_15msa_w*.csv")) +
               glob.glob(os.path.join(BASE, "building_features_15msa_r*.csv")))
print("找到分片:", [os.path.basename(p) for p in parts])
seen = {}; header = None
for p in parts:
    with open(p, encoding="utf-8-sig") as f:
        rd = csv.reader(f); header = next(rd)
        for row in rd:
            if row and row[0]:
                seen[row[0]] = row      # 按tract去重(w1里已完成的沿用)
out = os.path.join(BASE, "building_features_15msa.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(header)
    for k in sorted(seen):
        w.writerow(seen[k])
print(f"合并 {len(seen)} 个 tract -> {out}")
