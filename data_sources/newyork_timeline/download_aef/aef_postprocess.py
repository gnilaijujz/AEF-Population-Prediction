# -*- coding: utf-8 -*-
"""
把 GEE 导出的 aef_newyork_YYYY.csv 处理成与 2020 相同结构的一套文件：
  - 列改成 TRACT_ID, MSA_NAME, A00..A63
  - 按 2020 的 b0..b5 归属切成 6 个文件
  - 复制 2020 的 shapefile 附属文件
  - 输出到 data_aef_newyork_YYYY/
用法：  python aef_postprocess.py 2017 2018 2019 2021
       （需先把各年 GEE 导出的 aef_newyork_YYYY.csv 都放到本脚本同目录）
       不带参数时默认处理 2017 2018 2019 2021。
"""
import sys, os, csv, glob, shutil

YEARS = sys.argv[1:] if len(sys.argv) > 1 else ["2017", "2018", "2019", "2021"]
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)                          # 中期后/
SRC_2020 = os.path.join(ROOT, "data_aef_newyork")     # 2020 参照
BANDS = [f"A{i:02d}" for i in range(64)]

# 学习 2020 的 TRACT_ID -> b 文件 归属（只做一次，各年共用）
tid2b = {}
for f in sorted(glob.glob(os.path.join(SRC_2020, "aef_New_York_b*_2020.csv"))):
    b = os.path.basename(f).split("_b")[1][0]         # b 后面那位 0..5
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        tid2b[r["TRACT_ID"]] = b
print(f"2020 归属表：{len(tid2b)} 个 tract，分 {len(set(tid2b.values()))} 批\n")


def process(year):
    export = os.path.join(BASE, f"aef_newyork_{year}.csv")
    if not os.path.exists(export):
        print(f"[{year}] 找不到 {os.path.basename(export)}，跳过（先从 Drive 下载放到本目录）\n")
        return
    outdir = os.path.join(ROOT, f"data_aef_newyork_{year}")
    os.makedirs(outdir, exist_ok=True)

    rows = list(csv.DictReader(open(export, encoding="utf-8-sig")))
    key = next(k for k in rows[0] if k.lower().replace("_", "") in
               ("cb20203", "cb2020__3", "tractid") or k.startswith("cb_2020_"))
    print(f"[{year}] 键字段={key}，共 {len(rows)} 行")

    buckets = {str(b): [] for b in range(6)}
    missing = 0
    for r in rows:
        tid = r[key]
        b = tid2b.get(tid)
        if b is None:            # 该 tract 不在 2020 的 5031 里（理论上不该有）
            missing += 1; continue
        rec = {"TRACT_ID": tid, "MSA_NAME": "New York"}
        for band in BANDS:
            rec[band] = r.get(band, "")
        buckets[b].append(rec)

    hdr = ["TRACT_ID", "MSA_NAME"] + BANDS
    total = 0
    for b, recs in buckets.items():
        out = os.path.join(outdir, f"aef_New_York_b{b}_{year}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=hdr)
            w.writeheader(); w.writerows(recs)
        total += len(recs)
    for f in glob.glob(os.path.join(SRC_2020, "2020_MSA_in_track_newyork.*")):
        shutil.copy(f, outdir)
    print(f"[{year}] 写出 6 个 b 文件，合计 {total} 行；未匹配 {missing} 行；"
          f"已复制 shapefile → {os.path.basename(outdir)}\n")


for y in YEARS:
    process(y)
