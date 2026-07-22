import os
import pandas as pd

SRC_DIR = r"data_sources/newyork_timeline/download_aef/raw"
OUT_DIR = r"data_sources/newyork_timeline/download_aef"
OUT_PATH = os.path.join(OUT_DIR, "aef_newyork_2020.csv")

# 1. 合并 6 个 band 文件（按行堆叠）
bands = [f"aef_New_York_b{i}_2020.csv" for i in range(6)]
dfs = []
for b in bands:
    df = pd.read_csv(os.path.join(SRC_DIR, b))
    print(f"{b}: {df.shape}")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print(f"\n合并后（去重前）: {merged.shape}")

# 2. 统一为年度文件的列结构：ID 列名 cb_2020__3，丢弃 MSA_NAME
merged = merged.rename(columns={"TRACT_ID": "cb_2020__3"})
merged = merged.drop(columns=["MSA_NAME"], errors="ignore")

# 3. 检查并去除重复 tract
dup = merged["cb_2020__3"].duplicated().sum()
print(f"重复 TRACT_ID 数量: {dup}")
merged = merged.drop_duplicates(subset="cb_2020__3", keep="first")
print(f"去重后: {merged.shape}")

merged.to_csv(OUT_PATH, index=False)
print(f"\n已保存: {OUT_PATH}")

# 4. 与其它年份的 tract 对齐诊断（时序分析可行性）
print("\n===== 各年份 tract 对齐诊断 =====")
sets = {}
for year in [2017, 2018, 2019, 2020, 2021]:
    p = os.path.join(OUT_DIR, f"aef_newyork_{year}.csv")
    if os.path.exists(p):
        ids = set(pd.read_csv(p, usecols=[0]).iloc[:, 0])
        sets[year] = ids
        print(f"{year}: {len(ids)} 个 tract")

common = set.intersection(*sets.values())
union = set.union(*sets.values())
print(f"\n所有年份共有(交集) tract: {len(common)}")
print(f"并集 tract: {len(union)}")
for year, ids in sets.items():
    print(f"  {year}: 缺失公共集之外 {len(ids - common)} 个，缺少 {len(common - ids)} 个公共 tract")
