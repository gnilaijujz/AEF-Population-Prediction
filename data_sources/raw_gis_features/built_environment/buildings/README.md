# 建筑特征

> 第二块 · 人类活动与建成环境 / 建筑　|　整理自 `建筑特征_数据说明.docx`
> 脚本:`building_features_15msa_optimized.py`(单机/首轮)、`building_features_15msa_phase2.py`(多机第二轮再分配)、`merge_building_all.py`(合并)
> 输出:`building_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | OpenStreetMap (OSM) 建筑轮廓,经 ohsome API 查询(基于 OSM 全历史库 OSHDB) |
| 原文档所列来源 | Microsoft US Building Footprints;本实现改用 OSM/ohsome 以便直接按 tract 在线聚合(见选型说明) |
| 时间 | 2020 年(`time=2020-01-01`) |
| 下载/计算方式 | ohsome 聚合 API:`count`(建筑数)+ `area`(建筑面积),`groupBy/boundary`,tract 多边形作为 `bpolys` 传入,服务器端逐 tract 统计 |
| 过滤条件 | `building=* and geometry:polygon`(只统计面状建筑轮廓) |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS,15 个 MSA) |

## 二、特征字段

| 字段 | 含义 / 计算 | 单位/范围 | 数值大代表 |
|------|-----------|-----------|-----------|
| `building_count` | tract 内建筑轮廓数量(ohsome count) | 个,≥0 | 建筑越多,建成度/人口越高 |
| `building_area_m2` | tract 内建筑总占地面积(ohsome area) | m²,≥0 | 建筑占地越大 |
| `building_density` | `building_count ÷ tract 陆地面积(km²)` | 个/km²,≥0 | 单位面积建筑越密 → 城市化越高 |
| `coverage_ratio` | `building_area_m2 ÷ tract 陆地面积(m²)` | 0~1 | 建筑覆盖率越高 → 建成强度越大 |

陆地面积取自 tract 矢量 `ALAND` 字段(`cb_2020_11`,m²);密度用陆地面积而非含水总面积,避免临水 tract 被稀释。

## 三、执行流程(多机并行 + 两轮)

1. **首轮**(3 台按范围切分):按 tract 序号切成 3 段,三台机器各跑一段,分别写 `w0` / `w1` / `w2`。
2. **第二轮**(动态再分配):含密集城区(如纽约)的段最慢;先跑完的机器空闲后,用 phase2 脚本把"尚未完成的 tract"重新均分,写 `r0` / `r1` / `r2`。已完成的不重跑。
3. **合并**:`merge_building_all.py` 按 tract 去重合并所有 `w*` + `r*` → `building_features_15msa.csv`。

## 四、下载优化(重点)

- **递归二分批**:一批失败不整批丢弃,二分重试直到单个 tract,密集城区"重批"自动拆小。
- **超时→立即二分,错误→退避重试**:读超时=批太重→立即二分;500/SSL 瞬时错误→指数退避重试。
- **缩短超时+间隔**:单请求读超时 300s→90s;成功请求间隔 1.2s→0.4s。
- **多机动态再分配**(第二轮)消除单机瓶颈。
- **按需几何简化**:仅对顶点数 >1000 的超复杂 tract 用 ~11m 容差轻度抽稀,>99% tract 原样发送。
- **断点续跑**:每批即时落盘,重跑跳过已完成。
- **礼貌限速 + 服务器端聚合**。

## 五、局限与注意

- OSM 建筑完整度不均(大城市较全、中小城市/乡村稀疏),部分反映"OSM 是否被标注"而非纯真实建成度——尤其影响跨 MSA 迁移(是一种域偏移),建议在方法/讨论中注明。
- 几何简化仅作用于发给 ohsome 的边界,不影响 tract 对齐。
- `coverage_ratio` 理论 0~1,个别 tract 因 OSM 轮廓重叠/越界可能略 >1,可后期截断。
- 名单中约 21 个无几何的 tract 不参与计算。
- **选型说明**:文档原列 Microsoft Building Footprints(最完整);因本机地理库难安装,改用 OSM/ohsome 在线聚合,四指标一次到位;如需最完整口径可改用 Microsoft 数据。

## 附:关键参数

| 参数 | 取值 |
|------|------|
| `BATCH` | 8(初始批,失败自动二分) |
| `TIMEOUT` | 90s(单请求读超时;超时即二分) |
| `RETRY` | 3(仅对 500/SSL 等瞬时错误) |
| `SLEEP` | 0.4s(成功请求后间隔) |
| `SIMPLIFY_TOL / VERTEX_THRESHOLD` | 0.0001°(≈11m) / 1000 顶点 |
| `FILTER / TIME` | `building=* and geometry:polygon` / `2020-01-01` |
