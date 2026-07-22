# DEM 高程 / 坡度

> 第三块 · 地理与环境 / 地形　|　整理自 `DEM_数据说明.docx`
> 输出:`dem_features_tract.csv` / `dem_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | USGS 3DEP(经 Google Earth Engine,资产 `USGS/3DEP/10m`) |
| 原生分辨率 / 取样 | ~10m 原生;`reduceRegions` 按 30m 取样聚合到 tract |
| 时间 | 地形静态(3DEP 最优可用高程,年份不敏感) |
| 连接键 | `cb_2020_3`(`1400000US` + 11 位 tract 码) |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否(原始值,标准化交统一建模流程) |

## 二、特征字段

| 字段 | 含义 | 单位 | 数值大代表 |
|------|------|------|-----------|
| `elev_mean` | tract 内平均高程 | m | 地势越高 |
| `slope_mean` | tract 内平均坡度(`ee.Terrain.slope`) | 度 | 地形越陡 |

## 三、处理方法

GEE 里 `dem.addBands(slope)` 后 `reduceRegions(mean, scale=30)` 聚合到每个 tract,再 `Export.table.toDrive` 导出 CSV。

## 四、说明与注意

- 纯水体 tract 可能无高程值(空),保留不插补。
- 方法可写:高程/坡度来自 USGS 3DEP(~10m, sampled at 30m), via Google Earth Engine。
