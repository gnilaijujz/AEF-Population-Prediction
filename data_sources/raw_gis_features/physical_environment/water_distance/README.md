# 到水体距离 (Distance to Water)

> 第三块 · 地理与环境 / 水体　|　整理自 `水体距离_数据说明.docx`
> 输出:`water_dist_features_tract_combine.csv` / `water_dist_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | JRC Global Surface Water v1.4 逐年分类(经 GEE,资产 `JRC/GSW1_4/YearlyHistory`,`waterClass≥2` 为水) |
| 原生分辨率 | 30m |
| 时间 | 2020 年 |
| 与文档差异 | 文档列 nationalmap NHD(矢量、非 30m、不在 GEE);三要素不可兼得,改用 JRC 栅格水体 |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 | 单位 | 数值大代表 |
|------|------|------|-----------|
| `dist_water_m` | tract 内各像元到最近水体的平均直线距离 | m,≥0 | 离水越远(近水 tract≈0) |

## 三、处理方法

由 2020 水体二值图经 `fastDistanceTransform` 得到"到最近水像元距离图"(`.sqrt()`×像元边长→米),再 `reduceRegions` 求 tract 均值。

**优化:** 30m 版近水精确但远处会被搜索窗口截断出伪值;90m 版覆盖更远更省算力。最终合并——近处取 30m 精度、远处(30m 异常)取 90m 正确值,得 `combine` 版。

## 四、说明与注意

- 引用:Pekel et al., 2016, *Nature*(JRC GSW)。
- `combine` 版最大值约 15km(真实沙漠距离),无 410km 之类伪值。
