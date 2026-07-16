# NDVI 植被指数

> 第三块 · 地理与环境 / 植被　|　整理自 `NDVI_数据说明.docx`
> 输出:`ndvi_features_tract.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | Landsat C2 T1 L2 年度 NDVI 合成(经 GEE,资产 `LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI`) |
| 原生分辨率 | 30m(Landsat) |
| 时间 | 2020 年(`filterDate` 2020-01-01 ~ 2021-01-01) |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 | 范围 | 数值大代表 |
|------|------|------|-----------|
| `ndvi_mean` | tract 内平均 NDVI | -1 ~ 1 | 植被越茂密(城区/水面偏低,林区偏高) |

## 三、处理方法

取 2020 年度合成 `.mosaic()` 后 `reduceRegions` 求均值;单波段用 `ee.Reducer.mean().setOutputs([...])` 显式命名,避免单波段命名导致全空。

## 四、说明与注意

- 文档示例代码年份为 2017(官方 demo),本实现改为 2020。
