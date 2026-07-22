# 土地覆盖 (NLCD)

> 第二块 · 人类活动与建成环境 / 土地覆盖　|　整理自 `土地覆盖NLCD_数据说明.docx`
> 输出:`nlcd_features_tract.csv` / `nlcd_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | Annual NLCD 2020(MRLC,`Annual_NLCD_LndCov_2020_CU_C1V2.tif`),本地栅格分区统计 |
| 原生分辨率 | 30m |
| 时间 | 2020 年 |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否 |

## 二、特征字段(共 11 个)

| 字段 | 含义 / 计算 | 单位/范围 | 数值大代表 |
|------|-----------|-----------|-----------|
| `*_pct`(10 个类别) | 各土地覆盖类别在 tract 内的面积占比 | 0~100(%) | 该类地表占比越高 |
| `impervious_mean` | tract 内平均不透水面比例 | 0~100(%) | 不透水面越多 → 建成度越高 |

10 个类别:`water` / `developed` / `barren` / `forest` / `shrub` / `grassland` / `pasture_hay` / `cultivated_crops` / `woody_wetlands` / `herbaceous_wetlands`。

## 三、处理方法

基于 NLCD 土地覆盖栅格,对每个 tract 统计各类别像元占比;不透水面取 NLCD 不透水产品的均值。

## 四、说明与注意

- 此处 `water_pct` 与"到水体距离"是不同维度,可并存。
