# 树冠覆盖 (Tree Canopy Cover)

> 第三块 · 地理与环境 / 树冠　|　整理自 `树冠TCC_数据说明.docx`
> 输出:`tcc_features_tract.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | USFS/NLCD Tree Canopy Cover(经 GEE,资产 `USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5`,波段 `NLCD_Percent_Tree_Canopy_Cover`) |
| 原生分辨率 | 30m |
| 时间 | 2020 年(`calendarRange` 2020 + `study_area=CONUS`) |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 | 范围 | 数值大代表 |
|------|------|------|-----------|
| `tcc_mean` | tract 内平均树冠覆盖率 | 0 ~ 100(%) | 树冠越多 |

## 三、处理方法

先 `updateMask(值≤100)` 掩掉 254/255 填充值,再 `reduceRegions` 求均值,避免填充值抬高均值。

## 四、说明与注意

- 文档原列来源为 sciencebase 2025 增强版;此处取 GEE 上同属 NLCD 树冠体系的 2020 年度层,以满足年份/平台一致。
- 非树区域(城区/水面/农田)多为 0,正常。
