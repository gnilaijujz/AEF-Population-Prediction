# 夜间灯光 (NTL)

> 第二块 · 人类活动与建成环境 / 夜间灯光　|　整理自 `夜间灯光NTL_数据说明.docx`
> 输出:`ntl_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | VIIRS 夜间灯光年度合成 V2(masked 版,eogdata.mines.edu/products/vnl) |
| 原生分辨率 | 约 500m(VIIRS) |
| 时间 | 2020 年 |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 | 范围 | 数值大代表 |
|------|------|------|-----------|
| `ntl_sum` | tract 内总光量(最重要) | ≥0 | 经济活动/人口总量越大 |
| `ntl_mean` | tract 内平均亮度 | ≥0 | 平均越亮 |
| `ntl_intensity_per_km2` | 单位面积光量(与密度同构) | ≥0 | 越密集 |
| `ntl_std` | 亮度标准差(异质性) | ≥0 | 内部差异越大 |
| `ntl_cv` | 变异系数(离散程度) | ≥0 | 相对离散越大 |
| `ntl_max` / `ntl_min` / `ntl_count` | 最大/最小亮度、有效像元数 | — | 辅助 |

## 三、说明与注意

- ⚠️ `Albany_features.csv` 曾误取 Albany, GA(应为 Albany-Schenectady-Troy, NY, CBSA 10580),需用正确的纽约 Albany 重新生成后再合并,否则该 MSA 约 251 个 tract 缺 NTL。
- 15 个 per-MSA 文件合并后按名单裁剪;键 `cb_2020_3`。
- 来源标注:VIIRS Nighttime Lights Annual V2 (masked), 2020, Earth Observation Group, Colorado School of Mines。
