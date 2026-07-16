# 社会经济:收入 / 贫困 (NHGIS)

> 第一块 · 社会经济 / 收入与贫困　|　整理自 `社会经济NHGIS_数据说明.docx`
> 输出:`socioeconomic_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | IPUMS NHGIS([data2.nhgis.org](https://data2.nhgis.org)),底层 ACS 5-year 2016–2020,表 A88/AB2/BD5/CL6 |
| 时间 | 2016–2020 五年合并估计(一个值,非逐年) |
| 尺度 | census tract(原键 `GISJOIN` → 转换成 `cb_2020_3`) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 | 单位/范围 | 数值大代表 |
|------|------|-----------|-----------|
| `fam_inc_lt10k` … `fam_inc_50kplus` | 家庭收入 5 档的家庭数(<1万 / 1–1.5万 / 1.5–2.5万 / 2.5–5万 / 5万+) | 户数,≥0 | 该收入段家庭越多 |
| `median_family_income` | 家庭收入中位数 | 美元 | 越富裕 |
| `per_capita_income` | 人均收入 | 美元 | 越富裕 |
| `persons_below_poverty` | 贫困线以下人口数 | 人,≥0 | 贫困人口越多 |
| `*_moe` | 各特征对应误差范围(margin of error) | 同上 | ACS 抽样不确定性 |

## 三、说明与注意

- ⚠️ **NHGIS 原始数据禁止再分发**:公开仓库只放处理代码与来源说明,勿转发原始表(`nhgis0001*`)。
- 2016–2020 收入分档较粗(5 档),是为跨期可比而合并;也解决了康州 2022 县→规划区改制的对不齐(改用 2016–2020 口径后 CT 完整匹配)。
- 收入分档/贫困为原始计数,收入中位数、人均为美元;建模时可按需算率(贫困率、各档占比)。
- 引用:IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org)。
