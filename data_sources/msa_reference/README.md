# MSA 研究区与研究单元

> 整理自 `raw_msa_data/MSA 数据说明.docx`
> 脚本:`selected_15_msa/select_msa.py`(高/中/低分层抽样)

## 一、研究单元

- **Census Tract**(普查区,约 1,200–8,000 人,理想约 4,000 人)为研究单元。
- **人口标签**:美国官方统计(Census Bureau / ACS 表 `B01003`)。因 WorldPop(~1km)分辨率不足以支撑 tract 级预测,改用官方统计口径。

## 二、MSA 组成与筛选

**MSA 的县组成表**(list1_2020.xls,来自 Census [historical delineation files](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/historical-delineation-files.html))——用来确定"哪些县属于哪个 MSA"并做高/中/低分层。字段包括:

- `CBSA Code`、`CBSA Title`(MSA 名称)
- `Metropolitan/Micropolitan Statistical Area`(类型列 —— 筛出 MSA,剔除 Micropolitan)
- `Metropolitan Division`(大都市区次级划分,如纽约、洛杉矶)
- `FIPS State Code` + `FIPS County Code`(county FIPS,用来跟 tract 对齐)

**筛选流程**:筛出 tract 数 > 100 的 MSA,按人口分**高 / 中 / 低**三层,随机抽样得到 **15 个 MSA**(见 `selected_15_msa/selected_15_msa_over100.csv`)。

## 三、MSA 边界矢量

- TIGER/Line 2020 CBSA:[www2.census.gov/geo/tiger/TIGER2020/CBSA](https://www2.census.gov/geo/tiger/TIGER2020/CBSA/)(`tl_2020_us_cbsa`,画范围/出图用)。

## 四、目录内容

- `selected_15_msa/` — 筛选结果:`selected_15_msa_over100.csv`(15 MSA 名单)、`tracts_in_15msa_over100.csv`(tract 清单)、`msa_population_2020*.csv`(人口标签)、`select_msa.py`(抽样脚本)。
- `raw_msa_data/` — 原始下载(边界 shapefile、census 人口表、说明文档),**未纳入仓库**(体积大 / 可从官方获取)。
