# 道路特征

> 第二块 · 人类活动与建成环境 / 道路　|　整理自 `道路特征_数据说明.docx`
> 输出:`road_features_15msa.csv`(由 15 个 `tract_road_features_for_GNN_*.csv` 合并裁剪)

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | U.S. Census Bureau TIGER/Line 道路矢量 2020(census.gov shapefiles,`layergroup=Roads`) |
| 数据形态/尺度 | 矢量线数据;state 尺度含 primary+secondary road,county 尺度含全部 roads;按 tract 汇总 |
| 时间 | 2020 年 |
| 输出文件 | `road_features_15msa.csv`(11,158 个 tract) |
| 连接键 | `cb_2020_3` |
| 空间尺度 | census tract(美国本土 CONUS,15 个 MSA) |
| 是否归一化 | 否 |

## 二、特征字段

| 字段 | 含义 / 计算 | 单位/范围 | 数值大代表 |
|------|-----------|-----------|-----------|
| `road_len_total_m` | tract 内道路总长度 | m,≥0 | 路网总里程越大 |
| `road_density_local` | 地方道路(local)长度密度 | ≈ m/km²,≥0 | 地方路网越密 |
| `road_density_secondary` | 次要道路(secondary)长度密度 | ≈ m/km²,≥0 | 次干路网越密 |

## 三、处理方法

基于 TIGER 道路线矢量,与 tract 面叠加,按 tract 统计总道路长度,并按道路类别(local / secondary,TIGER MTFCC 分类)分别计算长度密度(长度 ÷ tract 面积)。文件命名带 `for_GNN`,表示已整理为图神经网络的节点特征。

## 四、说明与注意

- 本表为 15 个 per-MSA 文件合并后按名单裁剪;键 `cb_2020_3`。
- 与其它特征一致,名单中约 21 个无几何的 tract 不参与(未匹配)。
- 道路密度/长度是路网发达程度与建成度的代理,通常与人口/城市化正相关。
- 来源标注:Roads from U.S. Census Bureau TIGER/Line (2020)。
