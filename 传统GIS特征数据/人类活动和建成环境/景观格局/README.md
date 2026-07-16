# 景观格局指数 (Landscape Pattern Indices)

> 第二块 · 人类活动与建成环境 / 景观格局
> 输出:`landscape_indices_tract_*.csv`(每个 MSA 一份,尚未合并成 `*_15msa` 版)

## 概况

基于土地覆盖栅格,按 census tract 计算景观格局指数(landscape metrics),用于刻画建成/自然斑块的破碎化、聚集与形态,作为城市形态学维度的特征补充。

- 分析单元:census tract(15 个 MSA)
- 连接键:`cb_2020_3`
- 文件:`landscape_indices_tract_<MSA>.csv`(每 MSA 一份)

## 状态

- ⏳ 目前为 **per-MSA 中间文件**,暂未合并为 `landscape_features_15msa.csv`,因此暂未纳入 git 仓库(按"只追踪合并后 `*_15msa` 特征"的约定)。
- 📝 完整的字段口径 / 计算方法 `数据说明.docx` 待补;补齐后本 README 将同步更新字段表。
