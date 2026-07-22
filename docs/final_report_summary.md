# Final Report Summary

本文件与最终版结题汇报 `docs/report_materials/结题.pptx` 同步，用作项目的文字版研究说明。根目录中的 `结题.pptx` 是同一最终报告的项目入口副本。

## 1. Research Motivation

高分辨率人口空间分布是城市规划、公共服务配置、灾害应急和 SDGs 评估的基础。传统人口空间化依赖不透水面、夜间灯光、路网、POI 等 GIS 协变量，而 AlphaEarth Foundations (AEF) 等地理空间基础模型提供了全球一致、可复用的遥感语义表征。

本研究的核心判断是：AEF 的价值不能只用单城市拟合精度衡量，更应检验其是否能支持未见城市的人口预测。人口密度并非只由地表形态决定；相同遥感语义在不同城市可能对应不同人口响应。因此，跨城市空间可迁移能力必须被独立评估。

## 2. Research Questions

| Question | Operational Test | Expected Output |
| --- | --- | --- |
| Q1. AEF 能否进行人口预测？ | 使用 AEF 64D embedding 在每个 MSA 内进行 self-training。 | R2、RMSE、MAE、MAPE、Pearson r。 |
| Q2. AEF 是否具有空间可迁移能力？GIS 是否能提高迁移能力？ | source MSA 训练，target MSA 直接测试；比较 AEF、GIS、AEF+GIS 和 AEF+single GIS。 | source-target 迁移矩阵、CDS、avg_CDS。 |
| Q3. 迁移能力差异由什么因素导致？ | 结合 domain shift、源域健康指标、tract 级误差集聚和 AEF cluster。 | 特征距离回归、源域选择指标、conditional shift 解释。 |

## 3. Study Area And Data

研究区包含 15 个美国 MSA，按人口规模分层抽样，兼顾高/中/低人口规模城市。

| Group | MSA Examples |
| --- | --- |
| High population | New York, Chicago, Atlanta, Oklahoma City, Hartford |
| Medium population | Bridgeport, Albany, Knoxville, Baton Rouge, Jackson |
| Low population | Lansing, Modesto, Fort Wayne, Montgomery, Duluth |

研究单元为 census tract，共 11,179 个 tract，覆盖约 43.6M 2020 年人口。人口标签来自 ACS B01003 总人口表，并按 tract 面积换算人口密度。由于 WorldPop 约 1 km 分辨率不足以支撑 tract 级实验，本项目采用官方统计口径作为监督标签。

| Data Type | Source And Processing |
| --- | --- |
| Population label | ACS B01003 total population; converted to tract-level population density; `log1p` transform. |
| AEF feature | AlphaEarth Foundations Satellite Embedding, 64 dimensions, 10 m annual embedding, aggregated by tract mean. |
| GIS feature | 9 selected variables from built environment, activity intensity, network, vegetation, terrain, and location dimensions. |
| Validation data | 15 additional MSAs for external validation. |

## 4. Modeling Pipeline

1. Align MSA/tract boundaries, AEF pixels, GIS variables and ACS labels to the same census tract unit.
2. Build a tract graph using Queen contiguity.
3. Train GraphSAGE models for in-city prediction.
4. Run cross-city transfer by applying source-trained models directly to target cities.
5. Compute robust entropy-weighted performance scores and CDS.
6. Explain CDS with domain shift, source-domain health metrics and tract-level error patterns.

```text
AEF/GIS features + tract shapefile + population density
        -> Queen graph
        -> GraphSAGE training
        -> source-domain score + cross-domain score
        -> CDS and avg_CDS
        -> domain-shift and source-health attribution
```

## 5. Transferability Metric

本项目使用熵权法将多个回归指标合成为综合性能分数，再定义：

```text
CDS = Score_source - Score_cross
```

其中 `Score_source` 表示源域内性能，`Score_cross` 表示跨域迁移性能。`avg_CDS` 是一个 source 对所有 target 的平均迁移损失；数值越低，源城市越适合迁移。

AEF 配置下权重如下：

| Metric | Entropy Weight |
| --- | ---: |
| RMSE | 0.233 |
| MAE | 0.192 |
| R2 | 0.048 |
| MAPE | 0.066 |
| Pearson r | 0.462 |

## 6. Main Findings

### 6.1 AEF Contains Population-Relevant Spatial Semantics

城市内 self-training 结果表明，AEF embedding 能有效支持 tract 级人口密度预测。部分城市的 self-test R2 较高，例如 Albany_S 约 0.800、Baton 约 0.747、Bridgeport 约 0.712。该结果说明 AEF 表征包含人口相关空间语义，但并不意味着模型可以自动跨城市泛化。

### 6.2 Transferability Is Highly Source-Dependent

AEF 配置下，New York、Chicago、Atlanta、Bridgeport 通常表现为较优源域；Jackson_MS、Montgomery、Baton、Fort_wayne 等城市迁移损失较高。GIS-only 配置的源域排序与 AEF 不完全一致，说明传统 GIS 与 AEF 表征捕捉的空间相似性并不等价。

| Configuration | Lower-loss Source Cities | Higher-loss Source Cities | Interpretation |
| --- | --- | --- | --- |
| AEF | New York (-0.097), Chicago (-0.024), Atlanta (0.021), Bridgeport (0.052) | Jackson_MS (0.357), Montgomery (0.233), Baton (0.185), Fort_wayne (0.155) | AEF 迁移能力高度依赖源域，城市规模、表征覆盖度和局部形态响应都可能影响泛化。 |
| GIS | New York (0.062), Atlanta (0.093), Chicago (0.100), Bridgeport (0.130) | Jackson_MS (0.428) 等 | GIS 排序与 AEF 有重叠但不完全一致，说明传统 GIS 与遥感基础表征具有互补性。 |
| AEF+GIS9 | Chicago (0.071), New York (0.091), Bridgeport (0.098), Atlanta (0.111) | Jackson_MS (0.393) 等 | 全量 GIS 特征并非无条件提升，可能同时引入解释信息和噪声维度。 |

![AEF source ranking](../paper_figures/figure4_embedding_health/figure4_panels/1_Avg_CDS_Ranking.png)

### 6.3 GIS Adds Interpretation More Reliably Than Universal Performance Gain

GIS 特征的加入并不必然提升迁移泛化能力。NDVI、slope、POI diversity 等单一 GIS 特征在部分配置中可以增强 source ranking 稳定性，但全量 9 个 GIS 特征可能引入冗余或不稳定维度。

![Cross-configuration heatmap](../paper_figures/cross_config/avg_CDS_heatmap_cross_config.png)

![Ranking correlation](../paper_figures/cross_config/ranking_correlation_matrix.png)

### 6.4 Domain Shift Explains Part Of Transfer Loss

domain shift 与迁移损失存在统计关系，但解释力有限。Label shift 是关键贡献源，AEF_L2 提供独立增量解释力，而 GIS_L2 在控制 label shift 和 AEF shift 后贡献较小。

![AEF domain shift](../results/domain_shift/aef/plots/CDSI_heatmap.png)

![Domain-shift matrix correlation](../paper_figures/cross_config/L2_distance_matrix_correlation.png)

### 6.5 Source-Domain Structure Helps Low-Cost Source Selection

源域健康指标中，`n_nodes` 是最稳健的指标之一。旧城市内部回归中，`n_nodes` 对 `avg_CDS` 的 R2 为 0.466，p = 0.005。外部新城市验证中，精确数值预测不稳定，但排序预测可行，`n_nodes` 的 Spearman rho = 0.709，p = 0.022。

![Neighbor consistency vs CDS](../results/embedding_health_analysis/neighbor_consistency_vs_CDS_noNY.png)

| Feature | Old R2 | New Spearman rho | p-value | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `n_nodes` | 0.466 | 0.709 | 0.022 | 最稳健的外部排序指标。 |
| `embedding_diversity` | 0.220 | 0.612 | 0.060 | 有潜力但统计显著性不足。 |
| `neighbor_consistency` | 0.774 after outlier removal | 0.406 | 0.244 | 旧城市内部强，但外部泛化衰减。 |
| `isotropy` | 0.031 | -0.394 | 0.260 | 外部方向不稳定。 |

![External validation](../validation/results/external_apply_iqr/all_features_external_apply_iqr.png)

![n_nodes evidence](../validation/results/final_evidence/n_nodes_vs_HGG.png)

### 6.6 MSA-Level Metrics Miss Tract-Level Failure Mechanisms

迁移误差并非在 MSA 内随机分布，而是空间集聚并集中于特定 AEF cluster。高误差 cluster 往往对应紧凑高密度建成区，表现为建筑密度高、不透水面高、覆盖率高、NDVI 低。问题不仅是 `P(X)` 的 feature shift，更是 `P(Y|X)` 的 conditional/concept shift：相同形态在不同城市中对应不同人口密度。

## 7. Figure Index

| Figure | Path |
| --- | --- |
| AEF source ranking | `paper_figures/figure4_embedding_health/figure4_panels/1_Avg_CDS_Ranking.png` |
| Cross-configuration avg_CDS heatmap | `paper_figures/cross_config/avg_CDS_heatmap_cross_config.png` |
| Ranking correlation matrix | `paper_figures/cross_config/ranking_correlation_matrix.png` |
| AEF CDSI heatmap | `results/domain_shift/aef/plots/CDSI_heatmap.png` |
| L2 domain-shift matrix correlation | `paper_figures/cross_config/L2_distance_matrix_correlation.png` |
| Neighbor consistency vs CDS | `results/embedding_health_analysis/neighbor_consistency_vs_CDS_noNY.png` |
| External validation summary | `validation/results/external_apply_iqr/all_features_external_apply_iqr.png` |
| Final n_nodes evidence | `validation/results/final_evidence/n_nodes_vs_HGG.png` |

## 8. Conclusions

1. AEF 包含可用于人口预测的空间语义信息，但跨城市迁移存在明显 domain adaptation 难题。
2. 传统 GIS 特征主要提供解释和局部补充，而不是无条件提高泛化能力；不加筛选的高维特征堆叠可能稀释迁移信号。
3. MSA 级平均 domain shift 可以解释部分迁移差异，但不足以解释 tract 级局部失败机制。
4. 迁移失败集中在特定城市形态类型中，根源更接近 concept shift，即 AEF 表征与人口响应关系在城市间发生变化。
5. 源域规模和部分嵌入健康指标可支持低成本源域筛选；当前更适合用于排序判断，而非精确预测迁移损失数值。

## 9. Future Work

| Direction | Rationale |
| --- | --- |
| Spatiotemporal transfer | 结合多时相 AEF 与人口数据，分析不同年份和城市发展阶段下的动态迁移能力。 |
| Domain adaptation and causal modeling | 建模城市间 AEF-population 响应关系异质性，而不仅是度量特征距离。 |
| Global source-selection strategy | 构建融合表征相似性、人口差异和 concept-shift 风险的源域选择与目标域适配框架。 |
