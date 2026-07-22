# docs/report_materials

本文件夹保存项目汇报材料、实验方案记录和最终结题报告归档。当前项目的最终版结题报告为 `结题.pptx`；根目录下的 `结题.pptx` 是同一份报告的入口副本，便于打开项目时直接查看研究说明。

## 文件定位

| 文件 | 定位 | 使用说明 |
| --- | --- | --- |
| `结题.pptx` | 最终版结题报告 | 以 41 页最终汇报为准，系统说明研究背景、数据、方法、迁移评估、机制解释和结论。 |
| `../../结题.pptx` | 根目录入口副本 | 与本文件夹中的最终版 PPT 内容一致，用作项目总览入口。 |
| `../final_report_summary.md` | 最终报告文字版 | 将 `结题.pptx` 的研究逻辑整理为可检索、可维护的 Markdown 说明。 |
| `实验方案记录.docx` | 实验设计记录 | 保存中间阶段的问题设定、模型方案和实验计划。 |
| `AEF_人口开题汇报.pptx` | 开题汇报 | 记录项目早期研究动机、数据设想和技术路线。 |
| `AEF_人口阶段汇报0713.pptx` | 阶段汇报 | 记录 7 月 13 日前后的数据整理和初步实验进展。 |
| `AEF_汇报0716.pptx` | 阶段汇报 | 记录 7 月 16 日前后的模型与迁移实验调整。 |
| `AEF_汇报0720_1_带讲稿备注.pptx` | 阶段汇报 | 记录接近结题前的实验结果、讲稿备注和图表草稿。 |

## 最终报告结构

最终版 `结题.pptx` 的核心叙事可以概括为：从人口预测任务出发，评估 AEF 表征的跨城市空间可迁移能力，并解释迁移差异的来源。

| 报告部分 | 科学问题 | 对应项目材料 |
| --- | --- | --- |
| 研究背景与数据 | AEF embedding 是否包含人口相关空间语义？ | `../../README.md`、`../final_report_summary.md`、`../../data_sources/README.md` |
| 研究方法与结果 | AEF、GIS、AEF+GIS 在城市内预测和跨城市迁移中的表现如何？ | `../../scripts/README.md`、`../../results/README.md`、`../../paper_figures/README.md` |
| 结论与讨论 | 迁移能力差异能否由 domain shift、源域健康指标和 tract 级误差解释？ | `../../results/domain_shift/README.md`、`../../results/embedding_health_analysis/README.md`、`../../validation/README.md` |
| 小组分工 | 不同成员承担的数据、模型、验证和展示工作如何组织？ | `实验方案记录.docx` 与各阶段汇报材料 |

## 关键图件索引

最终 PPT 中使用或对应的核心图件主要来自 `paper_figures/`、`results/` 和 `validation/results/`。这些图件不重复存放在本文件夹中，以避免材料版本不一致。

| 图件 | 说明 | 路径 |
| --- | --- | --- |
| AEF source ranking | AEF 配置下不同源城市的平均迁移损失排序。 | `../../paper_figures/figure4_embedding_health/figure4_panels/1_Avg_CDS_Ranking.png` |
| Cross-configuration heatmap | AEF、GIS、AEF+GIS 及单 GIS 配置下的 avg_CDS 对比。 | `../../paper_figures/cross_config/avg_CDS_heatmap_cross_config.png` |
| Ranking correlation | 不同特征配置下源城市排序的一致性。 | `../../paper_figures/cross_config/ranking_correlation_matrix.png` |
| Domain shift heatmap | AEF domain shift 指标与 CDS 的相关结构。 | `../../results/domain_shift/aef/plots/CDSI_heatmap.png` |
| Embedding health | 源域 embedding 结构健康度与迁移表现的关系。 | `../../results/embedding_health_analysis/neighbor_consistency_vs_CDS_noNY.png` |
| External validation | 新 15 个 MSA 上的源域选择指标外部验证。 | `../../validation/results/external_apply_iqr/all_features_external_apply_iqr.png` |

## 维护规则

- 以后若修改最终汇报，应优先替换 `docs/report_materials/结题.pptx`，再同步根目录 `结题.pptx`。
- 若 PPT 中的结论、数值或图件发生变化，需要同步更新 `../../README.md` 和 `../final_report_summary.md`。
- 本文件夹只保存报告材料，不放可执行代码；实验脚本应放在 `../../scripts/`，模型输入应放在 `../../model_data/` 或 `../../data_sources/`。
- 本项目文档中的路径均以仓库根目录为基准，使用相对路径，避免写入本机绝对路径。
