# docs

本文件夹保存项目说明、最终报告同步文档、结构整理记录和汇报材料索引。这里的 Markdown 负责把最终版结题报告中的研究逻辑转化为可维护的项目文档；实验代码仍保存在 `../scripts/`，输入数据保存在 `../data_sources/` 与 `../model_data/`，输出结果保存在 `../results/` 与 `../validation/results/`。

## 文档结构

| 文件或文件夹 | 作用 | 维护重点 |
| --- | --- | --- |
| `../README.md` | 项目总说明 | 面向读者解释研究问题、数据体系、方法、主要发现、图件和复现实验入口。 |
| `final_report_summary.md` | 最终 PPT 同步文字版 | 与 `report_materials/结题.pptx` 保持一致，详细记录研究问题、模型流程、迁移指标、主要结果和结论。 |
| `report_materials/` | 汇报材料归档 | 保存最终结题报告、根目录副本说明、阶段汇报和实验方案记录。 |
| `report_materials.md` | 汇报材料总索引 | 从 docs 层级说明最终报告、历史材料、图件引用和相关 Markdown。 |
| `reorganization_notes.md` | 项目整理记录 | 记录本次目录重命名、数据归类、路径相对化和后续维护规则。 |
| `legacy_parent_README_20260716.md` | 历史说明备份 | 保留整理前的上级 README 内容，避免信息丢失。 |

## 最终报告同步关系

| 项目入口 | 文件 | 说明 |
| --- | --- | --- |
| 根目录 PPT | `../结题.pptx` | 最终结题报告的便捷入口，放在根目录是有意保留。 |
| 报告归档 PPT | `report_materials/结题.pptx` | 当前最终版结题报告主归档。 |
| 文字版报告 | `final_report_summary.md` | 将 PPT 内容整理为 Markdown，便于后续更新、引用和 GitHub 展示。 |
| 项目总 README | `../README.md` | 结合最终报告、数据说明和实验结果形成完整项目说明。 |

## 报告中的核心图表来源

这些图件在 Markdown 中通过相对路径引用，不复制到 `docs/`，以保证图表只有一个真实来源。

| 图表主题 | 路径 |
| --- | --- |
| AEF 源城市 avg_CDS 排序 | `../paper_figures/figure4_embedding_health/figure4_panels/1_Avg_CDS_Ranking.png` |
| 多特征配置 avg_CDS 热力图 | `../paper_figures/cross_config/avg_CDS_heatmap_cross_config.png` |
| 配置排序相关矩阵 | `../paper_figures/cross_config/ranking_correlation_matrix.png` |
| AEF domain shift 相关热力图 | `../results/domain_shift/aef/plots/CDSI_heatmap.png` |
| 源域健康指标解释图 | `../results/embedding_health_analysis/neighbor_consistency_vs_CDS_noNY.png` |
| 外部 MSA 验证图 | `../validation/results/external_apply_iqr/all_features_external_apply_iqr.png` |

## 维护规则

- 若最终 PPT 被替换，应同步检查 `../README.md`、`final_report_summary.md`、`report_materials.md` 和 `report_materials/README.md`。
- 若实验数值或图表被重跑，应优先更新 `results/` 或 `validation/results/` 中的真实输出，再更新文档中的表格和解释。
- 所有文档路径以仓库根目录为基准使用相对路径，不写入本机绝对路径。
- 本文件夹不包含可执行脚本；若以后新增文档生成脚本，应放入 `../scripts/` 并在相应 README 中说明。
