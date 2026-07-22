# Report Materials

`docs/report_materials/` 是项目汇报材料归档目录。当前最终版结题报告已经替换为 `docs/report_materials/结题.pptx`；根目录 `结题.pptx` 是同一份报告的项目入口副本。详细文字说明见 `docs/final_report_summary.md`。

## Current Final Deck

| 文件 | 定位 | 说明 |
| --- | --- | --- |
| `结题.pptx` | 根目录入口副本 | 便于在项目根目录直接查看最终结题报告。 |
| `docs/report_materials/结题.pptx` | 最终版结题报告归档 | 当前主版本，约 41 页，包含研究背景、数据系统、GraphSAGE 建模、跨城市迁移、domain shift、源域健康指标和外部验证。 |
| `docs/final_report_summary.md` | PPT 同步文字版 | 以最终 PPT 为主线，将研究问题、方法、结果、图件和结论整理为 Markdown。 |

## Historical Materials

| File | Role | Last Modified |
| --- | --- | --- |
| `docs/report_materials/AEF_人口开题汇报.pptx` | Opening proposal report | 2026-07-08 11:12 |
| `docs/report_materials/AEF_人口阶段汇报0713.pptx` | Stage progress report | 2026-07-15 23:43 |
| `docs/report_materials/AEF_汇报0716.pptx` | Stage progress report | 2026-07-16 02:21 |
| `docs/report_materials/AEF_汇报0720_1_带讲稿备注.pptx` | Stage progress report with speaker notes | 2026-07-21 12:40 |
| `docs/report_materials/实验方案记录.docx` | Experiment plan notes | 2026-07-16 14:54 |

## Scientific Narrative

| Final Report Section | Main Content | Related Markdown |
| --- | --- | --- |
| 研究背景与数据 | 人口空间化任务、AEF 基础模型表征、15 个美国 MSA 与 tract 级人口标签。 | `README.md`, `docs/final_report_summary.md`, `data_sources/README.md` |
| 研究方法与结果 | Queen contiguity 图、GraphSAGE、城市内预测、跨城市迁移矩阵和 CDS。 | `scripts/README.md`, `results/README.md` |
| 机制解释 | domain shift、feature L2 distance、源域健康指标、外部验证和 tract 级误差。 | `results/domain_shift/README.md`, `validation/README.md` |
| 结论与讨论 | AEF 可用于人口预测，但迁移能力存在显著空间异质性；GIS 信息具有选择性增益。 | `docs/final_report_summary.md` |

## Figure References

| Figure | Relative Path |
| --- | --- |
| Avg CDS ranking | `paper_figures/figure4_embedding_health/figure4_panels/1_Avg_CDS_Ranking.png` |
| Cross-configuration avg CDS heatmap | `paper_figures/cross_config/avg_CDS_heatmap_cross_config.png` |
| Configuration ranking correlation | `paper_figures/cross_config/ranking_correlation_matrix.png` |
| AEF domain-shift heatmap | `results/domain_shift/aef/plots/CDSI_heatmap.png` |
| Neighbor consistency vs CDS | `results/embedding_health_analysis/neighbor_consistency_vs_CDS_noNY.png` |
| External validation summary | `validation/results/external_apply_iqr/all_features_external_apply_iqr.png` |

These files are report materials rather than runnable code artifacts. All paths above are relative to the repository root.
