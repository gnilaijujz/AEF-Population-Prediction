# AEF_plus_GIS — AEF + 单个 GIS 特征的迁移实验结果

在纯 AEF(64 维)基础上**各加入一个 GIS/地形特征**,用 MLP 解码器重跑跨城迁移,观察该特征能否缓解域偏移、改善迁移性能。每个子目录是一次实验的结果。

## 子目录(按加入的特征区分)

| 子目录 | 加入的特征(`input_feature`) | 含义 |
|--------|------------------------------|------|
| `transfer_results/` | `elev_mean` | AEF + **海拔** |
| `plus_developed_pct/` | `developed_pct` | AEF + 建成用地占比 |
| `plus_poi_trans/` | `poi_trans` | AEF + 公共交通 POI 密度 |

## 每个子目录的文件

| 文件 | 内容 |
|------|------|
| `experiment_args.json` | 本次运行的参数(`input_feature`、超参、路径等) |
| `transfer_matrix_r2_mlp.csv` | 15×15 跨城迁移 R² 矩阵(行=源城,列=目标城) |
| `transfer_matrix_r_mlp.csv` | 皮尔逊 r 矩阵 |
| `transfer_matrix_rmse_mlp.csv` / `transfer_matrix_mae_mlp.csv` | RMSE / MAE 矩阵 |
| `transfer_metrics_mlp.csv` | 长表形式的逐对指标 |
| `transfer_embedding_distances_mlp.csv` | 源-目标域 L1/L2/余弦/MMD 嵌入距离 |

## 用法

这些结果由 `../MLP/MLP_transfer.py` 生成,是 `../Analysis/` 归因分析的对照输入——用于比较"纯 AEF"与"AEF+某特征"两种设定下的迁移矩阵与域距离,量化辅助特征对域偏移的缓解效果(参照基线斜率 β₀=-0.0189,见 `../Analysis/README.md`)。
