# MLP — MLP 解码器 + 迁移诊断

不使用图结构的 MLP 版本,是 GNN 的对照;`MLP_transfer.py` 还额外提供了一整套迁移性诊断工具。

## 文件与目录

| 路径 | 作用 |
|------|------|
| `MLP_regression.py` | 每城训练一个 3 层 MLP,保存 checkpoint |
| `MLP_transfer.py` | 跨城迁移 + Ridge 基线 + null 模型 + 均值校准 + 域距离 |
| `MLP_input/` | 每城「AEF64 + 单个 GIS 特征」的输入 CSV(每城约 31 个特征组合) |
| `MLP_input_pca/` | 每城「AEF64 → PCA18」输入 + PCA 元数据(components / explained_variance / standardization / all_cities scores) |

## MLP_regression.py

- **模型**:3 层 MLP(Linear→ReLU→dropout ×2 → 线性头,hidden 64,dropout 0.5)。
- **输入 / 目标 / 划分**:与 GNN 完全一致(同一数据加载与 checkpoint 格式),仅 **Adam lr 0.001**(比 GNN 的 0.01 小),batch 256。
- **输出**(`MLP_output/`):`<city>_MLP_predictions.csv`、`_tract_pred.shp`、`_training_history.csv`、`_loss_curves.png`、`<city>_MLP_model.pt`。
- **运行**:`python MLP_regression.py --all` 或 `--city <CityName>`。

## MLP_transfer.py

扫描 `--pretrained-dir`(默认 `MLP_output`)下的 `*_MLP_model.pt`,在此基础上提供:

- **迁移矩阵**:`transfer_matrix_{r2,rmse,mae,r}_<mode>.csv`、`transfer_metrics_<mode>.csv`。
- **Ridge 基线**(`--ridge`):用源城训练集拟合线性模型作对照。
- **null 模型**(`--null-model`):打乱目标域特征行、破坏 X–y 对应,检验输出偏置(理想 R²≈0)。
- **均值校准**(`--calibrate-mean`):`y_pred - mean(y_pred) + mean(y_true)`,输出 `..._calibrated.csv`。
- **域距离**:用第二隐层输出作为嵌入,算源-目标域 **L1 / L2 / 余弦 / MMD(RBF)** 距离 → `transfer_embedding_distances_<mode>.csv`(CKA 已写但注释掉)。
- **运行**:`python MLP_transfer.py --calibrate-mean --ridge --null-model`。

## MLP_input/ 与 MLP_input_pca/

- `MLP_input/<city>/<city>__AEF64_plus_<feature>.csv`:64 维 AEF + 1 个 GIS/社会经济特征(如 `developed_pct`、`elev_mean`、`poi_trans` 等),用于"AEF 加单特征"的对照实验。`aef64_plus_one_gis_summary.csv` 为汇总。
- `MLP_input_pca/`:把 64 维 AEF 降到 18 维主成分(`<city>__AEF64_PCA18.csv`),附带 PCA 载荷、解释方差、标准化参数与全城 scores,用于 MLP+PCA 实验。

> ⚠️ `MLP_input/` 是从 `data_AEF` + 各 GIS 特征 join 出来的**派生数据**(体积较大),复现时可由脚本重建。
