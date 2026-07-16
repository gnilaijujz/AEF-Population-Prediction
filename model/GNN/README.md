# GNN — GraphSAGE 解码器(自训练 + 跨城迁移)

## 文件

| 文件 | 作用 |
|------|------|
| `GNN_regression.py` | 每个城市训练一个 GraphSAGE 回归器,预测 tract 人口密度,保存 checkpoint |
| `GNN_transfer.py` | 加载已训练的 checkpoint,把每个源城模型在所有目标城上评估,构建跨城迁移矩阵(不训练) |

## GNN_regression.py

- **模型**:2 层 `SAGEConv`(mean 聚合,hidden 64)+ 线性输出头,ReLU + dropout 0.5。
- **输入**:`data_AEF/<city>/aef_*_b*_2020.csv`(列 `A00`–`A63`、`TRACT_ID`);ACS 人口 `ACSDT5Y2020.B01003-Data.csv`(`GEO_ID`、`B01003_001E`);各城 tract shapefile(自动探测)。
- **目标**:人口 ÷ tract 面积(km²);默认 `log1p` 变换后按训练集 `StandardScaler` 标准化。
- **划分**:随机 70/15/15(train/val/test,NeighborLoader minibatch `[10,5]`),Adam lr 0.01、wd 5e-4,按标准化后的验证 MSE 早停。
- **输出**(`GNN_output/output_try_log/`):`<city>_GraphSAGE_predictions.csv`、`_tract_pred.shp`、`_training_history.csv`、`_loss_curves.png`、`<city>_GraphSAGE_model.pt`(state_dict + scaler + 目标变换 + y 均值/方差 + feature_cols + args + 指标)。
- **运行**:`python GNN_regression.py --all` 或 `python GNN_regression.py --city <CityName>`。

## GNN_transfer.py

- 扫描 `--pretrained-dir`(默认 `GNN_output/output_try_log`)下的 `*_GraphSAGE_model.pt`,复用每个 checkpoint 里保存的 scaler / 目标变换 / 均值方差;**整个目标城作为测试集**。
- 跳过 feature 列不一致的城市对;指标 R²/RMSE/MAE/MAPE。
- **输出**(`GNN_output/transfer_results_try_log/`):`transfer_metrics_long.csv`、`transfer_matrix_{r2,rmse,mae}.csv`、`transfer_city_errors.csv`、`transfer_run_args.json`;`--save-pair-outputs` 时额外写 `pair_predictions/<src>__to__<tgt>/`。
- **运行**(需在本目录下运行,以便 `import GNN_regression`):`python GNN_transfer.py --save-pair-outputs`。

## 关系

`GNN_regression.py` 产出的 `_model.pt` 权重,正是 `GNN_transfer.py` 通过 `--pretrained-dir` 消费的对象。迁移矩阵进一步作为 `../Analysis/` 的输入。
