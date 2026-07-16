# model — 建模、迁移与归因

人口预测下游任务的建模代码与结果。两条并行的解码器(**GNN** / **MLP**)共享同一套数据加载与 checkpoint 格式,流程都是"**先在各城自训练 → 再跨城评估迁移 → 最后归因分析**"。

## 目录结构

```
model/
├── GNN/                 # GraphSAGE 解码器:自训练 + 跨城迁移
│   ├── GNN_regression.py    # 每城训练一个 GraphSAGE,保存 _model.pt
│   └── GNN_transfer.py      # 读取权重,做 15×15 跨城迁移矩阵
├── MLP/                 # MLP 解码器(无图结构),含更完整的迁移诊断
│   ├── MLP_regression.py    # 每城训练一个 3 层 MLP
│   ├── MLP_transfer.py      # 迁移 + Ridge 基线 + null 模型 + 均值校准 + 域距离
│   ├── MLP_input/           # 每城「AEF64 + 单个 GIS 特征」输入 CSV
│   └── MLP_input_pca/       # 每城「AEF64 降维到 PCA18」输入 + PCA 元数据
├── Analysis/            # 迁移矩阵的分析与可视化、"域距离→迁移性能"归因
├── AEF_plus_GIS/        # AEF + 单个 GIS 特征的迁移实验结果
└── README.md            # 本文件
```

> 目录曾名 `AEF&GIS`,因 `&` 在命令行/URL 中需转义,已重命名为 `AEF_plus_GIS`。

## 建模设置(GNN 与 MLP 一致)

| 项 | 设置 |
|----|------|
| 特征 | 64 维 AEF 嵌入(列 `A00`–`A63`) |
| 预测目标 | census tract 人口密度(人口 ÷ tract 面积) |
| 目标变换 | `log1p`(默认),再按训练集做标准化 |
| 数据划分 | 每城**随机 70/15/15**(train/val/test,固定随机种子)——注意这是城内划分,不是 leave-one-MSA-out |
| 跨城泛化 | 由 `*_transfer.py` 单独评估:源城模型 → 整个目标城当测试集 |
| 指标 | R² / RMSE / MAE / MAPE(在还原后的密度尺度上) |

## 两条流水线

**GNN**:`GNN_regression.py --all` 每城训练 GraphSAGE(2×`SAGEConv`+线性头,hidden 64,dropout 0.5),把 `<city>_GraphSAGE_model.pt`(含权重 + scaler + 目标变换 + 均值方差)存到 `GNN_output/`。→ `GNN_transfer.py --pretrained-dir ...` 扫描这些权重,逐 (源→目标) 评估,产出 `transfer_matrix_{r2,rmse,mae}.csv`。

**MLP**:`MLP_regression.py` 同理训练 3 层 MLP 存到 `MLP_output/`。→ `MLP_transfer.py` 除迁移矩阵外,额外给出:**Ridge 线性基线**、**null 模型**(打乱特征行破坏 X–y 对应,检验输出偏置)、**均值校准**(`y_pred - mean(y_pred) + mean(y_true)`)、以及源-目标域的 **L1/L2/余弦/MMD** 嵌入距离(`transfer_embedding_distances_*.csv`)。

**Analysis**:上述 `transfer_matrix_r2_*_calibrated.csv` 与 `transfer_embedding_distances_*.csv` 是分析脚本的输入,做 IQR 异常剔除、相关性、可视化,以及"域距离 → 迁移性能"回归归因(见 `Analysis/README.md`)。

## 期望的数据布局

脚本用相对路径,期望项目根下有:AEF 输入(`data_AEF/<city>/aef_*_b*_2020.csv`,列 `A00`–`A63` + `TRACT_ID`)、人口标签(ACS `B01003`)、以及输出目录 `GNN_output/`、`MLP_output/`。各 `experiment_args.json` 里记录了当次运行的绝对路径与超参(其中的 `E:\3S\...` 为跑实验机器的路径,复现时按实际改)。

## 输入数据说明

- **纯 AEF 输入**:`data_AEF/`(仓库根目录)
- **纯 GIS 输入**:`data_GIS/`(仓库根目录),其构建代码见 `data_GIS/create_dataset.ipynb`、`data_GIS/feature_corr.ipynb`
- **AEF + 单个 GIS 特征**:`model/MLP/MLP_input/`
