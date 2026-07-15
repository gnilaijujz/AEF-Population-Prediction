# 面向人口预测任务的地理空间基础模型空间可迁移能力评估

**Evaluating the Spatial Transferability of a Geospatial Foundation Model (AEF) for Population Prediction**

> 以美国大都市统计区(MSA)的人口估计为下游任务,评估 Google DeepMind **AlphaEarth Foundations (AEF)** 卫星嵌入的空间可迁移能力,并与传统 GIS 特征对比,归因人口预测误差的空间不均匀性。

- 指导老师:周智勇
- 组长:李书亦　组员:刘桂心瑜、于佳灵

---

## 1. 研究背景与科学问题

当前地理空间基础模型(GeoFM)研究重「外部评估」、缺「内部评估」;AEF 对人口估计的提升在**空间与尺度上并不均匀**。本研究以人口估计为下游任务,系统评估 AEF 的空间可迁移能力。

**两个核心任务:**

| 任务 | 内容 | 方法 | 产出 |
|------|------|------|------|
| **任务一 · 特征工程对照** | 三组特征对比:① 仅 AEF　② 仅 GIS　③ AEF+GIS 融合 | 解码器 GCN 图卷积(固定),备选 RF;空间留一交叉验证(leave-one-MSA-out) | R² / RMSE / MAE / KL;空间误差地图、残差 LISA 空间聚集分析 |
| **任务二 · 可迁移性** | 15×15 全连接迁移矩阵,每对 (source→target) 评估迁移性 | 域偏移量化(AEF 嵌入距离 L1/L2/余弦/分布 + GIS 特征距离);回归建立「特征距离 → 迁移性能」映射 | N×N 迁移矩阵;验证核心假设「距离越大,迁移越差」 |

---

## 2. 研究区与研究单元

- **研究单元:** Census Tract(普查区,约 1,200–8,000 人,理想约 4,000 人)
- **人口标签:** 美国官方统计数据(Census Bureau / ACS 表 `B01003`,`Estimate!!Total`);因 WorldPop(~1km)分辨率不足以支撑 tract 级预测,改用官方统计口径
- **抽样:** 从筛选出 tract 数 > 100 的 MSA 中,按人口分**高 / 中 / 低**三层随机抽样,得到 **15 个 MSA**

| 层级 | MSA(CBSA) | 2020 人口 | tract 数 |
|------|-----------|-----------|---------|
| high | New York-Newark-Jersey City, NY-NJ-PA (35620) | 19,261,570 | 4,953 |
| high | Chicago-Naperville-Elgin, IL-IN-WI (16980) | 9,478,801 | 2,335 |
| high | Atlanta-Sandy Springs-Alpharetta, GA (12060) | 5,947,008 | 1,500 |
| high | Oklahoma City, OK (36420) | 1,397,040 | 419 |
| high | Hartford-East Hartford-Middletown, CT (25540) | 1,205,842 | 308 |
| medium | Bridgeport-Stamford-Norwalk, CT (14860) | 944,306 | 227 |
| medium | Albany-Schenectady-Troy, NY (10580) | 880,766 | 251 |
| medium | Knoxville, TN (28940) | 861,872 | 225 |
| medium | Baton Rouge, LA (12940) | 856,779 | 215 |
| medium | Jackson, MS (27140) | 596,287 | 160 |
| low | Lansing-East Lansing, MI (29620) | 547,786 | 154 |
| low | Modesto, CA (33700) | 546,235 | 112 |
| low | Fort Wayne, IN (23060) | 409,419 | 104 |
| low | Montgomery, AL (33860) | 373,552 | 112 |
| low | Duluth, MN-WI (20260) | 289,276 | 104 |

---

## 3. 数据

### 3.1 AEF 嵌入(核心数据)

**AlphaEarth Foundations Satellite Embedding**(Google & Google DeepMind),经 Google Earth Engine 获取。提取 2020 年 AEF 影像,与各 MSA 的普查区边界叠加,对每个普查区内所有 10m 像素取平均,得到每区域 **64 维**特征向量,导出 CSV。

- 规格:64 维 · 10m · 年度(2017–2024,本研究用 2020)
- 预处理:尺度对齐到普查单元;AEF 暂不标准化;人口长尾则 log 变换;稀疏区 mask

### 3.2 传统 GIS 特征目录

所有特征均聚合到 census tract 级,连接键统一为 `cb_2020_3`(`1400000US` + 11 位 tract 码),原始值输出、标准化交统一建模流程。

**第一块 · 社会经济**

| 特征组 | 文件 | 主要字段 | 来源 / 尺度 / 时间 |
|--------|------|----------|-------------------|
| 人口结构 | `pop_features_15msa.csv` | `minor_ratio`(未成年占比)、`elderly_ratio`(老年占比)、`sex_ratio`(男/女×100) | ACS 5-year 2016–2020,表 DP05 |
| 收入 / 贫困 | `socioeconomic_features_15msa.csv` | 家庭收入 5 档户数、`median_family_income`、`per_capita_income`、`persons_below_poverty`(+ `*_moe`) | IPUMS NHGIS(底层 ACS 5yr 2016–2020,表 A88/AB2/BD5/CL6) |

**第二块 · 人类活动与建成环境**

| 特征组 | 文件 | 主要字段 | 来源 / 分辨率 / 时间 |
|--------|------|----------|---------------------|
| 土地覆盖 | `nlcd_features_15msa.csv` | 10 类 `*_pct` 面积占比 + `impervious_mean`(不透水面) | Annual NLCD 2020(MRLC),30m |
| 夜间灯光 | `ntl_features_15msa.csv` | `ntl_sum/mean/intensity_per_km2/std/cv/max/min/count` | VIIRS Nighttime Lights Annual V2 (masked) 2020,~500m |
| 建筑 | `building_features_15msa.csv` | `building_count`、`building_area_m2`、`building_density`、`coverage_ratio` | OpenStreetMap 建筑轮廓,经 ohsome API,2020 |
| POI | `poi_features_15msa.csv` | `poi_density` | OpenStreetMap,2020 |
| 道路 | (待整合) | `road_density_local/secondary`、`road_len_total_m` | Census TIGER 三级道路,2020 |

**第三块 · 地理与环境**

| 特征组 | 文件 | 主要字段 | 来源 / 分辨率 / 时间 |
|--------|------|----------|---------------------|
| 地形 | `dem_features_15msa.csv` | `elev_mean`(高程)、`slope_mean`(坡度) | USGS 3DEP(经 GEE,~10m 取样至 30m) |
| 植被 | `ndvi_features_15msa.csv` | `ndvi_mean` | Landsat C2 T1 L2 年度 NDVI 合成(GEE),30m,2020 |
| 树冠 | `tcc_features_15msa.csv` | `tcc_mean`(树冠覆盖率) | USFS/NLCD Tree Canopy Cover(GEE),30m,2020 |
| 水体距离 | `water_dist_features_15msa.csv` | `dist_water_m`(到最近水体平均距离) | JRC Global Surface Water v1.4(GEE),30m,2020 |

> 详细字段口径、处理方法与数据来源引用,见各特征目录下的 `*_数据说明.docx`(仅本地保存,不入库)。

---

## 4. 仓库结构

```
AEF-Population-Prediction/
├── README.md
├── .gitignore
├── GIS-feature-dataset/          # 合并后的建模数据集
│   ├── create_dataset.ipynb      #   特征合并流程
│   ├── feature_corr.ipynb        #   特征相关性分析
│   └── *_features_15msa.csv      #   8 组处理后特征表
├── MSA/
│   └── 15MSA/                    # 研究区筛选
│       ├── select_msa.py         #   高/中/低分层抽样脚本
│       └── *.csv                 #   15MSA 名单 / tract 清单 / 人口标签
├── 传统GIS特征数据/               # 各特征的提取代码
│   ├── 15msa/                    #   处理后 15MSA 特征表
│   ├── 社会经济数据/              #   NHGIS / 人口结构提取脚本
│   ├── 人类活动和建成环境/         #   建筑(.py)、POI、NLCD、NTL、道路
│   └── 地理数据和环境数据/         #   DEM / NDVI / 树冠 / 水体距离(GEE .js 脚本)
└── model/                        # 模型代码与结果(待导入)
```

> **未入库内容**(见 `.gitignore`):原始栅格/矢量大数据、census 原始下载、`*_数据说明.docx`、汇报 PPT/PDF,以及全国 tract 级中间 CSV。这些数据体积大或受再分发限制,请从下列来源自行获取。

---

## 5. 数据可用性与合规

- ⚠️ **NHGIS 原始数据禁止再分发**:本仓库仅提供处理代码与来源说明,不包含 NHGIS 原始表(`nhgis0001*`)。请自行从 [data2.nhgis.org](https://data2.nhgis.org) 获取。
- MSA 边界矢量:[TIGER/Line 2020 CBSA](https://www2.census.gov/geo/tiger/TIGER2020/CBSA/)
- 人口标签:[Census Bureau Data](https://data.census.gov/)(表 B01003 / DP05)
- 遥感与环境特征均可经 [Google Earth Engine](https://earthengine.google.com/) 复现(见各 `*_gee.js` 脚本)

**数据来源引用:**
- IPUMS NHGIS, University of Minnesota, [www.nhgis.org](https://www.nhgis.org)
- VIIRS Nighttime Lights Annual V2 (masked), Earth Observation Group, Colorado School of Mines
- Pekel et al., 2016, *Nature*(JRC Global Surface Water)
- USGS 3DEP;Annual NLCD (MRLC);USFS/NLCD Tree Canopy Cover;Landsat / OpenStreetMap

---

## 6. 研究进展(截至 2026-07)

- ✅ 完成全部特征数据准备
- ✅ 跑通第一组「仅 AEF」实验;多数城市 self-prediction 可收敛并取得正 R²,说明 AEF 具备人口预测能力
- ✅ 完成解码器选择、设计空间交叉验证方案、完成指标评估
- ✅ 获得第一个 15×15 迁移矩阵

**初步结论:** 空间可迁移性整体有限;部分城市对存在较好迁移潜力;存在极端负值(迁移风险警示);地理邻近未必带来迁移优势;城市规模与可迁移性具备一定关联。
