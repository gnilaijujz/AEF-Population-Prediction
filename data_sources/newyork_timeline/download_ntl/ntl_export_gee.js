// =====================================================================
//  夜间灯光 ntl_sum 逐年 tract 求和导出 · New York MSA · 时间维度
//  用法：粘到 GEE 代码编辑器 https://code.earthengine.google.com/
//        改第 22 行 tracts 资产路径 → Run 一次 → Tasks 面板出现 N 个任务
//        逐个点每个任务的 RUN（有几年就点几下），全部导到 Google Drive
//
//  数据：NOAA/VIIRS/DNB/ANNUAL_V21，波段 average_masked（= EOG VNL V2.1 masked 版）
//  口径对齐 2020：masked 版年度合成、tract 内像元辐亮度【求和】= ntl_sum、~500m 原生
//  ★本脚本把 2020 也一并重算★，让 2017–2021 五年全走同一 GEE 管线（而非本地接 GEE）
//  只跑 New York MSA（同一套 2020 tract 矢量，几何冻结），键 = cb_2020__3
//  引用：VIIRS Nighttime Lights Annual V2 (masked), Earth Observation Group, Colorado School of Mines
// =====================================================================

// 要跑的年份（2020 也重算，五年同管线可比）
var YEARS = [2017, 2018, 2019, 2020, 2021];   // ANNUAL_V21 覆盖 2012–2021；2022+ 在 V22，勿超

// ===== 0. 参数 =====
// 你上传的 2020 纽约 tract 矢量资产（与 AEF / NDVI / 水体用的同一个资产路径）
var TRACTS_ASSET = 'projects/YOUR_PROJECT/assets/newyork_tracts_2020';   // ← 改成你的资产
var BAND  = 'average_masked';   // ★masked 版，和 2020 一致（勿用未 masked 的 'average'）
var SCALE = 463.83;             // VIIRS V2.1 原生 ~500m（15 弧秒≈463.83m）；求和须在原生尺度

// 1) 载入 tract（NY 资产本身已是纽约 MSA，无需再按 CONUS 过滤）
var tracts = ee.FeatureCollection(TRACTS_ASSET);

// —— 先核对字段名：确认 tract 键是 'cb_2020__3'（双下划线，跑通后可注释掉）——
print('tract 数量', tracts.size());
print('第一条属性', tracts.first());

// ===== 主流程：每年一套 =====
YEARS.forEach(function (YEAR) {

  // ---- 2. 该年年度夜光影像（masked 波段）----
  var ntl = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
              .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
              .first()
              .select(BAND);

  // ---- 3. 每个 tract 求和 = ntl_sum ----
  var stats = ntl.reduceRegions({
    collection: tracts,
    // ⚠️单波段必须显式命名，否则会被命名成 'sum' 之外的默认名 → 对不上列
    reducer:    ee.Reducer.sum().setOutputs(['ntl_sum']),
    scale:      SCALE,
    tileScale:  16                       // 内存超限就调大到 32
  });

  // ---- 4. 只留 key + 特征，键名对齐 CSV（cb_2020_3，单下划线）----
  stats = stats.map(function (f) {
    return ee.Feature(null, {
      'cb_2020_3': f.get('cb_2020__3'),
      'ntl_sum':   f.get('ntl_sum')
    });
  });

  // ---- 5. 导出 ----
  Export.table.toDrive({
    collection:     stats,
    description:    'ntl_newyork_' + YEAR,
    folder:         'NTL_export',
    fileNamePrefix: 'ntl_newyork_' + YEAR,
    fileFormat:     'CSV',
    selectors:      ['cb_2020_3', 'ntl_sum']
  });
});

// 可视化自查（可选）：某一年的 masked 夜光
// Map.setCenter(-74.0, 40.7, 8);
// var w = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
//           .filterDate('2020-01-01','2021-01-01').first().select('average_masked');
// Map.addLayer(w, {min:0, max:60, palette:['000000','ffff00','ffffff']}, 'NTL masked 2020');
