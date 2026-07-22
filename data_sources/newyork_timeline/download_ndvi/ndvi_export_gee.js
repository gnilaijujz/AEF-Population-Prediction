// =====================================================================
//  NDVI (ndvi_mean) 逐年 tract 均值导出 · New York MSA · 时间维度
//  用法：粘到 GEE 代码编辑器 https://code.earthengine.google.com/
//        改第 12 行 YEARS 列表 → Run 一次 → Tasks 面板会出现 N 个任务
//        逐个点每个任务的 RUN（有几年就点几下），全部导到 Google Drive
//
//  数据：LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI（原生 30m 年度合成）
//  口径对齐 2020：scale=30、Reducer.mean() 显式命名成 'ndvi_mean'
//  只跑 New York MSA（同一套 2020 tract 矢量，几何冻结），键 = cb_2020__3
//  参考：中期前/ndvi_features_gee.js（当时是全 CONUS）+ download_aef/aef_export_gee.js
// =====================================================================

// 要跑的年份（2020 已有；若要让 2020 也重算成同一套口径，保留即可，多导一份不亏）
var YEARS = [2017, 2018, 2019, 2020, 2021];

// 1) 你上传的 2020 纽约 tract 矢量资产（与 AEF 用的同一个资产路径）
var tracts = ee.FeatureCollection('projects/YOUR_PROJECT/assets/newyork_tracts_2020');

var SCALE = 30;   // Landsat 原生 30m，与 2020 一致

// —— 先核对字段名：确认 tract 键是 'cb_2020__3'（跑通后可注释掉）——
print('tract 数量', tracts.size());
print('第一条属性', tracts.first());

// 2) 对每个年份各建一个导出任务
YEARS.forEach(function (YEAR) {
  var ndvi = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI')
               .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')  // 该年整年
               .filterBounds(tracts)
               .select('NDVI')
               .mosaic()               // 年度合成拼成一张无缝图
               .rename('ndvi');

  var stats = ndvi.reduceRegions({
    collection: tracts,
    // ⚠️单波段必须显式命名，否则会被命名成 'mean' 导致整列取不到 → 全空
    reducer:    ee.Reducer.mean().setOutputs(['ndvi_mean']),
    scale:      SCALE,
    tileScale:  16                      // 内存超限就调大到 32
  });

  // 只留 key + 特征，键名对齐 CSV（cb_2020_3，单下划线）
  stats = stats.map(function (f) {
    return ee.Feature(null, {
      'cb_2020_3': f.get('cb_2020__3'),
      'ndvi_mean': f.get('ndvi_mean')
    });
  });

  Export.table.toDrive({
    collection:     stats,
    description:    'ndvi_newyork_' + YEAR,
    folder:         'NDVI_export',
    fileNamePrefix: 'ndvi_newyork_' + YEAR,
    fileFormat:     'CSV',
    selectors:      ['cb_2020_3', 'ndvi_mean']
  });
});
