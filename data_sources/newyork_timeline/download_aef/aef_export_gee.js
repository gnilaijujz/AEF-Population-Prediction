// =====================================================================
//  AEF (AlphaEarth Satellite Embedding) 逐年 tract 均值导出 · New York MSA
//  用法：粘到 GEE 代码编辑器 https://code.earthengine.google.com/
//        改第 9 行 YEARS 列表 → Run 一次 → Tasks 面板会出现 N 个任务
//        逐个点每个任务的 RUN（有几年就点几下），全部导到 Google Drive
// =====================================================================

// 要跑的年份（2020 已有；如需让 2020 也重算成同一套 4941 tract，把 2020 加进来）
var YEARS = [2017, 2018, 2019, 2021];

// 1) 你上传的 2020 纽约 tract 矢量资产（改成你自己的 asset 路径）
var tracts = ee.FeatureCollection('projects/YOUR_PROJECT/assets/newyork_tracts_2020');

// —— 先核对字段名：确认 tract 键是 'cb_2020__3'（跑通后可注释掉）——
print('tract 数量', tracts.size());
print('第一条属性', tracts.first());

// 导出列：tract 键 + A00..A63
var bands = [];
for (var i = 0; i < 64; i++) { bands.push('A' + (i < 10 ? '0' + i : '' + i)); }
var cols = ['cb_2020__3'].concat(bands);   // ← 若键名不是 cb_2020__3，改这里

// 2) 对每个年份各建一个导出任务
YEARS.forEach(function (YEAR) {
  var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
              .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
              .filterBounds(tracts)
              .mosaic();                       // 该年 64 波段年度嵌入

  var out = emb.reduceRegions({
    collection: tracts,
    reducer:    ee.Reducer.mean(),
    scale:      10,
    tileScale:  16                             // 内存超限就调大到 32
  });

  Export.table.toDrive({
    collection:     out,
    description:    'aef_newyork_' + YEAR,
    folder:         'AEF_export',
    fileNamePrefix: 'aef_newyork_' + YEAR,
    fileFormat:     'CSV',
    selectors:      cols
  });
});
