// =====================================================================
//  不透水面 impervious_mean 逐年 tract 均值导出 · New York MSA · 时间维度
//  用法：粘到 GEE 代码编辑器 https://code.earthengine.google.com/
//        改第 22 行 tracts 资产路径 → Run 一次 → Tasks 面板出现 N 个任务
//        逐个点每个任务的 RUN（有几年就点几下），全部导到 Google Drive
//
//  数据：Annual NLCD · Fractional Impervious Surface（FctImp，0–100% 不透水比例，30m）
//        GEE 社区目录(sat-io)：projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/FRACTIONAL_IMPERVIOUS_SURFACE
//  口径对齐 2020：直接取【不透水产品】的 tract 均值 = impervious_mean（不是从 LndCov 反算）
//  ★本脚本把 2020 也一并重算★，让 2017–2021 五年全走同一 GEE/同一发布管线（而非本地接 GEE）
//  只跑 New York MSA（同一套 2020 tract 矢量，几何冻结），键 = cb_2020__3
//  引用：USGS, 2024, Annual NLCD Collection 1 Science Products, doi:10.5066/P94UXNTS
// =====================================================================

// 要跑的年份（2020 也重算，五年同管线可比）
var YEARS = [2017, 2018, 2019, 2020, 2021];   // Annual NLCD 覆盖 1985–2025，此区间均有

// ===== 0. 参数 =====
// 你上传的 2020 纽约 tract 矢量资产（与 AEF / NDVI / 水体 / NTL 用的同一个资产路径）
var TRACTS_ASSET = 'projects/YOUR_PROJECT/assets/newyork_tracts_2020';   // ← 改成你的资产
var IMP_COL = 'projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/FRACTIONAL_IMPERVIOUS_SURFACE';
var SCALE   = 30;   // NLCD 原生 30m（Albers/EPSG:5070），与 2020 一致

// 1) 载入 tract（NY 资产本身已是纽约 MSA，无需再按 CONUS 过滤）
var tracts = ee.FeatureCollection(TRACTS_ASSET);

// —— 先核对字段名：确认 tract 键是 'cb_2020__3'（双下划线，跑通后可注释掉）——
print('tract 数量', tracts.size());
print('第一条属性', tracts.first());
// 也顺便看一眼不透水集合的波段名/年份（跑通后可注释掉）
print('不透水集合示例', ee.Image(ee.ImageCollection(IMP_COL).first()));

// ===== 主流程：每年一套 =====
YEARS.forEach(function (YEAR) {

  // ---- 2. 该年不透水影像（0–100%）----
  var raw = ee.ImageCollection(IMP_COL)
              .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
              .first()
              .select(0)                 // 单波段，用序号取更稳（免猜波段名）
              .rename('impervious');
  // 只保留有效不透水值 0–100，剔除填充/nodata（>100 的值），防止污染均值
  var imp = raw.updateMask(raw.lte(100));

  // ---- 3. 每个 tract 求均值 = impervious_mean ----
  var stats = imp.reduceRegions({
    collection: tracts,
    // ⚠️单波段必须显式命名，否则会被命名成 'mean' 导致列名对不上
    reducer:    ee.Reducer.mean().setOutputs(['impervious_mean']),
    scale:      SCALE,
    tileScale:  16                       // 内存超限就调大到 32
  });

  // ---- 4. 只留 key + 特征，键名对齐 CSV（cb_2020_3，单下划线）----
  stats = stats.map(function (f) {
    return ee.Feature(null, {
      'cb_2020_3':       f.get('cb_2020__3'),
      'impervious_mean': f.get('impervious_mean')
    });
  });

  // ---- 5. 导出 ----
  Export.table.toDrive({
    collection:     stats,
    description:    'impervious_newyork_' + YEAR,
    folder:         'IMPERVIOUS_export',
    fileNamePrefix: 'impervious_newyork_' + YEAR,
    fileFormat:     'CSV',
    selectors:      ['cb_2020_3', 'impervious_mean']
  });
});

// 可视化自查（可选）：某一年的不透水面
// Map.setCenter(-74.0, 40.7, 9);
// var w = ee.ImageCollection(IMP_COL)
//           .filterDate('2020-01-01','2021-01-01').first().select(0);
// Map.addLayer(w.updateMask(w.lte(100)), {min:0, max:100, palette:['000000','ffff00','ff0000']}, '不透水% 2020');
