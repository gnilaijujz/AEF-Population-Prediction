// =====================================================================
//  dist_water_m 逐年 tract 均值导出 · New York MSA · 时间维度
//  用法：粘到 GEE 代码编辑器 https://code.earthengine.google.com/
//        改第 20 行 tracts 资产路径 + 第 17 行 YEARS → Run 一次
//        Tasks 面板会出现 N 个任务（每年 1 个 combine 任务）→ 逐个点 RUN
//        全部导到 Google Drive 的 dist_water_export 文件夹
//
//  数据：JRC/GSW1_4/YearlyHistory（waterClass≥2 计为水，原生 30m，覆盖 1984–2021）
//  口径对齐 2020 空间维度（water_dist_features_gee.js + 30m/90m combine）：
//    · 同一水体二值化阈值 waterClass≥2
//    · 同一 fastDistanceTransform(512) → sqrt → ×像元边长 → 米
//    · 同一 30m/90m combine：近处取 30m 精度、远处(30m 截断出伪值)取 90m 正确值
//      —— 已核实 2020 的 combine 规则就是逐 tract 取 min(30m, 90m)：
//         近水 tract 30m 更小更准 → 取 30m；远水 tract 30m 出 27万米伪值、90m≈正确 → 取 min 即 90m
//    · combine 最大值约 15km（真实距离），无 410km 之类搜索窗伪值
//  只跑 New York MSA（同一套 2020 tract 矢量，几何冻结），键 = cb_2020__3
//  引用：Pekel et al., 2016, Nature (JRC GSW v1.4)
// =====================================================================

// 要跑的年份（2020 已有；保留可重算成同一口径、多导一份不亏）
var YEARS = [2017, 2018, 2019, 2020, 2021];   // JRC YearlyHistory 止于 2021，勿超

// ===== 0. 参数（与 2020 空间维度逐字一致）=====
// 你上传的 2020 纽约 tract 矢量资产（与 AEF / NDVI 用的同一个资产路径）
var TRACTS_ASSET = 'projects/YOUR_PROJECT/assets/newyork_tracts_2020';   // ← 改成你的资产
var NEIGHBORHOOD = 512;   // fastDistanceTransform 搜索邻域(像元)，两个尺度都用 512（与 2020 一致）
var EXPORT_RAW   = false;  // true 时额外导出每年的 30m / 90m 原始两列，供验收自查(第八节)

// 1) 载入 tract（NY 资产本身已是纽约 MSA，无需再按 CONUS 过滤）
var tracts = ee.FeatureCollection(TRACTS_ASSET);

// —— 先核对字段名：确认 tract 键是 'cb_2020__3'（双下划线，跑通后可注释掉）——
print('tract 数量', tracts.size());
print('第一条属性', tracts.first());

// ===== 主流程：每年一套 =====
YEARS.forEach(function (YEAR) {

  // ---- 2. 该年水体二值(水=1，其余含 no-data=0) ----
  var yr = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
             .filter(ee.Filter.calendarRange(YEAR, YEAR, 'year'))
             .first()
             .select('waterClass');
  var water = yr.gte(2).unmask(0);          // 类2,3=水=1；其余=0（与 2020 一致）

  // ---- 3. 到最近水体距离图(米) ----
  //   fastDistanceTransform: 到最近"非零(水)"像元的平方像元距离
  //   .sqrt() → 像元距离；×像元边长(pixelArea^0.5) → 米（边长随 reduceRegions 的 scale 自适应）
  var distWater = water.fastDistanceTransform(NEIGHBORHOOD)
                       .sqrt()
                       .multiply(ee.Image.pixelArea().sqrt())
                       .rename('dist_water');

  // ---- 4. 两个尺度各求 tract 均值 ----
  //   同一 distWater 表达式，只靠 reduceRegions 的 scale 区分 30m / 90m：
  //   scale=30 → 512×30≈15km 窗，近处精确、远处截断出伪值；
  //   scale=90 → 512×90≈46km 窗，远处正确、近处较粗。
  //   链式：先在 tracts 上算 d30，再在带 d30 的结果上算 d90（reduceRegions 保留已有属性与几何）
  var f30 = distWater.reduceRegions({
    collection: tracts,
    reducer:    ee.Reducer.mean().setOutputs(['d30']),   // 显式命名，防单波段命名坑
    scale:      30,
    tileScale:  16                                        // 内存超限就调大到 32
  });
  var f3090 = distWater.reduceRegions({
    collection: f30,
    reducer:    ee.Reducer.mean().setOutputs(['d90']),
    scale:      90,
    tileScale:  16
  });

  // ---- 5. combine = 逐 tract min(d30, d90)（安全处理缺值：任一为空取另一个，都空则留空）----
  var out = f3090.map(function (f) {
    var d30 = f.get('d30');
    var d90 = f.get('d90');
    var dc  = ee.Algorithms.If(
                d30,
                ee.Algorithms.If(d90, ee.Number(d30).min(ee.Number(d90)), d30),
                d90);
    return ee.Feature(null, {
      'cb_2020_3':    f.get('cb_2020__3'),   // 键名对齐 CSV：单下划线
      'dist_water_m': dc,
      'd30_raw':      d30,                    // 仅在 EXPORT_RAW 时导出，供自查
      'd90_raw':      d90
    });
  });

  // ---- 6. 导出该年 combine（1 tract 1 行，列 cb_2020_3, dist_water_m）----
  Export.table.toDrive({
    collection:     out,
    description:    'water_dist_newyork_' + YEAR,
    folder:         'dist_water_export',
    fileNamePrefix: 'water_dist_newyork_' + YEAR,
    fileFormat:     'CSV',
    selectors:      ['cb_2020_3', 'dist_water_m']
  });

  // ---- 6b.（可选）额外导出 30m / 90m 原始两列，供验收自查 ----
  if (EXPORT_RAW) {
    Export.table.toDrive({
      collection:     out,
      description:    'water_dist_newyork_' + YEAR + '_raw3090',
      folder:         'dist_water_export',
      fileNamePrefix: 'water_dist_newyork_' + YEAR + '_raw3090',
      fileFormat:     'CSV',
      selectors:      ['cb_2020_3', 'd30_raw', 'd90_raw', 'dist_water_m']
    });
  }
});

// 可视化自查（可选）：某一年的距水距离图
// Map.setCenter(-74.0, 40.7, 8);
// var w = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
//           .filter(ee.Filter.calendarRange(2020, 2020, 'year')).first()
//           .select('waterClass').gte(2).unmask(0);
// var d = w.fastDistanceTransform(512).sqrt().multiply(ee.Image.pixelArea().sqrt());
// Map.addLayer(d, {min:0, max:5000, palette:['0000ff','00ffff','ffff00','ff0000']}, '距水距离(m) 2020');
