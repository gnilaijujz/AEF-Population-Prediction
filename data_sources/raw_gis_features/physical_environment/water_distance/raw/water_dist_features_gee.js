/**********************************************************************
 * 第三块 · 水体距离(Distance to Water)特征提取（GEE / JavaScript）
 * 目标：对全部 CONUS census tract 求 到最近地表水体的平均距离(米)
 * 说明：GEE 无现成"距水距离"栅格，用 JRC 水体掩膜 + fastDistanceTransform 现算
 * 水体源：JRC/GSW1_4/YearlyHistory 2020，waterClass≥2(季节+永久)，30m
 * 单元：2020_MSA_in_track（键 = cb_2020__3）
 * 输出：water_dist_features_tract.csv —— cb_2020_3 + dist_water_m(米)
 * 注：JRC 非 USGS 同源；引用 Pekel et al., 2016, Nature.
 **********************************************************************/

// ===== 0. 参数 =====
var TRACTS_ASSET = 'projects/poetic-tesla-502117-m4/assets/2020_MSA_in_track';
var DIST_SCALE  = 90;    // 距离计算/取样分辨率(米)。90m既省算力、搜索范围又更大
var NEIGHBORHOOD = 512;  // 距离变换搜索邻域(像元)。512×90m≈46km，足够覆盖西部干旱区

// ===== 1. tract，只留 CONUS =====
var tracts = ee.FeatureCollection(TRACTS_ASSET);
var CONUS = ['01','04','05','06','08','09','10','11','12','13','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29','30','31','32','33',
             '34','35','36','37','38','39','40','41','42','44','45','46','47','48',
             '49','50','51','53','54','55','56'];
tracts = tracts.filter(ee.Filter.inList('cb_2020_us', CONUS));

// ===== 2. 2020 水体二值(水=1，其余=0) =====
var yr = ee.ImageCollection('JRC/GSW1_4/YearlyHistory')
           .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
           .first()
           .select('waterClass');
var water = yr.gte(2).unmask(0);   // 类2,3=水=1；其余(含no-data)=0

// ===== 3. 到最近水体距离(米) =====
// fastDistanceTransform: 到最近"非零(水)"像元的平方距离(像元单位)
var distWater = water.fastDistanceTransform(NEIGHBORHOOD)
                     .sqrt()                             // → 像元距离
                     .multiply(ee.Image.pixelArea().sqrt())  // ×像元边长 → 米
                     .rename('dist_water');

// ===== 4. 每个 tract 的平均距水距离 =====
var stats = distWater.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.mean().setOutputs(['dist_water_m']),  // 显式命名，防单波段命名坑
  scale: DIST_SCALE,
  tileScale: 16
});

stats = stats.map(function (f) {
  return ee.Feature(null, {
    'cb_2020_3':    f.get('cb_2020__3'),
    'dist_water_m': f.get('dist_water_m')
  });
});

// ===== 5. 导出 =====
Export.table.toDrive({
  collection: stats,
  description: 'water_dist_features_tract',
  folder: 'water_dist_features',
  fileNamePrefix: 'water_dist_features_tract',
  fileFormat: 'CSV',
  selectors: ['cb_2020_3', 'dist_water_m']
});

// 可视化自查（可选，受限模式下建议注释掉）
Map.setCenter(-98.5, 39.8, 4);
// Map.addLayer(distWater, {min:0, max:5000, palette:['0000ff','00ffff','ffff00','ff0000']}, '距水距离(m)');
