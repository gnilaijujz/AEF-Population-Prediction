/**********************************************************************
 * 第三块 · 树冠(Tree Canopy Cover)特征提取（GEE Code Editor / JavaScript）
 * 目标：对全部 CONUS census tract 求 2020 年平均树冠覆盖率(%)
 * 数据：USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5（USFS/NLCD 树冠，30m，年度1985-2023）
 *       波段 NLCD_Percent_Tree_Canopy_Cover（0-100%，非树像元置0）
 * 单元：2020_MSA_in_track（键 = cb_2020__3）
 * 输出：tcc_features_tract.csv —— cb_2020_3 + tcc_mean
 *       （键/格式对齐 DEM、NDVI、NLCD，可直接按 key 拼）
 **********************************************************************/

// ===== 0. 参数 =====
var TRACTS_ASSET = projects/poetic-tesla-502117-m4/assets/2020_MSA_in_track;
var SCALE = 30;   // NLCD TCC 原生 30m

// ===== 1. 载入 tract 边界，只留 CONUS =====
var tracts = ee.FeatureCollection(TRACTS_ASSET);
var CONUS = ['01','04','05','06','08','09','10','11','12','13','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29','30','31','32','33',
             '34','35','36','37','38','39','40','41','42','44','45','46','47','48',
             '49','50','51','53','54','55','56'];
tracts = tracts.filter(ee.Filter.inList('cb_2020_us', CONUS));

// ===== 2. 树冠：取 2020 年 CONUS 那张 =====
var tccRaw = ee.ImageCollection('USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5')
            .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
            .filter('study_area == "CONUS"')
            .first()
            .select('NLCD_Percent_Tree_Canopy_Cover');
// 掩膜掉填充值(254/255=无数据/非处理区)，只保留有效树冠率 0-100
var tcc = tccRaw.updateMask(tccRaw.lte(100)).rename('tcc');

// ===== 3. 分区统计：每个 tract 的平均树冠% =====
var stats = tcc.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.mean().setOutputs(['tcc_mean']),  // 显式命名，避免单波段命名坑
  scale: SCALE,
  tileScale: 16
});

// 只留 key + 特征
stats = stats.map(function (f) {
  return ee.Feature(null, {
    'cb_2020_3': f.get('cb_2020__3'),
    'tcc_mean':  f.get('tcc_mean')
  });
});

// ===== 4. 导出 =====
Export.table.toDrive({
  collection: stats,
  description: 'tcc_features_tract',
  folder: 'tcc_features',
  fileNamePrefix: 'tcc_features_tract',
  fileFormat: 'CSV',
  selectors: ['cb_2020_3', 'tcc_mean']
});

// 可视化自查（可选，受限模式下建议注释掉）
Map.setCenter(-98.5, 39.8, 4);
// Map.addLayer(tcc, {min:0, max:100, palette:['ffffff','c0e8c0','006400']}, '树冠2020');
