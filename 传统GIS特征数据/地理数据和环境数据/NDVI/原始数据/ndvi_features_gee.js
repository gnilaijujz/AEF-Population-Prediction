/**********************************************************************
 * 第三块 · NDVI 特征提取（GEE Code Editor / JavaScript）
 * 目标：对全部 CONUS census tract 求 2020 年平均 NDVI
 * 数据：LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI（=30m）
 * 单元：2020_MSA_in_track（键 = cb_2020__3）
 * 输出：ndvi_features_tract.csv —— cb_2020_3 + ndvi_mean
 *       （键/格式对齐 DEM、NLCD，可直接按 key 拼）
 **********************************************************************/

// ===== 0. 参数 =====
var TRACTS_ASSET = 'projects/youtube-data-464807/assets/2020_MSA_in_track';
var SCALE = 30;   // Landsat 原生 30m

// ===== 1. 载入 tract 边界，只留 CONUS =====
var tracts = ee.FeatureCollection(TRACTS_ASSET);
var CONUS = ['01','04','05','06','08','09','10','11','12','13','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29','30','31','32','33',
             '34','35','36','37','38','39','40','41','42','44','45','46','47','48',
             '49','50','51','53','54','55','56'];
tracts = tracts.filter(ee.Filter.inList('cb_2020_us', CONUS));

// ===== 2. NDVI：取 2020 年度合成 =====
var ndvi = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI')
             .filterDate('2020-01-01', '2021-01-01')  // 2020整年
             .select('NDVI')
             .mosaic()               // 年度合成拼成一张无缝图
             .rename('ndvi');

// ===== 3. 分区统计：每个 tract 的平均 NDVI =====
var stats = ndvi.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.mean().setOutputs(['ndvi_mean']),  // 显式命名输出，避免单波段被命名成'mean'导致取不到
  scale: SCALE,
  tileScale: 16
});

// 只留 key + 特征，键名对齐她的CSV(cb_2020_3)
stats = stats.map(function (f) {
  return ee.Feature(null, {
    'cb_2020_3': f.get('cb_2020__3'),
    'ndvi_mean': f.get('ndvi_mean')
  });
});

// ===== 4. 导出 =====
Export.table.toDrive({
  collection: stats,
  description: 'ndvi_features_tract',
  folder: 'ndvi_features',
  fileNamePrefix: 'ndvi_features_tract',
  fileFormat: 'CSV',
  selectors: ['cb_2020_3', 'ndvi_mean']
});

/*  备用：若报超时/内存，按州分批(每州一个任务)：
    注意——分州需在上面 stats.map 之前保留 cb_2020_us 字段，
    可把 map 里加一行 'cb_2020_us': f.get('cb_2020_us')，再用下面这段：
CONUS.forEach(function (st) {
  Export.table.toDrive({
    collection: stats.filter(ee.Filter.eq('cb_2020_us', st)),
    description: 'ndvi_features_' + st,
    folder: 'ndvi_features',
    fileNamePrefix: 'ndvi_features_' + st,
    fileFormat: 'CSV',
    selectors: ['cb_2020_3', 'ndvi_mean']
  });
});
*/

// 可视化自查（可选）—— 受限模式下先注释掉，避免地图渲染反复烧配额
Map.setCenter(-98.5, 39.8, 4);
// Map.addLayer(ndvi, {min:0, max:1, palette:['ffffff','ce7e45','fcd163','74a901','207401','004c00','011301']}, 'NDVI 2020');
