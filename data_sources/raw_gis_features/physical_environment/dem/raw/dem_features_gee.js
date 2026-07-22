/**********************************************************************
 * 第三块 · DEM 特征提取（GEE Code Editor / JavaScript）
 * 目标：对全部 CONUS census tract 求 平均高程 + 平均坡度
 * 数据：USGS/3DEP/10m（nationalmap 3DEP 的 GEE 无缝版）
 * 单元：2020_MSA_in_track（75,068 个 tract，键 = cb_2020__3）
 * 输出：dem_features_tract.csv —— 首列 cb_2020_3 + elev_mean + slope_mean
 *       （可按 key 直接拼 AEF/NLCD）
 **********************************************************************/

// ===== 0. 参数 =====
var TRACTS_ASSET = 'projects/youtube-data-464807/assets/2020_MSA_in_track';  // 你的asset路径
var SCALE = 30;   // 分区统计分辨率(米)。tract都>300m，30m足够且省配额

// ===== 1. 载入 tract 边界 =====
var tracts = ee.FeatureCollection(TRACTS_ASSET);

// 只保留 CONUS 本土
var CONUS = ['01','04','05','06','08','09','10','11','12','13','16','17','18','19',
             '20','21','22','23','24','25','26','27','28','29','30','31','32','33',
             '34','35','36','37','38','39','40','41','42','44','45','46','47','48',
             '49','50','51','53','54','55','56'];
tracts = tracts.filter(ee.Filter.inList('cb_2020_us', CONUS));

// ===== 2. DEM + 派生坡度 =====
var dem   = ee.Image('USGS/3DEP/10m').select('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');   // 单位：度
var stack = dem.addBands(slope);                      // 两个波段: elevation, slope

// ===== 3. 分区统计：每个 tract 的均值 =====
var stats = stack.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.mean(),
  scale: SCALE,
  tileScale: 16          // 防内存超限，越大越稳(但越慢)
});

// 只留 key + 两个特征，并把键名改成 cb_2020_3（对齐她的CSV）
stats = stats.map(function (f) {
  return ee.Feature(null, {
    'cb_2020_3':  f.get('cb_2020__3'),
    'elev_mean':  f.get('elevation'),
    'slope_mean': f.get('slope')
  });
});

// ===== 4. 导出 =====
// 方案A（默认）：一次性导全部 CONUS tract
Export.table.toDrive({
  collection: stats,
  description: 'dem_features_tract',
  folder: 'dem_features',
  fileNamePrefix: 'dem_features_tract',
  fileFormat: 'CSV',
  selectors: ['cb_2020_3', 'elev_mean', 'slope_mean']
});

/*  方案B（备用）：若方案A报 "Computation timed out / memory" ，
    注释掉上面的 Export，改用下面按州分批（每个州一个导出任务）：

CONUS.forEach(function (st) {
  var sub = stats.filter(ee.Filter.eq('cb_2020_us', st)); // 注意:分州需在map前保留cb_2020_us
  Export.table.toDrive({
    collection: sub,
    description: 'dem_features_' + st,
    folder: 'dem_features',
    fileNamePrefix: 'dem_features_' + st,
    fileFormat: 'CSV',
    selectors: ['cb_2020_3', 'elev_mean', 'slope_mean']
  });
});
*/

// 可视化自查（可选）—— 用固定坐标定位美国本土，避免对整个FC求几何(会超边数上限)
Map.setCenter(-98.5, 39.8, 4);   // 美国本土中心
Map.addLayer(dem, {min:0, max:3000, palette:['006633','E5FFCC','662A00','D8D8D8','F5F5F5']}, 'DEM');
// 想在地图上看tract边界的话，只叠加少量(整体7.5万个太多)：
// Map.addLayer(tracts.limit(500), {color:'red'}, 'tract样例(前500)');
