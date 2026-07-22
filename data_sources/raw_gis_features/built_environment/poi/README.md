# POI 密度特征

> 第二块 · 人类活动与建成环境 / POI 密度　|　整理自 `POI特征_数据说明.docx`
> 脚本:`poi_density_15msa.py`　输出:`poi_features_tract.csv` / `poi_features_15msa.csv`

## 一、数据概况

| 项 | 内容 |
|----|------|
| 数据来源 | OpenStreetMap (OSM),经 ohsome API 查询(历史版本,时间点 2020-01-01 UTC) |
| 查询接口 | `POST /elements/count/groupBy/boundary`(按 tract 面逐个计数) |
| 时间 | 2020 年 |
| 分析单元 | census tract(美国本土 CONUS) |
| 连接键 | `cb_2020_3` |
| 密度计算 | POI 计数 ÷ tract 陆地面积 `ALAND`(km²)= 个/km² |
| 几何口径 | 仅点状 POI(`geometry:point`) |

## 二、特征字段

密度类单位均为「个/km²」;`poi_diversity` 无量纲(自然对数 Shannon 熵)。

| 字段 | 含义 | 数值大代表 |
|------|------|-----------|
| `poi_edu` | 教育设施密度(学校/学院/大学/幼儿园) | 居住/家庭密集、城市化 |
| `poi_comm` | 零售商业密度(shop + 集市) | 商业区/市中心 |
| `poi_med` | 医疗密度(医院/诊所/药店/医生) | 通常城市中心 |
| `poi_trans` | 公共交通密度(公交/地铁站点等) | 密集/城市 |
| `poi_pub` | 公共服务密度(警/消/政厅/邮局/图书馆) | 市政设施多 |
| `poi_food` | 餐饮密度(餐厅/咖啡/快餐/酒吧) | 生活活力强(人口/活力的强代理) |
| `poi_finance` | 金融密度(银行/ATM) | 商业中心 |
| `poi_total_all` | 综合 POI 密度(所有 amenity/shop/office/leisure/tourism) | 整体建成/城市化强度(最强人口密度代理之一) |
| `poi_diversity` | 功能多样性(7 类计数的 Shannon 熵) | 见下 |

## 三、`poi_diversity`(功能多样性)详解

对 tract 内 7 个命名类别(edu/comm/med/trans/pub/food/finance)的 POI 计数计算 Shannon 熵(自然对数),范围 `0 ~ ln(7)≈1.95`:
- **0** = 只有一类 POI(或没有)→ 功能单一(纯住宅/纯商业)
- **接近 1.95** = 7 类分布均匀 → 高度混合用途(住+商+服务混合社区)
- 衡量"类别是否均衡",不是数量:POI 多但全是餐饮 → 熵低;各类都有 → 熵高。混合度通常与城市活力/人口密度正相关。

## 四、整体解读

密度类越高 → 越城市化、经济越活跃 → 通常人口密度越高。`poi_total_all` 是最综合的城市化指数;`poi_food` 是最灵敏的活力代理;`poi_diversity` 捕捉密度类抓不到的"混合用途"维度。

## 五、重要局限(跨城市迁移需注意)

OSM 的 POI 映射完整度不均:大城市(纽约、旧金山)较全,中小城市/乡村稀疏。因此 POI 密度部分反映"OSM 是否被标注"而非纯真实密度。本研究核心是跨 MSA 可迁移性——若源城市 OSM 完整、目标城市稀疏,POI 特征会引入系统性偏差,建议在方法/讨论中注明,并可作为"域偏移"来源纳入分析。

## 附:各类别 OSM 标签口径(均附加 `and geometry:point`)

| 字段 | ohsome filter |
|------|---------------|
| `poi_edu` | `amenity in (school, college, university, kindergarten)` |
| `poi_comm` | `shop=* or amenity=marketplace` |
| `poi_med` | `amenity in (hospital, clinic, pharmacy, doctors)` |
| `poi_trans` | `public_transport=* or railway=station or amenity=bus_station` |
| `poi_pub` | `amenity in (police, fire_station, townhall, post_office, library)` |
| `poi_food` | `amenity in (restaurant, cafe, fast_food, bar, pub, food_court)` |
| `poi_finance` | `amenity in (bank, atm, bureau_de_change)` |
| `poi_total_all` | `amenity=* or shop=* or office=* or leisure=* or tourism=*` |
