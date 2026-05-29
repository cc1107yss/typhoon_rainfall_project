# Problem 2 Initial Rainfall Generation QC Report

## 1. 输入文件与输出文件
- target input: `data/processed/problem2_target_halfhour_inputs_safe.csv`
- Top-K 表: `data/processed/problem2_target_topk_similar_history.csv`
- 历史库: `data/processed/problem2_historical_halfhour_sample_library.csv`
- NPZ 输出: `data/processed/problem2_generated_initial_fields_topk_weighted.npz`
- index CSV 输出: `data/processed/problem2_generated_initial_fields_index.csv`
- figures 输出目录: `outputs/figures/problem2_initial_generation`

## 2. 运行参数
- GRID_SIZE: 201
- GRID_EXTENT_KM: 1000.0
- TOPK 实际值: 20
- MIN_VALID_TEMPLATES: 5
- CACHE_SIZE: 1000
- MAKE_FIGURES: True
- NAN_SKIP_THRESHOLD: 0.5

## 3. 目标样本统计
- 总目标时刻数: 974
```
              size                 min                 max
typhoon_name                                              
KONG-REY       421 2024-10-24 18:00:00 2024-11-02 12:00:00
MAN-YI         553 2024-11-08 12:00:00 2024-11-20 00:00:00
```
- NPZ rain_mmhr_initial shape: [974, 201, 201]
- NPZ log_rain_initial shape: [974, 201, 201]
- index CSV 行列数: 974 x 66

## 4. 模板读取与重采样统计
- Top-K 总行数: 19480
- 唯一 history_tif_path 数: 2166
- 成功读取/重采样模板数: 2445
- 读取尝试数: 2445
- 读取失败或重采样失败数: 0
- tif 不存在数: 0
- tif 全缺测数: 0
- 历史中心缺失数: 0
- history_move_dir_deg 缺失数: 0
- 重采样 NaN 比例过高数: 0
- cache hits/misses/final_size: 17035 / 2445 / 1000
- 平均 valid_template_count: 20.000000
- 最小 valid_template_count: 20
- low_valid_template_count 数量: 0
- 是否存在 generation_ok=False: False

## 5. 生成场基本检查
- NaN 数量: 0
- Inf 数量: 0
- 负值数量: 0
- 全零场数量: 0
- rain_max 分布: min=2.23022, mean=10.3183, p50=8.1781, p95=23.3285, max=29.4425
- rain_p95 分布: min=0.474522, mean=1.45525, p50=1.3959, p95=2.44893, max=3.32432
- rain_area_10_km2 分布: min=0, mean=6416.32, p50=0, p95=37800, max=44300
- centroid_offset 分布: min=5.48498, mean=110.288, p50=93.9465, p95=289.702, max=429.813
- anisotropy 分布: min=0.00984138, mean=0.219281, p50=0.214049, p95=0.379527, max=0.505319

## 6. 分台风统计
### KONG-REY
- 时刻数: 421
- initial_rain_max_mmhr 最大值: 24.2025 at 2024-10-30 07:00:00
- initial_rain_p95_mmhr 最大值: 2.54426 at 2024-10-30 09:30:00
- initial_rain_area_10_km2 最大值: 43400 at 2024-10-30 15:30:00
- 累计降水总量 proxy: 3.02551e+08
- 强降水持续时间 proxy: 92 hours
### MAN-YI
- 时刻数: 553
- initial_rain_max_mmhr 最大值: 29.4425 at 2024-11-16 18:00:00
- initial_rain_p95_mmhr 最大值: 3.32432 at 2024-11-14 05:00:00
- initial_rain_area_10_km2 最大值: 44300 at 2024-11-16 03:30:00
- 累计降水总量 proxy: 4.42042e+08
- 强降水持续时间 proxy: 87 hours

## 7. 平滑风险检查
- ratio_initial_to_topk_rain_max: min=0.0650402, mean=0.227619, p50=0.183633, p95=0.487152, max=0.674679
- ratio_initial_to_topk_rain_p95: min=0.283743, mean=0.513973, p50=0.508549, p95=0.673023, max=0.847698
- ratio_initial_to_topk_rain_area_10: min=0, mean=0.104126, p50=0, p95=0.559622, max=0.730522
- 判断: 初始模板场存在极端降水平滑倾向，需在后续极端分位数校准步骤中修正。

## 8. 时间连续性初检
### KONG-REY
- diff(initial_rain_max_mmhr): P95=2.58451, max=12.3704
- diff(initial_rain_p95_mmhr): P95=0.212623, max=0.67835
- diff(initial_rain_area_10_km2): P95=3705, max=24600
### MAN-YI
- diff(initial_rain_max_mmhr): P95=2.09511, max=7.98282
- diff(initial_rain_p95_mmhr): P95=0.164945, max=1.18384
- diff(initial_rain_area_10_km2): P95=3500, max=25200

## 9. 图件
- 图件数量: 8
```
    KONG_REY_initial_representative_fields.png
      MAN_YI_initial_representative_fields.png
KONG_REY_initial_cumulative_storm_relative.png
       KONG_REY_initial_max_storm_relative.png
  MAN_YI_initial_cumulative_storm_relative.png
         MAN_YI_initial_max_storm_relative.png
               KONG_REY_initial_timeseries.png
                 MAN_YI_initial_timeseries.png
```

## 10. 异常与日志样例
- 记录事件总数: 0
```
(none)
```

## 11. 防泄漏声明
本步骤使用 18 号 Top-K 检索结果读取历史 GPM 降水模板，生成 KONG-REY 和 MAN-YI 的初始降水场。目标台风输入不包含任何真实 GPM 降水信息；rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等历史降水字段仅用于历史模板指标对比和后续校准，不参与目标台风输入或相似度检索。
