# Problem 2 Extreme Quantile Calibration QC Report

## 1. 输入输出文件
- blended NPZ: `data/processed/problem2_env/problem2_generated_pca_blended_fields_env.npz`
- blended index: `data/processed/problem2_env/problem2_generated_pca_blended_fields_index_env.csv`
- Top-K 表: `data/processed/problem2_env/problem2_target_topk_similar_history_env.csv`
- 历史库: `data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- step-20 QC: `outputs/problem2_eof_pca_correction_qc_report.md`
- calibration targets 输出: `data/processed/problem2_env/problem2_extreme_calibration_targets_env.csv`
- calibrated NPZ 输出: `data/processed/problem2_env/problem2_generated_calibrated_fields_env.npz`
- calibrated index 输出: `data/processed/problem2_env/problem2_generated_calibrated_fields_index_env.csv`
- figures 目录: `outputs/figures/problem2_env/extreme_calibration`

## 2. 运行参数
- BETA_SOURCE: blend
- USE_WINSORIZED_TARGETS: True
- WINSOR_LOWER / WINSOR_UPPER: 0.05 / 0.95
- SCALE_MIN / SCALE_MAX / SCALE_MAX_MAX: 0.7 / 3.0 / 5.0
- TAIL_START_QUANTILE: 0.9
- AREA_CALIBRATION: True
- AREA_TOLERANCE: 0.25
- GLOBAL_RAIN_MAX_CAP_MMHR: 120.0
- PER_TARGET_CAP_FACTOR: 1.25

## 3. 校准目标统计
### 总体
- target_rain_max_mmhr: min=24.0004, mean=44.2828, p50=44.7812, p95=54.9177, max=59.3254
- target_rain_p95_mmhr: min=1.5447, mean=2.80643, p50=2.72837, p95=4.05349, max=4.99308
- target_rain_p99_mmhr: min=4.88974, mean=10.0536, p50=9.92644, p95=14.2423, max=17.1411
- target_rain_area_10_km2: min=12339, mean=45656.6, p50=44262.5, p95=73392.4, max=99868.2
- target_rain_area_20_km2: min=1194.21, mean=15071.8, p50=15018.1, p95=29162, max=37839.5
### KONG-REY
- target_rain_max_mmhr: min=24.0004, mean=43.2988, p50=44.2649, p95=51.6066, max=59.3254
- target_rain_p95_mmhr: min=2.00165, mean=2.67169, p50=2.53846, p95=3.70918, max=3.88092
- target_rain_p99_mmhr: min=6.20868, mean=9.7546, p50=9.53158, p95=13.4696, max=16.8213
- target_rain_area_10_km2: min=15313.2, mean=43023.9, p50=40937.9, p95=65521.4, max=88288.7
- target_rain_area_20_km2: min=1201.41, mean=13449.4, p50=13287.9, p95=25195.5, max=37184.7
### MAN-YI
- target_rain_max_mmhr: min=25.9629, mean=45.0319, p50=45.2659, p95=55.6054, max=58.2385
- target_rain_p95_mmhr: min=1.5447, mean=2.90901, p50=2.86369, p95=4.21581, max=4.99308
- target_rain_p99_mmhr: min=4.88974, mean=10.2813, p50=10.4991, p95=14.4026, max=17.1411
- target_rain_area_10_km2: min=12339, mean=47660.9, p50=46957.4, p95=76376, max=99868.2
- target_rain_area_20_km2: min=1194.21, mean=16307, p50=17235.2, p95=29925.6, max=37839.5
- Top-K weight_sum 分布: min=1, mean=1, p50=1, p95=1, max=1
- Top-K 权重和不等于 1 的目标数: 0
- 目标指标缺失数量:
```
target_rain_max_mmhr       0
target_rain_p95_mmhr       0
target_rain_p99_mmhr       0
target_rain_area_10_km2    0
target_rain_area_20_km2    0
```
- 目标构造补算/回退计数:
```
(none)
```

## 4. 校准场基本检查
- rain_mmhr_calibrated shape: [974, 201, 201]
- NaN 数量: 0
- Inf 数量: 0
- 负值数量: 0
- 全零场数量: 0
- cap 截断时次数: 0
- cap 截断格点数: 0
- calibration_ok=False 数量: 0

## 5. 校准前后对比
- blend_rain_max_mmhr: P50=7.18613, P95=21.5543, max=26.553
- calibrated_rain_max_mmhr: P50=35.9307, P95=49.7285, max=54.4054
- target_rain_max_mmhr: P50=44.7812, P95=54.9177, max=59.3254
- blend_rain_p95_mmhr: P50=1.3544, P95=2.39264, max=3.16419
- calibrated_rain_p95_mmhr: P50=2.72215, P95=4.05349, max=4.99308
- target_rain_p95_mmhr: P50=2.72837, P95=4.05349, max=4.99308
- blend_rain_p99_mmhr: P50=3.45342, P95=9.75509, max=10.8436
- calibrated_rain_p99_mmhr: P50=9.04098, P95=14.2378, max=17.1411
- target_rain_p99_mmhr: P50=9.92644, P95=14.2423, max=17.1411
- blend_rain_area_10_km2: P50=0, P95=38535, max=46100
- calibrated_rain_area_10_km2: P50=37050, P95=70335, max=115500
- target_rain_area_10_km2: P50=44262.5, P95=73392.4, max=99868.2

## 6. 目标接近程度
- ratio_calibrated_to_target_rain_max: P50=0.789742, P95=1
- ratio_calibrated_to_target_rain_p95: P50=1, P95=1
- ratio_calibrated_to_target_rain_p99: P50=1, P95=1
- ratio_calibrated_to_target_area_10: P50=0.853897, P95=1.08117
- Rmax: mean=0.225115, P50=0.210258, P95=0.600206
- P95: mean=0.00605452, P50=3.24211e-08, P95=9.56216e-08
- P99: mean=0.0874872, P50=6.9493e-08, P95=0.327469
- area10: mean=0.169056, P50=0.148113, P95=0.249818

## 7. 空间结构保持程度
- corr_blend_calibrated P50/P95/min: 0.947763 / 0.991074 / 0.848499
- rmse_blend_calibrated P50/P95/max: P50=1.26821, P95=1.71405, max=2.08341
- centroid_offset 变化分布: min=-75.47, mean=4.02425, p50=-2.04179, p95=54.1938, max=66.345
- anisotropy 变化分布: min=-0.0533738, mean=0.0104903, p50=0.000124283, p95=0.10057, max=0.173808
- r50 变化分布: min=-226.239, mean=-110.71, p50=-112.983, p95=-35.3748, max=-17.2468
- r80 变化分布: min=-327.136, mean=-167.925, p50=-167.132, p95=-76.7658, max=2.73919
- r90 变化分布: min=-194.211, mean=-95.5841, p50=-104.841, p95=-4.368, max=64.4049
- mean_scale_factor 分布: min=1.0258, mean=1.09062, p50=1.09175, p95=1.14452, max=1.15662
- p95_scale_factor 分布: min=1.22395, mean=2.01478, p50=1.94816, p95=2.86136, max=3
- max_scale_factor 分布: min=1.71632, mean=4.45347, p50=5, p95=5.72179, max=8.28185
- 说明: 警告：部分时次 corr_blend_calibrated 明显偏低，需检查是否需要更强的空间/时间平滑。

## 8. 分台风结果
### KONG-REY
- 时刻数: 421
- calibrated_rain_max_mmhr 最大值: 53.7328 at 2024-10-26 23:00:00
- calibrated_rain_p95_mmhr 最大值: 3.88092 at 2024-10-30 09:00:00
- calibrated_rain_p99_mmhr 最大值: 16.8213 at 2024-10-30 09:00:00
- calibrated_rain_area_10_km2 最大值: 87800 at 2024-10-30 09:00:00
- 累计降水 proxy: 4.92975e+08
- calibrated_rain_area_10_km2 > 0 持续时间 proxy: 210.5 hours
- calibrated_rain_area_20_km2 > 0 持续时间 proxy: 202.5 hours
- WND 最大时刻: 2024-10-30 00:00:00; Rmax 最大时刻: 2024-10-26 23:00:00; 时间差: 73 hours
### MAN-YI
- 时刻数: 553
- calibrated_rain_max_mmhr 最大值: 54.4054 at 2024-11-14 10:30:00
- calibrated_rain_p95_mmhr 最大值: 4.99308 at 2024-11-14 03:30:00
- calibrated_rain_p99_mmhr 最大值: 17.1411 at 2024-11-14 04:30:00
- calibrated_rain_area_10_km2 最大值: 115500 at 2024-11-14 04:00:00
- 累计降水 proxy: 6.86489e+08
- calibrated_rain_area_10_km2 > 0 持续时间 proxy: 276.5 hours
- calibrated_rain_area_20_km2 > 0 持续时间 proxy: 237 hours
- WND 最大时刻: 2024-11-16 00:00:00; Rmax 最大时刻: 2024-11-14 10:30:00; 时间差: 37.5 hours

## 9. 时间连续性检查
### KONG-REY
- diff(calibrated_rain_max_mmhr): P95=5.9894, max=15.0314
- diff(calibrated_rain_p95_mmhr): P95=0.252487, max=1.47613
- diff(calibrated_rain_p99_mmhr): P95=1.09356, max=3.93545
- diff(calibrated_rain_area_10_km2): P95=7515, max=22800
### MAN-YI
- diff(calibrated_rain_max_mmhr): P95=5.50344, max=20.9881
- diff(calibrated_rain_p95_mmhr): P95=0.297235, max=1.15565
- diff(calibrated_rain_p99_mmhr): P95=1.06955, max=3.85522
- diff(calibrated_rain_area_10_km2): P95=7345, max=45200
### 可能需要后续时间平滑关注的跳变
- KONG-REY 2024-10-27 18:30:00 calibrated_rain_max_mmhr large jump, value=32.5183
- KONG-REY 2024-10-29 12:30:00 calibrated_rain_p95_mmhr large jump, value=2.674
- KONG-REY 2024-10-31 10:30:00 calibrated_rain_p95_mmhr large jump, value=2.31589
- KONG-REY 2024-10-31 18:30:00 calibrated_rain_p95_mmhr large jump, value=3.15677
- KONG-REY 2024-11-01 21:30:00 calibrated_rain_p95_mmhr large jump, value=3.2841
- KONG-REY 2024-11-02 06:30:00 calibrated_rain_p95_mmhr large jump, value=2.36167
- KONG-REY 2024-10-30 06:30:00 calibrated_rain_p99_mmhr large jump, value=16.2848
- KONG-REY 2024-11-01 21:30:00 calibrated_rain_p99_mmhr large jump, value=6.79726
- KONG-REY 2024-11-02 06:30:00 calibrated_rain_p99_mmhr large jump, value=3.60092
- KONG-REY 2024-10-30 06:30:00 calibrated_rain_area_10_km2 large jump, value=80500
- KONG-REY 2024-10-30 09:30:00 calibrated_rain_area_10_km2 large jump, value=67800
- MAN-YI 2024-11-09 18:30:00 calibrated_rain_max_mmhr large jump, value=37.829
- MAN-YI 2024-11-13 18:30:00 calibrated_rain_max_mmhr large jump, value=23.4177
- MAN-YI 2024-11-14 00:30:00 calibrated_rain_max_mmhr large jump, value=44.0471
- MAN-YI 2024-11-18 12:30:00 calibrated_rain_max_mmhr large jump, value=18.7392
- MAN-YI 2024-11-11 00:30:00 calibrated_rain_p95_mmhr large jump, value=3.07315
- MAN-YI 2024-11-11 06:30:00 calibrated_rain_p95_mmhr large jump, value=2.30982
- MAN-YI 2024-11-14 00:30:00 calibrated_rain_p95_mmhr large jump, value=4.46561
- MAN-YI 2024-11-16 00:30:00 calibrated_rain_p95_mmhr large jump, value=2.92044
- MAN-YI 2024-11-17 18:30:00 calibrated_rain_p95_mmhr large jump, value=2.81795
- MAN-YI 2024-11-09 18:30:00 calibrated_rain_p99_mmhr large jump, value=10.6695
- MAN-YI 2024-11-11 06:30:00 calibrated_rain_p99_mmhr large jump, value=7.28703
- MAN-YI 2024-11-14 00:30:00 calibrated_rain_p99_mmhr large jump, value=13.1846
- MAN-YI 2024-11-14 03:30:00 calibrated_rain_p99_mmhr large jump, value=17.1323
- MAN-YI 2024-11-14 06:30:00 calibrated_rain_p99_mmhr large jump, value=13.5132
- MAN-YI 2024-11-14 00:30:00 calibrated_rain_area_10_km2 large jump, value=86000
- MAN-YI 2024-11-14 03:30:00 calibrated_rain_area_10_km2 large jump, value=114300
- MAN-YI 2024-11-14 06:30:00 calibrated_rain_area_10_km2 large jump, value=60300
- MAN-YI 2024-11-14 12:30:00 calibrated_rain_area_10_km2 large jump, value=75100
- MAN-YI 2024-11-15 15:30:00 calibrated_rain_area_10_km2 large jump, value=46800

## 10. 图件
- 图件数量: 19
```
       KONG_REY_extreme_calibration_timeseries.png
         MAN_YI_extreme_calibration_timeseries.png
KONG_REY_extreme_calibration_compare_20241024_1...
KONG_REY_extreme_calibration_compare_20241030_0...
KONG_REY_extreme_calibration_compare_20241026_1...
KONG_REY_extreme_calibration_compare_20241102_1...
MAN_YI_extreme_calibration_compare_20241108_120...
MAN_YI_extreme_calibration_compare_20241114_033...
MAN_YI_extreme_calibration_compare_20241110_000...
MAN_YI_extreme_calibration_compare_20241120_000...
 KONG_REY_calibrated_cumulative_storm_relative.png
        KONG_REY_calibrated_max_storm_relative.png
 KONG_REY_calibrated_duration10_storm_relative.png
 KONG_REY_calibrated_duration20_storm_relative.png
   MAN_YI_calibrated_cumulative_storm_relative.png
          MAN_YI_calibrated_max_storm_relative.png
   MAN_YI_calibrated_duration10_storm_relative.png
   MAN_YI_calibrated_duration20_storm_relative.png
          calibration_target_vs_output_scatter.png
```

## 11. 防泄漏声明
本步骤的校准目标来自目标时刻 Top-K 历史相似台风样本的加权降水指标，不使用 KONG-REY 和 MAN-YI 的真实 GPM 降水观测。目标台风输入仍仅包含路径、强度、移动、时间和海陆环境等安全特征。历史 rain_* 指标只用于校准生成场的极端分布，不参与目标台风输入构造或相似度检索。

## 12. 论文可写结论
- 极端校准前，20 号 blended 场的 Rmax/P99/P95 相比 Top-K 历史相似样本目标整体偏低，尤其 Rmax 尾部被 log 加权和 EOF/PCA 平滑明显压低。
- 校准后，P95/P99/Rmax 均向相似历史目标靠近，强降水尾部幅度有实质增强。
- corr_blend_calibrated 维持较高水平，说明校准主要增强强降水尾部，没有整体破坏 20 号的大尺度结构底图。
- 强降水面积和持续时间较 20 号 blended 场更接近历史相似台风过程，可为后续结果展示提供更合理的极端降水空间范围。
- calibrated 场可作为 22 号伪缺失验证、23 号最终图件整理以及问题三虚拟台风情景生成的基础。
