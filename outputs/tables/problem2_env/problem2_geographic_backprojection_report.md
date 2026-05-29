# Problem 2 Geographic Backprojection Report

## 1. 输入输出文件
- calibrated NPZ: `data/processed/problem2_env/problem2_generated_calibrated_fields_env.npz`
- calibrated index: `data/processed/problem2_env/problem2_generated_calibrated_fields_index_env.csv`
- final timeseries: `outputs/tables/problem2_env/problem2_final_timeseries_metrics.csv`
- final summary: `outputs/tables/problem2_env/problem2_final_typhoon_metrics_summary.csv`
- final key times: `outputs/tables/problem2_env/problem2_final_key_times.csv`
- geographic NPZ: `outputs/tables/problem2_env/problem2_final_geographic_fields.npz`
- geographic summary CSV: `outputs/tables/problem2_env/problem2_final_geographic_summary.csv`
- geographic key locations CSV: `outputs/tables/problem2_env/problem2_final_geographic_key_locations.csv`
- geographic timeseries metrics CSV: `outputs/tables/problem2_env/problem2_final_geographic_timeseries_metrics.csv`
- figures directory: `outputs/figures/problem2_env`

## 2. 运行参数
- GRID_MODE: `per_typhoon`
- GEO_RES_DEG: `0.1`
- GEO_MARGIN_DEG: `8.0`
- SAVE_TIMESLICE_GEO_FIELDS: `False`
- 是否使用 cartopy: `True`
- 插值方法: `scipy.interpolate.RegularGridInterpolator`, target-grid reverse sampling, out-of-bounds filled with 0.0
- storm-relative x_front_km range: -1000.0 to 1000.0 km
- storm-relative y_left_km range: -1000.0 to 1000.0 km

## 3. 数据完整性检查
- calibrated 场 shape: `(974, 201, 201)`
- calibrated index shape: `(974, 84)`
- final timeseries shape: `(974, 122)`
- final summary shape: `(2, 63)`
- final key times shape: `(16, 17)`
- KONG-REY 时刻数: 421
- MAN-YI 时刻数: 553
- 输入场 NaN/Inf/负值/全零检查: `{'input_nan': 0, 'input_inf': 0, 'input_negative': 0, 'input_all_zero_fields': 0}`
- KONG-REY 经纬度网格范围: lon 111.00 to 155.00, lat 4.00 to 43.00, shape (391, 441)
- KONG-REY 反投影后 NaN/Inf/负值/全零检查: `{'nan': 0, 'inf': 0, 'negative': 0, 'all_zero_timeslices': 0, 'sampled_points_total': 14424491, 'n_times': 421, 'result_all_zero': 0}`
- MAN-YI 经纬度网格范围: lon 103.00 to 172.00, lat 2.00 to 27.00, shape (251, 691)
- MAN-YI 反投影后 NaN/Inf/负值/全零检查: `{'nan': 0, 'inf': 0, 'negative': 0, 'all_zero_timeslices': 0, 'sampled_points_total': 18105035, 'n_times': 553, 'result_all_zero': 0}`

### 记录的问题
- None

## 4. 分台风地理结果摘要
### KONG-REY
- 地理累计降水最大值: 722.047 mm at (124.900, 18.600)
- 地理最大半小时雨强: 51.173 mm/hr at (132.200, 15.200), time 2024-10-27 00:00:00
- 最大 duration10: 27.000 h at (131.100, 15.500)
- 最大 duration20: 17.000 h at (127.700, 16.500)
- 累计降水 >=50/100/200 mm 面积: 2248696.2 / 1228918.2 / 515031.1 km2
- duration10 >=1/3/6 h 面积: 1374273.6 / 805546.8 / 464393.3 km2
- area_time_10 / area_time_20: 8075613.5 / 2330586.0 km2 h
### MAN-YI
- 地理累计降水最大值: 533.592 mm at (153.200, 13.900)
- 地理最大半小时雨强: 53.529 mm/hr at (135.400, 9.700), time 2024-11-14 10:00:00
- 最大 duration10: 23.500 h at (153.100, 13.800)
- 最大 duration20: 13.000 h at (154.800, 14.600)
- 累计降水 >=50/100/200 mm 面积: 3475292.3 / 2071050.5 / 789773.1 km2
- duration10 >=1/3/6 h 面积: 1738938.8 / 1307069.0 / 826625.0 km2
- area_time_10 / area_time_20: 10440287.6 / 2320139.8 km2 h

## 5. 与 storm-relative 结果关系说明
- storm-relative 图用于解释降水相对台风中心和移动方向的结构。
- geographic 图用于展示固定地理经纬度上的可能降水落区。
- 两者回答的问题不同，不能互相替代。
- 论文中应同时保留两类图：storm-relative 结构图负责机理解释，geographic 落区图负责真实空间影响表达。

## 6. 输出检查
- geographic summary shape: `(2, 49)`
- geographic key locations shape: `(12, 10)`
- geographic timeseries shape: `(974, 15)`
- generated figure count: 10
- `outputs/figures/problem2_env/KONG_REY_final_cumulative_rain_geographic.png`
- `outputs/figures/problem2_env/KONG_REY_final_duration10_geographic.png`
- `outputs/figures/problem2_env/KONG_REY_final_max_rain_geographic.png`
- `outputs/figures/problem2_env/KONG_REY_final_duration20_geographic.png`
- `outputs/figures/problem2_env/MAN_YI_final_cumulative_rain_geographic.png`
- `outputs/figures/problem2_env/MAN_YI_final_duration10_geographic.png`
- `outputs/figures/problem2_env/MAN_YI_final_max_rain_geographic.png`
- `outputs/figures/problem2_env/MAN_YI_final_duration20_geographic.png`
- `outputs/figures/problem2_env/problem2_geo_cumulative_compare.png`
- `outputs/figures/problem2_env/problem2_geo_duration10_compare.png`

## 7. 论文可写结论
- KONG-REY 地理累计降水高值中心位于约 124.90E, 18.60N，最大累计降水约 722.0 mm。
- MAN-YI 地理累计降水高值中心位于约 153.20E, 13.90N，最大累计降水约 533.6 mm。
- KONG-REY 的 duration10 地理最大持续时间更长，说明其固定地理落区上的强降水停留特征更突出。
- MAN-YI 的累计降水超过 100 mm 的地理面积更大，反映其可能影响范围更广。
- 地理图补足了 storm-relative 图不能表示固定经纬度落区的不足。
- 最终问题二结果应同时展示 storm-relative 结构图和 geographic 落区图，两者回答的问题不同，不能互相替代。
