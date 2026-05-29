# Problem 2 Top-K Retrieval QC Report

## 1. 输入文件
- 历史库路径: `data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- 目标路径/安全输入来源: `data/processed/problem3_scenario_inputs.csv` (problem3_scenario_inputs)
- 目标安全输入输出: `data/processed/problem2_target_halfhour_inputs_safe.csv`
- Top-K 输出: `data/processed/problem3/problem3_scenario_topk_analogs.csv`
- QC 报告: `outputs/reports/problem3/problem3_step2_topk_retrieval_qc_from_problem2_logic.md`

## 2. 目标输入表统计
- 总行数: 2841
- 分台风行数和时间范围:
```
              size                 min                 max
typhoon_name                                              
S0             421 2024-10-24 18:00:00 2024-11-02 12:00:00
S1             421 2024-10-24 18:00:00 2024-11-02 12:00:00
S2             421 2024-10-24 18:00:00 2024-11-02 12:00:00
S3             421 2024-10-24 18:00:00 2024-11-02 12:00:00
S4             647 2024-10-24 18:00:00 2024-11-07 05:00:00
S5             510 2024-10-24 18:00:00 2024-11-04 08:30:00
```
- WND/PRES/move_speed/move_dir 缺失率:
- WND: 0.000000
- PRES: 0.000000
- move_speed_kmh: 0.000000
- move_dir_deg: 0.000000
- 使用字段列表: target_row_id, scenario_id, scenario_name, target_id, typhoon_name, target_name_norm, time, source_file, event_uid, is_target, lat, lon, lon_180, WND, PRES, intensity, move_distance_km, move_speed_kmh, move_dir_deg, wind_change_rate, pressure_change_rate, dt_h, is_land, signed_coast_dist_km, coast_dist_km, landfrac_200km, landfrac_500km, terrain_mean_300km, terrain_max_300km, terrain_std_300km, year, month, day, hour, season, month_sin, month_cos, hour_sin, hour_cos, life_time_index, life_progress, move_dir_sin, move_dir_cos, target_time_window_flag, safe_input_flag
- 被明确排除的泄漏字段列表: rain_*, centroid_*, quad_*, anisotropy, rain_radius_*, rain_band_width_km, tif_path, gpm_center_lon/gpm_center_lat, center_match_distance_km
- 源文件中检测到的泄漏字段: []
- 未生成或全缺失环境字段: []

## 3. 历史库过滤统计
- 原始历史库行数: 32842
- 过滤后历史库行数: 32842
- 是否发现 KONG-REY/MAN-YI 名称: 否 (0)
- 是否发现目标时间窗样本: 否 (0)
- 最终参与检索历史样本数: 32842

## 4. 检索特征与权重
- TOPK: 20
- DIVERSIFY_BY_EVENT: True
- MAX_PER_HISTORY_EVENT: 3
- ENV_FEATURE_SET: base-old
- D_env 定义: selected environment features standardized by history mean/std, then averaged squared difference.
- 分量权重: {'track': 1.0, 'intensity': 1.5, 'motion': 1.2, 'environment': 1.2, 'time': 0.8, 'life_progress': 0.8}
- 实际参与检索的特征列表: ['lat', 'lon_180', 'WND', 'PRES', 'intensity', 'move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate', 'is_land', 'signed_coast_dist_km', 'coast_dist_km', 'month_sin', 'month_cos', 'life_progress']
- 按分量参与特征: {'track': ['lat', 'lon_180'], 'intensity': ['WND', 'PRES', 'intensity'], 'motion': ['move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate'], 'environment': ['is_land', 'signed_coast_dist_km', 'coast_dist_km'], 'time': ['month_sin', 'month_cos'], 'life_progress': ['life_progress']}
- 被跳过的候选特征及原因: {}
- 检索特征中误入泄漏字段: []
- 每个特征缺失率和填补策略:
```
             feature     component  history_missing_rate_before  target_missing_rate_before                                                            strategy  history_missing_rate_after  target_missing_rate_after
                 lat         track                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
             lon_180         track                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
                 WND     intensity                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
                PRES     intensity                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
           intensity     intensity                     0.000000                         0.0                                            mode_then_history_median                         0.0                        0.0
      move_speed_kmh        motion                     0.003319                         0.0 same_event_time_interpolation_then_event_median_then_history_median                         0.0                        0.0
        move_dir_sin        motion                     0.000000                         0.0 same_event_time_interpolation_then_event_median_then_history_median                         0.0                        0.0
        move_dir_cos        motion                     0.000000                         0.0 same_event_time_interpolation_then_event_median_then_history_median                         0.0                        0.0
    wind_change_rate        motion                     0.003319                         0.0 same_event_time_interpolation_then_event_median_then_history_median                         0.0                        0.0
pressure_change_rate        motion                     0.003319                         0.0 same_event_time_interpolation_then_event_median_then_history_median                         0.0                        0.0
             is_land   environment                     0.000000                         0.0                                            mode_then_history_median                         0.0                        0.0
signed_coast_dist_km   environment                     0.000000                         0.0               history_median_fill_for_available_environment_feature                         0.0                        0.0
       coast_dist_km   environment                     0.000000                         0.0               history_median_fill_for_available_environment_feature                         0.0                        0.0
           month_sin          time                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
           month_cos          time                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
       life_progress life_progress                     0.000000                         0.0                                                 history_median_fill                         0.0                        0.0
```

## 5. Top-K 结果统计
- 输出 Top-K 表行数: 56820
- 每个 target_id 是否都有 K 行: True
- 每个 target_id 权重和是否为 1: True
- similarity_distance: min=0.0318741, mean=1.35805, p50=1.0293, p95=3.81777, max=7.91692
- 每个目标时刻唯一历史事件数均值: 7.163675
- 每个目标时刻唯一历史事件数最小值: 7
- 是否存在某个目标时刻 Top-K 被单一台风主导: False
- history_tif_path 存在率: 1.000000

## 6. 分台风统计
### KONG-REY
- 目标时刻数: 0
- Top-K 行数: 0
- 平均 similarity_distance: nan
- rank=1 历史台风 Top 10:
(none)
- Top-K 历史台风 Top 10:
(none)
- 相似历史样本 WND/PRES/month 摘要:
```
       history_WND  history_PRES  history_month
count          0.0           0.0            0.0
mean           NaN           NaN            NaN
std            NaN           NaN            NaN
min            NaN           NaN            NaN
25%            NaN           NaN            NaN
50%            NaN           NaN            NaN
75%            NaN           NaN            NaN
max            NaN           NaN            NaN
```
- 目标样本 WND/PRES/month 摘要:
```
       WND  PRES  month
count  0.0   0.0    0.0
mean   NaN   NaN    NaN
std    NaN   NaN    NaN
min    NaN   NaN    NaN
25%    NaN   NaN    NaN
50%    NaN   NaN    NaN
75%    NaN   NaN    NaN
max    NaN   NaN    NaN
```
### MAN-YI
- 目标时刻数: 0
- Top-K 行数: 0
- 平均 similarity_distance: nan
- rank=1 历史台风 Top 10:
(none)
- Top-K 历史台风 Top 10:
(none)
- 相似历史样本 WND/PRES/month 摘要:
```
       history_WND  history_PRES  history_month
count          0.0           0.0            0.0
mean           NaN           NaN            NaN
std            NaN           NaN            NaN
min            NaN           NaN            NaN
25%            NaN           NaN            NaN
50%            NaN           NaN            NaN
75%            NaN           NaN            NaN
max            NaN           NaN            NaN
```
- 目标样本 WND/PRES/month 摘要:
```
       WND  PRES  month
count  0.0   0.0    0.0
mean   NaN   NaN    NaN
std    NaN   NaN    NaN
min    NaN   NaN    NaN
25%    NaN   NaN    NaN
50%    NaN   NaN    NaN
75%    NaN   NaN    NaN
max    NaN   NaN    NaN
```
## 7. 防泄漏声明
本步骤 Top-K 检索只使用路径、强度、移动、时间、海陆和地形等安全输入特征；rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等由 GPM 降水计算得到的字段未参与距离计算，仅随历史样本保留供后续模板生成、极端校准和伪缺失验证使用。

## 8. 其他说明
- 标准化矩阵维度: {'features': ['lat', 'lon_180', 'WND', 'PRES', 'intensity', 'move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate', 'is_land', 'signed_coast_dist_km', 'coast_dist_km', 'month_sin', 'month_cos', 'life_progress'], 'component_feature_counts': {'track': 2, 'intensity': 3, 'motion': 5, 'environment': 3, 'time': 2, 'life_progress': 1}, 'history_z_shape': [32842, 16], 'target_z_shape': [2841, 16]}
