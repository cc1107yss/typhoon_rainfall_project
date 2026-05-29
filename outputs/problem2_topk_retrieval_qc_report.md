# Problem 2 Top-K Retrieval QC Report

## 1. 输入文件
- 历史库路径: `data/processed/problem2_historical_halfhour_sample_library.csv`
- 目标路径/安全输入来源: `data/processed/target_typhoon_inputs_2024_halfhour_leakage_safe.csv` (existing_halfhour_safe_input)
- 目标安全输入输出: `data/processed/problem2_target_halfhour_inputs_safe.csv`
- Top-K 输出: `data/processed/problem2_target_topk_similar_history.csv`
- QC 报告: `outputs/problem2_topk_retrieval_qc_report.md`

## 2. 目标输入表统计
- 总行数: 974
- 分台风行数和时间范围:
```
              size                 min                 max
typhoon_name                                              
KONG-REY       421 2024-10-24 18:00:00 2024-11-02 12:00:00
MAN-YI         553 2024-11-08 12:00:00 2024-11-20 00:00:00
```
- WND/PRES/move_speed/move_dir 缺失率:
- WND: 0.000000
- PRES: 0.000000
- move_speed_kmh: 0.000000
- move_dir_deg: 0.000000
- 使用字段列表: target_id, typhoon_name, target_name_norm, time, source_file, event_uid, is_target, lat, lon, lon_180, WND, PRES, intensity, move_distance_km, move_speed_kmh, move_dir_deg, wind_change_rate, pressure_change_rate, dt_h, typhoon_id, storm_seq, typhoon_code, record_count, cadence_hours, is_land, coast_dist_km, signed_coast_dist_km, landfrac_100km, landfrac_200km, landfrac_300km, elev_mean_200km, elev_max_200km, terrain_std_200km, year, month, day, hour, season, month_sin, month_cos, hour_sin, hour_cos, life_time_index, life_progress, move_dir_sin, move_dir_cos, target_time_window_flag, safe_input_flag
- 被明确排除的泄漏字段列表: rain_*, centroid_*, quad_*, anisotropy, rain_radius_*, rain_band_width_km, tif_path, gpm_center_lon/gpm_center_lat, center_match_distance_km
- 源文件中检测到的泄漏字段: []
- 未生成或全缺失环境字段: ['landfrac_100km', 'landfrac_200km', 'landfrac_300km', 'elev_mean_200km', 'elev_max_200km', 'terrain_std_200km']

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
- 分量权重: {'track': 1.0, 'intensity': 1.5, 'motion': 1.2, 'environment': 1.2, 'time': 0.8, 'life_progress': 0.8}
- 实际参与检索的特征列表: ['lat', 'lon_180', 'WND', 'PRES', 'intensity', 'move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate', 'is_land', 'signed_coast_dist_km', 'coast_dist_km', 'month_sin', 'month_cos', 'life_progress']
- 按分量参与特征: {'track': ['lat', 'lon_180'], 'intensity': ['WND', 'PRES', 'intensity'], 'motion': ['move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate'], 'environment': ['is_land', 'signed_coast_dist_km', 'coast_dist_km'], 'time': ['month_sin', 'month_cos'], 'life_progress': ['life_progress']}
- 被跳过的候选特征及原因: {'landfrac_100km': 'history_all_missing', 'landfrac_200km': 'history_all_missing', 'landfrac_300km': 'history_all_missing', 'elev_mean_200km': 'history_all_missing', 'elev_max_200km': 'history_all_missing', 'terrain_std_200km': 'history_all_missing'}
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
- 输出 Top-K 表行数: 19480
- 每个 target_id 是否都有 K 行: True
- 每个 target_id 权重和是否为 1: True
- similarity_distance: min=0.0629807, mean=1.27098, p50=1.15564, p95=2.34497, max=6.60621
- 每个目标时刻唯一历史事件数均值: 7.165298
- 每个目标时刻唯一历史事件数最小值: 7
- 是否存在某个目标时刻 Top-K 被单一台风主导: False
- history_tif_path 存在率: 1.000000

## 6. 分台风统计
### KONG-REY
- 目标时刻数: 421
- Top-K 行数: 8420
- 平均 similarity_distance: 1.059143
- rank=1 历史台风 Top 10:
- DUJUAN: 130
- MUIFA: 76
- KALMAEGI: 40
- MALAKAS: 39
- MITAG: 32
- TALIM: 15
- FUNG-WONG: 15
- CIMARON: 14
- HAIMA: 13
- MALOU: 12
- Top-K 历史台风 Top 10:
- DUJUAN: 934
- MUIFA: 872
- MALAKAS: 682
- KOPPU: 657
- TALIM: 578
- MITAG: 544
- KALMAEGI: 525
- MERANTI: 503
- KOINU: 355
- CHANTHU: 253
- 相似历史样本 WND/PRES/month 摘要:
```
       history_WND  history_PRES  history_month
count  8420.000000   8420.000000    8420.000000
mean     30.051930    976.462074       9.032067
std      15.077938     27.276531       0.571873
min      10.000000    907.500000       7.000000
25%      18.000000    957.500000       9.000000
50%      24.666667    990.000000       9.000000
75%      40.833333    998.000000       9.000000
max      66.250000   1007.000000      11.000000
```
- 目标样本 WND/PRES/month 摘要:
```
              WND         PRES       month
count  421.000000   421.000000  421.000000
mean    30.719715   975.598575   10.173397
std     15.041931    27.911655    0.379040
min     15.000000   920.000000   10.000000
25%     18.000000   956.666667   10.000000
50%     24.500000   986.250000   10.000000
75%     41.333333   998.000000   10.000000
max     60.000000  1004.000000   11.000000
```
### MAN-YI
- 目标时刻数: 553
- Top-K 行数: 11060
- 平均 similarity_distance: 1.432256
- rank=1 历史台风 Top 10:
- KOPPU: 122
- SONGDA: 121
- HAIMA: 75
- YUTU: 45
- NALGAE: 38
- ATSANI: 31
- CHOI-WAN: 27
- SAUDEL: 24
- KOINU: 24
- SARIKA: 22
- Top-K 历史台风 Top 10:
- KOPPU: 1208
- HAIMA: 807
- YUTU: 676
- CHOI-WAN: 676
- DUJUAN: 626
- SONGDA: 575
- NALGAE: 564
- ATSANI: 506
- TALIM: 447
- KOINU: 383
- 相似历史样本 WND/PRES/month 摘要:
```
        history_WND  history_PRES  history_month
count  11060.000000  11060.000000   11060.000000
mean      27.573056    981.088841       9.776221
std       14.260435     25.008882       0.619275
min       13.000000    903.333333       8.000000
25%       18.000000    974.166667       9.000000
50%       23.000000    991.833333      10.000000
75%       34.666667    998.000000      10.000000
max       68.333333   1008.333333      11.000000
```
- 目标样本 WND/PRES/month 摘要:
```
              WND         PRES  month
count  553.000000   553.000000  553.0
mean    29.708861   980.739602   11.0
std     14.428932    25.811296    0.0
min     13.000000   920.000000   11.0
25%     20.000000   972.500000   11.0
50%     23.000000   995.000000   11.0
75%     35.083333   998.000000   11.0
max     62.000000  1005.000000   11.0
```
## 7. 防泄漏声明
本步骤 Top-K 检索只使用路径、强度、移动、时间、海陆和地形等安全输入特征；rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等由 GPM 降水计算得到的字段未参与距离计算，仅随历史样本保留供后续模板生成、极端校准和伪缺失验证使用。

## 8. 其他说明
- 标准化矩阵维度: {'features': ['lat', 'lon_180', 'WND', 'PRES', 'intensity', 'move_speed_kmh', 'move_dir_sin', 'move_dir_cos', 'wind_change_rate', 'pressure_change_rate', 'is_land', 'signed_coast_dist_km', 'coast_dist_km', 'month_sin', 'month_cos', 'life_progress'], 'component_feature_counts': {'track': 2, 'intensity': 3, 'motion': 5, 'environment': 3, 'time': 2, 'life_progress': 1}, 'history_z_shape': [32842, 16], 'target_z_shape': [974, 16]}
