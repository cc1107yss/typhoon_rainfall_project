# Problem 2 Final Results: KONG-REY / MAN-YI

## 1. 输入输出文件
- Target safe input table: `data/processed/problem2_target_halfhour_inputs_safe.csv`
- Top-K retrieval table: `data/processed/problem2_target_topk_similar_history.csv`
- Calibrated NPZ: `data/processed/problem2_generated_calibrated_fields.npz`
- Calibrated index: `data/processed/problem2_generated_calibrated_fields_index.csv`
- Step-21 QC report: `outputs/problem2_extreme_calibration_qc_report.md`
- Step-22 model validation summary: `data/processed/problem2_pseudo_validation_model_summary.csv`
- Step-22 event validation summary: `data/processed/problem2_pseudo_validation_event_summary.csv`
- Step-22 timeslice validation metrics: `data/processed/problem2_pseudo_validation_timeslice_metrics.csv`
- Step-22 QC report: `outputs/problem2_pseudo_validation_qc_report.md`
- Final timeseries CSV: `data/processed/problem2_final_timeseries_metrics.csv`
- Final summary CSV: `data/processed/problem2_final_typhoon_metrics_summary.csv`
- Final key times CSV: `data/processed/problem2_final_key_times.csv`
- Figures directory: `outputs/figures/problem2_final_results`

## 2. 数据完整性检查
- calibrated rain field shape: `(974, 201, 201)`
- target 时刻数: 974
- KONG-REY 时刻数: 421
- MAN-YI 时刻数: 553
- NaN/Inf/负值/全零场数量: {'nan': 0, 'inf': 0, 'negative': 0, 'all_zero': 0}
- calibration_ok 是否全 True: True
- final timeseries CSV 行列数: (974, 122)
- final summary CSV 行列数: (2, 63)
- final key times CSV 行列数: (16, 17)
- 生成图件数量: 19

## 3. KONG-REY 结果摘要
- 时间范围: 2024-10-24 18:00:00 to 2024-11-02 12:00:00
- 最大 WND: 60 at 2024-10-30 00:00:00
- 最小 PRES: 920 at 2024-10-30 00:00:00
- 最大 final_rain_p95: 3.88092 at 2024-10-30 09:00:00
- 最大 final_rain_p99: 16.8213 at 2024-10-30 09:00:00
- 最大 final_rain_max: 53.7328 at 2024-10-26 23:00:00
- 最大 final_rain_area_10: 87800 at 2024-10-30 09:00:00
- 最大 duration10: 146 h
- 累计降水 proxy: 4.92975e+08 mm km2
- 主要降水象限: front_left
- 简短解释: The P95 rainfall peak occurred 9.0 h after the WND peak, so the strongest extreme-rainfall stage was near intensity peak. The maximum area10 stage was 25.5 h before the closest-coast time, indicating it was before the closest-coast stage.

## 4. MAN-YI 结果摘要
- 时间范围: 2024-11-08 12:00:00 to 2024-11-20 00:00:00
- 最大 WND: 62 at 2024-11-16 00:00:00
- 最小 PRES: 920 at 2024-11-16 00:00:00
- 最大 final_rain_p95: 4.99308 at 2024-11-14 03:30:00
- 最大 final_rain_p99: 17.1411 at 2024-11-14 04:30:00
- 最大 final_rain_max: 54.4054 at 2024-11-14 10:30:00
- 最大 final_rain_area_10: 115500 at 2024-11-14 04:00:00
- 最大 duration10: 196 h
- 累计降水 proxy: 6.86489e+08 mm km2
- 主要降水象限: front_left
- 简短解释: The P95 rainfall peak occurred 44.5 h before the WND peak, so the strongest extreme-rainfall stage was before intensity peak. The maximum area10 stage was 76.5 h before the closest-coast time, indicating it was before the closest-coast stage.

## 5. 两场台风对比
- 最大强度更高: MAN-YI，以 WND_max 为依据。
- 极端降水 P95 更高: MAN-YI; P99 更高: MAN-YI。
- 强降水面积更大: MAN-YI，以 final_rain_area_10_km2_max 为依据。
- 强降水持续时间更长: MAN-YI，以 max_duration_10_h 为依据。
- 平均降水质心偏移更大: KONG-REY; 平均非对称/各向异性更强: KONG-REY。
- 路径、强度和距岸距离共同影响强降水阶段：强度峰值附近更容易出现极端分位数峰值，近岸或登陆附近阶段更容易扩大强降水面积和持续时间。

### 台风级核心指标表
| typhoon_name | duration_hours | WND_max | PRES_min | final_rain_p95_mmhr_max | final_rain_p99_mmhr_max | final_rain_max_mmhr_max | final_rain_area_10_km2_max | max_duration_10_h | total_volume_proxy_mm_km2 | dominant_quadrant_by_total_rain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KONG-REY | 210.5 | 60 | 920 | 3.88092 | 16.8213 | 53.7328 | 87800 | 146 | 4.92975e+08 | front_left |
| MAN-YI | 276.5 | 62 | 920 | 4.99308 | 17.1411 | 54.4054 | 115500 | 196 | 6.86489e+08 | front_left |

## 6. 模型验证引用
- Initial vs calibrated validation metrics:
  - P95 absolute error: 3.21097 -> 1.93514; improvement=39.73%.
  - P99 absolute error: 9.77556 -> 5.92285; improvement=39.41%.
  - Area10 absolute error: 62371.1 -> 39214.4; improvement=37.13%.
  - CSI10: 0.023035 -> 0.109664; absolute gain=0.0866288.
  - RMSE/Corr trade-off: RMSE 2.53666 -> 2.68153, Corr 0.366352 -> 0.353116. The calibration improves extreme quantiles and heavy-rain hit skill while slightly sacrificing whole-field RMSE/correlation.

## 7. 论文可写结论
1. KONG-REY 的条件生成降水过程在 2024-10-30 09:00:00 达到 P95 极端峰值，P95=3.881 mm/hr。
2. MAN-YI 的条件生成降水过程在 2024-11-14 03:30:00 达到 P95 极端峰值，P95=4.993 mm/hr。
3. 两场台风相比，MAN-YI 的 P95 极端降水更高，MAN-YI 的强降水面积峰值更大。
4. KONG-REY 的累计降水主要贡献象限为 front_left，MAN-YI 为 front_left，说明降水分布相对于移动方向存在明显不对称。
5. KONG-REY 与 MAN-YI 的最大 duration10 分别为 146.000 h 和 196.000 h，可用于讨论强降水持续性差异。
6. 两场台风平均降水质心偏移分别为 135.460 km 和 93.857 km，表明降水中心并不总与台风中心重合。
7. 极端分位数校准显著降低了 P95/P99 与 area10 误差，并提高了 10 mm/hr 阈值命中能力，适合作为最终结果版本。
8. 本结果是基于历史相似台风、路径、强度和环境条件生成的可能降水分布，不是 KONG-REY 和 MAN-YI 的真实观测降水。

## 8. 注意事项
- KONG-REY 和 MAN-YI 在本项目使用的 GPM 降水集中无真实记录。
- 本文结果是基于历史相似台风和路径-强度-环境条件生成的可能降水场。
- 不应将 final_ 结果表述为真实观测。
- final_ 结果来自 21 号 calibrated 场。
- 22 号伪缺失验证用于证明模型链条的合理性，不代表这两场目标台风的实测误差。

## 9. 图件清单
- `outputs/figures/problem2_final_results/KONG_REY_final_track_intensity_rain_timeseries.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_track_intensity_rain_timeseries.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_rain_metrics_timeseries.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_rain_metrics_timeseries.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_representative_fields.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_representative_fields.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_cumulative_rain_storm_relative.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_cumulative_rain_storm_relative.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_max_rain_storm_relative.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_max_rain_storm_relative.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_duration10_storm_relative.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_duration20_storm_relative.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_duration10_storm_relative.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_duration20_storm_relative.png`
- `outputs/figures/problem2_final_results/KONG_REY_final_quadrant_contribution.png`
- `outputs/figures/problem2_final_results/MAN_YI_final_quadrant_contribution.png`
- `outputs/figures/problem2_final_results/problem2_final_two_typhoons_timeseries_compare.png`
- `outputs/figures/problem2_final_results/problem2_final_two_typhoons_summary_compare.png`
- `outputs/figures/problem2_final_results/problem2_final_validation_model_comparison.png`

## 10. 运行记录问题
- 无异常。
