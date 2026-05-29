# Problem-3 Rainband Width Diagnostics

- Input NPZ: `data/processed/problem3/problem3_scenario_rainfields_calibrated.npz`
- Timeseries CSV: `outputs/tables/problem3/problem3_rainband_width_timeseries.csv`
- Summary CSV: `outputs/tables/problem3/problem3_rainband_width_summary.csv`
- Figure: `outputs/figures/problem3/problem3_rainband_width_bars.png`
- The metrics diagnose generated S0-S5 rainfall fields and are not safe-input variables.
- Field-quality summary: `{'rows': 2841, 'main_width_nan': 0, 'width10_nan': 1}`

| scenario_id | scenario_name | n_times | mean_rainband_width_km | median_rainband_width_km | max_rainband_width_km | mean_rainband_width10_km | max_rainband_width10_km | time_of_max_width | time_of_max_width10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | baseline_kong_rey | 421 | 510.7756 | 490.9603 | 1220.3180 | 218.3651 | 1132.3119 | 2024-11-01 00:30:00 | 2024-11-01 00:30:00 |
| S1 | intensity_enhanced | 421 | 534.3267 | 493.7832 | 1438.5804 | 250.3672 | 1868.9162 | 2024-11-01 07:00:00 | 2024-11-01 09:30:00 |
| S2 | near_coast_westward_path_shift | 421 | 513.8633 | 500.2773 | 986.7340 | 210.0770 | 1020.3357 | 2024-11-01 03:30:00 | 2024-11-01 03:30:00 |
| S3 | nearshore_segment_landfall_shift | 421 | 515.1620 | 490.7570 | 1267.5833 | 219.6611 | 1356.5000 | 2024-11-01 02:30:00 | 2024-11-01 00:30:00 |
| S4 | slower_translation | 647 | 521.3288 | 482.9768 | 1103.6784 | 222.3717 | 1152.7370 | 2024-11-04 21:00:00 | 2024-11-07 04:00:00 |
| S5 | compound_moderate_high_risk | 510 | 504.5119 | 471.9570 | 873.1413 | 205.8832 | 965.1749 | 2024-11-02 08:00:00 | 2024-11-04 02:00:00 |
