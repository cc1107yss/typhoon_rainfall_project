# Problem-2 Rainband Width Diagnostics

- Input NPZ: `data/processed/problem2_env/problem2_generated_calibrated_fields_env.npz`
- Timeseries CSV: `outputs/tables/problem2_env/problem2_rainband_width_timeseries.csv`
- Summary CSV: `outputs/tables/problem2_env/problem2_rainband_width_summary.csv`
- Figure: `outputs/figures/problem2_env/problem2_rainband_width_timeseries.png`
- The metrics are diagnostics computed after generation; they are not target-typhoon input variables.
- Field-quality summary: `{'rows': 974, 'main_width_nan': 0, 'width10_nan': 0, 'main_valid_min': 2349, 'heavy_valid_min': 18}`

| typhoon_name | n_times | mean_rainband_width_km | median_rainband_width_km | max_rainband_width_km | mean_rainband_width10_km | max_rainband_width10_km | time_of_max_width | time_of_max_width10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KONG-REY | 421 | 503.1983 | 477.0858 | 824.0419 | 205.7732 | 711.0488 | 2024-10-31 18:30:00 | 2024-11-02 10:00:00 |
| MAN-YI | 553 | 570.6941 | 536.3552 | 1439.1946 | 245.1036 | 2224.7694 | 2024-11-19 20:00:00 | 2024-11-19 16:30:00 |
