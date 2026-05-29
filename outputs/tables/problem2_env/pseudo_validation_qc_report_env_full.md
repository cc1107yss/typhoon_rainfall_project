# Problem 2 Pseudo-Missing Validation QC Report

## 1. Input and Output Files
- Historical library: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- EOF/PCA model: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_eof_pca_model.npz`
- Validation events CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_events_env_full.csv`
- Timeslice metrics CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_timeslice_metrics_env_full.csv`
- Event summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_event_summary_env_full.csv`
- Model summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_model_summary_env_full.csv`
- Representative fields NPZ: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_generated_fields_env_full.npz`
- Figures directory: `/Users/chenchen/typhoon_rainfall_project/outputs/figures/problem2_env/pseudo_validation_env_full`

## 2. Run Parameters
- N_VALIDATION_EVENTS: 8
- MAX_TIMES_PER_EVENT: 80
- TOPK: 20
- MAX_PER_HISTORY_EVENT: 3
- MIN_VALID_TEMPLATES: 5
- ENV_FEATURE_SET: env-full
- D_env: mean squared standardized difference across selected environment features
- BETA_BLEND: 0.3
- USE_EOF_MODEL: True
- USE_EXTREME_CALIBRATION: True
- RANDOM_SEED: 2026

## 3. Validation Event Selection
| validation_event_uid | typhoon_name | start_time | end_time | n_total_times | n_selected_times | WND_max | PRES_min | rain_p95_mmhr_max | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2016_0016 | MERANTI | 2016-09-09 00:00:00 | 2016-09-17 06:00:00 | 397 | 80 | 75 | 890 | 5.52 | strongest_wind |
| CH2024_0022 | TRAMI | 2024-10-20 00:00:00 | 2024-10-23 23:30:00 | 192 | 80 | 25 | 985 | 15.38 | highest_p95_rain |
| CH2019_0007 | MUN | 2019-07-02 00:00:00 | 2019-07-04 18:00:00 | 133 | 80 | 18 | 992 | 5.47 | most_land_or_landfall_influenced |
| CH2020_0003 | SINLAKU | 2020-07-31 06:00:00 | 2020-08-03 00:00:00 | 133 | 80 | 18 | 992 | 9.24 | near_coast_or_landfall |
| CH2019_0031 | FUNG-WONG | 2019-11-18 00:00:00 | 2019-11-24 00:00:00 | 289 | 80 | 30 | 980 | 10.57 | open_ocean |
| CH2014_0007 | MITAG | 2014-06-09 00:00:00 | 2014-06-12 00:00:00 | 145 | 80 | 20 | 994 | 7.6 | weak_intensity |
| CH2015_0014 | SOUDELOR | 2015-07-30 00:00:00 | 2015-08-12 06:00:00 | 637 | 80 | 68 | 905 | 8.77 | diversity_fill |
| CH2023_0003 | MAWAR | 2023-05-19 18:00:00 | 2023-06-03 12:00:00 | 709 | 80 | 68 | 905 | 9.53 | diversity_fill |

## 4. Filtering and Leakage Guard
- Training library excludes the current validation event for each run: yes
- Leakage fields used in retrieval: no
- Truth tif is read only in evaluation stage: yes
- KONG-REY / MAN-YI selected as validation events: no
- Top-K rows from the same validation event: 0
- Safe retrieval features: lat, lon_180, WND, PRES, intensity, move_speed_kmh, move_dir_sin, move_dir_cos, wind_change_rate, pressure_change_rate, is_land, signed_coast_dist_km, coast_dist_km, landfrac_200km, landfrac_500km, terrain_mean_300km, terrain_max_300km, terrain_std_300km, month_sin, month_cos, life_progress

## 5. Generation Process Statistics
- Total selected validation times: 640
- Timeslice metric rows: 1920
- Successful model-version rows: 1920
- Skipped/flagged model-version rows: 0
- Skip reasons: {}
- valid_template_count distribution: min=20, mean=20, p50=20, p95=20, max=20
- topk_unique_event_count distribution: min=7, mean=7.325, p50=7, p95=8, max=9
- Template cache counters: {'cache_misses': 4142, 'read_attempts': 4142, 'templates_ok': 4142, 'cache_hits': 8658}

## 6. Overall Performance by Version
| model_version | n_events | n_timeslices | rmse_mean | mae_mean | bias_mean | corr_mean | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_rain_max_mean | abs_error_area_10_mean | abs_error_area_20_mean | abs_error_centroid_offset_mean | abs_error_anisotropy_mean | csi10_mean | pod10_mean | far10_mean | f1_10_mean | csi20_mean | f1_20_mean | duration10_area_time_rel_error_median | duration20_area_time_rel_error_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | 8 | 640 | 2.53799 | 0.875519 | -0.544236 | 0.370638 | 3.22143 | 9.76296 | 36.6805 | 62664.5 | 19846.9 | 112.09 | 0.211245 | 0.0208411 | 0.0256755 | 0.154082 | 0.0355855 | 0.00015076 | 0.000292075 | 0.996103 | 1 |
| blend | 8 | 640 | 2.5348 | 0.869711 | -0.549356 | 0.383845 | 3.263 | 9.82926 | 37.7157 | 63047 | 19855.9 | 112.726 | 0.216094 | 0.017667 | 0.0217817 | 0.122199 | 0.0300765 | 0 | 0 | 0.999163 | 1 |
| calibrated | 8 | 640 | 2.69231 | 0.963661 | -0.317066 | 0.353678 | 1.97629 | 6.02059 | 18.8179 | 38861.9 | 16856.1 | 114.282 | 0.206702 | 0.108403 | 0.17458 | 0.766709 | 0.176201 | 0.043878 | 0.0722112 | 0.459333 | 0.743881 |

## 7. Module Gain Analysis
- Blend vs initial RMSE improvement: 0.13%
- Calibrated vs initial RMSE improvement: -6.08%
- Calibrated vs initial P95 error improvement: 38.65%
- Calibrated vs initial P99 error improvement: 38.33%
- Calibrated vs initial area10 error improvement: 37.98%
- Calibrated vs initial CSI10 change: 0.0875615
- Blend spatial structure: centroid error 112.726 vs initial 112.09; anisotropy error 0.216094 vs initial 0.211245.
- Calibration extremes: P95/P99/area10 errors are 1.97629, 6.02059, 38861.9.
- Calibration detection: CSI10/F1_10 are 0.108403, 0.176201.
- RMSE/correlation tradeoff: calibrated RMSE/corr are 2.69231, 0.353678; initial RMSE/corr are 2.53799, 0.370638.

## 8. Failure and Abnormal Samples
### Top 10 RMSE
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-20 15:30:00 | calibrated | 7.76826 | 13.0623 | 263700 | 0.076681 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.73929 | 15.9666 | 303300 | 0.484344 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73221 | 15.8813 | 301500 | 0.467985 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.72695 | 14.6009 | 296800 | 0.160669 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72591 | 14.6258 | 296800 | 0.171015 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47063 | 14.3966 | 266800 | 0.468757 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46766 | 14.3118 | 265000 | 0.448344 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.35661 | 16.313 | 333600 | 0.379802 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.3512 | 16.2573 | 331800 | 0.364565 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.33159 | 13.5694 | 250300 | 0.409413 | 18 | 995 | 295.547 |

### Top 10 P95 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.35661 | 16.313 | 333600 | 0.379802 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.3512 | 16.2573 | 331800 | 0.364565 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.73929 | 15.9666 | 303300 | 0.484344 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73221 | 15.8813 | 301500 | 0.467985 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72591 | 14.6258 | 296800 | 0.171015 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.72695 | 14.6009 | 296800 | 0.160669 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47063 | 14.3966 | 266800 | 0.468757 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46766 | 14.3118 | 265000 | 0.448344 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.03969 | 14.2104 | 281100 | 0.356767 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.33159 | 13.5694 | 250300 | 0.409413 | 18 | 995 | 295.547 |

### Top 10 Area10 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.35661 | 16.313 | 333600 | 0.379802 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.3512 | 16.2573 | 331800 | 0.364565 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.73929 | 15.9666 | 303300 | 0.484344 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73221 | 15.8813 | 301500 | 0.467985 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.72695 | 14.6009 | 296800 | 0.160669 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72591 | 14.6258 | 296800 | 0.171015 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.03969 | 14.2104 | 281100 | 0.356767 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 15:30:00 | blend | 6.74069 | 12.7157 | 275100 | 0.354004 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 15:30:00 | initial | 6.73608 | 12.6752 | 274400 | 0.340488 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47063 | 14.3966 | 266800 | 0.468757 | 18 | 995 | 300.021 |

### Lowest 10 Correlation
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2016_0016 | 2016-09-17 06:00:00 | initial | 2.50874 | 3.44736 | 64100 | -0.0457452 | 13 | 1004 | 173.369 |
| CH2016_0016 | 2016-09-17 06:00:00 | blend | 2.5043 | 3.4824 | 64100 | -0.0295194 | 13 | 1004 | 173.369 |
| CH2019_0031 | 2019-11-23 02:30:00 | calibrated | 1.32925 | 1.65956 | 19500 | -0.0142388 | 13 | 1008 | 71.2707 |
| CH2016_0016 | 2016-09-17 06:00:00 | calibrated | 2.51782 | 2.59315 | 64100 | -0.0129891 | 13 | 1004 | 173.369 |
| CH2019_0031 | 2019-11-23 02:00:00 | calibrated | 1.31343 | 1.5438 | 16700 | -0.0127237 | 13 | 1008 | 69.159 |
| CH2019_0031 | 2019-11-23 02:30:00 | initial | 0.773876 | 0.321476 | 2900 | -0.0127012 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 02:30:00 | blend | 0.761385 | 0.291315 | 2900 | -0.0113199 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 03:30:00 | calibrated | 1.28619 | 1.68559 | 16800 | -0.0106406 | 13 | 1008 | 75.4942 |
| CH2019_0031 | 2019-11-23 02:00:00 | initial | 0.815501 | 0.276505 | 3700 | -0.0100073 | 13 | 1008 | 69.159 |
| CH2019_0031 | 2019-11-23 02:00:00 | blend | 0.803621 | 0.253202 | 3700 | -0.00745733 | 13 | 1008 | 69.159 |

Likely causes include analog mismatch during rapid intensity change, coastal terrain discontinuity, compact convective cores that are hard to recover from analog means, and sparse heavy-rain truth coverage near the edge of the storm-relative tile.

## 9. Paper-Ready Conclusions
- The initial Top-K log1p analog field provides a no-target-rainfall baseline that recovers broad storm-relative rainfall placement.
- EOF/PCA blending acts as a large-scale structural regularizer and can be discussed through centroid, anisotropy, RMSE, and correlation changes.
- Extreme quantile calibration directly targets Top-K-derived P95/P99/Rmax and heavy-rain area constraints without reading validation-event rain metrics.
- P95/P99 and heavy-rain area are more stable validation targets than single-grid Rmax, which remains sensitive to small convective cores.
- Threshold metrics such as CSI10 and F1_10 quantify whether heavy-rain identification improves after calibration.
- The self-exclusion check confirms that validation-event samples do not enter Top-K templates.
- The pseudo-missing experiment shows whether the model can generate physically plausible typhoon rainfall structures when the target GPM field is unavailable.
- Generated figure count: 0
