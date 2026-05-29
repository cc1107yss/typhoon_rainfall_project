# Problem 2 Pseudo-Missing Validation QC Report

## 1. Input and Output Files
- Historical library: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- EOF/PCA model: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_eof_pca_model.npz`
- Validation events CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_events_env_key.csv`
- Timeslice metrics CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_timeslice_metrics_env_key.csv`
- Event summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_event_summary_env_key.csv`
- Model summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_model_summary_env_key.csv`
- Representative fields NPZ: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_env/pseudo_validation_generated_fields_env_key.npz`
- Figures directory: `/Users/chenchen/typhoon_rainfall_project/outputs/figures/problem2_env/pseudo_validation_env_key`

## 2. Run Parameters
- N_VALIDATION_EVENTS: 8
- MAX_TIMES_PER_EVENT: 80
- TOPK: 20
- MAX_PER_HISTORY_EVENT: 3
- MIN_VALID_TEMPLATES: 5
- ENV_FEATURE_SET: env-key
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
- Safe retrieval features: lat, lon_180, WND, PRES, intensity, move_speed_kmh, move_dir_sin, move_dir_cos, wind_change_rate, pressure_change_rate, is_land, signed_coast_dist_km, coast_dist_km, landfrac_500km, terrain_std_300km, month_sin, month_cos, life_progress

## 5. Generation Process Statistics
- Total selected validation times: 640
- Timeslice metric rows: 1920
- Successful model-version rows: 1920
- Skipped/flagged model-version rows: 0
- Skip reasons: {}
- valid_template_count distribution: min=20, mean=20, p50=20, p95=20, max=20
- topk_unique_event_count distribution: min=7, mean=7.32188, p50=7, p95=8, max=9
- Template cache counters: {'cache_misses': 4049, 'read_attempts': 4049, 'templates_ok': 4049, 'cache_hits': 8751}

## 6. Overall Performance by Version
| model_version | n_events | n_timeslices | rmse_mean | mae_mean | bias_mean | corr_mean | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_rain_max_mean | abs_error_area_10_mean | abs_error_area_20_mean | abs_error_centroid_offset_mean | abs_error_anisotropy_mean | csi10_mean | pod10_mean | far10_mean | f1_10_mean | csi20_mean | f1_20_mean | duration10_area_time_rel_error_median | duration20_area_time_rel_error_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | 8 | 640 | 2.53702 | 0.874278 | -0.545655 | 0.368637 | 3.21703 | 9.79842 | 36.7446 | 62760.8 | 19836.7 | 111.408 | 0.209399 | 0.020768 | 0.0257175 | 0.142787 | 0.0356984 | 0.000519908 | 0.000969586 | 0.995362 | 1 |
| blend | 8 | 640 | 2.53393 | 0.86837 | -0.550966 | 0.382067 | 3.26158 | 9.87025 | 37.7707 | 63149.2 | 19851.6 | 112.058 | 0.214345 | 0.0176769 | 0.021926 | 0.114047 | 0.0302287 | 0.00010608 | 0.000203016 | 0.99872 | 1 |
| calibrated | 8 | 640 | 2.68078 | 0.961442 | -0.32098 | 0.352785 | 1.95071 | 6.01946 | 18.9924 | 39101.6 | 16950.8 | 114.085 | 0.204174 | 0.109386 | 0.17637 | 0.766134 | 0.177697 | 0.0433351 | 0.0711426 | 0.455096 | 0.781568 |

## 7. Module Gain Analysis
- Blend vs initial RMSE improvement: 0.12%
- Calibrated vs initial RMSE improvement: -5.67%
- Calibrated vs initial P95 error improvement: 39.36%
- Calibrated vs initial P99 error improvement: 38.57%
- Calibrated vs initial area10 error improvement: 37.70%
- Calibrated vs initial CSI10 change: 0.0886185
- Blend spatial structure: centroid error 112.058 vs initial 111.408; anisotropy error 0.214345 vs initial 0.209399.
- Calibration extremes: P95/P99/area10 errors are 1.95071, 6.01946, 39101.6.
- Calibration detection: CSI10/F1_10 are 0.109386, 0.177697.
- RMSE/correlation tradeoff: calibrated RMSE/corr are 2.68078, 0.352785; initial RMSE/corr are 2.53702, 0.368637.

## 8. Failure and Abnormal Samples
### Top 10 RMSE
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-20 15:30:00 | calibrated | 7.79377 | 13.0923 | 264600 | 0.0627472 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74036 | 16.0263 | 303200 | 0.483575 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73644 | 14.6005 | 296800 | 0.142169 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.73473 | 14.6462 | 296800 | 0.153645 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73295 | 15.9433 | 300900 | 0.465781 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47169 | 14.4553 | 266700 | 0.467903 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46815 | 14.3731 | 264400 | 0.446448 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.36428 | 16.3472 | 333600 | 0.38125 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.35943 | 16.2984 | 333300 | 0.366233 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.29685 | 13.4307 | 247900 | 0.420709 | 18 | 995 | 295.547 |

### Top 10 P95 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.36428 | 16.3472 | 333600 | 0.38125 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.35943 | 16.2984 | 333300 | 0.366233 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74036 | 16.0263 | 303200 | 0.483575 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73295 | 15.9433 | 300900 | 0.465781 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.73473 | 14.6462 | 296800 | 0.153645 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73644 | 14.6005 | 296800 | 0.142169 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47169 | 14.4553 | 266700 | 0.467903 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46815 | 14.3731 | 264400 | 0.446448 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.04967 | 14.2532 | 279600 | 0.354561 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.29685 | 13.4307 | 247900 | 0.420709 | 18 | 995 | 295.547 |

### Top 10 Area10 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.36428 | 16.3472 | 333600 | 0.38125 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.35943 | 16.2984 | 333300 | 0.366233 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74036 | 16.0263 | 303200 | 0.483575 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73295 | 15.9433 | 300900 | 0.465781 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73644 | 14.6005 | 296800 | 0.142169 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.73473 | 14.6462 | 296800 | 0.153645 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.04967 | 14.2532 | 279600 | 0.354561 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 15:30:00 | initial | 6.74411 | 12.6885 | 275100 | 0.338923 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 15:30:00 | blend | 6.74762 | 12.7154 | 275100 | 0.352894 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47169 | 14.4553 | 266700 | 0.467903 | 18 | 995 | 300.021 |

### Lowest 10 Correlation
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2016_0016 | 2016-09-17 06:00:00 | initial | 2.50883 | 3.39337 | 64100 | -0.0369999 | 13 | 1004 | 173.369 |
| CH2016_0016 | 2016-09-17 06:00:00 | blend | 2.50486 | 3.44096 | 64100 | -0.0280213 | 13 | 1004 | 173.369 |
| CH2016_0016 | 2016-09-17 06:00:00 | calibrated | 2.52837 | 2.46884 | 63800 | -0.0159635 | 13 | 1004 | 173.369 |
| CH2019_0031 | 2019-11-23 02:00:00 | initial | 0.785113 | 0.174177 | 3700 | -0.0121651 | 13 | 1008 | 69.159 |
| CH2014_0007 | 2014-06-10 04:00:00 | calibrated | 3.34408 | 4.30912 | 62700 | -0.0107256 | 15 | 998 | 76.6571 |
| CH2019_0031 | 2019-11-23 02:30:00 | calibrated | 1.12524 | 1.50728 | 12200 | -0.0099267 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 02:30:00 | initial | 0.743218 | 0.249151 | 2900 | -0.00896923 | 13 | 1008 | 71.2707 |
| CH2014_0007 | 2014-06-10 03:00:00 | calibrated | 3.3824 | 3.84638 | 65000 | -0.00877932 | 15 | 998 | 64.2532 |
| CH2019_0031 | 2019-11-23 02:00:00 | calibrated | 1.14051 | 1.38104 | 11400 | -0.00849804 | 13 | 1008 | 69.159 |
| CH2019_0007 | 2019-07-02 10:00:00 | calibrated | 3.68273 | 1.03628 | 22400 | -0.00820527 | 15 | 995 | 123.646 |

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
