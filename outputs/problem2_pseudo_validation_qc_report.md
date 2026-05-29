# Problem 2 Pseudo-Missing Validation QC Report

## 1. Input and Output Files
- Historical library: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_historical_halfhour_sample_library.csv`
- EOF/PCA model: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_eof_pca_model.npz`
- Validation events CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_pseudo_validation_events.csv`
- Timeslice metrics CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_pseudo_validation_timeslice_metrics.csv`
- Event summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_pseudo_validation_event_summary.csv`
- Model summary CSV: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_pseudo_validation_model_summary.csv`
- Representative fields NPZ: `/Users/chenchen/typhoon_rainfall_project/data/processed/problem2_pseudo_validation_generated_fields.npz`
- Figures directory: `/Users/chenchen/typhoon_rainfall_project/outputs/figures/problem2_pseudo_validation`

## 2. Run Parameters
- N_VALIDATION_EVENTS: 8
- MAX_TIMES_PER_EVENT: 80
- TOPK: 20
- MAX_PER_HISTORY_EVENT: 3
- MIN_VALID_TEMPLATES: 5
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
- Safe retrieval features: lat, lon_180, WND, PRES, intensity, move_speed_kmh, move_dir_sin, move_dir_cos, wind_change_rate, pressure_change_rate, is_land, signed_coast_dist_km, coast_dist_km, month_sin, month_cos, life_progress

## 5. Generation Process Statistics
- Total selected validation times: 640
- Timeslice metric rows: 1920
- Successful model-version rows: 1920
- Skipped/flagged model-version rows: 0
- Skip reasons: {}
- valid_template_count distribution: min=20, mean=20, p50=20, p95=20, max=20
- topk_unique_event_count distribution: min=7, mean=7.15156, p50=7, p95=8, max=9
- Template cache counters: {'cache_misses': 4071, 'read_attempts': 4071, 'templates_ok': 4071, 'cache_hits': 8729}

## 6. Overall Performance by Version
| model_version | n_events | n_timeslices | rmse_mean | mae_mean | bias_mean | corr_mean | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_rain_max_mean | abs_error_area_10_mean | abs_error_area_20_mean | abs_error_centroid_offset_mean | abs_error_anisotropy_mean | csi10_mean | pod10_mean | far10_mean | f1_10_mean | csi20_mean | f1_20_mean | duration10_area_time_rel_error_median | duration20_area_time_rel_error_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial | 8 | 640 | 2.53666 | 0.874142 | -0.547111 | 0.366352 | 3.21097 | 9.77556 | 36.6279 | 62371.1 | 19825.9 | 110.474 | 0.215414 | 0.023035 | 0.0284668 | 0.147093 | 0.0389669 | 0.00061195 | 0.00114543 | 0.994805 | 1 |
| blend | 8 | 640 | 2.53305 | 0.867851 | -0.552509 | 0.380637 | 3.26075 | 9.85368 | 37.7123 | 62807.3 | 19847.7 | 111.166 | 0.220164 | 0.0202545 | 0.0250166 | 0.118851 | 0.0340292 | 0.000248525 | 0.00047352 | 0.997972 | 1 |
| calibrated | 8 | 640 | 2.68153 | 0.960706 | -0.323585 | 0.353116 | 1.93514 | 5.92285 | 19.0788 | 39214.4 | 16833.6 | 116.036 | 0.206522 | 0.109664 | 0.17634 | 0.764365 | 0.17733 | 0.043191 | 0.0712261 | 0.459276 | 0.796372 |

## 7. Module Gain Analysis
- Blend vs initial RMSE improvement: 0.14%
- Calibrated vs initial RMSE improvement: -5.71%
- Calibrated vs initial P95 error improvement: 39.73%
- Calibrated vs initial P99 error improvement: 39.41%
- Calibrated vs initial area10 error improvement: 37.13%
- Calibrated vs initial CSI10 change: 0.0866288
- Blend spatial structure: centroid error 111.166 vs initial 110.474; anisotropy error 0.220164 vs initial 0.215414.
- Calibration extremes: P95/P99/area10 errors are 1.93514, 5.92285, 39214.4.
- Calibration detection: CSI10/F1_10 are 0.109664, 0.17733.
- RMSE/correlation tradeoff: calibrated RMSE/corr are 2.68153, 0.353116; initial RMSE/corr are 2.53666, 0.366352.

## 8. Failure and Abnormal Samples
### Top 10 RMSE
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-20 15:30:00 | calibrated | 7.79422 | 13.0032 | 263500 | 0.0639476 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74309 | 16.0421 | 301400 | 0.456128 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73866 | 15.9812 | 299300 | 0.435498 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73125 | 14.5377 | 296800 | 0.142958 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72944 | 14.5927 | 296800 | 0.154712 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47082 | 14.4612 | 264800 | 0.442881 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46957 | 14.3981 | 262800 | 0.419341 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.38108 | 16.1787 | 333600 | 0.35418 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.37804 | 16.1393 | 333600 | 0.340871 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.34068 | 13.4547 | 246900 | 0.401183 | 18 | 995 | 295.547 |

### Top 10 P95 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.38108 | 16.1787 | 333600 | 0.35418 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.37804 | 16.1393 | 333600 | 0.340871 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74309 | 16.0421 | 301400 | 0.456128 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73866 | 15.9812 | 299300 | 0.435498 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72944 | 14.5927 | 296800 | 0.154712 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73125 | 14.5377 | 296800 | 0.142958 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47082 | 14.4612 | 264800 | 0.442881 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 20:30:00 | initial | 7.46957 | 14.3981 | 262800 | 0.419341 | 18 | 995 | 300.021 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.11576 | 14.1945 | 281700 | 0.329078 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | calibrated | 7.34068 | 13.4547 | 246900 | 0.401183 | 18 | 995 | 295.547 |

### Top 10 Area10 Error
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2024_0022 | 2024-10-21 16:30:00 | initial | 7.37804 | 16.1393 | 333600 | 0.340871 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 16:30:00 | blend | 7.38108 | 16.1787 | 333600 | 0.35418 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 21:00:00 | blend | 7.74309 | 16.0421 | 301400 | 0.456128 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-21 21:00:00 | initial | 7.73866 | 15.9812 | 299300 | 0.435498 | 18 | 995 | 295.547 |
| CH2024_0022 | 2024-10-20 15:30:00 | initial | 7.73125 | 14.5377 | 296800 | 0.142958 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-20 15:30:00 | blend | 7.72944 | 14.5927 | 296800 | 0.154712 | 13 | 1002 | 645.915 |
| CH2024_0022 | 2024-10-21 16:30:00 | calibrated | 7.11576 | 14.1945 | 281700 | 0.329078 | 17.25 | 996.25 | 347.742 |
| CH2024_0022 | 2024-10-21 15:30:00 | blend | 6.76282 | 12.6378 | 275100 | 0.328832 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 15:30:00 | initial | 6.76093 | 12.5953 | 275100 | 0.315414 | 16.75 | 997.083 | 364.645 |
| CH2024_0022 | 2024-10-21 20:30:00 | blend | 7.47082 | 14.4612 | 264800 | 0.442881 | 18 | 995 | 300.021 |

### Lowest 10 Correlation
| validation_event_uid | validation_time | model_version | rmse | abs_error_rain_p95 | abs_error_area_10 | corr | WND | PRES | signed_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CH2019_0031 | 2019-11-23 02:30:00 | initial | 0.72343 | 0.116361 | 2900 | -0.0202098 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 02:30:00 | calibrated | 1.00438 | 1.09325 | 8100 | -0.0155997 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 02:00:00 | initial | 0.768761 | 0.0566635 | 3700 | -0.0148572 | 13 | 1008 | 69.159 |
| CH2019_0031 | 2019-11-23 02:30:00 | blend | 0.715869 | 0.0811994 | 2900 | -0.0145731 | 13 | 1008 | 71.2707 |
| CH2019_0031 | 2019-11-23 02:00:00 | calibrated | 1.03621 | 1.03512 | 7400 | -0.0102478 | 13 | 1008 | 69.159 |
| CH2019_0031 | 2019-11-23 03:30:00 | calibrated | 0.997369 | 1.13376 | 7000 | -0.00996442 | 13 | 1008 | 75.4942 |
| CH2019_0031 | 2019-11-23 02:00:00 | blend | 0.761584 | 0.0220478 | 3700 | -0.00869139 | 13 | 1008 | 69.159 |
| CH2019_0031 | 2019-11-23 03:30:00 | initial | 0.716778 | 0.157398 | 4000 | -0.00759991 | 13 | 1008 | 75.4942 |
| CH2019_0007 | 2019-07-02 13:00:00 | calibrated | 3.2998 | 2.36415 | 42700 | -0.00572756 | 16 | 994.667 | 71.1185 |
| CH2019_0007 | 2019-07-02 13:00:00 | initial | 3.12883 | 3.80985 | 64700 | -0.0034512 | 16 | 994.667 | 71.1185 |

Likely causes include analog mismatch during rapid intensity change, coastal terrain discontinuity, compact convective cores that are hard to recover from analog means, and sparse heavy-rain truth coverage near the edge of the storm-relative tile.

## 9. Paper-Ready Conclusions
- The initial Top-K log1p analog field provides a no-target-rainfall baseline that recovers broad storm-relative rainfall placement.
- EOF/PCA blending acts as a large-scale structural regularizer and can be discussed through centroid, anisotropy, RMSE, and correlation changes.
- Extreme quantile calibration directly targets Top-K-derived P95/P99/Rmax and heavy-rain area constraints without reading validation-event rain metrics.
- P95/P99 and heavy-rain area are more stable validation targets than single-grid Rmax, which remains sensitive to small convective cores.
- Threshold metrics such as CSI10 and F1_10 quantify whether heavy-rain identification improves after calibration.
- The self-exclusion check confirms that validation-event samples do not enter Top-K templates.
- The pseudo-missing experiment shows whether the model can generate physically plausible typhoon rainfall structures when the target GPM field is unavailable.
- Generated figure count: 27
