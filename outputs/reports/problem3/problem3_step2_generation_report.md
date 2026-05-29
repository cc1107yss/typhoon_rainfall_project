# Problem 3 Step 2 Generation Report

## 1. Inputs read
- Scenario input main table: `data/processed/problem3_scenario_inputs.csv`
- Problem-3 Step-1 scenario summary: `data/processed/problem3_scenario_summary.csv`
- Problem-3 Step-1 validity audit: `outputs/tables/problem3/problem3_scenario_validity_audit.csv`
- Problem-2 env historical library: `data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- Problem-2 env EOF/PCA model artifact: `data/processed/problem2_env/problem2_eof_pca_model_env.npz`
- Problem-2 env calibrated baseline for S0 comparison: `data/processed/problem2_env/problem2_generated_calibrated_fields_env.npz` and `data/processed/problem2_env/problem2_generated_calibrated_fields_index_env.csv`

## 2. Reused Problem-2 logic
- Top-K retrieval: functions from `scripts/18_build_target_inputs_and_topk_retrieval.py`
- Storm-relative initial field generation: functions from `scripts/19_generate_initial_rainfall_fields_from_topk.py`
- EOF/PCA projection and beta blending: functions from `scripts/20_eof_pca_structure_correction.py`, using existing model artifact rather than refitting PCA
- Extreme quantile/tail calibration and physical caps: functions from `scripts/21_extreme_quantile_calibration.py`
- Problem-2 env feature setting reused for retrieval: `base-old`
- Selected safe feature count: 16
- Selected safe features: lat, lon_180, WND, PRES, intensity, move_speed_kmh, move_dir_sin, move_dir_cos, wind_change_rate, pressure_change_rate, is_land, signed_coast_dist_km, coast_dist_km, month_sin, month_cos, life_progress

## 3. Scenario counts
| scenario_id | n_timesteps |
| --- | --- |
| S0 | 421 |
| S1 | 421 |
| S2 | 421 |
| S3 | 421 |
| S4 | 647 |
| S5 | 510 |
- Total scenario input rows: 2841
- Historical library rows used for retrieval after Problem-2 filtering/imputation: 32842

## 4. Leakage guard
- Dropped forbidden scenario columns: []
- The retrieval feature list was checked against OWD, rain_*, centroid_*, anisotropy, asym_*, quad_*, r50/r80/r90, rainband, major/minor axis, orientation, rain_gini, and rain_entropy patterns.

## 5. Top-K distance statistics
| scenario_id | mean_topk_distance | p95_topk_distance | min_topk_count | qc_level |
| --- | --- | --- | --- | --- |
| S0 | 1.3373 | 3.8981 | 20 | WARNING |
| S1 | 1.5517 | 3.9953 | 20 | WARNING |
| S2 | 1.3427 | 4.0106 | 20 | WARNING |
| S3 | 1.3396 | 4.0331 | 20 | WARNING |
| S4 | 1.2871 | 3.1801 | 20 | PASS |
| S5 | 1.3332 | 3.6414 | 20 | WARNING |

## 6. Rainfield quality checks
| scenario_id | nan_count | inf_count | negative_count | all_zero_count | max_value_mmhr | p99_value_mmhr | qc_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0 | 0 | 0 | 0 | 53.7328 | 9.7476 | step1_out_of_support_ratio=0.0285 |
| S1 | 0 | 0 | 0 | 0 | 53.0507 | 10.0373 | step1_out_of_support_ratio=0.0285 |
| S2 | 0 | 0 | 0 | 0 | 52.4392 | 9.9081 | step1_out_of_support_ratio=0.0309 |
| S3 | 0 | 0 | 0 | 0 | 52.3672 | 9.6421 | step1_out_of_support_ratio=0.0285 |
| S4 | 0 | 0 | 0 | 0 | 54.2785 | 9.6162 | Generation quality checks passed. |
| S5 | 0 | 0 | 0 | 0 | 57.8708 | 10.0220 | step1_out_of_support_ratio=0.0078 |
- Nonnegative constraint was applied after calibration.
- Physical upper cap was enforced at 120.0 mm/hr, consistent with the Problem-2 calibration cap.

## 7. S0 baseline comparison
- Status: identical
- Note: S0 reproduces problem2 KONG-REY baseline.
- Shape match: True
- Time match: True
- Max absolute difference: 0.0
- Mean absolute difference: 0.0
- P95 max absolute difference: 0.0
- Area10 max absolute difference: 0.0

## 8. S4 slowdown checks
- S4 time strictly increasing: True
- S4 adjacent timestep is 30 min: True
- S0 mean move speed: 27.387707 km/h
- S4 mean move speed: 17.798633 km/h
- S4 slower than S0: True
- Historical speed P05/P95: 5.559746 / 39.127383 km/h
- S4 speed within historical min-max: True
- S4 P95 Top-K distance: 3.180067
- Other-scenario median P95 Top-K distance: 3.995281
- S4 Top-K distance warning: False

## 9. Units and scope
- The generated rainfields are half-hourly rainfall intensity fields in mm/hr.
- This step does not convert mm/hr to accumulated rainfall. The third step should use 0.5 x R_t for half-hour accumulation.
- `rain_sum_mmhr_grid` in the timeslice metrics is a quality-control grid sum of intensity values, not accumulated precipitation.
- This step did not compute final scenario conclusions, cumulative rainfall maps, duration10 geographic maps, or final scenario-comparison tables.

## 10. Outputs
- Top-K analogs: `data/processed/problem3/problem3_scenario_topk_analogs.csv`
- Initial rainfields: `data/processed/problem3/problem3_scenario_rainfields_initial.npz`
- Calibrated rainfields: `data/processed/problem3/problem3_scenario_rainfields_calibrated.npz`
- Timeslice metrics: `data/processed/problem3/problem3_scenario_timeslice_metrics.csv`
- Generation QC: `data/processed/problem3/problem3_scenario_generation_qc.csv`
- Paper QC summary: `outputs/tables/problem3/problem3_step2_generation_qc_summary.csv`

Final run status: **WARNING**
