# Problem 3 Step 3 Event Metrics Report

## 1. Inputs read
- Calibrated scenario rainfields: `data/processed/problem3/problem3_scenario_rainfields_calibrated.npz`
- Timeslice metrics: `data/processed/problem3/problem3_scenario_timeslice_metrics.csv`
- Step-2 generation QC: `data/processed/problem3/problem3_scenario_generation_qc.csv`
- Top-K analog table: `data/processed/problem3/problem3_scenario_topk_analogs.csv`
- Scenario input table: `data/processed/problem3_scenario_inputs.csv`
- Scenario summary table: `data/processed/problem3_scenario_summary.csv`
- Problem-2 geographic logic reused from: `scripts/24_geographic_backprojection_results.py`
- Storm-relative x/y grid read from: `data/processed/problem2_env/problem2_eof_pca_model_env.npz`

## 2. Geographic backprojection
- Geographic backprojection was performed because the Step-2 fields are storm-relative 201 x 201 grids centered on the typhoon.
- The reverse-sampling formula and `RegularGridInterpolator` logic follow the Problem-2 geographic backprojection script.
- Common geographic grid: lon 110.0 to 155.0, lat 4.0 to 43.0.
- Resolution: 0.1 deg x 0.1 deg; grid shape: (391, 451).
- Grid-cell area uses latitude correction: 111.32^2 x 0.1 x 0.1 x cos(lat), in km2.
- Cartopy/Natural Earth overlay used: `True`. If false, metrics and maps still use the fixed lon-lat grid.

## 3. Unit conversion
- The rainfields are rainfall intensity fields in mm/hr.
- Half-hour rainfall is H_t(i,j) = 0.5 x R_t(i,j).
- Event accumulation is C(i,j) = 0.5 x sum_t R_t(i,j).
- Fixed-grid duration is D10(i,j) = 0.5 x sum_t 1{R_t(i,j) >= 10 mm/hr}; D20 is defined analogously.
- This script never directly sums mm/hr as mm.

## 4. Scenario event metrics
| scenario_id | n_timesteps | max_accum_rain_mm | max_area10_km2 | geo_D10_max_h | area_accum_ge_100mm_km2 | total_event_rain_volume_index | qc_level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 421 | 723.479 | 74508.148 | 24.500 | 1213807.241 | 498451759.629 | WARNING |
| S1 | 421 | 746.935 | 79501.254 | 26.000 | 1262705.347 | 515906270.507 | WARNING |
| S2 | 421 | 798.343 | 71587.795 | 26.500 | 1203443.114 | 509748095.915 | WARNING |
| S3 | 421 | 705.525 | 74772.724 | 23.500 | 1223517.922 | 497134915.301 | WARNING |
| S4 | 647 | 1208.094 | 75003.966 | 39.000 | 1672827.851 | 746000833.285 | PASS |
| S5 | 510 | 947.659 | 82748.914 | 31.500 | 1452824.889 | 600093791.904 | WARNING |

## 5. Relative changes vs S0
| scenario_id | delta_max_accum_rain_pct | delta_max_area10_pct | delta_geo_D10_max_pct | delta_total_event_rain_volume_pct | main_sensitive_output |
| --- | --- | --- | --- | --- | --- |
| S0 | 0.000 | 0.000 | 0.000 | 0.000 | baseline |
| S1 | 3.242 | 6.701 | 6.122 | 3.502 | rain-rate tail and heavy-rain area |
| S2 | 10.348 | -3.920 | 8.163 | 2.266 | geographic rain footprint |
| S3 | -2.482 | 0.355 | -4.082 | -0.264 | coastal impact location |
| S4 | 66.984 | 0.665 | 59.184 | 49.664 | accumulation and duration |
| S5 | 30.986 | 11.060 | 28.571 | 20.392 | compound risk |

## 6. S0 baseline consistency check
- Status: `WARNING-but-usable`
- Note: S0 differs from the Problem-2 baseline but remains usable; Step-2 already recorded non-identical S0 fields.
- Problem-2 KONG-REY max accumulation: 722.047 mm.
- Problem-3 S0 max accumulation: 723.479 mm.
- Max-accumulation location shift: 369.396 km.
- Problem-2 / Problem-3 S0 geo D10 max: 27.000 / 24.500 h.
- Problem-2 / Problem-3 S0 max area10: 88526.323 / 74508.148 km2.
- If differences are visible, scenario interpretation should use the internally consistent Problem-3 S0 control rather than claiming exact reproduction.

## 7. Physical interpretation
| scenario_id | control_factor | most_sensitive_metric | observed_change | physical_explanation | paper_sentence |
| --- | --- | --- | --- | --- | --- |
| S1 | Intensity enhancement | area10 | 6.701 | Higher WND with lower PRES changes the intensity-sensitive rain-rate tail and heavy-rain footprint. | S1 mainly modifies rainfall intensity; P99 changes by 4.0% relative to S0. |
| S2 | Near-coast/westward path shift | centroid_shift_km | 104.759 | Track displacement changes where the storm-relative rain shield is projected on the fixed map. | S2 changes the geographic footprint; the accum>=100 mm IoU versus S0 is 0.67. |
| S3 | Nearshore segment/landfall shift | 1-IoU_D10_3h | 22.268 | A localized path perturbation transfers the coastal impact belt while retaining the broader storm evolution. | S3 relocates nearshore exposure; the rainfall centroid shifts 22.0 km. |
| S4 | Slower translation | max_accum | 66.984 | Reduced translation speed extends exposure time at fixed locations, amplifying accumulation and D10 duration. | S4 is most interpretable as a residence-time scenario; geo D10 changes by 59.2%. |
| S5 | Compound moderate high-risk perturbation | max_accum | 30.986 | Moderate intensity, path, and speed perturbations combine rate, footprint, and residence-time effects. | S5 combines multiple controls; total rainfall volume index changes by 20.4%. |

## 8. Sensitivity summary
- S1 is read through intensity-sensitive metrics: max rain rate, P99, area10, and area20.
- S2 is read through geographic footprint metrics: rainfall centroid shift, IoU, and area with accumulation >=100 mm.
- S3 is read through relocation of the nearshore impact belt and centroid/IoU changes.
- S4 is read through residence-time metrics: max accumulation, geo_D10_max, and area_D10_ge_3h.
- S5 is read as the compound case; it should be described by which of rate, footprint, duration, and volume jointly increase.

## 9. QC status
| scenario_id | n_timesteps | max_accum_rain_mm | max_duration10_h | baseline_consistency_status | qc_level | qc_note |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 421 | 723.479 | 24.500 | WARNING-but-usable | WARNING | step2_qc_warning:step1_out_of_support_ratio=0.0285; S0_baseline_difference; baseline_consistency:WARNING-but-usable |
| S1 | 421 | 746.935 | 26.000 | not_applicable | WARNING | step2_qc_warning:step1_out_of_support_ratio=0.0285 |
| S2 | 421 | 798.343 | 26.500 | not_applicable | WARNING | step2_qc_warning:step1_out_of_support_ratio=0.0309 |
| S3 | 421 | 705.525 | 23.500 | not_applicable | WARNING | step2_qc_warning:step1_out_of_support_ratio=0.0285 |
| S4 | 647 | 1208.094 | 39.000 | not_applicable | PASS | Step-3 metrics quality checks passed. |
| S5 | 510 | 947.659 | 31.500 | not_applicable | WARNING | step2_qc_warning:step1_out_of_support_ratio=0.0078 |
- Calibrated rainfield shape read: `(2841, 201, 201)`.
- Figures generated: 15.
- Final run status: **WARNING**.

## 10. Outputs
- Event metrics: `data/processed/problem3/problem3_scenario_event_metrics.csv`
- Relative changes: `data/processed/problem3/problem3_scenario_relative_changes.csv`
- Geographic accumulation/duration NPZ: `data/processed/problem3/problem3_scenario_geographic_accum_duration.npz`
- Final comparison table: `outputs/tables/problem3/problem3_final_scenario_comparison_table.csv`
- Sensitivity interpretation table: `outputs/tables/problem3/problem3_sensitivity_interpretation_table.csv`
- Step-3 QC: `data/processed/problem3/problem3_step3_metrics_qc.csv`
- Figures directory: `outputs/figures/problem3`

## 11. Next step
- Step 4 should select the formal tables and maps, then write the Problem-3 paper text around the S0-controlled scenario comparison.
- The report text should explicitly distinguish fixed-geographic accumulation/duration maps from storm-relative structure maps.
