# Problem 3 Step 1 Scenario Design Report

## 1. Inputs read
- `data/processed/problem2_env/problem2_target_halfhour_inputs_safe_env.csv`
- `data/processed/env_added/target_typhoon_inputs_2024_halfhour_leakage_safe_env.csv`
- `data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- `data/external/naturalearth/ne_50m_land/ne_50m_land.shp` for is_land and signed coast distance recalculation
- `data/external/etopo2022/ETOPO_2022_v1_30s_N90W180_bed.tif` plus `scripts/27_add_landfrac_terrain_features.py` for land fraction and terrain recalculation

## 2. Why KONG-REY is S0
KONG-REY is used as the main baseline because the current Problem-2 chain already has complete half-hour target inputs and final calibrated/geographic rainfall outputs for it. It also has a near-coast segment and a strong intensity evolution, so it is suitable as the main text case for virtual typhoon perturbation. MAN-YI can remain a later robustness comparison.

## 3. Scenario rules
- S0: unperturbed KONG-REY Problem-2 input.
- S1: WND is increased by up to 8.00 m/s and capped at historical WND P99=66.50; PRES is lowered synchronously using the historical WND-PRES fit.
- S2: the whole path is shifted 100 km westward; motion, coast distance, land/terrain variables are recomputed.
- S3: only the closest-coast segment is shifted smoothly northward with a Gaussian peak of 75 km and sigma 24 h; motion and environmental variables are recomputed.
- S4: the time axis is stretched with gamma=0.651; half-hour input times are regenerated and motion/intensity rates are recomputed.
- S5: uses half of the S1 intensity increment, half of the S2 westward shift, and a moderate slowdown gamma=0.825.

## 4. Directly perturbed vs recomputed variables
Direct perturbations are limited to WND/PRES for S1, lon for S2, lat for S3, time-axis/path sampling for S4, and moderate WND/PRES/lon/time-axis changes for S5. For path and time-axis scenarios, move_speed_kmh, move_dir_deg, wind_change_rate, pressure_change_rate, is_land, coast_dist_km, signed_coast_dist_km, landfrac_200km, landfrac_500km, terrain_mean_300km, terrain_std_300km, and terrain_max_300km are recomputed as applicable.

## 5. WND and PRES handling
The comparable historical WND-PRES linear fit uses 13330 rows. The fitted slope is -1.7482096844941974, so pressure decreases when WND increases. The script caps enhanced WND at historical P99 and floors PRES at historical P01 to avoid unbounded intensification.

## 6. OWD and leakage exclusions
OWD is absent from the required input set and is not used. The output scenario input table is checked before writing so that OWD, rain_*, centroid_*, anisotropy, asym_*, quad_*, r50/r80/r90, rainband, major/minor axis, orientation, rain_gini, and rain_entropy columns are not written.

## 7. Historical comparable sample
- Source: `data/processed/problem2_env/problem2_historical_halfhour_sample_library_env.csv`
- Valid historical rows before regional/seasonal filtering: 32733
- Strict sample rows: 13330
- Relaxed-month rows: 20349
- Broad WNP rows: 21248
- Final sample rule: strict: months 9-11, latitude within KONG-REY range +/-5 deg, longitude within KONG-REY range +/-15 deg
- Final sample rows: 13330

## 8. Scenario summary
| scenario_id | n_timesteps | start_time | end_time | max_WND | min_PRES | mean_move_speed_kmh | min_coast_dist_km |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 421 | 2024-10-24 18:00:00 | 2024-11-02 12:00:00 | 60.0000 | 920.0000 | 27.3877 | 5.6388 |
| S1 | 421 | 2024-10-24 18:00:00 | 2024-11-02 12:00:00 | 66.5000 | 908.6366 | 27.3877 | 5.6388 |
| S2 | 421 | 2024-10-24 18:00:00 | 2024-11-02 12:00:00 | 60.0000 | 920.0000 | 27.3794 | 0.0384 |
| S3 | 421 | 2024-10-24 18:00:00 | 2024-11-02 12:00:00 | 60.0000 | 920.0000 | 27.4177 | 0.9754 |
| S4 | 647 | 2024-10-24 18:00:00 | 2024-11-07 05:00:00 | 60.0000 | 920.0000 | 17.7986 | 1.6552 |
| S5 | 510 | 2024-10-24 18:00:00 | 2024-11-04 08:30:00 | 64.0000 | 913.0072 | 22.5847 | 0.2199 |

## 9. Historical support audit
| scenario_id | out_of_support_count | out_of_support_ratio | out_of_support_variables | validity_level |
| --- | --- | --- | --- | --- |
| S0 | 12 | 0.0285 | move_speed_kmh=12 | limited_speed_tail_support |
| S1 | 12 | 0.0285 | move_speed_kmh=12 | limited_speed_tail_support |
| S2 | 13 | 0.0309 | move_speed_kmh=12; coast_dist_km=2 | limited_speed_or_coast_tail_support |
| S3 | 12 | 0.0285 | move_speed_kmh=12 | limited_speed_tail_support |
| S4 | 0 | 0.0000 | none | within_historical_min_max |
| S5 | 4 | 0.0078 | coast_dist_km=4 | limited_speed_or_coast_tail_support |
- Total out-of-support audited time steps: 53

## 10. Next step
Problem 3 Step 2 should feed S0-S5 rows from `data/processed/problem3_scenario_inputs.csv` into the established Problem-2 path-intensity-environment rainfall generation chain, then generate scenario rainfall fields and compute extreme precipitation indicators such as cumulative rainfall, P95/P99, heavy-rain area, and duration.
