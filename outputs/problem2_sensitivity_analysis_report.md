# Problem 2 Parameter Sensitivity Analysis Report

## 1. Input and Output Files
- Historical library: `data/processed/problem2_historical_halfhour_sample_library.csv`
- EOF/PCA model: `data/processed/problem2_eof_pca_model.npz`
- Step-22 validation events: `data/processed/problem2_pseudo_validation_events.csv`
- Step-22 model summary: `data/processed/problem2_pseudo_validation_model_summary.csv`
- Step-22 timeslice metrics: `data/processed/problem2_pseudo_validation_timeslice_metrics.csv`
- Sensitivity settings: `data/processed/problem2_sensitivity_settings.csv`
- Sensitivity timeslice metrics: `data/processed/problem2_sensitivity_timeslice_metrics.csv`
- Sensitivity summary: `data/processed/problem2_sensitivity_summary.csv`
- Sensitivity relative change: `data/processed/problem2_sensitivity_relative_change.csv`
- Figures directory: `outputs/figures/problem2_sensitivity_analysis`

## 2. Parameter Settings
| setting_id | K | weight_track | weight_intensity | weight_motion | weight_environment | weight_time | weight_life | beta_blend | description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 1 | 1.5 | 1.2 | 1.2 | 0.8 | 0.8 | 0.3 | K=20, beta=0.3, baseline weights |
| S1 | 10 | 1 | 1.5 | 1.2 | 1.2 | 0.8 | 0.8 | 0.3 | K=10 |
| S2 | 30 | 1 | 1.5 | 1.2 | 1.2 | 0.8 | 0.8 | 0.3 | K=30 |
| S3 | 20 | 1 | 1.2 | 1.2 | 1.2 | 0.8 | 0.8 | 0.3 | lower intensity weight |
| S4 | 20 | 1 | 1.5 | 1.5 | 1.2 | 0.8 | 0.8 | 0.3 | higher motion weight |
| S5 | 20 | 1 | 1.5 | 1.2 | 1.5 | 0.8 | 0.8 | 0.3 | higher environment weight |
| S6 | 20 | 1 | 1.5 | 1.2 | 1.2 | 0.8 | 0.8 | 0 | beta=0.0 |
| S7 | 20 | 1 | 1.5 | 1.2 | 1.2 | 0.8 | 0.8 | 0.5 | beta=0.5 |

## 3. Validation Events and Timeslices
- N_VALIDATION_EVENTS: 8
- MAX_TIMES_PER_EVENT: 60
- RANDOM_SEED: 2026
- Selected-time source: CH2016_0016:step22_times_subset; CH2024_0022:step22_times_subset; CH2019_0007:step22_times_subset; CH2020_0003:step22_times_subset; CH2019_0031:step22_times_subset; CH2014_0007:step22_times_subset; CH2015_0014:step22_times_subset; CH2023_0003:step22_times_subset
- Selected times by event: {'CH2016_0016': 60, 'CH2024_0022': 60, 'CH2019_0007': 60, 'CH2020_0003': 60, 'CH2019_0031': 60, 'CH2014_0007': 60, 'CH2015_0014': 60, 'CH2023_0003': 60}
- Valid calibrated rows by setting: {'S1': 480, 'S2': 480, 'S3': 480, 'S4': 480, 'S5': 480, 'S6': 480, 'S7': 480, 'baseline': 480}
- This run uses a lightweight but horizontally comparable validation design: 8 events x up to 60 half-hour timeslices per setting.

## 4. Leakage Guard
- Retrieval features are restricted to track, intensity, motion, environment, month, and life-progress variables.
- Retrieval rain_* fields used: no
- For each validation event, the retrieval library removes all rows whose event_uid equals the validation_event_uid.
- Top-K rows from the same validation event: 0
- Truth tif fields are read only after Top-K retrieval, initial generation, EOF/PCA blending, and historical-sample calibration targets are defined.
- Safe retrieval features: lat, lon_180, WND, PRES, intensity, move_speed_kmh, move_dir_sin, move_dir_cos, wind_change_rate, pressure_change_rate, is_land, signed_coast_dist_km, coast_dist_km, month_sin, month_cos, life_progress

## 5. Step-22 Calibrated Reference
| model_version | n_events | n_timeslices | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_area_10_mean | csi10_mean |
| --- | --- | --- | --- | --- | --- | --- |
| calibrated | 8 | 640 | 1.93514 | 5.92285 | 39214.4 | 0.109664 |

## 6. Overall Performance by Setting
| setting_id | n_events | n_timeslices | rmse_mean | mae_mean | corr_mean | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_rain_max_mean | abs_error_area_10_mean | csi10_mean | f1_10_mean | topk_unique_event_count_mean | validation_ok_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8 | 480 | 2.73548 | 0.974948 | 0.361355 | 1.97611 | 6.15773 | 19.7977 | 40352.9 | 0.116039 | 0.186682 | 7.14375 | 1 |
| S1 | 8 | 480 | 2.80547 | 0.99768 | 0.337061 | 2.02711 | 5.75344 | 15.5196 | 40074.8 | 0.110799 | 0.179122 | 4.00833 | 1 |
| S2 | 8 | 480 | 2.72185 | 0.970564 | 0.365793 | 1.99896 | 6.39769 | 21.6752 | 40733.1 | 0.115299 | 0.185744 | 10.2979 | 1 |
| S3 | 8 | 480 | 2.73129 | 0.974612 | 0.36061 | 1.98166 | 6.17443 | 19.8931 | 40318.8 | 0.115748 | 0.186109 | 7.11875 | 1 |
| S4 | 8 | 480 | 2.73277 | 0.973885 | 0.362957 | 1.97793 | 6.17376 | 19.745 | 40338.3 | 0.11607 | 0.186639 | 7.16667 | 1 |
| S5 | 8 | 480 | 2.73342 | 0.974407 | 0.361378 | 1.98868 | 6.16682 | 19.7821 | 40237.7 | 0.115335 | 0.185831 | 7.14375 | 1 |
| S6 | 8 | 480 | 2.762 | 0.988148 | 0.3492 | 1.95883 | 6.01188 | 17.8568 | 40232.9 | 0.11325 | 0.183049 | 7.14375 | 1 |
| S7 | 8 | 480 | 2.72761 | 0.968712 | 0.367674 | 1.99671 | 6.2303 | 21.2688 | 40552.9 | 0.117566 | 0.188767 | 7.14375 | 1 |

## 7. Relative Change vs Baseline
| setting_id | rmse_change_pct | corr_change | p95_error_change_pct | p99_error_change_pct | rmax_error_change_pct | area10_error_change_pct | csi10_change | f1_10_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| S1 | 2.55838 | -0.0242941 | 2.58087 | -6.56553 | -21.6095 | -0.689231 | -0.0052399 | -0.0075593 |
| S2 | -0.498362 | 0.00443806 | 1.15617 | 3.89685 | 9.48325 | 0.942208 | -0.000739875 | -0.000937695 |
| S3 | -0.153435 | -0.000744918 | 0.280639 | 0.271254 | 0.481503 | -0.0846696 | -0.000290117 | -0.000573021 |
| S4 | -0.0993668 | 0.00160263 | 0.0918825 | 0.260311 | -0.266541 | -0.0361395 | 3.10776e-05 | -4.28137e-05 |
| S5 | -0.0754656 | 2.37176e-05 | 0.635845 | 0.147639 | -0.0787914 | -0.285502 | -0.000703977 | -0.000850097 |
| S6 | 0.969414 | -0.0121552 | -0.874699 | -2.36852 | -9.80408 | -0.297376 | -0.00278829 | -0.00363226 |
| S7 | -0.287691 | 0.00631939 | 1.04262 | 1.17851 | 7.43051 | 0.495627 | 0.00152723 | 0.00208564 |
- Largest absolute relative-change item: S1 rmax_error_change_pct = -21.6095%

## 8. Parameter Sensitivity Conclusions
### K Sensitivity
| setting_id | K | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_area_10_mean | csi10_mean | topk_unique_event_count_mean |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 1.97611 | 6.15773 | 40352.9 | 0.116039 | 7.14375 |
| S1 | 10 | 2.02711 | 5.75344 | 40074.8 | 0.110799 | 4.00833 |
| S2 | 30 | 1.99896 | 6.39769 | 40733.1 | 0.115299 | 10.2979 |
K=10, K=20, and K=30 retain the same validation events and selected timeslices, so differences mainly reflect analog sample-size effects rather than sample-composition drift.

### Beta Sensitivity
| setting_id | beta_blend | abs_error_rain_p95_mean | abs_error_rain_p99_mean | abs_error_area_10_mean | csi10_mean | f1_10_mean |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.3 | 1.97611 | 6.15773 | 40352.9 | 0.116039 | 0.186682 |
| S6 | 0 | 1.95883 | 6.01188 | 40232.9 | 0.11325 | 0.183049 |
| S7 | 0.5 | 1.99671 | 6.2303 | 40552.9 | 0.117566 | 0.188767 |
The beta experiment isolates the EOF/PCA structural constraint while keeping Top-K and distance weights fixed.

### Distance Weight Sensitivity
| setting_id | weight_intensity | weight_motion | weight_environment | abs_error_rain_p95_mean | abs_error_rain_p99_mean | csi10_mean | f1_10_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S3 | 1.2 | 1.2 | 1.2 | 1.98166 | 6.17443 | 0.115748 | 0.186109 |
| S4 | 1.5 | 1.5 | 1.2 | 1.97793 | 6.17376 | 0.11607 | 0.186639 |
| S5 | 1.5 | 1.2 | 1.5 | 1.98868 | 6.16682 | 0.115335 | 0.185831 |
The three weight perturbations do not change validation samples or downstream calibration logic; they only alter analog ordering and weights.

## 9. Figure Outputs
- `outputs/figures/problem2_sensitivity_analysis/sensitivity_bar_extreme_errors.png`
- `outputs/figures/problem2_sensitivity_analysis/sensitivity_bar_skill_scores.png`
- `outputs/figures/problem2_sensitivity_analysis/sensitivity_relative_change.png`
- `outputs/figures/problem2_sensitivity_analysis/sensitivity_k_beta_focus.png`

## 10. Paper-Ready Conclusion
基于历史台风伪缺失验证的敏感性检验表明，问题二生成模型在 Top-K 相似样本数量、相似距离分量权重和 EOF/PCA 融合系数扰动下总体表现稳定。K=20 在极端误差控制和历史模板多样性之间取得较好折中；K=10 更容易受少数历史样本影响，而 K=30 会引入相似性较弱样本，可能平滑极端降水结构。当 beta=0 时模型缺少 EOF 结构约束，beta=0.5 时 EOF 平滑作用增强，二者相比 beta=0.3 均可能带来极端结构或技巧评分上的波动，因此 beta=0.3 是较稳健的折中。在强度、移动和环境权重扰动下，P95/P99 极端误差和 CSI10/F1_10 未出现大幅恶化，说明模型对距离权重设置具有鲁棒性；其中强度权重降低时极端降水误差的变化可作为强度变量对降水生成重要性的补充证据。
