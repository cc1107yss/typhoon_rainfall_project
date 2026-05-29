# Rainband Width Integration Notes

## Modified Scripts
- `scripts/rainband_width_utils.py`: added shared covariance-based rainband width computation for main (`>=1 mm/hr`) and heavy (`>=10 mm/hr`) rainbands.
- `scripts/17_add_motion_relative_rain_features.py`: added `rainband_width_km`, `rainband_length_km`, `rainband_aspect_ratio`, `rainband_width10_km`, `rainband_length10_km`, and `rainband_aspect_ratio10` to the existing motion-relative rainfall feature extractor.
- `scripts/10_problem1_analysis_enhanced.py`: added `rainband_width_km` as a Problem-1 target and added dedicated width summary tables and figures.
- `scripts/12_group_validation_problem1.py`: added `rainband_width_km` to random-row vs event-group validation targets.
- `scripts/13_repeated_group_validation.py`: aligned repeated group validation with the final `problem1_env` table and added `rainband_width_km`.
- `scripts/16_field_audit_and_leakage_guard.py`: added rainband-width fields to leakage guards and audited the final env-added historical table.
- `scripts/18_build_target_inputs_and_topk_retrieval.py`: added rainband-width fields to defensive leakage patterns.
- `scripts/22_pseudo_missing_validation.py`: added rainband-width fields to defensive leakage patterns.

## New Scripts
- `scripts/32_add_rainband_width_features.py`: computes rainband-width features for the final historical `problem1_env` table.
- `scripts/33_problem2_rainband_width_diagnostics.py`: computes Problem-2 generated-field rainband-width diagnostics.
- `scripts/34_problem3_rainband_width_diagnostics.py`: computes Problem-3 S0--S5 generated-field rainband-width diagnostics.

## Added Metrics
- `rainband_width_km`: main rainband equivalent transverse width `B`, from the minor eigenvalue of the rainfall-weighted covariance matrix over `R >= 1 mm/hr`.
- `rainband_length_km`: main rainband equivalent major-axis length `L`, from the major eigenvalue over `R >= 1 mm/hr`.
- `rainband_aspect_ratio`: `L / (B + eps)`.
- `rainband_width10_km`, `rainband_length10_km`, `rainband_aspect_ratio10`: same diagnostics over `R >= 10 mm/hr`.

## Commands Run
- `cp 1.0.tex 1.0_before_rainband_width_update.tex`

## New Result Files
- Pending execution.

## Warnings / QC
- Pending execution.

## Leakage Check
- Pending execution.
