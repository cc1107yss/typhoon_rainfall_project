#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Problem-2 environment-feature ablations and final KONG-REY / MAN-YI generation.

The workflow keeps the original rainfall/template fields, merges the new
environment variables from data/processed/env_added, evaluates three D_env
feature sets with pseudo-missing validation, selects the most balanced setting,
and reruns the Problem-2 target generation pipeline with that setting.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

DATA_ENV_DIR = PROJECT_ROOT / "data/processed/problem2_env"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/problem2_env"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_env"

HIST_ENV_SOURCE = PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env.csv"
TARGET_ENV_SOURCE = PROJECT_ROOT / "data/processed/env_added/target_typhoon_inputs_2024_halfhour_leakage_safe_env.csv"
OLD_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"
OLD_VALIDATION_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_model_summary.csv"

ENV_LIBRARY_PATH = DATA_ENV_DIR / "problem2_historical_halfhour_sample_library_env.csv"
ENV_LIBRARY_REPORT_PATH = TABLE_DIR / "env_historical_library_merge_report.csv"
ABLATION_CSV_PATH = TABLE_DIR / "env_ablation_validation.csv"
ABLATION_COMPARE_CSV_PATH = TABLE_DIR / "env_ablation_vs_old_problem2.csv"
RUN_SUMMARY_JSON_PATH = TABLE_DIR / "problem2_env_run_summary.json"

ENV_MODES = ["base-old", "env-full", "env-key"]
BASE_ENV_FEATURES = ["is_land", "signed_coast_dist_km", "coast_dist_km"]
NEW_ENV_FEATURES = [
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",
]
ENV_FEATURE_SETS = {
    "base-old": BASE_ENV_FEATURES,
    "env-full": BASE_ENV_FEATURES + NEW_ENV_FEATURES,
    "env-key": BASE_ENV_FEATURES + ["landfrac_500km", "terrain_std_300km"],
}

REQUESTED_METRICS = {
    "RMSE": "rmse_mean",
    "MAE": "mae_mean",
    "Corr": "corr_mean",
    "P95_error": "abs_error_rain_p95_mean",
    "P99_error": "abs_error_rain_p99_mean",
    "Rmax_error": "abs_error_rain_max_mean",
    "A10_error": "abs_error_area_10_mean",
    "A20_error": "abs_error_area_20_mean",
    "CSI10": "csi10_mean",
    "F1_10": "f1_10_mean",
}


def load_module(script: str, tag: str):
    path = PROJECT_ROOT / "scripts" / script
    name = f"problem2_env_{Path(script).stem}_{tag}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for path in [DATA_ENV_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def canonical_time(df: pd.DataFrame, col: str = "time") -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.floor("s")


def build_env_historical_library() -> pd.DataFrame:
    if not OLD_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Missing old Problem-2 library: {OLD_LIBRARY_PATH}")
    if not HIST_ENV_SOURCE.exists():
        raise FileNotFoundError(f"Missing env-added historical table: {HIST_ENV_SOURCE}")

    library = pd.read_csv(OLD_LIBRARY_PATH, encoding="utf-8-sig", low_memory=False)
    env_header = pd.read_csv(HIST_ENV_SOURCE, nrows=0, encoding="utf-8-sig").columns.tolist()
    requested = [
        "track_event_uid",
        "time",
        "track_is_land",
        "track_signed_coast_dist_km",
        "track_coast_dist_km",
        *NEW_ENV_FEATURES,
    ]
    usecols = [c for c in requested if c in env_header]
    env = pd.read_csv(HIST_ENV_SOURCE, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    env = env.rename(
        columns={
            "track_event_uid": "event_uid",
            "track_is_land": "is_land",
            "track_signed_coast_dist_km": "signed_coast_dist_km",
            "track_coast_dist_km": "coast_dist_km",
        }
    )
    library["time"] = canonical_time(library)
    env["time"] = canonical_time(env)
    env = env.dropna(subset=["event_uid", "time"]).drop_duplicates(["event_uid", "time"], keep="last")

    merged = library.merge(env, on=["event_uid", "time"], how="left", suffixes=("", "_envsrc"))
    report_rows: List[Dict[str, object]] = []
    for feature in BASE_ENV_FEATURES + NEW_ENV_FEATURES:
        src = f"{feature}_envsrc" if f"{feature}_envsrc" in merged.columns else feature
        if src in merged.columns:
            before_missing = float(pd.to_numeric(merged.get(feature), errors="coerce").isna().mean()) if feature in merged.columns else 1.0
            if feature in merged.columns and src != feature:
                merged[feature] = pd.to_numeric(merged[src], errors="coerce").combine_first(
                    pd.to_numeric(merged[feature], errors="coerce")
                )
            else:
                merged[feature] = pd.to_numeric(merged[src], errors="coerce")
            after_missing = float(pd.to_numeric(merged[feature], errors="coerce").isna().mean())
            report_rows.append(
                {
                    "feature": feature,
                    "source_column": src,
                    "missing_rate_before": before_missing,
                    "missing_rate_after": after_missing,
                    "non_missing_count": int(pd.to_numeric(merged[feature], errors="coerce").notna().sum()),
                }
            )
        else:
            report_rows.append(
                {
                    "feature": feature,
                    "source_column": "",
                    "missing_rate_before": np.nan,
                    "missing_rate_after": 1.0,
                    "non_missing_count": 0,
                }
            )

    drop_cols = [c for c in merged.columns if c.endswith("_envsrc")]
    merged = merged.drop(columns=drop_cols)
    merged["time"] = pd.to_datetime(merged["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    merged.to_csv(ENV_LIBRARY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_csv(ENV_LIBRARY_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"[28] Env historical library: {ENV_LIBRARY_PATH} {merged.shape}")
    return merged


def configure_env_feature_mode(module, mode: str) -> List[str]:
    if hasattr(module, "configure_environment_features"):
        return list(module.configure_environment_features(mode))
    features = list(ENV_FEATURE_SETS[mode])
    module.SAFE_FEATURE_COMPONENTS["environment"] = features
    if hasattr(module, "ENVIRONMENT_FIELDS"):
        module.ENVIRONMENT_FIELDS = features
    if hasattr(module, "ENV_FEATURE_SET"):
        module.ENV_FEATURE_SET = mode
    return features


def metric_row_from_summary(summary_path: Path, mode: str, features: Sequence[str]) -> Dict[str, object]:
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    row = summary.loc[summary["model_version"].astype(str).eq("calibrated")].iloc[0]
    out: Dict[str, object] = {
        "env_setting": mode,
        "env_features": ",".join(features),
        "env_features_n": int(len(features)),
        "model_version": "calibrated",
        "model_summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
    }
    for out_col, source_col in REQUESTED_METRICS.items():
        out[out_col] = float(pd.to_numeric(pd.Series([row.get(source_col)]), errors="coerce").iloc[0])
    return out


def run_pseudo_validation(mode: str, eof_model_path: Path, tag: str, make_figures: bool = False) -> Tuple[Dict[str, object], Path]:
    print(f"[28] Running pseudo-missing validation: {mode}")
    p22 = load_module("22_pseudo_missing_validation.py", f"p22_{tag}_{mode.replace('-', '_')}")
    features = configure_env_feature_mode(p22, mode)
    safe = mode.replace("-", "_")
    p22.HISTORICAL_LIBRARY_PATH = ENV_LIBRARY_PATH
    p22.EOF_MODEL_PATH = eof_model_path
    p22.VALIDATION_EVENTS_OUTPUT_PATH = DATA_ENV_DIR / f"pseudo_validation_events_{safe}.csv"
    p22.TIMESLICE_METRICS_OUTPUT_PATH = DATA_ENV_DIR / f"pseudo_validation_timeslice_metrics_{safe}.csv"
    p22.EVENT_SUMMARY_OUTPUT_PATH = DATA_ENV_DIR / f"pseudo_validation_event_summary_{safe}.csv"
    p22.MODEL_SUMMARY_OUTPUT_PATH = DATA_ENV_DIR / f"pseudo_validation_model_summary_{safe}.csv"
    p22.GENERATED_FIELDS_OUTPUT_PATH = DATA_ENV_DIR / f"pseudo_validation_generated_fields_{safe}.npz"
    p22.QC_REPORT_PATH = TABLE_DIR / f"pseudo_validation_qc_report_{safe}.md"
    p22.FIGURE_DIR = FIGURE_DIR / f"pseudo_validation_{safe}"
    p22.MAKE_FIGURES = bool(make_figures)
    p22.main()
    return metric_row_from_summary(p22.MODEL_SUMMARY_OUTPUT_PATH, mode, features), p22.MODEL_SUMMARY_OUTPUT_PATH


def score_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    scored = ablation.copy()
    low_cols = ["RMSE", "MAE", "P95_error", "P99_error", "Rmax_error", "A10_error", "A20_error"]
    high_cols = ["Corr", "CSI10", "F1_10"]
    score = np.zeros(len(scored), dtype=float)
    for col in low_cols:
        vals = pd.to_numeric(scored[col], errors="coerce").to_numpy(dtype=float)
        span = np.nanmax(vals) - np.nanmin(vals)
        score += 0.0 if not np.isfinite(span) or span <= 1e-12 else (vals - np.nanmin(vals)) / span
    for col in high_cols:
        vals = pd.to_numeric(scored[col], errors="coerce").to_numpy(dtype=float)
        span = np.nanmax(vals) - np.nanmin(vals)
        score += 0.0 if not np.isfinite(span) or span <= 1e-12 else (np.nanmax(vals) - vals) / span
    scored["stability_score"] = score
    scored["selected_as_final"] = False
    best_idx = int(np.nanargmin(score))
    scored.loc[best_idx, "selected_as_final"] = True
    return scored


def compare_to_old(ablation: pd.DataFrame) -> pd.DataFrame:
    if not OLD_VALIDATION_SUMMARY_PATH.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_VALIDATION_SUMMARY_PATH, encoding="utf-8-sig")
    old_row = old.loc[old["model_version"].astype(str).eq("calibrated")].iloc[0]
    rows = []
    for _, row in ablation.iterrows():
        out: Dict[str, object] = {"env_setting": row["env_setting"]}
        for out_col, source_col in REQUESTED_METRICS.items():
            old_value = float(pd.to_numeric(pd.Series([old_row.get(source_col)]), errors="coerce").iloc[0])
            new_value = float(row[out_col])
            out[f"{out_col}_old"] = old_value
            out[f"{out_col}_new"] = new_value
            out[f"{out_col}_delta"] = new_value - old_value
            if np.isfinite(old_value) and abs(old_value) > 1e-12:
                out[f"{out_col}_pct_delta"] = (new_value - old_value) / old_value * 100.0
            else:
                out[f"{out_col}_pct_delta"] = np.nan
        rows.append(out)
    compare = pd.DataFrame(rows)
    compare.to_csv(ABLATION_COMPARE_CSV_PATH, index=False, encoding="utf-8-sig")
    return compare


def configure_step18(mode: str):
    mod = load_module("18_build_target_inputs_and_topk_retrieval.py", f"step18_{mode.replace('-', '_')}")
    configure_env_feature_mode(mod, mode)
    mod.HISTORICAL_LIBRARY_PATH = ENV_LIBRARY_PATH
    mod.TARGET_HALFHOUR_SAFE_CANDIDATES = [TARGET_ENV_SOURCE]
    mod.TARGET_TRACK_POINT_CANDIDATES = []
    mod.TARGET_OUTPUT_PATH = DATA_ENV_DIR / "problem2_target_halfhour_inputs_safe_env.csv"
    mod.TOPK_OUTPUT_PATH = DATA_ENV_DIR / "problem2_target_topk_similar_history_env.csv"
    mod.QC_REPORT_PATH = TABLE_DIR / "problem2_topk_retrieval_qc_report.md"
    return mod


def configure_step19():
    mod = load_module("19_generate_initial_rainfall_fields_from_topk.py", "step19")
    mod.TARGET_INPUT_PATH = DATA_ENV_DIR / "problem2_target_halfhour_inputs_safe_env.csv"
    mod.TOPK_TABLE_PATH = DATA_ENV_DIR / "problem2_target_topk_similar_history_env.csv"
    mod.HISTORICAL_LIBRARY_PATH = ENV_LIBRARY_PATH
    mod.INDEX_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_initial_fields_index_env.csv"
    mod.NPZ_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_initial_fields_topk_weighted_env.npz"
    mod.QC_REPORT_PATH = TABLE_DIR / "problem2_initial_generation_qc_report.md"
    mod.FIGURE_DIR = FIGURE_DIR / "initial_generation"
    return mod


def configure_step20():
    mod = load_module("20_eof_pca_structure_correction.py", "step20")
    mod.INITIAL_NPZ_PATH = DATA_ENV_DIR / "problem2_generated_initial_fields_topk_weighted_env.npz"
    mod.INITIAL_INDEX_PATH = DATA_ENV_DIR / "problem2_generated_initial_fields_index_env.csv"
    mod.TOPK_TABLE_PATH = DATA_ENV_DIR / "problem2_target_topk_similar_history_env.csv"
    mod.HISTORICAL_LIBRARY_PATH = ENV_LIBRARY_PATH
    mod.TARGET_INPUT_PATH = DATA_ENV_DIR / "problem2_target_halfhour_inputs_safe_env.csv"
    mod.MODEL_OUTPUT_PATH = DATA_ENV_DIR / "problem2_eof_pca_model_env.npz"
    mod.COEFF_OUTPUT_PATH = DATA_ENV_DIR / "problem2_target_eof_coefficients_env.csv"
    mod.BLENDED_NPZ_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_pca_blended_fields_env.npz"
    mod.BLENDED_INDEX_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_pca_blended_fields_index_env.csv"
    mod.QC_REPORT_PATH = TABLE_DIR / "problem2_eof_pca_correction_qc_report.md"
    mod.FIGURE_DIR = FIGURE_DIR / "eof_pca_correction"
    return mod


def configure_step21():
    mod = load_module("21_extreme_quantile_calibration.py", "step21")
    mod.HISTORICAL_LIBRARY_PATH = ENV_LIBRARY_PATH
    mod.TOPK_TABLE_PATH = DATA_ENV_DIR / "problem2_target_topk_similar_history_env.csv"
    mod.BLENDED_NPZ_PATH = DATA_ENV_DIR / "problem2_generated_pca_blended_fields_env.npz"
    mod.BLENDED_INDEX_PATH = DATA_ENV_DIR / "problem2_generated_pca_blended_fields_index_env.csv"
    mod.TARGETS_OUTPUT_PATH = DATA_ENV_DIR / "problem2_extreme_calibration_targets_env.csv"
    mod.CALIBRATED_NPZ_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_env.npz"
    mod.CALIBRATED_INDEX_OUTPUT_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_index_env.csv"
    mod.QC_REPORT_PATH = TABLE_DIR / "problem2_extreme_calibration_qc_report.md"
    mod.FIGURE_DIR = FIGURE_DIR / "extreme_calibration"
    return mod


def configure_step23(validation_summary_path: Path):
    mod = load_module("23_finalize_problem2_results.py", "step23")
    mod.TARGET_SAFE_INPUT_PATH = DATA_ENV_DIR / "problem2_target_halfhour_inputs_safe_env.csv"
    mod.TOPK_TABLE_PATH = DATA_ENV_DIR / "problem2_target_topk_similar_history_env.csv"
    mod.CALIBRATED_NPZ_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_env.npz"
    mod.CALIBRATED_INDEX_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_index_env.csv"
    mod.CALIBRATION_QC_REPORT_PATH = TABLE_DIR / "problem2_extreme_calibration_qc_report.md"
    mod.VALIDATION_MODEL_SUMMARY_PATH = validation_summary_path
    mod.VALIDATION_EVENT_SUMMARY_PATH = DATA_ENV_DIR / "pseudo_validation_event_summary_selected.csv"
    mod.VALIDATION_TIMESLICE_METRICS_PATH = DATA_ENV_DIR / "pseudo_validation_timeslice_metrics_selected.csv"
    mod.VALIDATION_QC_REPORT_PATH = TABLE_DIR / "pseudo_validation_qc_report_selected.md"
    mod.FINAL_TIMESERIES_PATH = TABLE_DIR / "problem2_final_timeseries_metrics.csv"
    mod.FINAL_SUMMARY_PATH = TABLE_DIR / "problem2_final_typhoon_metrics_summary.csv"
    mod.FINAL_KEY_TIMES_PATH = TABLE_DIR / "problem2_final_key_times.csv"
    mod.FINAL_REPORT_PATH = TABLE_DIR / "problem2_final_results_report.md"
    mod.RUN_LOG_PATH = TABLE_DIR / "problem2_final_results_run_log.txt"
    mod.FIGURE_DIR = FIGURE_DIR
    return mod


def configure_step24():
    mod = load_module("24_geographic_backprojection_results.py", "step24")
    mod.CALIBRATED_NPZ_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_env.npz"
    mod.CALIBRATED_INDEX_PATH = DATA_ENV_DIR / "problem2_generated_calibrated_fields_index_env.csv"
    mod.FINAL_TIMESERIES_PATH = TABLE_DIR / "problem2_final_timeseries_metrics.csv"
    mod.FINAL_SUMMARY_PATH = TABLE_DIR / "problem2_final_typhoon_metrics_summary.csv"
    mod.FINAL_KEY_TIMES_PATH = TABLE_DIR / "problem2_final_key_times.csv"
    mod.GEOGRAPHIC_NPZ_PATH = TABLE_DIR / "problem2_final_geographic_fields.npz"
    mod.GEOGRAPHIC_SUMMARY_PATH = TABLE_DIR / "problem2_final_geographic_summary.csv"
    mod.GEOGRAPHIC_KEY_LOCATIONS_PATH = TABLE_DIR / "problem2_final_geographic_key_locations.csv"
    mod.GEOGRAPHIC_TIMESERIES_PATH = TABLE_DIR / "problem2_final_geographic_timeseries_metrics.csv"
    mod.GEOGRAPHIC_REPORT_PATH = TABLE_DIR / "problem2_geographic_backprojection_report.md"
    mod.FIGURE_DIR = FIGURE_DIR
    mod.RUN_LOG_PATH = TABLE_DIR / "problem2_geographic_backprojection_run_log.txt"
    return mod


def copy_selected_validation_outputs(mode: str) -> None:
    safe = mode.replace("-", "_")
    for stem in ["model_summary", "event_summary", "timeslice_metrics"]:
        src = DATA_ENV_DIR / f"pseudo_validation_{stem}_{safe}.csv"
        dst = DATA_ENV_DIR / f"pseudo_validation_{stem}_selected.csv"
        if src.exists():
            shutil.copy2(src, dst)
    src_report = TABLE_DIR / f"pseudo_validation_qc_report_{safe}.md"
    dst_report = TABLE_DIR / "pseudo_validation_qc_report_selected.md"
    if src_report.exists():
        shutil.copy2(src_report, dst_report)


def run_final_pipeline(mode: str, validation_summary_path: Path) -> None:
    print(f"[28] Running final Problem-2 generation with ENV setting: {mode}")
    step18 = configure_step18(mode)
    step18.main()
    step19 = configure_step19()
    step19.main()
    step20 = configure_step20()
    step20.main()
    step21 = configure_step21()
    step21.main()
    copy_selected_validation_outputs(mode)
    step23 = configure_step23(validation_summary_path)
    step23.main()
    step24 = configure_step24()
    step24.main()


def field_quality_npz(npz_path: Path) -> Dict[str, int]:
    quality = {"nan": 0, "inf": 0, "negative": 0, "all_zero_fields": 0}
    with np.load(npz_path, allow_pickle=True) as z:
        arrays = []
        if "rain_mmhr_calibrated" in z.files:
            arrays.append(np.asarray(z["rain_mmhr_calibrated"], dtype=np.float64))
        else:
            for name in z.files:
                if name.endswith(("geo_cumulative_rain_mm", "geo_max_rain_mmhr", "geo_duration10_h", "geo_duration20_h")):
                    arrays.append(np.asarray(z[name], dtype=np.float64))
        for arr in arrays:
            quality["nan"] += int(np.count_nonzero(np.isnan(arr)))
            quality["inf"] += int(np.count_nonzero(np.isinf(arr)))
            quality["negative"] += int(np.count_nonzero(np.isfinite(arr) & (arr < 0.0)))
            if arr.ndim == 3:
                quality["all_zero_fields"] += int(np.count_nonzero(np.all(np.nan_to_num(arr, nan=0.0) == 0.0, axis=(1, 2))))
            elif arr.ndim == 2:
                quality["all_zero_fields"] += int(np.all(np.nan_to_num(arr, nan=0.0) == 0.0))
    return quality


def check_geographic_figures() -> Dict[str, object]:
    expected = []
    for name in ["KONG_REY", "MAN_YI"]:
        for suffix in [
            "final_cumulative_rain_geographic",
            "final_max_rain_geographic",
            "final_duration10_geographic",
        ]:
            expected.append(FIGURE_DIR / f"{name}_{suffix}.png")
    return {
        "expected": [str(p.relative_to(PROJECT_ROOT)) for p in expected],
        "missing": [str(p.relative_to(PROJECT_ROOT)) for p in expected if not p.exists()],
        "zero_size": [str(p.relative_to(PROJECT_ROOT)) for p in expected if p.exists() and p.stat().st_size <= 0],
    }


def write_run_summary(final_mode: str, compare: pd.DataFrame) -> None:
    final_summary_path = TABLE_DIR / "problem2_final_typhoon_metrics_summary.csv"
    geo_summary_path = TABLE_DIR / "problem2_final_geographic_summary.csv"
    calibrated_npz = DATA_ENV_DIR / "problem2_generated_calibrated_fields_env.npz"
    geo_npz = TABLE_DIR / "problem2_final_geographic_fields.npz"
    selected_compare = {}
    if not compare.empty:
        sub = compare.loc[compare["env_setting"].astype(str).eq(final_mode)]
        selected_compare = sub.iloc[0].to_dict() if len(sub) else {}
    payload = {
        "final_env_setting": final_mode,
        "final_env_features": ENV_FEATURE_SETS[final_mode],
        "ablation_csv": str(ABLATION_CSV_PATH.relative_to(PROJECT_ROOT)),
        "comparison_csv": str(ABLATION_COMPARE_CSV_PATH.relative_to(PROJECT_ROOT)) if ABLATION_COMPARE_CSV_PATH.exists() else None,
        "selected_vs_old": selected_compare,
        "final_summary_csv": str(final_summary_path.relative_to(PROJECT_ROOT)),
        "geographic_summary_csv": str(geo_summary_path.relative_to(PROJECT_ROOT)),
        "field_quality": {
            "calibrated": field_quality_npz(calibrated_npz),
            "geographic": field_quality_npz(geo_npz),
        },
        "geographic_figures": check_geographic_figures(),
    }
    RUN_SUMMARY_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[28] Run summary: {RUN_SUMMARY_JSON_PATH}")


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    build_env_historical_library()

    ablation_rows = []
    summary_paths: Dict[str, Path] = {}
    for mode in ENV_MODES:
        row, summary_path = run_pseudo_validation(mode, PROJECT_ROOT / "data/processed/problem2_eof_pca_model.npz", "ablation")
        ablation_rows.append(row)
        summary_paths[mode] = summary_path

    ablation = score_ablation(pd.DataFrame(ablation_rows))
    ablation.to_csv(ABLATION_CSV_PATH, index=False, encoding="utf-8-sig")
    compare = compare_to_old(ablation)
    final_mode = str(ablation.loc[ablation["selected_as_final"].astype(bool), "env_setting"].iloc[0])
    print(f"[28] Selected final ENV setting: {final_mode}")

    run_final_pipeline(final_mode, summary_paths[final_mode])
    write_run_summary(final_mode, compare)
    print(f"[28] Done in {(time.time() - t0) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
