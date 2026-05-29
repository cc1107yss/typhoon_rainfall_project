#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem 3 step 2: generate S0-S5 virtual-scenario rainfields.

This script reuses the Problem-2 env generation logic and writes only
Problem-3-specific outputs. It does not retrain Problem-1/Problem-2 models and
does not overwrite any problem2_env artifacts.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROBLEM3_DATA_DIR = PROCESSED_DIR / "problem3"
PROBLEM3_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "problem3"
PROBLEM3_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "problem3"
PROBLEM3_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "problem3"

SCENARIO_INPUT_PATH = PROCESSED_DIR / "problem3_scenario_inputs.csv"
SCENARIO_SUMMARY_PATH = PROCESSED_DIR / "problem3_scenario_summary.csv"
STEP1_AUDIT_PATH = PROBLEM3_TABLE_DIR / "problem3_scenario_validity_audit.csv"
HISTORY_PATH = PROCESSED_DIR / "problem2_env" / "problem2_historical_halfhour_sample_library_env.csv"
P2_ENV_RUN_SUMMARY_PATH = PROBLEM3_TABLE_DIR.parent / "problem2_env" / "problem2_env_run_summary.json"
P2_EOF_MODEL_PATH = PROCESSED_DIR / "problem2_env" / "problem2_eof_pca_model_env.npz"
P2_CALIBRATED_NPZ_PATH = PROCESSED_DIR / "problem2_env" / "problem2_generated_calibrated_fields_env.npz"
P2_CALIBRATED_INDEX_PATH = PROCESSED_DIR / "problem2_env" / "problem2_generated_calibrated_fields_index_env.csv"

TARGET_SAFE_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_target_inputs_safe.csv"
TOPK_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_topk_analogs.csv"
INITIAL_NPZ_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_rainfields_initial.npz"
INITIAL_INDEX_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_initial_index.csv"
COEFF_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_eof_coefficients.csv"
BLENDED_INDEX_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_pca_blended_index.csv"
CALIBRATION_TARGETS_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_calibration_targets.csv"
CALIBRATED_INDEX_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_calibrated_index.csv"
CALIBRATED_NPZ_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_rainfields_calibrated.npz"
TIMESLICE_METRICS_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_timeslice_metrics.csv"
GENERATION_QC_PATH = PROBLEM3_DATA_DIR / "problem3_scenario_generation_qc.csv"
QC_SUMMARY_PATH = PROBLEM3_TABLE_DIR / "problem3_step2_generation_qc_summary.csv"
REPORT_PATH = PROBLEM3_REPORT_DIR / "problem3_step2_generation_report.md"

SCRIPT18_PATH = PROJECT_ROOT / "scripts" / "18_build_target_inputs_and_topk_retrieval.py"
SCRIPT19_PATH = PROJECT_ROOT / "scripts" / "19_generate_initial_rainfall_fields_from_topk.py"
SCRIPT20_PATH = PROJECT_ROOT / "scripts" / "20_eof_pca_structure_correction.py"
SCRIPT21_PATH = PROJECT_ROOT / "scripts" / "21_extreme_quantile_calibration.py"

EXPECTED_SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5"]
EXPECTED_FIELD_SHAPE = (201, 201)
TOPK = 20
AREA10_THRESHOLD = 10.0
AREA20_THRESHOLD = 20.0
PHYSICAL_CAP_MMHR = 120.0
EPS = 1e-12

LEAKAGE_PATTERNS = [
    re.compile(r"(^|_)rain(_|$)", re.IGNORECASE),
    re.compile(r"centroid", re.IGNORECASE),
    re.compile(r"anisotropy", re.IGNORECASE),
    re.compile(r"(^|_)asym(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)quad(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)r50($|_)", re.IGNORECASE),
    re.compile(r"(^|_)r80($|_)", re.IGNORECASE),
    re.compile(r"(^|_)r90($|_)", re.IGNORECASE),
    re.compile(r"rainband", re.IGNORECASE),
    re.compile(r"band_width", re.IGNORECASE),
    re.compile(r"major_axis", re.IGNORECASE),
    re.compile(r"minor_axis", re.IGNORECASE),
    re.compile(r"orientation_deg", re.IGNORECASE),
    re.compile(r"rain_gini", re.IGNORECASE),
    re.compile(r"rain_entropy", re.IGNORECASE),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def import_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_time(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(value) else ""


def is_leakage_col(col: str) -> bool:
    return str(col).upper() == "OWD" or any(pattern.search(str(col)) for pattern in LEAKAGE_PATTERNS)


def leakage_columns(columns: Iterable[str]) -> list[str]:
    return [str(c) for c in columns if is_leakage_col(str(c))]


def load_env_feature_setting() -> str:
    if not P2_ENV_RUN_SUMMARY_PATH.exists():
        return "base-old"
    with P2_ENV_RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    value = str(data.get("final_env_setting") or "base-old").strip().lower()
    return value or "base-old"


def prepare_target_safe_inputs(scenario: pd.DataFrame, dropped_leakage: list[str]) -> pd.DataFrame:
    required = [
        "scenario_id",
        "scenario_name",
        "target_id",
        "time",
        "lat",
        "lon",
        "lon_180",
        "WND",
        "PRES",
        "intensity",
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "dt_h",
        "is_land",
        "signed_coast_dist_km",
        "coast_dist_km",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "move_dir_sin",
        "move_dir_cos",
        "life_progress",
    ]
    missing = [col for col in required if col not in scenario.columns]
    if missing:
        raise RuntimeError(f"Scenario input missing required fields: {missing}")

    bad = leakage_columns(scenario.columns)
    dropped_leakage.extend(bad)
    safe = scenario.drop(columns=bad, errors="ignore").copy()
    safe["time"] = pd.to_datetime(safe["time"], errors="coerce")
    safe = safe.sort_values(["scenario_id", "time", "scenario_time_index" if "scenario_time_index" in safe.columns else "target_id"]).reset_index(drop=True)
    safe["target_row_id"] = np.arange(len(safe), dtype=int)
    safe["typhoon_name"] = safe["scenario_id"].astype(str)
    safe["target_name_norm"] = safe["scenario_id"].astype(str)
    safe["event_uid"] = safe["scenario_id"].astype(str)
    safe["source_file"] = "problem3_scenario_inputs.csv"
    safe["is_target"] = True
    safe["safe_input_flag"] = True
    safe["target_time_window_flag"] = True

    for col in [
        "lat",
        "lon",
        "lon_180",
        "WND",
        "PRES",
        "intensity",
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "dt_h",
        "is_land",
        "signed_coast_dist_km",
        "coast_dist_km",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "move_dir_sin",
        "move_dir_cos",
        "life_progress",
    ]:
        safe[col] = numeric(safe[col])

    keep_cols = [
        "target_row_id",
        "scenario_id",
        "scenario_name",
        "target_id",
        "typhoon_name",
        "target_name_norm",
        "time",
        "source_file",
        "event_uid",
        "is_target",
        "lat",
        "lon",
        "lon_180",
        "WND",
        "PRES",
        "intensity",
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "dt_h",
        "is_land",
        "signed_coast_dist_km",
        "coast_dist_km",
        "landfrac_200km",
        "landfrac_500km",
        "terrain_mean_300km",
        "terrain_max_300km",
        "terrain_std_300km",
        "year",
        "month",
        "day",
        "hour",
        "season",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "life_time_index",
        "life_progress",
        "move_dir_sin",
        "move_dir_cos",
        "target_time_window_flag",
        "safe_input_flag",
    ]
    for col in keep_cols:
        if col not in safe.columns:
            safe[col] = np.nan
    out = safe[keep_cols].copy()
    out.to_csv(TARGET_SAFE_PATH, index=False, encoding="utf-8-sig")
    return out


def build_problem3_topk(step18, target: pd.DataFrame, env_setting: str) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    step18.HISTORICAL_LIBRARY_PATH = HISTORY_PATH
    step18.TOPK_OUTPUT_PATH = TOPK_PATH
    step18.QC_REPORT_PATH = PROBLEM3_REPORT_DIR / "problem3_step2_topk_retrieval_qc_from_problem2_logic.md"
    step18.TOPK = TOPK
    step18.configure_environment_features(env_setting)

    history, history_report = step18.load_historical_library()
    selected_components, skipped = step18.select_safe_retrieval_features(history, target)
    selected_features = [feature for features in selected_components.values() for feature in features]
    if not selected_features:
        raise RuntimeError("No safe retrieval features were selected.")
    leaked_features = [feature for feature in selected_features if step18.is_leakage_feature(feature)]
    if leaked_features:
        raise RuntimeError(f"Leakage features selected for Top-K retrieval: {leaked_features}")

    history_imp, target_imp, impute_report = step18.impute_retrieval_features(history, target, selected_components)
    params = step18.compute_standardization_params(history_imp, selected_features)
    topk, diagnostics = step18.build_topk_table(history_imp, target_imp, selected_components, params)

    target_meta = target[
        [
            "target_id",
            "target_row_id",
            "scenario_id",
            "scenario_name",
            "time",
            "lat",
            "lon",
            "lon_180",
            "WND",
            "PRES",
        ]
    ].copy()
    target_meta["time"] = pd.to_datetime(target_meta["time"], errors="coerce")
    topk = topk.merge(target_meta, on="target_id", how="left", suffixes=("", "_scenario"))
    topk["scenario_id"] = topk["scenario_id"].fillna(topk["target_typhoon_name"])
    topk["scenario_name"] = topk["scenario_name"].fillna(topk["scenario_id"])
    topk["time"] = pd.to_datetime(topk["time"], errors="coerce").map(format_time)
    topk["analog_event_uid"] = topk["history_event_uid"]
    topk["analog_time"] = pd.to_datetime(topk["history_time"], errors="coerce").map(format_time)
    topk["analog_typhoon_name"] = topk["history_typhoon_name"]
    topk["analog_lat"] = topk["history_lat"]
    topk["analog_lon"] = topk["history_lon_180"]
    topk["analog_WND"] = topk["history_WND"]
    topk["analog_PRES"] = topk["history_PRES"]
    topk["analog_distance"] = topk["similarity_distance"]
    topk["analog_weight"] = topk["similarity_weight"]

    first_cols = [
        "scenario_id",
        "scenario_name",
        "time",
        "target_row_id",
        "target_id",
        "rank",
        "analog_event_uid",
        "analog_time",
        "analog_typhoon_name",
        "analog_lat",
        "analog_lon",
        "analog_WND",
        "analog_PRES",
        "analog_distance",
        "analog_weight",
    ]
    ordered = [c for c in first_cols if c in topk.columns] + [c for c in topk.columns if c not in first_cols]
    topk = topk[ordered].copy()
    topk.to_csv(TOPK_PATH, index=False, encoding="utf-8-sig")

    target_source_report = {
        "source_type": "problem3_scenario_inputs",
        "source_path": rel(SCENARIO_INPUT_PATH),
        "source_shape": [int(target.shape[0]), int(target.shape[1])],
        "leakage_columns_in_source": [],
        "missing_environment_fields": [],
    }
    step18.write_qc_report(
        history_imp,
        target_imp,
        topk,
        target_source_report,
        history_report,
        selected_components,
        skipped,
        impute_report,
        diagnostics,
    )
    return topk, selected_features, history_imp


def save_initial_npz(
    rain_initial: np.ndarray,
    log_initial: np.ndarray,
    target: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> None:
    INITIAL_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INITIAL_NPZ_PATH,
        rainfields_initial_mmhr=np.asarray(rain_initial, dtype=np.float32),
        log_rain_initial=np.asarray(log_initial, dtype=np.float32),
        target_id=target["target_id"].astype(str).to_numpy(dtype="U"),
        target_row_id=pd.to_numeric(target["target_row_id"], errors="coerce").to_numpy(dtype=np.int32),
        scenario_id=target["scenario_id"].astype(str).to_numpy(dtype="U"),
        scenario_name=target["scenario_name"].astype(str).to_numpy(dtype="U"),
        time=pd.to_datetime(target["time"], errors="coerce").map(format_time).astype(str).to_numpy(dtype="U"),
        lat=pd.to_numeric(target["lat"], errors="coerce").to_numpy(dtype=np.float32),
        lon=pd.to_numeric(target["lon"], errors="coerce").to_numpy(dtype=np.float32),
        lon_180=pd.to_numeric(target["lon_180"], errors="coerce").to_numpy(dtype=np.float32),
        x_front_km=np.asarray(x_front_km, dtype=np.float32),
        y_left_km=np.asarray(y_left_km, dtype=np.float32),
    )


def load_eof_model() -> dict[str, np.ndarray]:
    if not P2_EOF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Problem-2 env EOF/PCA model not found: {P2_EOF_MODEL_PATH}")
    with np.load(P2_EOF_MODEL_PATH, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def build_coefficients_table(initial_index: pd.DataFrame, coefficients: np.ndarray, recon_diag: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i in range(coefficients.shape[0]):
        row = {
            "field_index": int(i),
            "target_id": initial_index.iloc[i].get("target_id"),
            "typhoon_name": initial_index.iloc[i].get("typhoon_name"),
            "time": initial_index.iloc[i].get("time"),
            "eof_reconstruction_rmse_log": float(recon_diag["eof_reconstruction_rmse_log"][i]),
            "eof_reconstruction_corr_log": float(recon_diag["eof_reconstruction_corr_log"][i]),
            "eof_reconstruction_energy_ratio": float(recon_diag["eof_reconstruction_energy_ratio"][i]),
        }
        for j in range(coefficients.shape[1]):
            row[f"eof_coeff_{j + 1:02d}"] = float(coefficients[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_existing_eof_pca(step20, rain_initial: np.ndarray, log_initial: np.ndarray, initial_index: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    model = load_eof_model()
    x_front_km = np.asarray(model["x_front_km"], dtype=np.float32)
    y_left_km = np.asarray(model["y_left_km"], dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_front_km, y_left_km)
    step20.N_COMPONENTS = int(np.asarray(model["n_components"]).item())
    step20.BETA_BLEND = float(np.asarray(model["beta_blend"]).item())
    coefficients = step20.project_target_initial_fields(log_initial, model)
    rain_eof, log_eof, recon_diag = step20.reconstruct_eof_fields(log_initial, coefficients, model)
    rain_blend, log_blend = step20.blend_initial_and_eof_fields(rain_initial, rain_eof, step20.BETA_BLEND)
    coeff_df = build_coefficients_table(initial_index, coefficients, recon_diag)
    coeff_df.to_csv(COEFF_PATH, index=False, encoding="utf-8-sig")
    cumulative_evr = float(np.asarray(model["cumulative_explained_variance_ratio"])[-1])
    blended_index = step20.build_blended_index_table(
        initial_index,
        rain_initial,
        rain_eof,
        rain_blend,
        x_front_km,
        y_left_km,
        x_grid,
        y_grid,
        cumulative_evr,
    )
    blended_index.to_csv(BLENDED_INDEX_PATH, index=False, encoding="utf-8-sig")
    del rain_eof, log_eof, log_blend, coefficients
    gc.collect()
    return rain_blend, blended_index


def calibrate_fields(step21, topk: pd.DataFrame, history: pd.DataFrame, rain_blend: np.ndarray, blended_index: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    targets, _ = step21.compute_weighted_calibration_targets(topk, history, blended_index)
    if not np.array_equal(targets["target_id"].astype(str).to_numpy(), blended_index["target_id"].astype(str).to_numpy()):
        raise RuntimeError("Calibration target order mismatch.")
    targets_table = step21.build_calibration_targets_table(targets)
    targets_table.to_csv(CALIBRATION_TARGETS_PATH, index=False, encoding="utf-8-sig")
    x_front_km = np.asarray(load_eof_model()["x_front_km"], dtype=np.float32)
    y_left_km = np.asarray(load_eof_model()["y_left_km"], dtype=np.float32)
    rain_calibrated, scale_field, diag_df = step21.calibrate_all_fields(rain_blend, targets, x_front_km, y_left_km)
    rain_calibrated = np.nan_to_num(rain_calibrated, nan=0.0, posinf=PHYSICAL_CAP_MMHR, neginf=0.0).astype(np.float32)
    rain_calibrated[rain_calibrated < 0.0] = 0.0
    rain_calibrated[rain_calibrated > PHYSICAL_CAP_MMHR] = PHYSICAL_CAP_MMHR
    calibrated_index = step21.build_calibrated_index_table(
        blended_index,
        targets,
        diag_df,
        rain_blend,
        rain_calibrated,
        scale_field,
        x_front_km,
        y_left_km,
    )
    calibrated_index.to_csv(CALIBRATED_INDEX_PATH, index=False, encoding="utf-8-sig")
    del scale_field
    gc.collect()
    return rain_calibrated, calibrated_index


def reuse_problem2_kongrey_calibrated_baseline(rain_calibrated: np.ndarray, target: pd.DataFrame) -> tuple[np.ndarray, Dict[str, object]]:
    """Replace Problem-3 S0 fields with the finalized Problem-2 KONG-REY baseline."""
    info: Dict[str, object] = {
        "status": "not_checked",
        "note": "Problem-2 KONG-REY calibrated baseline was not reused.",
        "reused_timesteps": 0,
        "max_abs_diff_after_reuse": np.nan,
    }
    if not P2_CALIBRATED_NPZ_PATH.exists() or not P2_CALIBRATED_INDEX_PATH.exists():
        info["status"] = "missing"
        info["note"] = "Problem-2 env calibrated field or index is missing."
        return rain_calibrated, info

    p2_index = read_csv(P2_CALIBRATED_INDEX_PATH)
    p2_index["time"] = pd.to_datetime(p2_index["time"], errors="coerce")
    kong = p2_index.loc[p2_index["typhoon_name"].astype(str).eq("KONG-REY")].sort_values("time").reset_index(drop=True)
    s0 = target.loc[target["scenario_id"].astype(str).eq("S0")].sort_values("time").copy()
    if kong.empty or len(kong) != len(s0):
        info["status"] = "failed"
        info["note"] = f"Problem-2 KONG-REY rows={len(kong)}, S0 rows={len(s0)}."
        return rain_calibrated, info

    p2_times = pd.to_datetime(kong["time"], errors="coerce").astype("int64").to_numpy()
    s0_times = pd.to_datetime(s0["time"], errors="coerce").astype("int64").to_numpy()
    if not np.array_equal(p2_times, s0_times):
        info["status"] = "failed"
        info["note"] = "Problem-2 KONG-REY and Problem-3 S0 times do not match."
        return rain_calibrated, info

    with np.load(P2_CALIBRATED_NPZ_PATH, allow_pickle=True) as z:
        p2_fields = np.asarray(z["rain_mmhr_calibrated"], dtype=np.float32)
    p2_kong = p2_fields[kong["field_index"].to_numpy(dtype=int)]
    if p2_kong.shape != (len(s0),) + tuple(rain_calibrated.shape[1:]):
        info["status"] = "failed"
        info["note"] = f"Problem-2 KONG-REY field shape {p2_kong.shape} cannot replace S0 shape {(len(s0),) + tuple(rain_calibrated.shape[1:])}."
        return rain_calibrated, info

    out = np.asarray(rain_calibrated, dtype=np.float32).copy()
    s0_indices = s0.index.to_numpy(dtype=int)
    out[s0_indices] = p2_kong
    diff = out[s0_indices].astype(np.float32) - p2_kong.astype(np.float32)
    info.update(
        {
            "status": "reused",
            "note": "S0 calibrated rainfields directly reuse Problem-2 env KONG-REY calibrated baseline.",
            "reused_timesteps": int(len(s0_indices)),
            "max_abs_diff_after_reuse": float(np.max(np.abs(diff))),
        }
    )
    return out, info


def save_calibrated_npz(rain_calibrated: np.ndarray, target: pd.DataFrame) -> None:
    np.savez_compressed(
        CALIBRATED_NPZ_PATH,
        rainfields_calibrated_mmhr=np.asarray(rain_calibrated, dtype=np.float32),
        target_id=target["target_id"].astype(str).to_numpy(dtype="U"),
        target_row_id=pd.to_numeric(target["target_row_id"], errors="coerce").to_numpy(dtype=np.int32),
        scenario_id=target["scenario_id"].astype(str).to_numpy(dtype="U"),
        scenario_name=target["scenario_name"].astype(str).to_numpy(dtype="U"),
        time=pd.to_datetime(target["time"], errors="coerce").map(format_time).astype(str).to_numpy(dtype="U"),
        lat=pd.to_numeric(target["lat"], errors="coerce").to_numpy(dtype=np.float32),
        lon=pd.to_numeric(target["lon"], errors="coerce").to_numpy(dtype=np.float32),
        lon_180=pd.to_numeric(target["lon_180"], errors="coerce").to_numpy(dtype=np.float32),
        WND=pd.to_numeric(target["WND"], errors="coerce").to_numpy(dtype=np.float32),
        PRES=pd.to_numeric(target["PRES"], errors="coerce").to_numpy(dtype=np.float32),
    )


def cell_area_from_grid(x_front_km: np.ndarray, y_left_km: np.ndarray) -> float:
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    return dx * dy


def dominant_quadrant(field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> str:
    rain = np.nan_to_num(np.asarray(field, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    quadrants = {
        "front_left": float(np.sum(rain[(x_grid > 0.0) & (y_grid > 0.0)])),
        "front_right": float(np.sum(rain[(x_grid > 0.0) & (y_grid < 0.0)])),
        "back_left": float(np.sum(rain[(x_grid < 0.0) & (y_grid > 0.0)])),
        "back_right": float(np.sum(rain[(x_grid < 0.0) & (y_grid < 0.0)])),
    }
    if max(quadrants.values()) <= EPS:
        return "none"
    return max(quadrants, key=quadrants.get)


def build_timeslice_metrics(rain_calibrated: np.ndarray, target: pd.DataFrame) -> pd.DataFrame:
    model = load_eof_model()
    x_front_km = np.asarray(model["x_front_km"], dtype=np.float32)
    y_left_km = np.asarray(model["y_left_km"], dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_front_km, y_left_km)
    cell_area = cell_area_from_grid(x_front_km, y_left_km)
    rows = []
    for i in range(rain_calibrated.shape[0]):
        field = np.asarray(rain_calibrated[i], dtype=np.float32)
        finite = field[np.isfinite(field)]
        rows.append(
            {
                "scenario_id": target.iloc[i]["scenario_id"],
                "scenario_name": target.iloc[i]["scenario_name"],
                "time": format_time(target.iloc[i]["time"]),
                "lat": float(target.iloc[i]["lat"]),
                "lon": float(target.iloc[i]["lon"]),
                "max_rain_mmhr": float(np.max(finite)) if finite.size else np.nan,
                "p95_mmhr": float(np.percentile(finite, 95)) if finite.size else np.nan,
                "p99_mmhr": float(np.percentile(finite, 99)) if finite.size else np.nan,
                "area10_km2": float(np.count_nonzero(np.isfinite(field) & (field >= AREA10_THRESHOLD)) * cell_area),
                "area20_km2": float(np.count_nonzero(np.isfinite(field) & (field >= AREA20_THRESHOLD)) * cell_area),
                "rain_sum_mmhr_grid": float(np.sum(np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0))),
                "nonzero_area_km2": float(np.count_nonzero(np.isfinite(field) & (field > 0.0)) * cell_area),
                "dominant_quadrant": dominant_quadrant(field, x_grid, y_grid),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TIMESLICE_METRICS_PATH, index=False, encoding="utf-8-sig")
    return out


def count_quality(fields: np.ndarray) -> Dict[str, int | float | str]:
    arr = np.asarray(fields)
    return {
        "field_shape": str(tuple(arr.shape[1:])),
        "nan_count": int(np.count_nonzero(np.isnan(arr))),
        "inf_count": int(np.count_nonzero(np.isinf(arr))),
        "negative_count": int(np.count_nonzero(np.isfinite(arr) & (arr < 0.0))),
        "all_zero_count": int(np.count_nonzero(np.all(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) == 0.0, axis=(1, 2)))),
        "max_value_mmhr": float(np.nanmax(arr)),
        "p99_value_mmhr": float(np.nanpercentile(arr, 99)),
    }


def compare_s0_with_problem2(rain_calibrated: np.ndarray, target: pd.DataFrame, metrics: pd.DataFrame) -> Dict[str, object]:
    result: Dict[str, object] = {
        "status": "not_checked",
        "note": "Problem-2 env calibrated field or index not found.",
        "shape_match": False,
        "time_match": False,
        "max_abs_diff": np.nan,
        "mean_abs_diff": np.nan,
        "p95_abs_diff": np.nan,
        "area10_abs_diff": np.nan,
    }
    if not P2_CALIBRATED_NPZ_PATH.exists() or not P2_CALIBRATED_INDEX_PATH.exists():
        return result
    p2_index = read_csv(P2_CALIBRATED_INDEX_PATH)
    p2_index["time"] = pd.to_datetime(p2_index["time"], errors="coerce")
    kong = p2_index.loc[p2_index["typhoon_name"].astype(str).eq("KONG-REY")].sort_values("time").reset_index(drop=True)
    s0_target = target.loc[target["scenario_id"].astype(str).eq("S0")].sort_values("time").reset_index()
    if kong.empty or len(kong) != len(s0_target):
        result["note"] = f"Problem-2 KONG-REY rows={len(kong)}, S0 rows={len(s0_target)}."
        result["status"] = "warning"
        return result
    with np.load(P2_CALIBRATED_NPZ_PATH, allow_pickle=True) as z:
        p2_fields = np.asarray(z["rain_mmhr_calibrated"], dtype=np.float32)
    p2_kong = p2_fields[kong["field_index"].to_numpy(dtype=int)]
    s0_fields = rain_calibrated[s0_target["index"].to_numpy(dtype=int)]
    result["shape_match"] = bool(p2_kong.shape == s0_fields.shape)
    p2_times = pd.to_datetime(kong["time"], errors="coerce").astype("int64").to_numpy()
    s0_times = pd.to_datetime(s0_target["time"], errors="coerce").astype("int64").to_numpy()
    result["time_match"] = bool(np.array_equal(p2_times, s0_times))
    if not result["shape_match"] or not result["time_match"]:
        result["status"] = "warning"
        result["note"] = "S0 and problem2 KONG-REY shape/time alignment differs."
        return result
    diff = s0_fields.astype(np.float32) - p2_kong.astype(np.float32)
    result["max_abs_diff"] = float(np.max(np.abs(diff)))
    result["mean_abs_diff"] = float(np.mean(np.abs(diff)))
    p2_p95 = np.percentile(p2_kong.reshape(len(p2_kong), -1), 95, axis=1)
    s0_p95 = np.percentile(s0_fields.reshape(len(s0_fields), -1), 95, axis=1)
    result["p95_abs_diff"] = float(np.max(np.abs(s0_p95 - p2_p95)))
    cell_area = 100.0
    p2_area10 = np.count_nonzero(p2_kong >= AREA10_THRESHOLD, axis=(1, 2)) * cell_area
    s0_area10 = np.count_nonzero(s0_fields >= AREA10_THRESHOLD, axis=(1, 2)) * cell_area
    result["area10_abs_diff"] = float(np.max(np.abs(s0_area10 - p2_area10)))
    if result["max_abs_diff"] <= 1e-5 and result["mean_abs_diff"] <= 1e-7:
        result["status"] = "identical"
        result["note"] = "S0 reproduces problem2 KONG-REY baseline."
    elif result["max_abs_diff"] <= 1e-3 and result["mean_abs_diff"] <= 1e-5:
        result["status"] = "close"
        result["note"] = "S0 is numerically close to problem2 KONG-REY baseline."
    else:
        result["status"] = "warning"
        result["note"] = "S0 differs from problem2 KONG-REY baseline; likely caused by retrieval/model/config differences."
    return result


def s4_checks(target: pd.DataFrame, topk: pd.DataFrame, history: pd.DataFrame, qc_base: pd.DataFrame) -> Dict[str, object]:
    s0 = target.loc[target["scenario_id"].eq("S0")].copy()
    s4 = target.loc[target["scenario_id"].eq("S4")].copy()
    s4_times = pd.to_datetime(s4["time"], errors="coerce").sort_values()
    diffs_min = s4_times.diff().dropna().dt.total_seconds() / 60.0
    s0_speed = float(numeric(s0["move_speed_kmh"]).mean(skipna=True))
    s4_speed = float(numeric(s4["move_speed_kmh"]).mean(skipna=True))
    hist_speed = numeric(history["move_speed_kmh"]) if "move_speed_kmh" in history.columns else pd.Series(dtype=float)
    s4_dist = topk.loc[topk["scenario_id"].eq("S4"), "analog_distance"]
    other_p95 = qc_base.loc[qc_base["scenario_id"].ne("S4"), "p95_topk_distance"]
    s4_p95 = float(numeric(s4_dist).quantile(0.95)) if len(s4_dist) else np.nan
    other_median_p95 = float(numeric(other_p95).median(skipna=True)) if len(other_p95) else np.nan
    distance_warning = bool(np.isfinite(s4_p95) and np.isfinite(other_median_p95) and s4_p95 > 1.5 * other_median_p95)
    return {
        "s4_time_strictly_increasing": bool(s4_times.is_monotonic_increasing and not s4_times.duplicated().any()),
        "s4_all_30min": bool(len(diffs_min) > 0 and np.allclose(diffs_min.to_numpy(dtype=float), 30.0)),
        "s4_mean_move_speed_kmh": s4_speed,
        "s0_mean_move_speed_kmh": s0_speed,
        "s4_slower_than_s0": bool(s4_speed < s0_speed),
        "historical_speed_q05": float(hist_speed.quantile(0.05)) if len(hist_speed.dropna()) else np.nan,
        "historical_speed_q95": float(hist_speed.quantile(0.95)) if len(hist_speed.dropna()) else np.nan,
        "s4_speed_within_historical_minmax": bool(len(hist_speed.dropna()) and s4_speed >= float(hist_speed.min()) and s4_speed <= float(hist_speed.max())),
        "s4_p95_topk_distance": s4_p95,
        "other_scenario_median_p95_topk_distance": other_median_p95,
        "s4_topk_distance_warning": distance_warning,
    }


def build_qc_tables(
    rain_calibrated: np.ndarray,
    target: pd.DataFrame,
    topk: pd.DataFrame,
    metrics: pd.DataFrame,
    step1_audit: pd.DataFrame,
    s0_compare: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = target.groupby("scenario_id").size().to_dict()
    step1_lookup = (
        step1_audit.set_index("scenario_id")["out_of_support_ratio"].to_dict()
        if "scenario_id" in step1_audit.columns and "out_of_support_ratio" in step1_audit.columns
        else {}
    )
    rows = []
    summary_rows = []
    scenario_p95_dist = topk.groupby("scenario_id")["analog_distance"].quantile(0.95)
    median_p95_dist = float(scenario_p95_dist.median(skipna=True)) if len(scenario_p95_dist) else np.nan
    for scenario_id in EXPECTED_SCENARIOS:
        idx = target.index[target["scenario_id"].eq(scenario_id)].to_numpy(dtype=int)
        sub_fields = rain_calibrated[idx]
        q = count_quality(sub_fields)
        sub_topk = topk.loc[topk["scenario_id"].eq(scenario_id)].copy()
        counts = sub_topk.groupby("target_id").size()
        mean_dist = float(numeric(sub_topk["analog_distance"]).mean(skipna=True)) if len(sub_topk) else np.nan
        p95_dist = float(numeric(sub_topk["analog_distance"]).quantile(0.95)) if len(sub_topk) else np.nan
        min_topk_count = int(counts.min()) if len(counts) else 0
        actual = int(len(idx))
        expected_n = int(expected.get(scenario_id, 0))
        hard_fail = (
            actual != expected_n
            or tuple(sub_fields.shape[1:]) != EXPECTED_FIELD_SHAPE
            or q["nan_count"] > 0
            or q["inf_count"] > 0
            or q["negative_count"] > 0
            or q["all_zero_count"] > 0
            or min_topk_count == 0
        )
        warnings = []
        if min_topk_count < TOPK:
            warnings.append(f"min_topk_count={min_topk_count}<20")
        out_ratio = float(step1_lookup.get(scenario_id, 0.0) or 0.0)
        if out_ratio > 0.0:
            warnings.append(f"step1_out_of_support_ratio={out_ratio:.4f}")
        if np.isfinite(median_p95_dist) and np.isfinite(p95_dist) and p95_dist > 1.5 * median_p95_dist:
            warnings.append("p95_topk_distance_high_vs_scenario_median")
        if scenario_id == "S0" and s0_compare.get("status") not in {"identical", "close"}:
            warnings.append("S0_baseline_difference")

        if hard_fail:
            level = "FAIL"
            note = "Field shape/count or generated field quality failed."
        elif warnings:
            level = "WARNING"
            note = "; ".join(warnings)
        else:
            level = "PASS"
            note = "Generation quality checks passed."

        row = {
            "scenario_id": scenario_id,
            "n_timesteps": actual,
            "expected_timesteps": expected_n,
            "actual_timesteps": actual,
            "field_shape": q["field_shape"],
            "nan_count": q["nan_count"],
            "inf_count": q["inf_count"],
            "negative_count": q["negative_count"],
            "all_zero_count": q["all_zero_count"],
            "max_value_mmhr": q["max_value_mmhr"],
            "p99_value_mmhr": q["p99_value_mmhr"],
            "mean_topk_distance": mean_dist,
            "p95_topk_distance": p95_dist,
            "min_topk_count": min_topk_count,
            "out_of_support_ratio_from_step1": out_ratio,
            "qc_level": level,
            "qc_note": note,
        }
        rows.append(row)

        sub_metrics = metrics.loc[metrics["scenario_id"].eq(scenario_id)]
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": str(target.loc[target["scenario_id"].eq(scenario_id), "scenario_name"].iloc[0]),
                "n_timesteps": actual,
                "max_rain_mmhr": float(numeric(sub_metrics["max_rain_mmhr"]).max(skipna=True)),
                "max_p95_mmhr": float(numeric(sub_metrics["p95_mmhr"]).max(skipna=True)),
                "max_p99_mmhr": float(numeric(sub_metrics["p99_mmhr"]).max(skipna=True)),
                "max_area10_km2": float(numeric(sub_metrics["area10_km2"]).max(skipna=True)),
                "max_area20_km2": float(numeric(sub_metrics["area20_km2"]).max(skipna=True)),
                "mean_topk_distance": mean_dist,
                "p95_topk_distance": p95_dist,
                "nan_count": q["nan_count"],
                "negative_count": q["negative_count"],
                "all_zero_count": q["all_zero_count"],
                "qc_level": level,
                "brief_note": note,
            }
        )
    qc = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    qc.to_csv(GENERATION_QC_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(QC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return qc, summary


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_report(
    env_setting: str,
    selected_features: Sequence[str],
    scenario_counts: Mapping[str, int],
    history_rows: int,
    dropped_leakage: Sequence[str],
    qc: pd.DataFrame,
    qc_summary: pd.DataFrame,
    s0_compare: Mapping[str, object],
    s4: Mapping[str, object],
    final_status: str,
) -> None:
    topk_stats = qc[["scenario_id", "mean_topk_distance", "p95_topk_distance", "min_topk_count", "qc_level"]].copy()
    quality = qc[["scenario_id", "nan_count", "inf_count", "negative_count", "all_zero_count", "max_value_mmhr", "p99_value_mmhr", "qc_note"]].copy()
    lines = [
        "# Problem 3 Step 2 Generation Report",
        "",
        "## 1. Inputs read",
        f"- Scenario input main table: `{rel(SCENARIO_INPUT_PATH)}`",
        f"- Problem-3 Step-1 scenario summary: `{rel(SCENARIO_SUMMARY_PATH)}`",
        f"- Problem-3 Step-1 validity audit: `{rel(STEP1_AUDIT_PATH)}`",
        f"- Problem-2 env historical library: `{rel(HISTORY_PATH)}`",
        f"- Problem-2 env EOF/PCA model artifact: `{rel(P2_EOF_MODEL_PATH)}`",
        f"- Problem-2 env calibrated baseline for S0 comparison: `{rel(P2_CALIBRATED_NPZ_PATH)}` and `{rel(P2_CALIBRATED_INDEX_PATH)}`",
        "",
        "## 2. Reused Problem-2 logic",
        f"- Top-K retrieval: functions from `{rel(SCRIPT18_PATH)}`",
        f"- Storm-relative initial field generation: functions from `{rel(SCRIPT19_PATH)}`",
        f"- EOF/PCA projection and beta blending: functions from `{rel(SCRIPT20_PATH)}`, using existing model artifact rather than refitting PCA",
        f"- Extreme quantile/tail calibration and physical caps: functions from `{rel(SCRIPT21_PATH)}`",
        f"- Problem-2 env feature setting reused for retrieval: `{env_setting}`",
        f"- Selected safe feature count: {len(selected_features)}",
        f"- Selected safe features: {', '.join(selected_features)}",
        "",
        "## 3. Scenario counts",
        markdown_table(pd.DataFrame({"scenario_id": list(scenario_counts.keys()), "n_timesteps": list(scenario_counts.values())})),
        f"- Total scenario input rows: {sum(scenario_counts.values())}",
        f"- Historical library rows used for retrieval after Problem-2 filtering/imputation: {history_rows}",
        "",
        "## 4. Leakage guard",
        f"- Dropped forbidden scenario columns: {sorted(set(dropped_leakage))}",
        "- The retrieval feature list was checked against OWD, rain_*, centroid_*, anisotropy, asym_*, quad_*, r50/r80/r90, rainband, major/minor axis, orientation, rain_gini, and rain_entropy patterns.",
        "",
        "## 5. Top-K distance statistics",
        markdown_table(topk_stats),
        "",
        "## 6. Rainfield quality checks",
        markdown_table(quality),
        "- Nonnegative constraint was applied after calibration.",
        f"- Physical upper cap was enforced at {PHYSICAL_CAP_MMHR:.1f} mm/hr, consistent with the Problem-2 calibration cap.",
        "",
        "## 7. S0 baseline comparison",
        f"- Status: {s0_compare.get('status')}",
        f"- Note: {s0_compare.get('note')}",
        f"- Shape match: {s0_compare.get('shape_match')}",
        f"- Time match: {s0_compare.get('time_match')}",
        f"- Max absolute difference: {s0_compare.get('max_abs_diff')}",
        f"- Mean absolute difference: {s0_compare.get('mean_abs_diff')}",
        f"- P95 max absolute difference: {s0_compare.get('p95_abs_diff')}",
        f"- Area10 max absolute difference: {s0_compare.get('area10_abs_diff')}",
        "",
        "## 8. S4 slowdown checks",
        f"- S4 time strictly increasing: {s4.get('s4_time_strictly_increasing')}",
        f"- S4 adjacent timestep is 30 min: {s4.get('s4_all_30min')}",
        f"- S0 mean move speed: {s4.get('s0_mean_move_speed_kmh'):.6f} km/h",
        f"- S4 mean move speed: {s4.get('s4_mean_move_speed_kmh'):.6f} km/h",
        f"- S4 slower than S0: {s4.get('s4_slower_than_s0')}",
        f"- Historical speed P05/P95: {s4.get('historical_speed_q05'):.6f} / {s4.get('historical_speed_q95'):.6f} km/h",
        f"- S4 speed within historical min-max: {s4.get('s4_speed_within_historical_minmax')}",
        f"- S4 P95 Top-K distance: {s4.get('s4_p95_topk_distance'):.6f}",
        f"- Other-scenario median P95 Top-K distance: {s4.get('other_scenario_median_p95_topk_distance'):.6f}",
        f"- S4 Top-K distance warning: {s4.get('s4_topk_distance_warning')}",
        "",
        "## 9. Units and scope",
        "- The generated rainfields are half-hourly rainfall intensity fields in mm/hr.",
        "- This step does not convert mm/hr to accumulated rainfall. The third step should use 0.5 x R_t for half-hour accumulation.",
        "- `rain_sum_mmhr_grid` in the timeslice metrics is a quality-control grid sum of intensity values, not accumulated precipitation.",
        "- This step did not compute final scenario conclusions, cumulative rainfall maps, duration10 geographic maps, or final scenario-comparison tables.",
        "",
        "## 10. Outputs",
        f"- Top-K analogs: `{rel(TOPK_PATH)}`",
        f"- Initial rainfields: `{rel(INITIAL_NPZ_PATH)}`",
        f"- Calibrated rainfields: `{rel(CALIBRATED_NPZ_PATH)}`",
        f"- Timeslice metrics: `{rel(TIMESLICE_METRICS_PATH)}`",
        f"- Generation QC: `{rel(GENERATION_QC_PATH)}`",
        f"- Paper QC summary: `{rel(QC_SUMMARY_PATH)}`",
        "",
        f"Final run status: **{final_status}**",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in [PROBLEM3_DATA_DIR, PROBLEM3_TABLE_DIR, PROBLEM3_FIGURE_DIR, PROBLEM3_REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("[30] Loading scenario inputs")
    scenario = read_csv(SCENARIO_INPUT_PATH)
    scenario["time"] = pd.to_datetime(scenario["time"], errors="coerce")
    scenario_counts = scenario.groupby("scenario_id").size().reindex(EXPECTED_SCENARIOS).to_dict()
    if int(sum(scenario_counts.values())) != 2841:
        raise RuntimeError(f"Expected 2841 scenario rows, got {sum(scenario_counts.values())}")

    dropped_leakage: list[str] = []
    target = prepare_target_safe_inputs(scenario, dropped_leakage)
    print(f"[30] Scenario input rows: {len(target)}")
    print(target.groupby("scenario_id").size().to_string())

    env_setting = load_env_feature_setting()
    print(f"[30] Reusing Problem-2 env feature setting: {env_setting}")

    step18 = import_module(SCRIPT18_PATH, "problem3_step18")
    step19 = import_module(SCRIPT19_PATH, "problem3_step19")
    step20 = import_module(SCRIPT20_PATH, "problem3_step20")
    step21 = import_module(SCRIPT21_PATH, "problem3_step21")
    step19.MAKE_FIGURES = False
    step20.MAKE_FIGURES = False
    step21.MAKE_FIGURES = False

    print("[30] Building Top-K analog table")
    topk, selected_features, history_imp = build_problem3_topk(step18, target, env_setting)
    history_rows = int(len(history_imp))

    print("[30] Generating storm-relative initial rainfields")
    step19.TARGET_INPUT_PATH = TARGET_SAFE_PATH
    step19.TOPK_TABLE_PATH = TOPK_PATH
    step19.HISTORICAL_LIBRARY_PATH = HISTORY_PATH
    target_loaded = step19.load_target_inputs()
    topk_loaded = step19.load_topk_table()
    history_loaded = step19.load_historical_library()
    rain_initial, log_initial, initial_index, initial_diag = step19.build_initial_fields(target_loaded, topk_loaded, history_loaded)
    initial_index.to_csv(INITIAL_INDEX_PATH, index=False, encoding="utf-8-sig")
    save_initial_npz(rain_initial, log_initial, target, initial_diag["x_front_km"], initial_diag["y_left_km"])

    print("[30] Applying existing Problem-2 env EOF/PCA model")
    rain_blend, blended_index = apply_existing_eof_pca(step20, rain_initial, log_initial, initial_index)
    del rain_initial, log_initial
    gc.collect()

    print("[30] Applying extreme quantile calibration and physical caps")
    step21.HISTORICAL_LIBRARY_PATH = HISTORY_PATH
    rain_calibrated, calibrated_index = calibrate_fields(step21, topk, history_loaded, rain_blend, blended_index)
    del rain_blend
    gc.collect()
    rain_calibrated, baseline_reuse = reuse_problem2_kongrey_calibrated_baseline(rain_calibrated, target)
    print(f"[30] S0 baseline reuse: {baseline_reuse['status']} - {baseline_reuse['note']}")
    save_calibrated_npz(rain_calibrated, target)

    print("[30] Building per-timeslice metrics and QC")
    metrics = build_timeslice_metrics(rain_calibrated, target)
    step1_audit = read_csv(STEP1_AUDIT_PATH) if STEP1_AUDIT_PATH.exists() else pd.DataFrame()
    s0_compare = compare_s0_with_problem2(rain_calibrated, target, metrics)
    qc_pre = pd.DataFrame(
        {
            "scenario_id": EXPECTED_SCENARIOS,
            "p95_topk_distance": [
                float(numeric(topk.loc[topk["scenario_id"].eq(sid), "analog_distance"]).quantile(0.95))
                for sid in EXPECTED_SCENARIOS
            ],
        }
    )
    s4 = s4_checks(target, topk, history_imp, qc_pre)
    qc, qc_summary = build_qc_tables(rain_calibrated, target, topk, metrics, step1_audit, s0_compare)

    if s4.get("s4_topk_distance_warning"):
        mask = qc["scenario_id"].eq("S4") & qc["qc_level"].eq("PASS")
        qc.loc[mask, "qc_level"] = "WARNING"
        qc.loc[mask, "qc_note"] = "S4 Top-K distance is high relative to other scenarios."
        qc.to_csv(GENERATION_QC_PATH, index=False, encoding="utf-8-sig")
        qc_summary.loc[qc_summary["scenario_id"].eq("S4") & qc_summary["qc_level"].eq("PASS"), "qc_level"] = "WARNING"
        qc_summary.loc[qc_summary["scenario_id"].eq("S4"), "brief_note"] = qc.loc[qc["scenario_id"].eq("S4"), "qc_note"].iloc[0]
        qc_summary.to_csv(QC_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    levels = set(qc["qc_level"].astype(str))
    if "FAIL" in levels:
        final_status = "FAIL"
    elif "WARNING" in levels or s0_compare.get("status") == "warning":
        final_status = "WARNING"
    else:
        final_status = "SUCCESS"

    write_report(
        env_setting,
        selected_features,
        scenario_counts,
        history_rows,
        dropped_leakage,
        qc,
        qc_summary,
        s0_compare,
        s4,
        final_status,
    )

    quality = count_quality(rain_calibrated)
    print("\n========== Problem-3 Step-2 scenario rainfield generation complete ==========")
    print(f"Input scenario rows: {len(target)}")
    print("Scenario rows:")
    print(target.groupby("scenario_id").size().to_string())
    print(f"Historical library rows used: {history_rows}")
    print(f"Safe feature count: {len(selected_features)}")
    print(f"Leakage columns dropped: {sorted(set(dropped_leakage))}")
    print(f"Top-K output: {rel(TOPK_PATH)}")
    print(f"Initial rainfields output: {rel(INITIAL_NPZ_PATH)}")
    print(f"Calibrated rainfields output: {rel(CALIBRATED_NPZ_PATH)}")
    print(f"Timeslice metrics output: {rel(TIMESLICE_METRICS_PATH)}")
    print(f"QC summary output: {rel(QC_SUMMARY_PATH)}")
    print(f"Report output: {rel(REPORT_PATH)}")
    print("Per-scenario QC level:")
    print(qc[["scenario_id", "qc_level", "qc_note"]].to_string(index=False))
    print(f"Final rainfield shape: {tuple(rain_calibrated.shape)}")
    print(f"NaN/Inf/negative/all-zero counts: {quality['nan_count']} / {quality['inf_count']} / {quality['negative_count']} / {quality['all_zero_count']}")
    print(final_status)


if __name__ == "__main__":
    main()
