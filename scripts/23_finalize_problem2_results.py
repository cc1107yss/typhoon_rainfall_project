#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 final result assembly for 2024 KONG-REY and MAN-YI.

This step only reads the step-21 calibrated rainfall fields and step-22
pseudo-missing validation outputs. It does not redo retrieval, EOF/PCA,
extreme calibration, or validation.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_SAFE_INPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_halfhour_inputs_safe.csv"
TOPK_TABLE_PATH = PROJECT_ROOT / "data/processed/problem2_target_topk_similar_history.csv"
CALIBRATED_NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields.npz"
CALIBRATED_INDEX_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields_index.csv"
CALIBRATION_QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_extreme_calibration_qc_report.md"

VALIDATION_MODEL_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_model_summary.csv"
VALIDATION_EVENT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_event_summary.csv"
VALIDATION_TIMESLICE_METRICS_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_timeslice_metrics.csv"
VALIDATION_QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_pseudo_validation_qc_report.md"

FINAL_TIMESERIES_PATH = PROJECT_ROOT / "data/processed/problem2_final_timeseries_metrics.csv"
FINAL_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_final_typhoon_metrics_summary.csv"
FINAL_KEY_TIMES_PATH = PROJECT_ROOT / "data/processed/problem2_final_key_times.csv"
FINAL_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_final_results_report.md"
RUN_LOG_PATH = PROJECT_ROOT / "outputs/problem2_final_results_run_log.txt"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_final_results"

EXPECTED_TOTAL_TIMES = 974
EXPECTED_TYPHOON_COUNTS = {"KONG-REY": 421, "MAN-YI": 553}
GRID_SIZE = 201
HALFHOUR_HOURS = 0.5
AREA10_THRESHOLD = 10.0
AREA20_THRESHOLD = 20.0
EPS = 1e-12

FINAL_PREFIX_MAP = {
    "calibrated_rain_mean_mmhr": "final_rain_mean_mmhr",
    "calibrated_rain_max_mmhr": "final_rain_max_mmhr",
    "calibrated_rain_p50_mmhr": "final_rain_p50_mmhr",
    "calibrated_rain_p75_mmhr": "final_rain_p75_mmhr",
    "calibrated_rain_p90_mmhr": "final_rain_p90_mmhr",
    "calibrated_rain_p95_mmhr": "final_rain_p95_mmhr",
    "calibrated_rain_p99_mmhr": "final_rain_p99_mmhr",
    "calibrated_rain_sum_halfhour_mm": "final_rain_sum_halfhour_mm",
    "calibrated_rain_volume_proxy_mm_km2": "final_rain_volume_proxy_mm_km2",
    "calibrated_rain_area_1_km2": "final_rain_area_1_km2",
    "calibrated_rain_area_5_km2": "final_rain_area_5_km2",
    "calibrated_rain_area_10_km2": "final_rain_area_10_km2",
    "calibrated_rain_area_20_km2": "final_rain_area_20_km2",
    "calibrated_heavy_rain_fraction_10": "final_heavy_rain_fraction_10",
    "calibrated_centroid_x_front_km": "final_centroid_x_front_km",
    "calibrated_centroid_y_left_km": "final_centroid_y_left_km",
    "calibrated_centroid_offset_km": "final_centroid_offset_km",
    "calibrated_centroid_angle_deg": "final_centroid_angle_deg",
    "calibrated_asym_front_back_ratio": "final_asym_front_back_ratio",
    "calibrated_asym_left_right_ratio": "final_asym_left_right_ratio",
    "calibrated_quad_front_left_sum": "final_quad_front_left_sum",
    "calibrated_quad_front_right_sum": "final_quad_front_right_sum",
    "calibrated_quad_back_left_sum": "final_quad_back_left_sum",
    "calibrated_quad_back_right_sum": "final_quad_back_right_sum",
    "calibrated_quad_front_left_ratio": "final_quad_front_left_ratio",
    "calibrated_quad_front_right_ratio": "final_quad_front_right_ratio",
    "calibrated_quad_back_left_ratio": "final_quad_back_left_ratio",
    "calibrated_quad_back_right_ratio": "final_quad_back_right_ratio",
    "calibrated_anisotropy": "final_anisotropy",
    "calibrated_rain_radius_r50_km": "final_rain_radius_r50_km",
    "calibrated_rain_radius_r80_km": "final_rain_radius_r80_km",
    "calibrated_rain_radius_r90_km": "final_rain_radius_r90_km",
    "calibrated_rain_band_width_km": "final_rain_band_width_km",
}


# =========================
# Helpers
# =========================


def resolve_project_path(path: object = PROJECT_ROOT) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def safe_name(value: object) -> str:
    return str(value).replace("-", "_").replace(" ", "_")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def format_time(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def stats_summary(values: object) -> Dict[str, float]:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return {"min": np.nan, "mean": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "min": float(s.min()),
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def stats_text(values: object) -> str:
    stats = stats_summary(values)
    return ", ".join(f"{k}={v:.6g}" if np.isfinite(v) else f"{k}=NA" for k, v in stats.items())


def cell_area_from_grid(x_front_km: np.ndarray, y_left_km: np.ndarray) -> float:
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    return dx * dy


def count_field_quality(fields: np.ndarray) -> Dict[str, int]:
    arr = np.asarray(fields)
    return {
        "nan": int(np.count_nonzero(np.isnan(arr))),
        "inf": int(np.count_nonzero(np.isinf(arr))),
        "negative": int(np.count_nonzero(np.isfinite(arr) & (arr < 0.0))),
        "all_zero": int(np.count_nonzero(np.all(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) == 0.0, axis=(1, 2)))),
    }


def argmax_row(df: pd.DataFrame, col: str) -> pd.Series:
    s = numeric(df, col)
    if s.notna().sum() == 0:
        raise RuntimeError(f"Cannot identify maximum row for {col}: all values are NaN.")
    return df.loc[s.idxmax()]


def argmin_row(df: pd.DataFrame, col: str) -> pd.Series:
    s = numeric(df, col)
    if s.notna().sum() == 0:
        raise RuntimeError(f"Cannot identify minimum row for {col}: all values are NaN.")
    return df.loc[s.idxmin()]


def absmin_row(df: pd.DataFrame, col: str) -> pd.Series:
    s = numeric(df, col).abs()
    if s.notna().sum() == 0:
        raise RuntimeError(f"Cannot identify abs-minimum row for {col}: all values are NaN.")
    return df.loc[s.idxmin()]


def add_issue(issues: List[str], message: str) -> None:
    issues.append(message)
    print(f"[issue] {message}")


def write_run_log(issues: Sequence[str], status: str, extra: Optional[str] = None) -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Problem-2 Final Results Run Log",
        "",
        f"- status: {status}",
        f"- project_root: {PROJECT_ROOT}",
    ]
    if extra:
        lines.extend(["", "## Extra", extra])
    lines.append("")
    lines.append("## Issues")
    if issues:
        lines.extend(f"- {item}" for item in issues)
    else:
        lines.append("- None")
    RUN_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Loaders
# =========================


def load_calibrated_npz(issues: List[str]) -> Dict[str, np.ndarray]:
    if not CALIBRATED_NPZ_PATH.exists():
        add_issue(issues, f"calibrated NPZ 缺失: {CALIBRATED_NPZ_PATH}")
        raise FileNotFoundError(f"Missing calibrated NPZ: {CALIBRATED_NPZ_PATH}")

    required = [
        "rain_mmhr_calibrated",
        "log_rain_calibrated",
        "calibration_scale_field",
        "target_id",
        "typhoon_name",
        "time",
        "lat",
        "lon_180",
        "move_dir_deg",
        "x_front_km",
        "y_left_km",
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "target_rain_area_20_km2",
    ]
    optional = ["rain_mmhr_blend"]
    with np.load(CALIBRATED_NPZ_PATH, allow_pickle=True) as z:
        missing = [k for k in required if k not in z.files]
        if missing:
            add_issue(issues, f"calibrated NPZ 缺少必要数组: {missing}")
            raise RuntimeError(f"Calibrated NPZ is missing required arrays: {missing}")
        out = {k: z[k] for k in required + [k for k in optional if k in z.files]}

    rain = np.asarray(out["rain_mmhr_calibrated"], dtype=np.float32)
    log_rain = np.asarray(out["log_rain_calibrated"], dtype=np.float32)
    scale = np.asarray(out["calibration_scale_field"], dtype=np.float32)
    if rain.ndim != 3 or rain.shape[1:] != (GRID_SIZE, GRID_SIZE):
        add_issue(issues, f"calibrated 场 shape 异常: {rain.shape}")
        raise RuntimeError(f"Unexpected calibrated rain shape: {rain.shape}")
    if rain.shape != log_rain.shape or rain.shape != scale.shape:
        add_issue(issues, f"NPZ rain/log/scale shape 不一致: rain={rain.shape}, log={log_rain.shape}, scale={scale.shape}")
        raise RuntimeError("Calibrated NPZ array shapes are inconsistent.")
    if rain.shape[0] != EXPECTED_TOTAL_TIMES:
        add_issue(issues, f"calibrated 场时刻数异常: {rain.shape[0]}，期望 {EXPECTED_TOTAL_TIMES}")
        raise RuntimeError(f"Unexpected calibrated time count: {rain.shape[0]}")

    out["rain_mmhr_calibrated"] = rain
    out["log_rain_calibrated"] = log_rain
    out["calibration_scale_field"] = scale
    if "rain_mmhr_blend" in out:
        out["rain_mmhr_blend"] = np.asarray(out["rain_mmhr_blend"], dtype=np.float32)
    return out


def load_calibrated_index(issues: List[str]) -> pd.DataFrame:
    if not CALIBRATED_INDEX_PATH.exists():
        add_issue(issues, f"calibrated index 缺失: {CALIBRATED_INDEX_PATH}")
        raise FileNotFoundError(f"Missing calibrated index: {CALIBRATED_INDEX_PATH}")
    df = pd.read_csv(CALIBRATED_INDEX_PATH)
    if len(df) != EXPECTED_TOTAL_TIMES:
        add_issue(issues, f"calibrated index 行数异常: {len(df)}，期望 {EXPECTED_TOTAL_TIMES}")
        raise RuntimeError(f"Unexpected calibrated index row count: {len(df)}")
    if "field_index" not in df.columns or "target_id" not in df.columns or "typhoon_name" not in df.columns:
        add_issue(issues, "calibrated index 缺少 field_index/target_id/typhoon_name 基础字段")
        raise RuntimeError("Calibrated index is missing required identifier columns.")
    return df


def load_validation_summary(issues: List[str]) -> pd.DataFrame:
    if not VALIDATION_MODEL_SUMMARY_PATH.exists():
        add_issue(issues, f"22 号模型验证 summary 缺失: {VALIDATION_MODEL_SUMMARY_PATH}")
        raise FileNotFoundError(f"Missing validation model summary: {VALIDATION_MODEL_SUMMARY_PATH}")
    df = pd.read_csv(VALIDATION_MODEL_SUMMARY_PATH)
    if "model_version" not in df.columns:
        add_issue(issues, "22 号模型验证 summary 缺少 model_version 字段")
        raise RuntimeError("Validation summary is missing model_version.")
    if not df["model_version"].astype(str).eq("calibrated").any():
        add_issue(issues, "22 号模型验证 summary 中没有 calibrated 行")
        raise RuntimeError("Validation summary has no calibrated row.")
    return df


def validate_input_consistency(
    calibrated: Mapping[str, np.ndarray],
    index_df: pd.DataFrame,
    issues: List[str],
) -> None:
    rain = np.asarray(calibrated["rain_mmhr_calibrated"])
    if rain.shape[0] != len(index_df):
        add_issue(issues, f"NPZ shape 与 index 行数不一致: {rain.shape[0]} vs {len(index_df)}")
        raise RuntimeError("NPZ time dimension and calibrated index row count mismatch.")

    index_target = index_df.sort_values("field_index")["target_id"].astype(str).to_numpy()
    npz_target = np.asarray(calibrated["target_id"]).astype(str)
    if len(index_target) != len(npz_target) or not np.array_equal(index_target, npz_target):
        add_issue(issues, "target_id 顺序不一致: calibrated NPZ 与 calibrated index 排序后不完全一致")
        raise RuntimeError("target_id order mismatch between NPZ and index.")

    counts = index_df["typhoon_name"].value_counts().to_dict()
    for name, expected in EXPECTED_TYPHOON_COUNTS.items():
        actual = int(counts.get(name, 0))
        if actual != expected:
            add_issue(issues, f"{name} 行数异常: {actual}，期望 {expected}")
            raise RuntimeError(f"Unexpected row count for {name}: {actual}")
    unexpected = sorted(set(counts) - set(EXPECTED_TYPHOON_COUNTS))
    if unexpected:
        add_issue(issues, f"发现非目标台风名称: {unexpected}")
        raise RuntimeError(f"Unexpected typhoon names in calibrated index: {unexpected}")

    quality = count_field_quality(rain)
    if quality["nan"] > 0 or quality["inf"] > 0 or quality["negative"] > 0:
        add_issue(issues, f"rain_mmhr_calibrated 存在 NaN/Inf/负值: {quality}")
        raise RuntimeError("Calibrated rainfall has NaN, Inf, or negative values.")
    if quality["all_zero"] > 0:
        add_issue(issues, f"rain_mmhr_calibrated 存在全零场数量: {quality['all_zero']}")
        raise RuntimeError("Calibrated rainfall has all-zero fields.")

    if "calibration_ok" not in index_df.columns:
        add_issue(issues, "calibrated index 缺少 calibration_ok 字段")
        raise RuntimeError("Calibrated index is missing calibration_ok.")
    calibration_ok = bool_series(index_df, "calibration_ok")
    if not calibration_ok.all():
        n_false = int((~calibration_ok).sum())
        add_issue(issues, f"calibration_ok 存在 False: {n_false} 行")
        raise RuntimeError("Some calibrated fields are marked calibration_ok=False.")


# =========================
# Final tables
# =========================


def compute_final_timeseries_metrics(
    calibrated: Mapping[str, np.ndarray],
    index_df: pd.DataFrame,
    issues: List[str],
) -> pd.DataFrame:
    df = index_df.sort_values("field_index").reset_index(drop=True).copy()
    missing = [c for c in FINAL_PREFIX_MAP if c not in df.columns]
    if missing:
        add_issue(issues, f"calibrated index 缺少用于 final_ 的字段: {missing}")
        raise RuntimeError("Calibrated index is missing final metric source columns.")

    for source, target in FINAL_PREFIX_MAP.items():
        df[target] = df[source]

    for name, sub_idx in df.groupby("typhoon_name", sort=False).groups.items():
        order = df.loc[sub_idx].sort_values("time").index
        df.loc[order, "cumulative_volume_proxy_mm_km2"] = numeric(df.loc[order], "final_rain_volume_proxy_mm_km2").cumsum()
        df.loc[order, "cumulative_area_time_10_km2_h"] = (numeric(df.loc[order], "final_rain_area_10_km2") * HALFHOUR_HOURS).cumsum()
        df.loc[order, "cumulative_area_time_20_km2_h"] = (numeric(df.loc[order], "final_rain_area_20_km2") * HALFHOUR_HOURS).cumsum()
        df.loc[order, "cumulative_max_rain_mmhr_so_far"] = numeric(df.loc[order], "final_rain_max_mmhr").cummax()
        df.loc[order, "cumulative_max_p95_mmhr_so_far"] = numeric(df.loc[order], "final_rain_p95_mmhr").cummax()

    required_order = [
        "field_index",
        "target_id",
        "typhoon_name",
        "time",
        "lat",
        "lon_180",
        "WND",
        "PRES",
        "intensity",
        "move_speed_kmh",
        "move_dir_deg",
        "signed_coast_dist_km",
        "is_land",
        "life_progress",
        "final_rain_mean_mmhr",
        "final_rain_max_mmhr",
        "final_rain_p50_mmhr",
        "final_rain_p75_mmhr",
        "final_rain_p90_mmhr",
        "final_rain_p95_mmhr",
        "final_rain_p99_mmhr",
        "final_rain_sum_halfhour_mm",
        "final_rain_volume_proxy_mm_km2",
        "final_rain_area_1_km2",
        "final_rain_area_5_km2",
        "final_rain_area_10_km2",
        "final_rain_area_20_km2",
        "final_heavy_rain_fraction_10",
        "final_centroid_x_front_km",
        "final_centroid_y_left_km",
        "final_centroid_offset_km",
        "final_centroid_angle_deg",
        "final_asym_front_back_ratio",
        "final_asym_left_right_ratio",
        "final_quad_front_left_sum",
        "final_quad_front_right_sum",
        "final_quad_back_left_sum",
        "final_quad_back_right_sum",
        "final_quad_front_left_ratio",
        "final_quad_front_right_ratio",
        "final_quad_back_left_ratio",
        "final_quad_back_right_ratio",
        "final_anisotropy",
        "final_rain_radius_r50_km",
        "final_rain_radius_r80_km",
        "final_rain_radius_r90_km",
        "final_rain_band_width_km",
        "cumulative_volume_proxy_mm_km2",
        "cumulative_area_time_10_km2_h",
        "cumulative_area_time_20_km2_h",
        "cumulative_max_rain_mmhr_so_far",
        "cumulative_max_p95_mmhr_so_far",
        "blend_rain_max_mmhr",
        "blend_rain_p95_mmhr",
        "calibrated_rain_max_mmhr",
        "calibrated_rain_p95_mmhr",
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "corr_blend_calibrated",
        "max_scale_factor",
        "calibration_ok",
    ]
    missing_required = [c for c in required_order if c not in df.columns]
    if missing_required:
        add_issue(issues, f"最终逐时次表缺少必需字段: {missing_required}")
        raise RuntimeError("Final timeseries table is missing required columns.")

    extra = [c for c in df.columns if c not in required_order]
    return df[required_order + extra].copy()


def compute_cumulative_fields(
    rain_calibrated: np.ndarray,
    timeseries_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    fields: Dict[str, np.ndarray] = {}
    for name, sub in timeseries_df.groupby("typhoon_name", sort=False):
        indices = sub.sort_values("time")["field_index"].astype(int).to_numpy()
        fields[str(name)] = np.sum(rain_calibrated[indices] * HALFHOUR_HOURS, axis=0)
    return fields


def compute_max_fields(
    rain_calibrated: np.ndarray,
    timeseries_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    fields: Dict[str, np.ndarray] = {}
    for name, sub in timeseries_df.groupby("typhoon_name", sort=False):
        indices = sub.sort_values("time")["field_index"].astype(int).to_numpy()
        fields[str(name)] = np.max(rain_calibrated[indices], axis=0)
    return fields


def compute_duration_fields(
    rain_calibrated: np.ndarray,
    timeseries_df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    fields: Dict[str, Dict[str, np.ndarray]] = {}
    for name, sub in timeseries_df.groupby("typhoon_name", sort=False):
        indices = sub.sort_values("time")["field_index"].astype(int).to_numpy()
        storm = rain_calibrated[indices]
        fields[str(name)] = {
            "duration10_h": np.sum(storm >= AREA10_THRESHOLD, axis=0).astype(np.float32) * HALFHOUR_HOURS,
            "duration20_h": np.sum(storm >= AREA20_THRESHOLD, axis=0).astype(np.float32) * HALFHOUR_HOURS,
        }
    return fields


def compute_quadrant_contribution(
    cumulative_field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> Dict[str, float]:
    x_grid, y_grid = np.meshgrid(x_front_km, y_left_km)
    weights = np.where(np.isfinite(cumulative_field) & (cumulative_field > 0.0), cumulative_field, 0.0)
    total = float(np.sum(weights))
    front = x_grid > 0.0
    back = x_grid < 0.0
    left = y_grid > 0.0
    right = y_grid < 0.0
    sums = {
        "front_left": float(np.sum(weights[front & left])),
        "front_right": float(np.sum(weights[front & right])),
        "back_left": float(np.sum(weights[back & left])),
        "back_right": float(np.sum(weights[back & right])),
    }
    ratios = {f"{k}_ratio": safe_div(v, total) for k, v in sums.items()}
    dominant = max(sums, key=sums.get) if total > EPS else "NA"
    return {**sums, **ratios, "total": total, "dominant": dominant}


def calibrated_validation_row(validation_summary: pd.DataFrame) -> pd.Series:
    return validation_summary.loc[validation_summary["model_version"].astype(str).eq("calibrated")].iloc[0]


def compute_typhoon_summary(
    timeseries_df: pd.DataFrame,
    rain_calibrated: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    validation_summary: pd.DataFrame,
    cumulative_fields: Mapping[str, np.ndarray],
    duration_fields: Mapping[str, Mapping[str, np.ndarray]],
    issues: List[str],
) -> pd.DataFrame:
    cell_area = cell_area_from_grid(x_front_km, y_left_km)
    validation = calibrated_validation_row(validation_summary)
    validation_cols = {
        "validation_calibrated_rmse_mean": "rmse_mean",
        "validation_calibrated_corr_mean": "corr_mean",
        "validation_calibrated_p95_abs_error_mean": "abs_error_rain_p95_mean",
        "validation_calibrated_p99_abs_error_mean": "abs_error_rain_p99_mean",
        "validation_calibrated_area10_abs_error_mean": "abs_error_area_10_mean",
        "validation_calibrated_csi10_mean": "csi10_mean",
        "validation_calibrated_f1_10_mean": "f1_10_mean",
    }

    rows: List[Dict[str, object]] = []
    for name, sub in timeseries_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        if sub["time_dt"].isna().any():
            add_issue(issues, f"{name} 存在无法解析的 time")
            raise RuntimeError(f"Cannot parse times for {name}.")

        wnd_row = argmax_row(sub, "WND")
        pres_row = argmin_row(sub, "PRES")
        p95_row = argmax_row(sub, "final_rain_p95_mmhr")
        p99_row = argmax_row(sub, "final_rain_p99_mmhr")
        rmax_row = argmax_row(sub, "final_rain_max_mmhr")
        area10_row = argmax_row(sub, "final_rain_area_10_km2")
        area20_row = argmax_row(sub, "final_rain_area_20_km2")

        cumulative = np.asarray(cumulative_fields[str(name)], dtype=float)
        duration10 = np.asarray(duration_fields[str(name)]["duration10_h"], dtype=float)
        duration20 = np.asarray(duration_fields[str(name)]["duration20_h"], dtype=float)
        quadrant = compute_quadrant_contribution(cumulative, x_front_km, y_left_km)

        row: Dict[str, object] = {
            "typhoon_name": name,
            "start_time": format_time(sub["time_dt"].iloc[0]),
            "end_time": format_time(sub["time_dt"].iloc[-1]),
            "n_times": int(len(sub)),
            "duration_hours": float(len(sub) * HALFHOUR_HOURS),
            "lat_min": float(numeric(sub, "lat").min()),
            "lat_max": float(numeric(sub, "lat").max()),
            "lon_min": float(numeric(sub, "lon_180").min()),
            "lon_max": float(numeric(sub, "lon_180").max()),
            "WND_max": float(wnd_row["WND"]),
            "WND_max_time": format_time(wnd_row["time"]),
            "PRES_min": float(pres_row["PRES"]),
            "PRES_min_time": format_time(pres_row["time"]),
            "mean_signed_coast_dist_km": float(numeric(sub, "signed_coast_dist_km").mean()),
            "land_time_fraction": float(numeric(sub, "is_land").mean()),
            "final_rain_max_mmhr_max": float(rmax_row["final_rain_max_mmhr"]),
            "final_rain_max_mmhr_time": format_time(rmax_row["time"]),
            "final_rain_p95_mmhr_max": float(p95_row["final_rain_p95_mmhr"]),
            "final_rain_p95_mmhr_time": format_time(p95_row["time"]),
            "final_rain_p99_mmhr_max": float(p99_row["final_rain_p99_mmhr"]),
            "final_rain_p99_mmhr_time": format_time(p99_row["time"]),
            "final_rain_area_10_km2_max": float(area10_row["final_rain_area_10_km2"]),
            "final_rain_area_10_km2_time": format_time(area10_row["time"]),
            "final_rain_area_20_km2_max": float(area20_row["final_rain_area_20_km2"]),
            "final_rain_area_20_km2_time": format_time(area20_row["time"]),
            "total_volume_proxy_mm_km2": float(numeric(sub, "final_rain_volume_proxy_mm_km2").sum()),
            "total_sum_halfhour_mm": float(numeric(sub, "final_rain_sum_halfhour_mm").sum()),
            "max_grid_cumulative_halfhour_mm": float(np.nanmax(cumulative)),
            "mean_grid_cumulative_halfhour_mm": float(np.nanmean(cumulative)),
            "p95_grid_cumulative_halfhour_mm": float(np.nanpercentile(cumulative, 95)),
            "p99_grid_cumulative_halfhour_mm": float(np.nanpercentile(cumulative, 99)),
            "max_duration_10_h": float(np.nanmax(duration10)),
            "mean_duration_10_h": float(np.nanmean(duration10)),
            "area_with_duration_10_ge_1h_km2": float(np.count_nonzero(duration10 >= 1.0) * cell_area),
            "area_with_duration_10_ge_3h_km2": float(np.count_nonzero(duration10 >= 3.0) * cell_area),
            "area_with_duration_10_ge_6h_km2": float(np.count_nonzero(duration10 >= 6.0) * cell_area),
            "max_duration_20_h": float(np.nanmax(duration20)),
            "mean_duration_20_h": float(np.nanmean(duration20)),
            "area_with_duration_20_ge_1h_km2": float(np.count_nonzero(duration20 >= 1.0) * cell_area),
            "area_with_duration_20_ge_3h_km2": float(np.count_nonzero(duration20 >= 3.0) * cell_area),
            "area_with_duration_20_ge_6h_km2": float(np.count_nonzero(duration20 >= 6.0) * cell_area),
            "centroid_offset_km_mean": float(numeric(sub, "final_centroid_offset_km").mean()),
            "centroid_offset_km_max": float(numeric(sub, "final_centroid_offset_km").max()),
            "anisotropy_mean": float(numeric(sub, "final_anisotropy").mean()),
            "anisotropy_max": float(numeric(sub, "final_anisotropy").max()),
            "rain_radius_r50_km_mean": float(numeric(sub, "final_rain_radius_r50_km").mean()),
            "rain_radius_r80_km_mean": float(numeric(sub, "final_rain_radius_r80_km").mean()),
            "rain_radius_r90_km_mean": float(numeric(sub, "final_rain_radius_r90_km").mean()),
            "rain_band_width_km_mean": float(numeric(sub, "final_rain_band_width_km").mean()),
            "front_back_ratio_mean": float(numeric(sub, "final_asym_front_back_ratio").mean()),
            "left_right_ratio_mean": float(numeric(sub, "final_asym_left_right_ratio").mean()),
            "dominant_quadrant_by_total_rain": quadrant["dominant"],
            "quadrant_front_left_ratio_total": quadrant["front_left_ratio"],
            "quadrant_front_right_ratio_total": quadrant["front_right_ratio"],
            "quadrant_back_left_ratio_total": quadrant["back_left_ratio"],
            "quadrant_back_right_ratio_total": quadrant["back_right_ratio"],
        }
        for out_col, source_col in validation_cols.items():
            row[out_col] = float(validation.get(source_col, np.nan))
        rows.append(row)

    return pd.DataFrame(rows)


def identify_key_times(timeseries_df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name, sub in timeseries_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        selectors: List[Tuple[str, pd.Series, str]] = [
            ("start", sub.iloc[0], "first half-hour field of the target storm period"),
            ("wnd_max", argmax_row(sub, "WND"), "maximum WND"),
            ("pres_min", argmin_row(sub, "PRES"), "minimum PRES"),
            ("rain_p95_max", argmax_row(sub, "final_rain_p95_mmhr"), "maximum final_rain_p95_mmhr"),
            ("rain_max_max", argmax_row(sub, "final_rain_max_mmhr"), "maximum final_rain_max_mmhr"),
            ("area10_max", argmax_row(sub, "final_rain_area_10_km2"), "maximum final_rain_area_10_km2"),
            ("near_land_or_min_coast_dist", absmin_row(sub, "signed_coast_dist_km"), "minimum absolute signed coast distance"),
            ("end", sub.iloc[-1], "last half-hour field of the target storm period"),
        ]
        for key_type, row, reason in selectors:
            try:
                rows.append(
                    {
                        "typhoon_name": name,
                        "key_type": key_type,
                        "time": format_time(row["time"]),
                        "field_index": int(row["field_index"]),
                        "lat": row.get("lat", np.nan),
                        "lon_180": row.get("lon_180", np.nan),
                        "WND": row.get("WND", np.nan),
                        "PRES": row.get("PRES", np.nan),
                        "signed_coast_dist_km": row.get("signed_coast_dist_km", np.nan),
                        "final_rain_max_mmhr": row.get("final_rain_max_mmhr", np.nan),
                        "final_rain_p95_mmhr": row.get("final_rain_p95_mmhr", np.nan),
                        "final_rain_p99_mmhr": row.get("final_rain_p99_mmhr", np.nan),
                        "final_rain_area_10_km2": row.get("final_rain_area_10_km2", np.nan),
                        "final_rain_area_20_km2": row.get("final_rain_area_20_km2", np.nan),
                        "final_centroid_offset_km": row.get("final_centroid_offset_km", np.nan),
                        "final_anisotropy": row.get("final_anisotropy", np.nan),
                        "reason": reason,
                    }
                )
            except Exception as exc:
                add_issue(issues, f"{name} 关键时刻 {key_type} 无法识别: {exc}")
                raise
    out = pd.DataFrame(rows)
    for name in EXPECTED_TYPHOON_COUNTS:
        got = set(out.loc[out["typhoon_name"].eq(name), "key_type"])
        need = {
            "start",
            "wnd_max",
            "pres_min",
            "rain_p95_max",
            "rain_max_max",
            "area10_max",
            "near_land_or_min_coast_dist",
            "end",
        }
        missing = sorted(need - got)
        if missing:
            add_issue(issues, f"{name} 关键时刻类型缺失: {missing}")
            raise RuntimeError(f"Missing key times for {name}: {missing}")
    return out


def save_final_tables(
    timeseries_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    key_times_df: pd.DataFrame,
    issues: List[str],
) -> None:
    try:
        FINAL_TIMESERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        timeseries_df.to_csv(FINAL_TIMESERIES_PATH, index=False)
        summary_df.to_csv(FINAL_SUMMARY_PATH, index=False)
        key_times_df.to_csv(FINAL_KEY_TIMES_PATH, index=False)
    except Exception as exc:
        add_issue(issues, f"输出文件写入失败: {exc}")
        raise


# =========================
# Figures
# =========================


def _prepare_figure_dir() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _storm_sorted(df: pd.DataFrame, name: object) -> pd.DataFrame:
    sub = df.loc[df["typhoon_name"].eq(name)].copy()
    sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
    return sub.sort_values("time_dt")


def _format_time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
    ax.grid(True, linewidth=0.35, alpha=0.35)


def _elapsed_hours(sub: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(sub["time"], errors="coerce")
    return ((t - t.iloc[0]).dt.total_seconds() / 3600.0).to_numpy()


def _plot_field(
    ax,
    field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    im = ax.imshow(
        field,
        origin="lower",
        extent=[float(x_front_km[0]), float(x_front_km[-1]), float(y_left_km[0]), float(y_left_km[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    line_color = "white" if cmap not in {"RdBu_r", "coolwarm"} else "black"
    ax.axvline(0, color=line_color, linestyle="--", linewidth=0.8, alpha=0.75)
    ax.axhline(0, color=line_color, linestyle="--", linewidth=0.8, alpha=0.75)
    ax.scatter([0], [0], marker="+", color="black", s=55, linewidths=1.2, zorder=4)
    ax.set_xlabel("x_front_km")
    ax.set_ylabel("y_left_km")
    ax.set_title(title, fontsize=8.5)
    return im


def make_track_intensity_rain_timeseries(timeseries_df: pd.DataFrame) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    for name in EXPECTED_TYPHOON_COUNTS:
        sub = _storm_sorted(timeseries_df, name)
        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
        panels = [
            ("WND", "WND"),
            ("PRES", "PRES"),
            ("signed_coast_dist_km", "Coast dist km"),
            ("final_rain_p95_mmhr", "Final P95 mm/hr"),
            ("final_rain_area_10_km2", "Area >=10 km2"),
        ]
        for ax, (col, ylabel) in zip(axes, panels):
            ax.plot(sub["time_dt"], numeric(sub, col), linewidth=1.2)
            ax.set_ylabel(ylabel)
            _format_time_axis(ax)
        axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.6)
        if "is_land" in sub.columns:
            land = numeric(sub, "is_land") > 0
            if land.any():
                axes[2].fill_between(sub["time_dt"], axes[2].get_ylim()[0], axes[2].get_ylim()[1], where=land, color="0.85", alpha=0.45, step="mid")
        axes[0].set_title(f"{name} track, intensity and final rainfall metrics")
        axes[-1].set_xlabel("Time")
        path = FIGURE_DIR / f"{safe_name(name)}_final_track_intensity_rain_timeseries.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def make_final_rain_metrics_timeseries(timeseries_df: pd.DataFrame) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    for name in EXPECTED_TYPHOON_COUNTS:
        sub = _storm_sorted(timeseries_df, name)
        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
        panels = [
            ("final_rain_max_mmhr", "Rmax mm/hr"),
            ("final_rain_p95_mmhr", "P95 mm/hr"),
            ("final_rain_p99_mmhr", "P99 mm/hr"),
            ("final_rain_area_10_km2", "Area >=10 km2"),
            ("final_rain_area_20_km2", "Area >=20 km2"),
        ]
        for ax, (col, ylabel) in zip(axes, panels):
            ax.plot(sub["time_dt"], numeric(sub, col), linewidth=1.2)
            ax.set_ylabel(ylabel)
            _format_time_axis(ax)
        axes[0].set_title(f"{name} final rainfall metric time series")
        axes[-1].set_xlabel("Time")
        path = FIGURE_DIR / f"{safe_name(name)}_final_rain_metrics_timeseries.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _representative_key_rows(key_times_df: pd.DataFrame, name: object) -> pd.DataFrame:
    sub = key_times_df.loc[key_times_df["typhoon_name"].eq(name)].copy()
    ordered_types = ["start", "wnd_max", "rain_p95_max", "rain_max_max", "area10_max", "end"]
    rows = []
    for key_type in ordered_types:
        part = sub.loc[sub["key_type"].eq(key_type)]
        if len(part) == 0 and key_type == "wnd_max":
            part = sub.loc[sub["key_type"].eq("pres_min")]
        if len(part) == 0:
            raise RuntimeError(f"Missing representative key type {key_type} for {name}.")
        rows.append(part.iloc[0])
    return pd.DataFrame(rows)


def make_representative_field_figures(
    timeseries_df: pd.DataFrame,
    key_times_df: pd.DataFrame,
    rain_calibrated: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    for name in EXPECTED_TYPHOON_COUNTS:
        rep = _representative_key_rows(key_times_df, name)
        indices = rep["field_index"].astype(int).to_numpy()
        selected = rain_calibrated[indices]
        vmax = max(float(np.nanpercentile(selected, 99.5)), 1.0)
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
        for ax, (_, row), field in zip(axes.ravel(), rep.iterrows(), selected):
            title = (
                f"{name} {row['key_type']}\n"
                f"{row['time']} WND={float(row['WND']):.1f} PRES={float(row['PRES']):.0f}\n"
                f"P95={float(row['final_rain_p95_mmhr']):.2f} Rmax={float(row['final_rain_max_mmhr']):.2f}"
            )
            im = _plot_field(ax, field, x_front_km, y_left_km, title, vmax=vmax)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label="rain mm/hr")
        path = FIGURE_DIR / f"{safe_name(name)}_final_representative_fields.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def make_cumulative_figures(
    cumulative_fields: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    for name in EXPECTED_TYPHOON_COUNTS:
        field = np.asarray(cumulative_fields[name])
        vmax = max(float(np.nanpercentile(field, 99.3)), 1.0)
        fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
        im = _plot_field(ax, field, x_front_km, y_left_km, f"{name} cumulative rainfall", cmap="YlGnBu", vmax=vmax)
        fig.colorbar(im, ax=ax, label="cumulative half-hour mm")
        path = FIGURE_DIR / f"{safe_name(name)}_final_cumulative_rain_storm_relative.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def make_max_rain_figures(
    max_fields: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    for name in EXPECTED_TYPHOON_COUNTS:
        field = np.asarray(max_fields[name])
        vmax = max(float(np.nanpercentile(field, 99.5)), 1.0)
        fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
        im = _plot_field(ax, field, x_front_km, y_left_km, f"{name} maximum half-hour rain rate", cmap="viridis", vmax=vmax)
        fig.colorbar(im, ax=ax, label="max rain mm/hr")
        path = FIGURE_DIR / f"{safe_name(name)}_final_max_rain_storm_relative.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def make_duration_figures(
    duration_fields: Mapping[str, Mapping[str, np.ndarray]],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    specs = [
        ("duration10_h", "duration >=10 mm/hr", "duration10"),
        ("duration20_h", "duration >=20 mm/hr", "duration20"),
    ]
    for name in EXPECTED_TYPHOON_COUNTS:
        for key, title, suffix in specs:
            field = np.asarray(duration_fields[name][key])
            vmax = max(float(np.nanpercentile(field, 99.5)), 0.5)
            fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
            im = _plot_field(ax, field, x_front_km, y_left_km, f"{name} {title}", cmap="plasma", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, label="hours")
            path = FIGURE_DIR / f"{safe_name(name)}_final_{suffix}_storm_relative.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path)
    return paths


def make_quadrant_figures(
    cumulative_fields: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []
    labels = ["front_left", "front_right", "back_left", "back_right"]
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    for name in EXPECTED_TYPHOON_COUNTS:
        q = compute_quadrant_contribution(cumulative_fields[name], x_front_km, y_left_km)
        values = [float(q[f"{label}_ratio"]) * 100.0 for label in labels]
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        ax.bar([label.replace("_", "\n") for label in labels], values, color=colors)
        ax.set_ylabel("Contribution (%)")
        ax.set_ylim(0, max(values) * 1.25 if values else 100)
        ax.set_title(f"{name} cumulative rainfall quadrant contribution")
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        path = FIGURE_DIR / f"{safe_name(name)}_final_quadrant_contribution.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def make_two_typhoon_compare_figures(
    timeseries_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> List[Path]:
    _prepare_figure_dir()
    paths: List[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    for name in EXPECTED_TYPHOON_COUNTS:
        sub = _storm_sorted(timeseries_df, name)
        hours = _elapsed_hours(sub)
        axes[0].plot(hours, numeric(sub, "final_rain_p95_mmhr"), linewidth=1.2, label=name)
        axes[1].plot(hours, numeric(sub, "final_rain_area_10_km2"), linewidth=1.2, label=name)
    axes[0].set_ylabel("Final P95 mm/hr")
    axes[1].set_ylabel("Area >=10 km2")
    axes[1].set_xlabel("Hours since storm-period start")
    for ax in axes:
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.legend()
    axes[0].set_title("KONG-REY vs MAN-YI final rainfall time series")
    path = FIGURE_DIR / "problem2_final_two_typhoons_timeseries_compare.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    metrics = [
        ("final_rain_p95_mmhr_max", "Max P95 mm/hr"),
        ("final_rain_area_10_km2_max", "Max area >=10 km2"),
        ("total_volume_proxy_mm_km2", "Total volume proxy"),
        ("max_duration_10_h", "Max duration >=10 h"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    names = list(EXPECTED_TYPHOON_COUNTS)
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = [float(summary_df.loc[summary_df["typhoon_name"].eq(name), col].iloc[0]) for name in names]
        ax.bar(names, vals, color=["#4E79A7", "#F28E2B"])
        ax.set_title(title)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    fig.suptitle("Two-typhoon final summary comparison")
    path = FIGURE_DIR / "problem2_final_two_typhoons_summary_compare.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def make_validation_comparison_figure(validation_summary: pd.DataFrame) -> List[Path]:
    _prepare_figure_dir()
    df = validation_summary.copy()
    order = ["initial", "blend", "calibrated"]
    df["model_version"] = pd.Categorical(df["model_version"].astype(str), categories=order, ordered=True)
    df = df.sort_values("model_version")
    panels = [
        ("abs_error_rain_p95_mean", "Mean absolute P95 error"),
        ("abs_error_rain_p99_mean", "Mean absolute P99 error"),
        ("abs_error_area_10_mean", "Mean absolute area10 error"),
        ("csi10_mean", "CSI10"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    colors = ["#9C755F", "#4E79A7", "#59A14F"]
    for ax, (col, title) in zip(axes.ravel(), panels):
        vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        ax.bar(df["model_version"].astype(str), vals, color=colors[: len(df)])
        ax.set_title(title)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    fig.suptitle("Pseudo-missing validation model comparison")
    path = FIGURE_DIR / "problem2_final_validation_model_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return [path]


# =========================
# Report
# =========================


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _metric_row(summary_df: pd.DataFrame, name: str) -> pd.Series:
    rows = summary_df.loc[summary_df["typhoon_name"].eq(name)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one summary row for {name}, got {len(rows)}.")
    return rows.iloc[0]


def _winner(summary_df: pd.DataFrame, col: str, higher_is_larger: bool = True) -> str:
    values = pd.to_numeric(summary_df[col], errors="coerce")
    idx = values.idxmax() if higher_is_larger else values.idxmin()
    return str(summary_df.loc[idx, "typhoon_name"])


def _storm_explanation(summary_df: pd.DataFrame, timeseries_df: pd.DataFrame, name: str) -> str:
    row = _metric_row(summary_df, name)
    sub = timeseries_df.loc[timeseries_df["typhoon_name"].eq(name)].copy()
    wnd_time = pd.Timestamp(row["WND_max_time"])
    p95_time = pd.Timestamp(row["final_rain_p95_mmhr_time"])
    area_time = pd.Timestamp(row["final_rain_area_10_km2_time"])
    coast_row = absmin_row(sub, "signed_coast_dist_km")
    coast_time = pd.Timestamp(coast_row["time"])
    p95_gap = (p95_time - wnd_time).total_seconds() / 3600.0
    area_gap = (area_time - coast_time).total_seconds() / 3600.0
    stage = "near intensity peak" if abs(p95_gap) <= 12 else ("after intensity peak" if p95_gap > 0 else "before intensity peak")
    coast_stage = "close to the near-coast stage" if abs(area_gap) <= 24 else ("after the closest-coast stage" if area_gap > 0 else "before the closest-coast stage")
    return (
        f"The P95 rainfall peak occurred {abs(p95_gap):.1f} h {('after' if p95_gap >= 0 else 'before')} "
        f"the WND peak, so the strongest extreme-rainfall stage was {stage}. "
        f"The maximum area10 stage was {abs(area_gap):.1f} h {('after' if area_gap >= 0 else 'before')} "
        f"the closest-coast time, indicating it was {coast_stage}."
    )


def _validation_comparison_lines(validation_summary: pd.DataFrame) -> List[str]:
    initial = validation_summary.loc[validation_summary["model_version"].astype(str).eq("initial")].iloc[0]
    calibrated = calibrated_validation_row(validation_summary)

    def val(row: pd.Series, col: str) -> float:
        return float(pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0])

    lines = [
        "- Initial vs calibrated validation metrics:",
        f"  - P95 absolute error: {val(initial, 'abs_error_rain_p95_mean'):.6g} -> {val(calibrated, 'abs_error_rain_p95_mean'):.6g}; improvement={val(calibrated, 'p95_error_improvement_vs_initial'):.2%}.",
        f"  - P99 absolute error: {val(initial, 'abs_error_rain_p99_mean'):.6g} -> {val(calibrated, 'abs_error_rain_p99_mean'):.6g}; improvement={val(calibrated, 'p99_error_improvement_vs_initial'):.2%}.",
        f"  - Area10 absolute error: {val(initial, 'abs_error_area_10_mean'):.6g} -> {val(calibrated, 'abs_error_area_10_mean'):.6g}; improvement={val(calibrated, 'area10_error_improvement_vs_initial'):.2%}.",
        f"  - CSI10: {val(initial, 'csi10_mean'):.6g} -> {val(calibrated, 'csi10_mean'):.6g}; absolute gain={val(calibrated, 'csi10_improvement_vs_initial'):.6g}.",
        f"  - RMSE/Corr trade-off: RMSE {val(initial, 'rmse_mean'):.6g} -> {val(calibrated, 'rmse_mean'):.6g}, Corr {val(initial, 'corr_mean'):.6g} -> {val(calibrated, 'corr_mean'):.6g}. The calibration improves extreme quantiles and heavy-rain hit skill while slightly sacrificing whole-field RMSE/correlation.",
    ]
    return lines


def _summary_markdown_table(summary_df: pd.DataFrame) -> str:
    cols = [
        "typhoon_name",
        "duration_hours",
        "WND_max",
        "PRES_min",
        "final_rain_p95_mmhr_max",
        "final_rain_p99_mmhr_max",
        "final_rain_max_mmhr_max",
        "final_rain_area_10_km2_max",
        "max_duration_10_h",
        "total_volume_proxy_mm_km2",
        "dominant_quadrant_by_total_rain",
    ]
    tmp = summary_df[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in tmp.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def write_final_report(
    calibrated: Mapping[str, np.ndarray],
    timeseries_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    key_times_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    figure_paths: Sequence[Path],
    issues: Sequence[str],
) -> None:
    rain = np.asarray(calibrated["rain_mmhr_calibrated"])
    quality = count_field_quality(rain)
    calibration_ok_all = bool_series(timeseries_df, "calibration_ok").all()
    counts = timeseries_df["typhoon_name"].value_counts().to_dict()

    kong = _metric_row(summary_df, "KONG-REY")
    manyi = _metric_row(summary_df, "MAN-YI")
    p95_winner = _winner(summary_df, "final_rain_p95_mmhr_max")
    p99_winner = _winner(summary_df, "final_rain_p99_mmhr_max")
    area_winner = _winner(summary_df, "final_rain_area_10_km2_max")
    duration_winner = _winner(summary_df, "max_duration_10_h")
    intensity_winner = _winner(summary_df, "WND_max")
    offset_winner = _winner(summary_df, "centroid_offset_km_mean")
    anisotropy_winner = _winner(summary_df, "anisotropy_mean")

    lines: List[str] = [
        "# Problem 2 Final Results: KONG-REY / MAN-YI",
        "",
        "## 1. 输入输出文件",
        f"- Target safe input table: `{rel_path(TARGET_SAFE_INPUT_PATH)}`",
        f"- Top-K retrieval table: `{rel_path(TOPK_TABLE_PATH)}`",
        f"- Calibrated NPZ: `{rel_path(CALIBRATED_NPZ_PATH)}`",
        f"- Calibrated index: `{rel_path(CALIBRATED_INDEX_PATH)}`",
        f"- Step-21 QC report: `{rel_path(CALIBRATION_QC_REPORT_PATH)}`",
        f"- Step-22 model validation summary: `{rel_path(VALIDATION_MODEL_SUMMARY_PATH)}`",
        f"- Step-22 event validation summary: `{rel_path(VALIDATION_EVENT_SUMMARY_PATH)}`",
        f"- Step-22 timeslice validation metrics: `{rel_path(VALIDATION_TIMESLICE_METRICS_PATH)}`",
        f"- Step-22 QC report: `{rel_path(VALIDATION_QC_REPORT_PATH)}`",
        f"- Final timeseries CSV: `{rel_path(FINAL_TIMESERIES_PATH)}`",
        f"- Final summary CSV: `{rel_path(FINAL_SUMMARY_PATH)}`",
        f"- Final key times CSV: `{rel_path(FINAL_KEY_TIMES_PATH)}`",
        f"- Figures directory: `{rel_path(FIGURE_DIR)}`",
        "",
        "## 2. 数据完整性检查",
        f"- calibrated rain field shape: `{rain.shape}`",
        f"- target 时刻数: {rain.shape[0]}",
        f"- KONG-REY 时刻数: {int(counts.get('KONG-REY', 0))}",
        f"- MAN-YI 时刻数: {int(counts.get('MAN-YI', 0))}",
        f"- NaN/Inf/负值/全零场数量: {quality}",
        f"- calibration_ok 是否全 True: {bool(calibration_ok_all)}",
        f"- final timeseries CSV 行列数: {timeseries_df.shape}",
        f"- final summary CSV 行列数: {summary_df.shape}",
        f"- final key times CSV 行列数: {key_times_df.shape}",
        f"- 生成图件数量: {len(figure_paths)}",
        "",
        "## 3. KONG-REY 结果摘要",
        f"- 时间范围: {kong['start_time']} to {kong['end_time']}",
        f"- 最大 WND: {kong['WND_max']:.6g} at {kong['WND_max_time']}",
        f"- 最小 PRES: {kong['PRES_min']:.6g} at {kong['PRES_min_time']}",
        f"- 最大 final_rain_p95: {kong['final_rain_p95_mmhr_max']:.6g} at {kong['final_rain_p95_mmhr_time']}",
        f"- 最大 final_rain_p99: {kong['final_rain_p99_mmhr_max']:.6g} at {kong['final_rain_p99_mmhr_time']}",
        f"- 最大 final_rain_max: {kong['final_rain_max_mmhr_max']:.6g} at {kong['final_rain_max_mmhr_time']}",
        f"- 最大 final_rain_area_10: {kong['final_rain_area_10_km2_max']:.6g} at {kong['final_rain_area_10_km2_time']}",
        f"- 最大 duration10: {kong['max_duration_10_h']:.6g} h",
        f"- 累计降水 proxy: {kong['total_volume_proxy_mm_km2']:.6g} mm km2",
        f"- 主要降水象限: {kong['dominant_quadrant_by_total_rain']}",
        f"- 简短解释: {_storm_explanation(summary_df, timeseries_df, 'KONG-REY')}",
        "",
        "## 4. MAN-YI 结果摘要",
        f"- 时间范围: {manyi['start_time']} to {manyi['end_time']}",
        f"- 最大 WND: {manyi['WND_max']:.6g} at {manyi['WND_max_time']}",
        f"- 最小 PRES: {manyi['PRES_min']:.6g} at {manyi['PRES_min_time']}",
        f"- 最大 final_rain_p95: {manyi['final_rain_p95_mmhr_max']:.6g} at {manyi['final_rain_p95_mmhr_time']}",
        f"- 最大 final_rain_p99: {manyi['final_rain_p99_mmhr_max']:.6g} at {manyi['final_rain_p99_mmhr_time']}",
        f"- 最大 final_rain_max: {manyi['final_rain_max_mmhr_max']:.6g} at {manyi['final_rain_max_mmhr_time']}",
        f"- 最大 final_rain_area_10: {manyi['final_rain_area_10_km2_max']:.6g} at {manyi['final_rain_area_10_km2_time']}",
        f"- 最大 duration10: {manyi['max_duration_10_h']:.6g} h",
        f"- 累计降水 proxy: {manyi['total_volume_proxy_mm_km2']:.6g} mm km2",
        f"- 主要降水象限: {manyi['dominant_quadrant_by_total_rain']}",
        f"- 简短解释: {_storm_explanation(summary_df, timeseries_df, 'MAN-YI')}",
        "",
        "## 5. 两场台风对比",
        f"- 最大强度更高: {intensity_winner}，以 WND_max 为依据。",
        f"- 极端降水 P95 更高: {p95_winner}; P99 更高: {p99_winner}。",
        f"- 强降水面积更大: {area_winner}，以 final_rain_area_10_km2_max 为依据。",
        f"- 强降水持续时间更长: {duration_winner}，以 max_duration_10_h 为依据。",
        f"- 平均降水质心偏移更大: {offset_winner}; 平均非对称/各向异性更强: {anisotropy_winner}。",
        "- 路径、强度和距岸距离共同影响强降水阶段：强度峰值附近更容易出现极端分位数峰值，近岸或登陆附近阶段更容易扩大强降水面积和持续时间。",
        "",
        "### 台风级核心指标表",
        _summary_markdown_table(summary_df),
        "",
        "## 6. 模型验证引用",
        *_validation_comparison_lines(validation_summary),
        "",
        "## 7. 论文可写结论",
        f"1. KONG-REY 的条件生成降水过程在 {kong['final_rain_p95_mmhr_time']} 达到 P95 极端峰值，P95={kong['final_rain_p95_mmhr_max']:.3f} mm/hr。",
        f"2. MAN-YI 的条件生成降水过程在 {manyi['final_rain_p95_mmhr_time']} 达到 P95 极端峰值，P95={manyi['final_rain_p95_mmhr_max']:.3f} mm/hr。",
        f"3. 两场台风相比，{p95_winner} 的 P95 极端降水更高，{area_winner} 的强降水面积峰值更大。",
        f"4. KONG-REY 的累计降水主要贡献象限为 {kong['dominant_quadrant_by_total_rain']}，MAN-YI 为 {manyi['dominant_quadrant_by_total_rain']}，说明降水分布相对于移动方向存在明显不对称。",
        f"5. KONG-REY 与 MAN-YI 的最大 duration10 分别为 {kong['max_duration_10_h']:.3f} h 和 {manyi['max_duration_10_h']:.3f} h，可用于讨论强降水持续性差异。",
        f"6. 两场台风平均降水质心偏移分别为 {kong['centroid_offset_km_mean']:.3f} km 和 {manyi['centroid_offset_km_mean']:.3f} km，表明降水中心并不总与台风中心重合。",
        "7. 极端分位数校准显著降低了 P95/P99 与 area10 误差，并提高了 10 mm/hr 阈值命中能力，适合作为最终结果版本。",
        "8. 本结果是基于历史相似台风、路径、强度和环境条件生成的可能降水分布，不是 KONG-REY 和 MAN-YI 的真实观测降水。",
        "",
        "## 8. 注意事项",
        "- KONG-REY 和 MAN-YI 在本项目使用的 GPM 降水集中无真实记录。",
        "- 本文结果是基于历史相似台风和路径-强度-环境条件生成的可能降水场。",
        "- 不应将 final_ 结果表述为真实观测。",
        "- final_ 结果来自 21 号 calibrated 场。",
        "- 22 号伪缺失验证用于证明模型链条的合理性，不代表这两场目标台风的实测误差。",
        "",
        "## 9. 图件清单",
    ]
    lines.extend(f"- `{rel_path(path)}`" for path in figure_paths)
    lines.extend(["", "## 10. 运行记录问题"])
    if issues:
        lines.extend(f"- {item}" for item in issues)
    else:
        lines.append("- 无异常。")

    try:
        FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FINAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        add_issue(list(issues), f"最终报告写入失败: {exc}")
        raise


# =========================
# Main
# =========================


def main() -> None:
    issues: List[str] = []
    try:
        print("[23] Loading step-21 calibrated fields and index...")
        calibrated = load_calibrated_npz(issues)
        index_df = load_calibrated_index(issues)
        validation_summary = load_validation_summary(issues)

        print("[23] Checking input consistency...")
        validate_input_consistency(calibrated, index_df, issues)
        rain_calibrated = np.asarray(calibrated["rain_mmhr_calibrated"], dtype=np.float32)
        x_front_km = np.asarray(calibrated["x_front_km"], dtype=np.float32)
        y_left_km = np.asarray(calibrated["y_left_km"], dtype=np.float32)

        print("[23] Computing final time-series and storm-level metrics...")
        timeseries_df = compute_final_timeseries_metrics(calibrated, index_df, issues)
        cumulative_fields = compute_cumulative_fields(rain_calibrated, timeseries_df)
        max_fields = compute_max_fields(rain_calibrated, timeseries_df)
        duration_fields = compute_duration_fields(rain_calibrated, timeseries_df)
        summary_df = compute_typhoon_summary(
            timeseries_df,
            rain_calibrated,
            x_front_km,
            y_left_km,
            validation_summary,
            cumulative_fields,
            duration_fields,
            issues,
        )
        key_times_df = identify_key_times(timeseries_df, issues)

        print("[23] Saving final CSV tables...")
        save_final_tables(timeseries_df, summary_df, key_times_df, issues)

        print("[23] Making final figures...")
        figure_paths: List[Path] = []
        figure_steps = [
            ("track/intensity/rain time series", lambda: make_track_intensity_rain_timeseries(timeseries_df)),
            ("final rain metric time series", lambda: make_final_rain_metrics_timeseries(timeseries_df)),
            ("representative rainfall fields", lambda: make_representative_field_figures(timeseries_df, key_times_df, rain_calibrated, x_front_km, y_left_km)),
            ("cumulative rainfall fields", lambda: make_cumulative_figures(cumulative_fields, x_front_km, y_left_km)),
            ("maximum rainfall fields", lambda: make_max_rain_figures(max_fields, x_front_km, y_left_km)),
            ("duration fields", lambda: make_duration_figures(duration_fields, x_front_km, y_left_km)),
            ("quadrant contribution figures", lambda: make_quadrant_figures(cumulative_fields, x_front_km, y_left_km)),
            ("two-typhoon comparison figures", lambda: make_two_typhoon_compare_figures(timeseries_df, summary_df)),
            ("validation comparison figure", lambda: make_validation_comparison_figure(validation_summary)),
        ]
        for label, maker in figure_steps:
            try:
                figure_paths.extend(maker())
            except Exception as exc:
                add_issue(issues, f"图件生成失败: {label}: {exc}")
                raise

        print("[23] Writing final report...")
        write_final_report(calibrated, timeseries_df, summary_df, key_times_df, validation_summary, figure_paths, issues)
        write_run_log(issues, "SUCCESS")

        print("[23] Done.")
        print(f"  script: {rel_path(Path(__file__))}")
        print(f"  timeseries: {rel_path(FINAL_TIMESERIES_PATH)} {timeseries_df.shape}")
        print(f"  summary: {rel_path(FINAL_SUMMARY_PATH)} {summary_df.shape}")
        print(f"  key_times: {rel_path(FINAL_KEY_TIMES_PATH)} {key_times_df.shape}")
        print(f"  report: {rel_path(FINAL_REPORT_PATH)}")
        print(f"  figures: {rel_path(FIGURE_DIR)} count={len(figure_paths)}")
    except Exception as exc:
        tb = traceback.format_exc()
        add_issue(issues, f"FATAL: {type(exc).__name__}: {exc}")
        write_run_log(issues, "FAILED", tb)
        raise


if __name__ == "__main__":
    main()
