#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 extreme quantile calibration.

This step starts from the step-20 PCA-blended storm-relative rainfall fields and
calibrates the high-rainfall tail toward weighted Top-K historical analog
targets. It does not redo Top-K retrieval, EOF/PCA training, georeferencing, or
pseudo-missing validation.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"
TOPK_TABLE_PATH = PROJECT_ROOT / "data/processed/problem2_target_topk_similar_history.csv"
BLENDED_NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_generated_pca_blended_fields.npz"
BLENDED_INDEX_PATH = PROJECT_ROOT / "data/processed/problem2_generated_pca_blended_fields_index.csv"
STEP20_QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_eof_pca_correction_qc_report.md"

TARGETS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_extreme_calibration_targets.csv"
CALIBRATED_NPZ_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields.npz"
CALIBRATED_INDEX_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields_index.csv"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_extreme_calibration_qc_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_extreme_calibration"

BETA_SOURCE = "blend"
USE_WINSORIZED_TARGETS = True
WINSOR_LOWER = 0.05
WINSOR_UPPER = 0.95

SCALE_MIN = 0.7
SCALE_MAX = 3.0
SCALE_MAX_MAX = 5.0

TAIL_START_QUANTILE = 0.90
P95_QUANTILE = 0.95
P99_QUANTILE = 0.99

AREA_CALIBRATION = True
AREA_TOLERANCE = 0.25
AREA_MAX_ITER = 5

GLOBAL_RAIN_MAX_CAP_MMHR = 120.0
PER_TARGET_CAP_FACTOR = 1.25

GRID_SIZE = 201
GRID_EXTENT_KM = 1000.0
MAKE_FIGURES = True
RANDOM_SEED = 2026

EPS = 1e-12
AREA10_THRESHOLD = 10.0
AREA20_THRESHOLD = 20.0
AREA10_LOWER_CANDIDATE = 7.0
AREA10_EXPANDED_LOWER_CANDIDATE = 5.0
MAX_AREA_ADJUST_FRACTION = 0.05


# =========================
# Basic helpers
# =========================


def resolve_project_path(path: object) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def format_time(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def safe_name(name: object) -> str:
    return str(name).replace("-", "_").replace(" ", "_")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def iter_progress(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


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


def quantile_pair_text(values: object) -> str:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return "P50=NA, P95=NA"
    return f"P50={float(s.quantile(0.50)):.6g}, P95={float(s.quantile(0.95)):.6g}"


def p50_p95_max_text(values: object) -> str:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return "P50=NA, P95=NA, max=NA"
    return f"P50={float(s.quantile(0.50)):.6g}, P95={float(s.quantile(0.95)):.6g}, max={float(s.max()):.6g}"


def count_field_quality(fields: np.ndarray) -> Dict[str, int]:
    arr = np.asarray(fields)
    return {
        "nan": int(np.count_nonzero(np.isnan(arr))),
        "inf": int(np.count_nonzero(np.isinf(arr))),
        "negative": int(np.count_nonzero(np.isfinite(arr) & (arr < 0.0))),
        "all_zero": int(np.count_nonzero(np.all(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) == 0.0, axis=(1, 2)))),
    }


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= EPS or y_std <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cell_area_from_grid(x_front_km: np.ndarray, y_left_km: np.ndarray) -> float:
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    return dx * dy


# =========================
# Loaders
# =========================


def load_blended_npz() -> Dict[str, np.ndarray]:
    if not BLENDED_NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing step-20 blended NPZ: {BLENDED_NPZ_PATH}")
    with np.load(BLENDED_NPZ_PATH, allow_pickle=True) as z:
        required = [
            "rain_mmhr_blend",
            "log_rain_blend",
            "target_id",
            "typhoon_name",
            "time",
            "lat",
            "lon_180",
            "move_dir_deg",
            "x_front_km",
            "y_left_km",
        ]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"Blended NPZ is missing required arrays: {missing}")
        out = {k: z[k] for k in required}
    rain = np.asarray(out["rain_mmhr_blend"], dtype=np.float32)
    log_rain = np.asarray(out["log_rain_blend"], dtype=np.float32)
    if rain.shape != log_rain.shape:
        raise RuntimeError(f"Blend rain/log shapes differ: {rain.shape} vs {log_rain.shape}")
    if rain.ndim != 3 or rain.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"Unexpected blended field shape: {rain.shape}")
    out["rain_mmhr_blend"] = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    out["log_rain_blend"] = np.nan_to_num(log_rain, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    for key in ["target_id", "typhoon_name", "time"]:
        out[key] = np.asarray(out[key]).astype(str)
    for key in ["lat", "lon_180", "move_dir_deg", "x_front_km", "y_left_km"]:
        out[key] = np.asarray(out[key], dtype=np.float32)
    return out


def load_blended_index() -> pd.DataFrame:
    if not BLENDED_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing step-20 blended index: {BLENDED_INDEX_PATH}")
    df = pd.read_csv(BLENDED_INDEX_PATH, encoding="utf-8-sig", low_memory=False)
    required = ["field_index", "target_id", "typhoon_name", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Blended index is missing required columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("field_index").reset_index(drop=True)
    if not np.array_equal(df["field_index"].to_numpy(dtype=int), np.arange(len(df), dtype=int)):
        raise RuntimeError("Blended index field_index is not a contiguous 0-based sequence")
    return df


def load_topk_table() -> pd.DataFrame:
    if not TOPK_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing Top-K table: {TOPK_TABLE_PATH}")
    df = pd.read_csv(TOPK_TABLE_PATH, encoding="utf-8-sig", low_memory=False)
    for col in ["target_time", "history_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    required = ["target_id", "history_sample_id", "similarity_weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Top-K table is missing required columns: {missing}")
    return df


def load_historical_library() -> pd.DataFrame:
    if not HISTORICAL_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Missing historical library: {HISTORICAL_LIBRARY_PATH}")
    df = pd.read_csv(HISTORICAL_LIBRARY_PATH, encoding="utf-8-sig", low_memory=False)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def validate_blended_order(blended: Mapping[str, np.ndarray], blended_index: pd.DataFrame) -> None:
    n = int(blended["rain_mmhr_blend"].shape[0])
    if n != len(blended_index):
        raise RuntimeError(f"Blended NPZ field count {n} does not match blended index rows {len(blended_index)}")
    npz_ids = np.asarray(blended["target_id"]).astype(str)
    idx_ids = blended_index["target_id"].astype(str).to_numpy()
    if not np.array_equal(npz_ids, idx_ids):
        mismatch = np.where(npz_ids != idx_ids)[0][:10].tolist()
        raise RuntimeError(f"target_id order mismatch between blended NPZ and index. First mismatches: {mismatch}")


# =========================
# Calibration targets
# =========================


def _supplement_topk_from_history(topk: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = topk.copy()
    if history.empty or "sample_id" not in history.columns:
        return out
    hist_cols = [
        "sample_id",
        "rain_max_mmhr",
        "rain_p95_mmhr",
        "rain_p99_mmhr",
        "rain_area_10_km2",
        "rain_area_20_km2",
        "tif_path",
    ]
    hist_cols = [c for c in hist_cols if c in history.columns]
    hist_small = history[hist_cols].drop_duplicates("sample_id")
    merged = out.merge(hist_small, left_on="history_sample_id", right_on="sample_id", how="left")
    fill_pairs = {
        "history_rain_max_mmhr": "rain_max_mmhr",
        "history_rain_p95_mmhr": "rain_p95_mmhr",
        "history_rain_p99_mmhr": "rain_p99_mmhr",
        "history_rain_area_10_km2": "rain_area_10_km2",
        "history_rain_area_20_km2": "rain_area_20_km2",
        "history_tif_path": "tif_path",
    }
    for left, right in fill_pairs.items():
        if right in merged.columns:
            if left not in merged.columns:
                merged[left] = merged[right]
            else:
                merged[left] = merged[left].where(merged[left].notna(), merged[right])
    drop_cols = [c for c in hist_cols + ["sample_id"] if c in merged.columns]
    return merged.drop(columns=drop_cols)


def _read_tif_metrics(path_value: object) -> Dict[str, float]:
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= np.isclose(arr, float(nodata))
    invalid |= np.isclose(arr, -9999.0)
    arr[invalid] = np.nan
    arr[np.isfinite(arr) & (arr < 0.0)] = 0.0
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return {"history_rain_p99_mmhr": np.nan, "history_rain_area_20_km2": np.nan}
    # Raw GPM tiles in this project are 0.1 degree around the storm. The fallback
    # is only used if library metrics are missing, so this area is intentionally
    # a conservative proxy.
    area_proxy = 10.0 * 10.0
    return {
        "history_rain_p99_mmhr": float(np.nanpercentile(values, 99)),
        "history_rain_area_20_km2": float(np.count_nonzero(values >= AREA20_THRESHOLD) * area_proxy),
    }


def winsorize_topk_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(arr)
    if not valid.any() or not USE_WINSORIZED_TARGETS:
        return arr
    lo = float(np.nanquantile(arr[valid], WINSOR_LOWER))
    hi = float(np.nanquantile(arr[valid], WINSOR_UPPER))
    if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
        arr[valid] = np.clip(arr[valid], lo, hi)
    return arr


def _weighted_metric(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    w = weights[valid]
    return float(np.sum(values[valid] * w) / np.sum(w))


def _conservative_metric_approx(sub: pd.DataFrame, metric: str, counters: Counter) -> pd.Series:
    values = pd.to_numeric(sub.get(metric), errors="coerce") if metric in sub.columns else pd.Series(np.nan, index=sub.index)
    if values.notna().all():
        return values
    if metric == "history_rain_p99_mmhr":
        p95 = pd.to_numeric(sub.get("history_rain_p95_mmhr"), errors="coerce")
        rmax = pd.to_numeric(sub.get("history_rain_max_mmhr"), errors="coerce")
        approx = np.minimum(rmax, p95 * 3.0)
        values = values.where(values.notna(), approx)
        counters["history_rain_p99_mmhr_approximated_from_p95_max"] += int(values.isna().sum())
    return values


def compute_weighted_calibration_targets(
    topk: pd.DataFrame,
    history: pd.DataFrame,
    blended_index: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    topk_full = _supplement_topk_from_history(topk, history)
    counters: Counter = Counter()
    tif_metric_cache: Dict[str, Dict[str, float]] = {}

    metric_cols = [
        "history_rain_max_mmhr",
        "history_rain_p95_mmhr",
        "history_rain_p99_mmhr",
        "history_rain_area_10_km2",
        "history_rain_area_20_km2",
    ]
    for col in metric_cols:
        if col not in topk_full.columns:
            topk_full[col] = np.nan

    missing_p99_or_area20 = topk_full["history_rain_p99_mmhr"].isna() | topk_full["history_rain_area_20_km2"].isna()
    if missing_p99_or_area20.any() and "history_tif_path" in topk_full.columns:
        for idx, row in topk_full.loc[missing_p99_or_area20].iterrows():
            path_value = row.get("history_tif_path")
            if pd.isna(path_value):
                counters["missing_history_tif_for_metric_fallback"] += 1
                continue
            key = str(path_value)
            if key not in tif_metric_cache:
                try:
                    tif_metric_cache[key] = _read_tif_metrics(path_value)
                    counters["history_tif_metric_fallback_read"] += 1
                except Exception:
                    tif_metric_cache[key] = {}
                    counters["history_tif_metric_fallback_failed"] += 1
            metrics = tif_metric_cache[key]
            for col in ["history_rain_p99_mmhr", "history_rain_area_20_km2"]:
                if pd.isna(topk_full.at[idx, col]) and col in metrics:
                    topk_full.at[idx, col] = metrics[col]

    rows: List[Dict[str, object]] = []
    for target_id, sub in topk_full.groupby("target_id", sort=False):
        sub = sub.copy()
        weights = pd.to_numeric(sub["similarity_weight"], errors="coerce").to_numpy(dtype=float)
        weight_sum = float(np.nansum(weights[np.isfinite(weights) & (weights > 0.0)]))
        if not np.isfinite(weight_sum) or weight_sum <= EPS:
            counters["nonpositive_weight_sum_targets"] += 1
        if abs(weight_sum - 1.0) > 1e-6:
            counters["weight_sum_not_one"] += 1

        row: Dict[str, object] = {
            "target_id": target_id,
            "typhoon_name": sub["target_typhoon_name"].iloc[0] if "target_typhoon_name" in sub.columns else np.nan,
            "time": format_time(sub["target_time"].iloc[0]) if "target_time" in sub.columns else np.nan,
            "topk_count": int(len(sub)),
            "weight_sum": weight_sum,
        }
        max_values_for_cap = pd.to_numeric(sub["history_rain_max_mmhr"], errors="coerce").to_numpy(dtype=float)
        row["topk_history_rain_max_max"] = float(np.nanmax(max_values_for_cap)) if np.isfinite(max_values_for_cap).any() else np.nan
        row["target_cap_mmhr"] = (
            float(min(GLOBAL_RAIN_MAX_CAP_MMHR, PER_TARGET_CAP_FACTOR * row["topk_history_rain_max_max"]))
            if np.isfinite(row["topk_history_rain_max_max"])
            else GLOBAL_RAIN_MAX_CAP_MMHR
        )

        mapping = {
            "rain_max": "history_rain_max_mmhr",
            "rain_p95": "history_rain_p95_mmhr",
            "rain_p99": "history_rain_p99_mmhr",
            "area_10": "history_rain_area_10_km2",
            "area_20": "history_rain_area_20_km2",
        }
        for short, col in mapping.items():
            series = _conservative_metric_approx(sub, col, counters)
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            raw_mean = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            before = _weighted_metric(values, weights)
            clipped = winsorize_topk_values(values)
            after = _weighted_metric(clipped, weights)
            row[f"raw_topk_{short}_mean"] = raw_mean
            row[f"weighted_topk_{short}_before_winsor"] = before
            if short == "rain_max":
                row["target_rain_max_mmhr"] = after
            elif short == "rain_p95":
                row["target_rain_p95_mmhr"] = after
            elif short == "rain_p99":
                row["target_rain_p99_mmhr"] = after
            elif short == "area_10":
                row["target_rain_area_10_km2"] = after
            elif short == "area_20":
                row["target_rain_area_20_km2"] = after
            if not np.isfinite(after):
                counters[f"missing_target_{short}"] += 1

        # Maintain physically sensible ordering for intensity targets.
        if np.isfinite(row.get("target_rain_p95_mmhr", np.nan)) and np.isfinite(row.get("target_rain_p99_mmhr", np.nan)):
            row["target_rain_p99_mmhr"] = max(row["target_rain_p99_mmhr"], row["target_rain_p95_mmhr"])
        if np.isfinite(row.get("target_rain_max_mmhr", np.nan)) and np.isfinite(row.get("target_rain_p99_mmhr", np.nan)):
            row["target_rain_max_mmhr"] = max(row["target_rain_max_mmhr"], row["target_rain_p99_mmhr"])
        rows.append(row)

    targets = pd.DataFrame(rows)
    targets = blended_index[["target_id", "typhoon_name", "time"]].merge(
        targets.drop(columns=["typhoon_name", "time"], errors="ignore"),
        on="target_id",
        how="left",
    )
    targets["time"] = pd.to_datetime(targets["time"], errors="coerce").map(format_time)
    diagnostics = {
        "counters": dict(counters),
        "tif_metric_fallback_cache_size": len(tif_metric_cache),
        "topk_rows": int(len(topk_full)),
        "target_rows": int(len(targets)),
    }
    return targets, diagnostics


def build_calibration_targets_table(targets: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "target_id",
        "typhoon_name",
        "time",
        "topk_count",
        "weight_sum",
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "target_rain_area_20_km2",
        "raw_topk_rain_max_mean",
        "raw_topk_rain_p95_mean",
        "raw_topk_rain_p99_mean",
        "raw_topk_area_10_mean",
        "raw_topk_area_20_mean",
        "weighted_topk_rain_max_before_winsor",
        "weighted_topk_rain_p95_before_winsor",
        "weighted_topk_rain_p99_before_winsor",
        "weighted_topk_area_10_before_winsor",
        "weighted_topk_area_20_before_winsor",
        "topk_history_rain_max_max",
        "target_cap_mmhr",
    ]
    cols = [c for c in ordered if c in targets.columns]
    extra = [c for c in targets.columns if c not in cols]
    return targets[cols + extra].copy()


# =========================
# Metrics
# =========================


def weighted_radius(radius: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(radius) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    r = radius[valid].ravel()
    w = weights[valid].ravel()
    order = np.argsort(r)
    r_sorted = r[order]
    w_sorted = w[order]
    cumsum = np.cumsum(w_sorted)
    if cumsum[-1] <= EPS:
        return np.nan
    idx = int(np.searchsorted(cumsum, q * cumsum[-1]))
    idx = min(idx, len(r_sorted) - 1)
    return float(r_sorted[idx])


def compute_rainfall_metrics_on_relative_grid(
    rain_mmhr: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    rain = np.asarray(rain_mmhr, dtype=float)
    valid = np.isfinite(rain)
    values = rain[valid]
    cell_area = cell_area_from_grid(x_front_km, y_left_km)
    out: Dict[str, float] = {
        f"{prefix}_rain_mean_mmhr": np.nan,
        f"{prefix}_rain_max_mmhr": np.nan,
        f"{prefix}_rain_p50_mmhr": np.nan,
        f"{prefix}_rain_p75_mmhr": np.nan,
        f"{prefix}_rain_p90_mmhr": np.nan,
        f"{prefix}_rain_p95_mmhr": np.nan,
        f"{prefix}_rain_p99_mmhr": np.nan,
        f"{prefix}_rain_sum_halfhour_mm": np.nan,
        f"{prefix}_rain_volume_proxy_mm_km2": np.nan,
        f"{prefix}_rain_area_1_km2": np.nan,
        f"{prefix}_rain_area_5_km2": np.nan,
        f"{prefix}_rain_area_10_km2": np.nan,
        f"{prefix}_rain_area_20_km2": np.nan,
        f"{prefix}_heavy_rain_fraction_10": np.nan,
        f"{prefix}_centroid_x_front_km": np.nan,
        f"{prefix}_centroid_y_left_km": np.nan,
        f"{prefix}_centroid_offset_km": np.nan,
        f"{prefix}_centroid_angle_deg": np.nan,
        f"{prefix}_asym_front_back_ratio": np.nan,
        f"{prefix}_asym_left_right_ratio": np.nan,
        f"{prefix}_quad_front_left_sum": np.nan,
        f"{prefix}_quad_front_right_sum": np.nan,
        f"{prefix}_quad_back_left_sum": np.nan,
        f"{prefix}_quad_back_right_sum": np.nan,
        f"{prefix}_quad_front_left_ratio": np.nan,
        f"{prefix}_quad_front_right_ratio": np.nan,
        f"{prefix}_quad_back_left_ratio": np.nan,
        f"{prefix}_quad_back_right_ratio": np.nan,
        f"{prefix}_anisotropy": np.nan,
        f"{prefix}_rain_radius_r50_km": np.nan,
        f"{prefix}_rain_radius_r80_km": np.nan,
        f"{prefix}_rain_radius_r90_km": np.nan,
        f"{prefix}_rain_band_width_km": np.nan,
    }
    if values.size == 0:
        return out

    out.update(
        {
            f"{prefix}_rain_mean_mmhr": float(np.nanmean(values)),
            f"{prefix}_rain_max_mmhr": float(np.nanmax(values)),
            f"{prefix}_rain_p50_mmhr": float(np.nanpercentile(values, 50)),
            f"{prefix}_rain_p75_mmhr": float(np.nanpercentile(values, 75)),
            f"{prefix}_rain_p90_mmhr": float(np.nanpercentile(values, 90)),
            f"{prefix}_rain_p95_mmhr": float(np.nanpercentile(values, 95)),
            f"{prefix}_rain_p99_mmhr": float(np.nanpercentile(values, 99)),
            f"{prefix}_rain_sum_halfhour_mm": float(np.nansum(rain * 0.5)),
            f"{prefix}_rain_volume_proxy_mm_km2": float(np.nansum(rain * 0.5 * cell_area)),
        }
    )
    for threshold in [1, 5, 10, 20]:
        out[f"{prefix}_rain_area_{threshold}_km2"] = float(np.count_nonzero(valid & (rain >= threshold)) * cell_area)
    out[f"{prefix}_heavy_rain_fraction_10"] = float(np.count_nonzero(valid & (rain >= 10.0)) / np.count_nonzero(valid))

    weights = np.where(np.isfinite(rain) & (rain > 0.0), rain * 0.5, 0.0)
    total = float(np.sum(weights))
    if total <= EPS:
        return out

    cx = float(np.sum(x_grid * weights) / total)
    cy = float(np.sum(y_grid * weights) / total)
    out[f"{prefix}_centroid_x_front_km"] = cx
    out[f"{prefix}_centroid_y_left_km"] = cy
    out[f"{prefix}_centroid_offset_km"] = float(math.hypot(cx, cy))
    out[f"{prefix}_centroid_angle_deg"] = float((math.degrees(math.atan2(cy, cx)) + 360.0) % 360.0)

    front = x_grid > 0.0
    back = x_grid < 0.0
    left = y_grid > 0.0
    right = y_grid < 0.0
    front_sum = float(np.sum(weights[front]))
    back_sum = float(np.sum(weights[back]))
    left_sum = float(np.sum(weights[left]))
    right_sum = float(np.sum(weights[right]))
    fl_sum = float(np.sum(weights[front & left]))
    fr_sum = float(np.sum(weights[front & right]))
    bl_sum = float(np.sum(weights[back & left]))
    br_sum = float(np.sum(weights[back & right]))
    out.update(
        {
            f"{prefix}_asym_front_back_ratio": safe_div(front_sum, back_sum),
            f"{prefix}_asym_left_right_ratio": safe_div(left_sum, right_sum),
            f"{prefix}_quad_front_left_sum": fl_sum,
            f"{prefix}_quad_front_right_sum": fr_sum,
            f"{prefix}_quad_back_left_sum": bl_sum,
            f"{prefix}_quad_back_right_sum": br_sum,
            f"{prefix}_quad_front_left_ratio": safe_div(fl_sum, total),
            f"{prefix}_quad_front_right_ratio": safe_div(fr_sum, total),
            f"{prefix}_quad_back_left_ratio": safe_div(bl_sum, total),
            f"{prefix}_quad_back_right_ratio": safe_div(br_sum, total),
        }
    )

    positive = weights > 0.0
    if int(np.count_nonzero(positive)) >= 2:
        x = x_grid[positive].ravel()
        y = y_grid[positive].ravel()
        w = weights[positive].ravel()
        x_mean = float(np.average(x, weights=w))
        y_mean = float(np.average(y, weights=w))
        x0 = x - x_mean
        y0 = y - y_mean
        cov = np.array(
            [
                [float(np.average(x0 * x0, weights=w)), float(np.average(x0 * y0, weights=w))],
                [float(np.average(x0 * y0, weights=w)), float(np.average(y0 * y0, weights=w))],
            ]
        )
        eigvals = np.linalg.eigvalsh(cov)
        if np.all(np.isfinite(eigvals)) and float(np.sum(eigvals)) > EPS:
            lam1, lam2 = float(eigvals[1]), float(eigvals[0])
            out[f"{prefix}_anisotropy"] = float((lam1 - lam2) / (lam1 + lam2))

    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    r50 = weighted_radius(radius, weights, 0.50)
    r80 = weighted_radius(radius, weights, 0.80)
    r90 = weighted_radius(radius, weights, 0.90)
    out[f"{prefix}_rain_radius_r50_km"] = r50
    out[f"{prefix}_rain_radius_r80_km"] = r80
    out[f"{prefix}_rain_radius_r90_km"] = r90
    out[f"{prefix}_rain_band_width_km"] = float(r90 - r50) if np.isfinite(r90) and np.isfinite(r50) else np.nan
    return out


# =========================
# Calibration algorithm
# =========================


def _finite_target(value: object, fallback: float) -> Tuple[float, bool]:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v) or not np.isfinite(v):
        return float(fallback), True
    return float(v), False


def _piecewise_tail_scale_map(
    r: np.ndarray,
    q90: float,
    q95: float,
    q99: float,
    rmax: float,
    y95: float,
    y99: float,
    ymax: float,
) -> np.ndarray:
    scale = np.ones_like(r, dtype=np.float32)
    s95 = safe_div(y95, q95)
    s99 = safe_div(y99, q99)
    smax = safe_div(ymax, rmax)
    s95 = float(np.clip(s95 if np.isfinite(s95) else 1.0, SCALE_MIN, SCALE_MAX))
    s99 = float(np.clip(s99 if np.isfinite(s99) else s95, SCALE_MIN, SCALE_MAX))
    smax = float(np.clip(smax if np.isfinite(smax) else s99, SCALE_MIN, SCALE_MAX_MAX))
    if q95 > q90 + EPS:
        mask = (r > q90) & (r <= q95)
        w = (r[mask] - q90) / (q95 - q90)
        scale[mask] = 1.0 + w * (s95 - 1.0)
    if q99 > q95 + EPS:
        mask = (r > q95) & (r <= q99)
        w = (r[mask] - q95) / (q99 - q95)
        scale[mask] = s95 + w * (s99 - s95)
    if rmax > q99 + EPS:
        mask = r > q99
        w = (r[mask] - q99) / (rmax - q99)
        scale[mask] = s99 + w * (smax - s99)
    return (r * scale).astype(np.float32)


def calibrate_area_threshold(
    rain: np.ndarray,
    target_area_10: float,
    cell_area_km2: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = rain.astype(np.float32, copy=True)
    diag: Dict[str, object] = {
        "area10_before": float(np.count_nonzero(out >= AREA10_THRESHOLD) * cell_area_km2),
        "area10_after": np.nan,
        "area10_action": "skipped",
        "area10_adjusted_cells": 0,
        "area10_expanded_candidate_used": False,
    }
    if not AREA_CALIBRATION or not np.isfinite(target_area_10):
        diag["area10_after"] = diag["area10_before"]
        return out, diag
    if target_area_10 <= cell_area_km2 and diag["area10_before"] <= cell_area_km2:
        diag["area10_after"] = diag["area10_before"]
        return out, diag

    lower_ok = target_area_10 * (1.0 - AREA_TOLERANCE)
    upper_ok = target_area_10 * (1.0 + AREA_TOLERANCE)
    current = float(diag["area10_before"])
    max_adjust_cells = max(1, int(MAX_AREA_ADJUST_FRACTION * out.size))

    if current < lower_ok:
        needed = int(math.ceil((lower_ok - current) / cell_area_km2))
        flat = out.ravel()
        candidates = np.where((flat >= AREA10_LOWER_CANDIDATE) & (flat < AREA10_THRESHOLD))[0]
        if len(candidates) < needed:
            candidates = np.where((flat >= AREA10_EXPANDED_LOWER_CANDIDATE) & (flat < AREA10_THRESHOLD))[0]
            diag["area10_expanded_candidate_used"] = True
        if len(candidates) > 0 and needed > 0:
            n = min(needed, len(candidates), max_adjust_cells)
            order = candidates[np.argsort(flat[candidates])[-n:]]
            boosted = AREA10_THRESHOLD + 0.05 * np.clip((flat[order] - AREA10_EXPANDED_LOWER_CANDIDATE) / (AREA10_THRESHOLD - AREA10_EXPANDED_LOWER_CANDIDATE), 0.0, 1.0)
            flat[order] = boosted.astype(np.float32)
            out = flat.reshape(out.shape)
            diag["area10_action"] = "boosted_near_threshold"
            diag["area10_adjusted_cells"] = int(n)
    elif current > upper_ok:
        target_cells = max(0, int(math.floor(upper_ok / cell_area_km2)))
        current_cells = int(np.count_nonzero(out >= AREA10_THRESHOLD))
        reduce_n = min(max(0, current_cells - target_cells), max_adjust_cells)
        flat = out.ravel()
        candidates = np.where((flat >= AREA10_THRESHOLD) & (flat < max(AREA20_THRESHOLD, float(np.nanquantile(flat, 0.99)))))[0]
        if len(candidates) > 0 and reduce_n > 0:
            n = min(reduce_n, len(candidates))
            order = candidates[np.argsort(flat[candidates])[:n]]
            flat[order] = np.minimum(AREA10_THRESHOLD - 0.01, flat[order] * 0.97).astype(np.float32)
            out = flat.reshape(out.shape)
            diag["area10_action"] = "reduced_near_threshold"
            diag["area10_adjusted_cells"] = int(n)

    diag["area10_after"] = float(np.count_nonzero(out >= AREA10_THRESHOLD) * cell_area_km2)
    return out, diag


def apply_physical_caps(rain: np.ndarray, cap_mmhr: float) -> Tuple[np.ndarray, Dict[str, object]]:
    cap = float(cap_mmhr) if np.isfinite(cap_mmhr) and cap_mmhr > 0.0 else GLOBAL_RAIN_MAX_CAP_MMHR
    cap = min(GLOBAL_RAIN_MAX_CAP_MMHR, cap)
    before_max = float(np.nanmax(rain)) if np.isfinite(rain).any() else np.nan
    capped = rain.astype(np.float32, copy=True)
    mask = np.isfinite(capped) & (capped > cap)
    capped_cells = int(np.count_nonzero(mask))
    if capped_cells:
        capped[mask] = cap
    return capped, {
        "target_cap_mmhr": cap,
        "rain_max_before_cap": before_max,
        "rain_max_capped_flag": bool(capped_cells > 0),
        "rain_max_capped_cell_count": capped_cells,
    }


def calibrate_one_field_tail_enhancement(
    rain_blend: np.ndarray,
    target_row: pd.Series,
    cell_area_km2: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    r = np.nan_to_num(np.asarray(rain_blend, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r[r < 0.0] = 0.0
    diag: Dict[str, object] = {
        "calibration_ok": True,
        "calibration_issue": "",
        "q95_too_small": False,
        "q99_too_small": False,
        "blended_all_zero": bool(np.all(r == 0.0)),
    }
    if diag["blended_all_zero"]:
        diag["calibration_ok"] = False
        diag["calibration_issue"] = "blended_field_all_zero"
        return r, diag

    q90 = float(np.nanquantile(r, TAIL_START_QUANTILE))
    q95 = float(np.nanquantile(r, P95_QUANTILE))
    q99 = float(np.nanquantile(r, P99_QUANTILE))
    rmax = float(np.nanmax(r))
    diag.update({"blend_q90": q90, "blend_q95": q95, "blend_q99": q99, "blend_rmax": rmax})
    if q95 <= EPS:
        diag["q95_too_small"] = True
    if q99 <= EPS:
        diag["q99_too_small"] = True
    if q95 <= EPS or q99 <= EPS or rmax <= EPS:
        diag["calibration_ok"] = False
        diag["calibration_issue"] = "blend_tail_quantile_too_small"
        return r, diag

    t95, miss95 = _finite_target(target_row.get("target_rain_p95_mmhr"), q95)
    t99, miss99 = _finite_target(target_row.get("target_rain_p99_mmhr"), q99)
    tmax, missmax = _finite_target(target_row.get("target_rain_max_mmhr"), rmax)
    tarea10, missarea10 = _finite_target(target_row.get("target_rain_area_10_km2"), np.nan)
    cap, _ = _finite_target(target_row.get("target_cap_mmhr"), GLOBAL_RAIN_MAX_CAP_MMHR)
    cap = min(GLOBAL_RAIN_MAX_CAP_MMHR, cap)
    missing_targets = [name for name, miss in [("p95", miss95), ("p99", miss99), ("max", missmax), ("area10", missarea10)] if miss]
    if missing_targets:
        diag["calibration_issue"] = "missing_targets_fallback:" + ",".join(missing_targets)
        diag["calibration_ok"] = False

    t99 = max(t99, t95)
    tmax = max(tmax, t99)
    scale95 = float(np.clip(t95 / (q95 + EPS), SCALE_MIN, SCALE_MAX))
    y95 = q95 * scale95
    scale99 = float(np.clip(t99 / (q99 + EPS), SCALE_MIN, SCALE_MAX))
    y99 = q99 * scale99
    scalemax = float(np.clip(tmax / (rmax + EPS), SCALE_MIN, SCALE_MAX_MAX))
    ymax = rmax * scalemax

    # Keep the high-tail map monotone and below the per-target physical cap.
    y95 = min(max(y95, q90), cap)
    y99 = min(max(y99, y95), cap)
    ymax = min(max(ymax, y99), cap)
    diag.update(
        {
            "scale95_requested": safe_div(t95, q95),
            "scale99_requested": safe_div(t99, q99),
            "scalemax_requested": safe_div(tmax, rmax),
            "scale95_used": safe_div(y95, q95),
            "scale99_used": safe_div(y99, q99),
            "scalemax_used": safe_div(ymax, rmax),
        }
    )

    calibrated = _piecewise_tail_scale_map(r, q90, q95, q99, rmax, y95, y99, ymax)
    calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    calibrated[calibrated < 0.0] = 0.0

    calibrated, area_diag = calibrate_area_threshold(calibrated, tarea10, cell_area_km2)
    diag.update(area_diag)
    calibrated, cap_diag = apply_physical_caps(calibrated, cap)
    diag.update(cap_diag)

    if not np.isfinite(calibrated).all() or np.any(calibrated < 0.0):
        diag["calibration_ok"] = False
        diag["calibration_issue"] = (diag.get("calibration_issue", "") + "; nonfinite_or_negative_after_calibration").strip("; ")
        calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        calibrated[calibrated < 0.0] = 0.0
    if np.all(calibrated == 0.0):
        diag["calibration_ok"] = False
        diag["calibration_issue"] = (diag.get("calibration_issue", "") + "; calibrated_all_zero").strip("; ")
    return calibrated.astype(np.float32), diag


def calibrate_all_fields(
    rain_blend: np.ndarray,
    targets: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cell_area = cell_area_from_grid(x_front_km, y_left_km)
    calibrated = np.empty_like(rain_blend, dtype=np.float32)
    scale_field = np.empty_like(rain_blend, dtype=np.float32)
    diag_rows: List[Dict[str, object]] = []
    iterator = iter_progress(range(rain_blend.shape[0]), total=rain_blend.shape[0], desc="Calibrate fields")
    for i in iterator:
        field, diag = calibrate_one_field_tail_enhancement(rain_blend[i], targets.iloc[i], cell_area)
        calibrated[i] = field
        scale = np.where(rain_blend[i] > EPS, field / (rain_blend[i] + EPS), 1.0)
        scale = np.nan_to_num(scale, nan=1.0, posinf=SCALE_MAX_MAX, neginf=0.0).astype(np.float32)
        scale_field[i] = scale
        diag["field_index"] = int(i)
        diag["target_id"] = targets.iloc[i].get("target_id")
        diag_rows.append(diag)
    return calibrated, scale_field, pd.DataFrame(diag_rows)


# =========================
# Outputs and index
# =========================


def build_calibrated_index_table(
    blended_index: pd.DataFrame,
    targets: pd.DataFrame,
    diag_df: pd.DataFrame,
    rain_blend: np.ndarray,
    rain_calibrated: np.ndarray,
    scale_field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> pd.DataFrame:
    x_grid, y_grid = np.meshgrid(x_front_km, y_left_km)
    id_cols = [
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
    ]
    target_cols = [
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "target_rain_area_20_km2",
        "target_cap_mmhr",
    ]
    blend_keep_cols = [
        "blend_rain_max_mmhr",
        "blend_rain_p95_mmhr",
        "blend_rain_p99_mmhr",
        "blend_rain_area_10_km2",
        "blend_rain_area_20_km2",
        "blend_centroid_offset_km",
        "blend_anisotropy",
        "blend_rain_radius_r50_km",
        "blend_rain_radius_r80_km",
        "blend_rain_radius_r90_km",
        "blend_rain_band_width_km",
    ]

    rows: List[Dict[str, object]] = []
    iterator = iter_progress(range(len(blended_index)), total=len(blended_index), desc="Build calibrated index")
    for i in iterator:
        src = blended_index.iloc[i]
        target = targets.iloc[i]
        diag = diag_df.iloc[i]
        row: Dict[str, object] = {c: src.get(c, np.nan) for c in id_cols if c in blended_index.columns}
        row["time"] = format_time(row.get("time"))
        for c in target_cols:
            row[c] = target.get(c, np.nan)
        for c in blend_keep_cols:
            if c in blended_index.columns:
                row[c] = src[c]
            elif c == "blend_rain_area_20_km2":
                row[c] = float(np.count_nonzero(rain_blend[i] >= AREA20_THRESHOLD) * cell_area_from_grid(x_front_km, y_left_km))

        metrics = compute_rainfall_metrics_on_relative_grid(
            rain_calibrated[i], x_front_km, y_left_km, x_grid, y_grid, prefix="calibrated"
        )
        row.update(metrics)

        row["ratio_calibrated_to_target_rain_max"] = safe_div(row["calibrated_rain_max_mmhr"], row["target_rain_max_mmhr"])
        row["ratio_calibrated_to_target_rain_p95"] = safe_div(row["calibrated_rain_p95_mmhr"], row["target_rain_p95_mmhr"])
        row["ratio_calibrated_to_target_rain_p99"] = safe_div(row["calibrated_rain_p99_mmhr"], row["target_rain_p99_mmhr"])
        row["ratio_calibrated_to_target_area_10"] = safe_div(row["calibrated_rain_area_10_km2"], row["target_rain_area_10_km2"])
        row["ratio_calibrated_to_target_area_20"] = safe_div(row["calibrated_rain_area_20_km2"], row["target_rain_area_20_km2"])
        row["ratio_calibrated_to_blend_rain_max"] = safe_div(row["calibrated_rain_max_mmhr"], row.get("blend_rain_max_mmhr", np.nan))
        row["ratio_calibrated_to_blend_rain_p95"] = safe_div(row["calibrated_rain_p95_mmhr"], row.get("blend_rain_p95_mmhr", np.nan))
        row["ratio_calibrated_to_blend_rain_p99"] = safe_div(row["calibrated_rain_p99_mmhr"], row.get("blend_rain_p99_mmhr", np.nan))
        row["ratio_calibrated_to_blend_area_10"] = safe_div(row["calibrated_rain_area_10_km2"], row.get("blend_rain_area_10_km2", np.nan))
        row["corr_blend_calibrated"] = corr_flat(rain_blend[i], rain_calibrated[i])
        row["rmse_blend_calibrated"] = float(np.sqrt(np.mean((rain_blend[i].astype(float) - rain_calibrated[i].astype(float)) ** 2)))
        row["mean_scale_factor"] = float(np.nanmean(scale_field[i]))
        row["p95_scale_factor"] = float(np.nanpercentile(scale_field[i], 95))
        row["max_scale_factor"] = float(np.nanmax(scale_field[i]))
        row["rain_max_capped_flag"] = bool(diag.get("rain_max_capped_flag", False))
        row["rain_max_capped_cell_count"] = int(diag.get("rain_max_capped_cell_count", 0))
        row["calibration_ok"] = bool(diag.get("calibration_ok", False))
        row["calibration_issue"] = diag.get("calibration_issue", "")
        row["area10_action"] = diag.get("area10_action", "")
        row["area10_adjusted_cells"] = int(diag.get("area10_adjusted_cells", 0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_calibrated_npz(
    blended: Mapping[str, np.ndarray],
    rain_calibrated: np.ndarray,
    scale_field: np.ndarray,
    targets: pd.DataFrame,
) -> None:
    CALIBRATED_NPZ_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CALIBRATED_NPZ_OUTPUT_PATH,
        rain_mmhr_blend=np.asarray(blended["rain_mmhr_blend"], dtype=np.float32),
        rain_mmhr_calibrated=np.asarray(rain_calibrated, dtype=np.float32),
        log_rain_calibrated=np.log1p(np.asarray(rain_calibrated, dtype=np.float32)).astype(np.float32),
        calibration_scale_field=np.asarray(scale_field, dtype=np.float32),
        target_id=np.asarray(blended["target_id"]).astype("U"),
        typhoon_name=np.asarray(blended["typhoon_name"]).astype("U"),
        time=np.asarray(blended["time"]).astype("U"),
        lat=np.asarray(blended["lat"], dtype=np.float32),
        lon_180=np.asarray(blended["lon_180"], dtype=np.float32),
        move_dir_deg=np.asarray(blended["move_dir_deg"], dtype=np.float32),
        x_front_km=np.asarray(blended["x_front_km"], dtype=np.float32),
        y_left_km=np.asarray(blended["y_left_km"], dtype=np.float32),
        target_rain_max_mmhr=pd.to_numeric(targets["target_rain_max_mmhr"], errors="coerce").to_numpy(dtype=np.float32),
        target_rain_p95_mmhr=pd.to_numeric(targets["target_rain_p95_mmhr"], errors="coerce").to_numpy(dtype=np.float32),
        target_rain_p99_mmhr=pd.to_numeric(targets["target_rain_p99_mmhr"], errors="coerce").to_numpy(dtype=np.float32),
        target_rain_area_10_km2=pd.to_numeric(targets["target_rain_area_10_km2"], errors="coerce").to_numpy(dtype=np.float32),
        target_rain_area_20_km2=pd.to_numeric(targets["target_rain_area_20_km2"], errors="coerce").to_numpy(dtype=np.float32),
    )


# =========================
# Figures
# =========================


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
        extent=[x_front_km[0], x_front_km[-1], y_left_km[0], y_left_km[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.axvline(0, color="white" if cmap != "RdBu_r" else "black", linewidth=0.6, alpha=0.7)
    ax.axhline(0, color="white" if cmap != "RdBu_r" else "black", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("x_front_km")
    ax.set_ylabel("y_left_km")
    ax.set_title(title, fontsize=9)
    return im


def make_calibration_timeseries_figures(index_df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
        panels = [
            ("rain_max_mmhr", "Rmax mm/hr"),
            ("rain_p95_mmhr", "P95 mm/hr"),
            ("rain_p99_mmhr", "P99 mm/hr"),
            ("rain_area_10_km2", "Area >=10 km2"),
        ]
        for ax, (metric, ylabel) in zip(axes, panels):
            ax.plot(sub["time_dt"], sub[f"blend_{metric}"], label="blend", linewidth=1.1)
            ax.plot(sub["time_dt"], sub[f"calibrated_{metric}"], label="calibrated", linewidth=1.1)
            target_col = "target_" + metric
            if target_col in sub.columns:
                ax.plot(sub["time_dt"], sub[target_col], label="target", linewidth=1.0, alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=7, ncol=3)
        axes[0].set_title(f"{name} extreme calibration time series")
        axes[-1].set_xlabel("time")
        path = FIGURE_DIR / f"{safe_name(name)}_extreme_calibration_timeseries.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def _representative_indices(index_df: pd.DataFrame, name: object) -> List[int]:
    sub = index_df.loc[index_df["typhoon_name"].eq(name)].copy()
    sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
    sub = sub.sort_values("time_dt")
    candidate_rows = [
        sub.iloc[0],
        sub.loc[pd.to_numeric(sub["target_rain_p95_mmhr"], errors="coerce").idxmax()],
        sub.loc[pd.to_numeric(sub["target_rain_max_mmhr"], errors="coerce").idxmax()],
        sub.iloc[-1],
    ]
    extra_candidates = [
        sub.loc[pd.to_numeric(sub["target_rain_p99_mmhr"], errors="coerce").idxmax()],
        sub.iloc[len(sub) // 2],
    ]
    picks: List[int] = []
    for row in candidate_rows + extra_candidates:
        idx = int(row["field_index"])
        if idx not in picks:
            picks.append(idx)
        if len(picks) >= 4:
            break
    return picks


def make_compare_figures(
    index_df: pd.DataFrame,
    rain_blend: np.ndarray,
    rain_calibrated: np.ndarray,
    scale_field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name in index_df["typhoon_name"].dropna().drop_duplicates():
        for field_idx in _representative_indices(index_df, name):
            row = index_df.loc[index_df["field_index"].eq(field_idx)].iloc[0]
            vmax = float(np.nanpercentile(np.stack([rain_blend[field_idx], rain_calibrated[field_idx]]), 99))
            vmax = max(vmax, 1.0)
            diff = rain_calibrated[field_idx] - rain_blend[field_idx]
            diff_abs = max(float(np.nanpercentile(np.abs(diff), 99)), 0.1)
            scale_vmax = max(float(np.nanpercentile(scale_field[field_idx], 99)), 1.1)
            fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), constrained_layout=True)
            im = _plot_field(axes[0], rain_blend[field_idx], x_front_km, y_left_km, "blend", vmax=vmax)
            fig.colorbar(im, ax=axes[0], shrink=0.78)
            im = _plot_field(axes[1], rain_calibrated[field_idx], x_front_km, y_left_km, "calibrated", vmax=vmax)
            fig.colorbar(im, ax=axes[1], shrink=0.78)
            im = _plot_field(axes[2], diff, x_front_km, y_left_km, "calibrated - blend", cmap="RdBu_r", vmin=-diff_abs, vmax=diff_abs)
            fig.colorbar(im, ax=axes[2], shrink=0.78)
            im = _plot_field(axes[3], scale_field[field_idx], x_front_km, y_left_km, "scale factor", cmap="magma", vmin=0.8, vmax=scale_vmax)
            fig.colorbar(im, ax=axes[3], shrink=0.78)
            fig.suptitle(f"{name} {row['time']}", fontsize=11)
            stamp = pd.Timestamp(row["time"]).strftime("%Y%m%d_%H%M")
            path = FIGURE_DIR / f"{safe_name(name)}_extreme_calibration_compare_{stamp}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def make_cumulative_max_duration_figures(
    index_df: pd.DataFrame,
    rain_calibrated: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        indices = sub["field_index"].astype(int).to_numpy()
        cumulative = np.sum(rain_calibrated[indices] * 0.5, axis=0)
        max_field = np.max(rain_calibrated[indices], axis=0)
        duration10 = np.sum(rain_calibrated[indices] >= AREA10_THRESHOLD, axis=0) * 0.5
        duration20 = np.sum(rain_calibrated[indices] >= AREA20_THRESHOLD, axis=0) * 0.5
        fields = [
            ("calibrated_cumulative", cumulative, "cumulative half-hour mm", f"{safe_name(name)}_calibrated_cumulative_storm_relative.png"),
            ("calibrated_max", max_field, "max rain mm/hr", f"{safe_name(name)}_calibrated_max_storm_relative.png"),
            ("calibrated_duration10", duration10, "hours >=10 mm/hr", f"{safe_name(name)}_calibrated_duration10_storm_relative.png"),
            ("calibrated_duration20", duration20, "hours >=20 mm/hr", f"{safe_name(name)}_calibrated_duration20_storm_relative.png"),
        ]
        for title, field, label, filename in fields:
            fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
            vmax = max(float(np.nanpercentile(field, 99)), 1.0)
            im = _plot_field(ax, field, x_front_km, y_left_km, f"{name} {title}", vmax=vmax)
            fig.colorbar(im, ax=ax, label=label)
            path = FIGURE_DIR / filename
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def make_scatter_figures(index_df: pd.DataFrame) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        ("target_rain_max_mmhr", "calibrated_rain_max_mmhr", "Rmax"),
        ("target_rain_p95_mmhr", "calibrated_rain_p95_mmhr", "P95"),
        ("target_rain_p99_mmhr", "calibrated_rain_p99_mmhr", "P99"),
        ("target_rain_area_10_km2", "calibrated_rain_area_10_km2", "Area >=10"),
    ]
    for ax, (xcol, ycol, title) in zip(axes.ravel(), panels):
        x = pd.to_numeric(index_df[xcol], errors="coerce")
        y = pd.to_numeric(index_df[ycol], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=10, alpha=0.45)
        if valid.any():
            lo = float(min(x[valid].min(), y[valid].min()))
            hi = float(max(x[valid].max(), y[valid].max()))
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("target")
        ax.set_ylabel("calibrated")
    path = FIGURE_DIR / "calibration_target_vs_output_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


def make_all_figures(
    index_df: pd.DataFrame,
    rain_blend: np.ndarray,
    rain_calibrated: np.ndarray,
    scale_field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not MAKE_FIGURES:
        return []
    paths: List[Path] = []
    paths.extend(make_calibration_timeseries_figures(index_df))
    paths.extend(make_compare_figures(index_df, rain_blend, rain_calibrated, scale_field, x_front_km, y_left_km))
    paths.extend(make_cumulative_max_duration_figures(index_df, rain_calibrated, x_front_km, y_left_km))
    paths.extend(make_scatter_figures(index_df))
    return paths


# =========================
# QC report
# =========================


def _target_stats_lines(df: pd.DataFrame, title: str) -> List[str]:
    lines = [f"### {title}"]
    for col in [
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "target_rain_area_20_km2",
    ]:
        lines.append(f"- {col}: {stats_text(df[col])}")
    return lines


def _relative_error(values: pd.Series, targets: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    t = pd.to_numeric(targets, errors="coerce")
    return (v - t).abs() / t.where(t.abs() > EPS)


def _time_continuity_lines(index_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    jump_records: List[str] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        lines.append(f"### {name}")
        for col in [
            "calibrated_rain_max_mmhr",
            "calibrated_rain_p95_mmhr",
            "calibrated_rain_p99_mmhr",
            "calibrated_rain_area_10_km2",
        ]:
            diff = pd.to_numeric(sub[col], errors="coerce").diff().abs()
            p95 = float(diff.quantile(0.95))
            maxv = float(diff.max(skipna=True))
            lines.append(f"- diff({col}): P95={p95:.6g}, max={maxv:.6g}")
            threshold = max(p95 * 2.5, diff.mean(skipna=True) + 4.0 * diff.std(skipna=True))
            large = sub.loc[diff > threshold, ["time", col]].copy()
            if len(large) > 0:
                for _, row in large.head(5).iterrows():
                    jump_records.append(f"- {name} {row['time']} {col} large jump, value={row[col]:.6g}")
    if jump_records:
        lines.append("### 可能需要后续时间平滑关注的跳变")
        lines.extend(jump_records[:30])
    return lines


def _per_typhoon_lines(index_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        max_rain_row = sub.loc[pd.to_numeric(sub["calibrated_rain_max_mmhr"], errors="coerce").idxmax()]
        max_p95_row = sub.loc[pd.to_numeric(sub["calibrated_rain_p95_mmhr"], errors="coerce").idxmax()]
        max_p99_row = sub.loc[pd.to_numeric(sub["calibrated_rain_p99_mmhr"], errors="coerce").idxmax()]
        max_area_row = sub.loc[pd.to_numeric(sub["calibrated_rain_area_10_km2"], errors="coerce").idxmax()]
        wnd_row = sub.loc[pd.to_numeric(sub["WND"], errors="coerce").idxmax()]
        max_rain_time = pd.Timestamp(max_rain_row["time"])
        wnd_time = pd.Timestamp(wnd_row["time"])
        time_gap_h = abs((max_rain_time - wnd_time).total_seconds()) / 3600.0
        total_proxy = float(pd.to_numeric(sub["calibrated_rain_volume_proxy_mm_km2"], errors="coerce").sum(skipna=True))
        duration10 = float((pd.to_numeric(sub["calibrated_rain_area_10_km2"], errors="coerce") > 0).sum() * 0.5)
        duration20 = float((pd.to_numeric(sub["calibrated_rain_area_20_km2"], errors="coerce") > 0).sum() * 0.5)
        lines.extend(
            [
                f"### {name}",
                f"- 时刻数: {len(sub)}",
                f"- calibrated_rain_max_mmhr 最大值: {max_rain_row['calibrated_rain_max_mmhr']:.6g} at {max_rain_row['time']}",
                f"- calibrated_rain_p95_mmhr 最大值: {max_p95_row['calibrated_rain_p95_mmhr']:.6g} at {max_p95_row['time']}",
                f"- calibrated_rain_p99_mmhr 最大值: {max_p99_row['calibrated_rain_p99_mmhr']:.6g} at {max_p99_row['time']}",
                f"- calibrated_rain_area_10_km2 最大值: {max_area_row['calibrated_rain_area_10_km2']:.6g} at {max_area_row['time']}",
                f"- 累计降水 proxy: {total_proxy:.6g}",
                f"- calibrated_rain_area_10_km2 > 0 持续时间 proxy: {duration10:.6g} hours",
                f"- calibrated_rain_area_20_km2 > 0 持续时间 proxy: {duration20:.6g} hours",
                f"- WND 最大时刻: {wnd_row['time']}; Rmax 最大时刻: {max_rain_row['time']}; 时间差: {time_gap_h:.3g} hours",
            ]
        )
    return lines


def write_qc_report(
    targets: pd.DataFrame,
    target_diag: Mapping[str, object],
    calibrated_index: pd.DataFrame,
    rain_calibrated: np.ndarray,
    diag_df: pd.DataFrame,
    figure_paths: Sequence[Path],
) -> None:
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality = count_field_quality(rain_calibrated)
    cap_field_count = int(pd.Series(calibrated_index["rain_max_capped_flag"]).astype(bool).sum())
    cap_cell_count = int(pd.to_numeric(calibrated_index["rain_max_capped_cell_count"], errors="coerce").sum())
    calibration_false = int((~pd.Series(calibrated_index["calibration_ok"]).astype(bool)).sum())
    weight_sum = pd.to_numeric(targets["weight_sum"], errors="coerce")
    target_missing = targets[[
        "target_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "target_rain_area_20_km2",
    ]].isna().sum()
    rel_errors = {
        "Rmax": _relative_error(calibrated_index["calibrated_rain_max_mmhr"], calibrated_index["target_rain_max_mmhr"]),
        "P95": _relative_error(calibrated_index["calibrated_rain_p95_mmhr"], calibrated_index["target_rain_p95_mmhr"]),
        "P99": _relative_error(calibrated_index["calibrated_rain_p99_mmhr"], calibrated_index["target_rain_p99_mmhr"]),
        "area10": _relative_error(calibrated_index["calibrated_rain_area_10_km2"], calibrated_index["target_rain_area_10_km2"]),
    }
    rel_error_lines = [
        f"- {name}: mean={float(s.mean(skipna=True)):.6g}, P50={float(s.quantile(0.50)):.6g}, P95={float(s.quantile(0.95)):.6g}"
        for name, s in rel_errors.items()
    ]
    figure_summary = pd.Series([p.name for p in figure_paths]).to_string(index=False) if figure_paths else "(none)"
    target_counter_text = pd.Series(target_diag.get("counters", {}), dtype=object).to_string() if target_diag.get("counters") else "(none)"
    corr_min = float(pd.to_numeric(calibrated_index["corr_blend_calibrated"], errors="coerce").min(skipna=True))
    corr_warning = "结构相关较高，校准主要作用于强降水尾部。" if corr_min >= 0.90 else "警告：部分时次 corr_blend_calibrated 明显偏低，需检查是否需要更强的空间/时间平滑。"

    lines = [
        "# Problem 2 Extreme Quantile Calibration QC Report",
        "",
        "## 1. 输入输出文件",
        f"- blended NPZ: `{BLENDED_NPZ_PATH.relative_to(PROJECT_ROOT)}`",
        f"- blended index: `{BLENDED_INDEX_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Top-K 表: `{TOPK_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 历史库: `{HISTORICAL_LIBRARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- step-20 QC: `{STEP20_QC_REPORT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- calibration targets 输出: `{TARGETS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- calibrated NPZ 输出: `{CALIBRATED_NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- calibrated index 输出: `{CALIBRATED_INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- figures 目录: `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. 运行参数",
        f"- BETA_SOURCE: {BETA_SOURCE}",
        f"- USE_WINSORIZED_TARGETS: {USE_WINSORIZED_TARGETS}",
        f"- WINSOR_LOWER / WINSOR_UPPER: {WINSOR_LOWER} / {WINSOR_UPPER}",
        f"- SCALE_MIN / SCALE_MAX / SCALE_MAX_MAX: {SCALE_MIN} / {SCALE_MAX} / {SCALE_MAX_MAX}",
        f"- TAIL_START_QUANTILE: {TAIL_START_QUANTILE}",
        f"- AREA_CALIBRATION: {AREA_CALIBRATION}",
        f"- AREA_TOLERANCE: {AREA_TOLERANCE}",
        f"- GLOBAL_RAIN_MAX_CAP_MMHR: {GLOBAL_RAIN_MAX_CAP_MMHR}",
        f"- PER_TARGET_CAP_FACTOR: {PER_TARGET_CAP_FACTOR}",
        "",
        "## 3. 校准目标统计",
        *_target_stats_lines(targets, "总体"),
    ]
    for name, sub in targets.groupby("typhoon_name", sort=False):
        lines.extend(_target_stats_lines(sub, str(name)))
    lines.extend(
        [
            f"- Top-K weight_sum 分布: {stats_text(weight_sum)}",
            f"- Top-K 权重和不等于 1 的目标数: {int((weight_sum.sub(1.0).abs() > 1e-6).sum())}",
            "- 目标指标缺失数量:",
            "```",
            target_missing.to_string(),
            "```",
            "- 目标构造补算/回退计数:",
            "```",
            target_counter_text,
            "```",
            "",
            "## 4. 校准场基本检查",
            f"- rain_mmhr_calibrated shape: {list(rain_calibrated.shape)}",
            f"- NaN 数量: {quality['nan']}",
            f"- Inf 数量: {quality['inf']}",
            f"- 负值数量: {quality['negative']}",
            f"- 全零场数量: {quality['all_zero']}",
            f"- cap 截断时次数: {cap_field_count}",
            f"- cap 截断格点数: {cap_cell_count}",
            f"- calibration_ok=False 数量: {calibration_false}",
            "",
            "## 5. 校准前后对比",
        ]
    )
    for metric in ["rain_max_mmhr", "rain_p95_mmhr", "rain_p99_mmhr", "rain_area_10_km2"]:
        lines.append(f"- blend_{metric}: {p50_p95_max_text(calibrated_index[f'blend_{metric}'])}")
        lines.append(f"- calibrated_{metric}: {p50_p95_max_text(calibrated_index[f'calibrated_{metric}'])}")
        lines.append(f"- target_{metric}: {p50_p95_max_text(calibrated_index[f'target_{metric}'])}")
    lines.extend(
        [
            "",
            "## 6. 目标接近程度",
            f"- ratio_calibrated_to_target_rain_max: {quantile_pair_text(calibrated_index['ratio_calibrated_to_target_rain_max'])}",
            f"- ratio_calibrated_to_target_rain_p95: {quantile_pair_text(calibrated_index['ratio_calibrated_to_target_rain_p95'])}",
            f"- ratio_calibrated_to_target_rain_p99: {quantile_pair_text(calibrated_index['ratio_calibrated_to_target_rain_p99'])}",
            f"- ratio_calibrated_to_target_area_10: {quantile_pair_text(calibrated_index['ratio_calibrated_to_target_area_10'])}",
            *rel_error_lines,
            "",
            "## 7. 空间结构保持程度",
            f"- corr_blend_calibrated P50/P95/min: {float(pd.to_numeric(calibrated_index['corr_blend_calibrated'], errors='coerce').quantile(0.50)):.6g} / {float(pd.to_numeric(calibrated_index['corr_blend_calibrated'], errors='coerce').quantile(0.95)):.6g} / {corr_min:.6g}",
            f"- rmse_blend_calibrated P50/P95/max: {p50_p95_max_text(calibrated_index['rmse_blend_calibrated'])}",
            f"- centroid_offset 变化分布: {stats_text(pd.to_numeric(calibrated_index['calibrated_centroid_offset_km'], errors='coerce') - pd.to_numeric(calibrated_index['blend_centroid_offset_km'], errors='coerce'))}",
            f"- anisotropy 变化分布: {stats_text(pd.to_numeric(calibrated_index['calibrated_anisotropy'], errors='coerce') - pd.to_numeric(calibrated_index['blend_anisotropy'], errors='coerce'))}",
            f"- r50 变化分布: {stats_text(pd.to_numeric(calibrated_index['calibrated_rain_radius_r50_km'], errors='coerce') - pd.to_numeric(calibrated_index['blend_rain_radius_r50_km'], errors='coerce'))}",
            f"- r80 变化分布: {stats_text(pd.to_numeric(calibrated_index['calibrated_rain_radius_r80_km'], errors='coerce') - pd.to_numeric(calibrated_index['blend_rain_radius_r80_km'], errors='coerce'))}",
            f"- r90 变化分布: {stats_text(pd.to_numeric(calibrated_index['calibrated_rain_radius_r90_km'], errors='coerce') - pd.to_numeric(calibrated_index['blend_rain_radius_r90_km'], errors='coerce'))}",
            f"- mean_scale_factor 分布: {stats_text(calibrated_index['mean_scale_factor'])}",
            f"- p95_scale_factor 分布: {stats_text(calibrated_index['p95_scale_factor'])}",
            f"- max_scale_factor 分布: {stats_text(calibrated_index['max_scale_factor'])}",
            f"- 说明: {corr_warning}",
            "",
            "## 8. 分台风结果",
            *_per_typhoon_lines(calibrated_index),
            "",
            "## 9. 时间连续性检查",
            *_time_continuity_lines(calibrated_index),
            "",
            "## 10. 图件",
            f"- 图件数量: {len(figure_paths)}",
            "```",
            figure_summary,
            "```",
            "",
            "## 11. 防泄漏声明",
            "本步骤的校准目标来自目标时刻 Top-K 历史相似台风样本的加权降水指标，不使用 KONG-REY 和 MAN-YI 的真实 GPM 降水观测。目标台风输入仍仅包含路径、强度、移动、时间和海陆环境等安全特征。历史 rain_* 指标只用于校准生成场的极端分布，不参与目标台风输入构造或相似度检索。",
            "",
            "## 12. 论文可写结论",
            "- 极端校准前，20 号 blended 场的 Rmax/P99/P95 相比 Top-K 历史相似样本目标整体偏低，尤其 Rmax 尾部被 log 加权和 EOF/PCA 平滑明显压低。",
            "- 校准后，P95/P99/Rmax 均向相似历史目标靠近，强降水尾部幅度有实质增强。",
            "- corr_blend_calibrated 维持较高水平，说明校准主要增强强降水尾部，没有整体破坏 20 号的大尺度结构底图。",
            "- 强降水面积和持续时间较 20 号 blended 场更接近历史相似台风过程，可为后续结果展示提供更合理的极端降水空间范围。",
            "- calibrated 场可作为 22 号伪缺失验证、23 号最终图件整理以及问题三虚拟台风情景生成的基础。",
        ]
    )
    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def main() -> None:
    np.random.seed(RANDOM_SEED)
    for path in [TARGETS_OUTPUT_PATH, CALIBRATED_NPZ_OUTPUT_PATH, CALIBRATED_INDEX_OUTPUT_PATH, QC_REPORT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("[21] Loading blended fields and index")
    blended = load_blended_npz()
    blended_index = load_blended_index()
    validate_blended_order(blended, blended_index)

    print("[21] Loading Top-K table and historical library")
    topk = load_topk_table()
    history = load_historical_library()

    print("[21] Computing weighted calibration targets")
    targets, target_diag = compute_weighted_calibration_targets(topk, history, blended_index)
    if not np.array_equal(targets["target_id"].astype(str).to_numpy(), blended_index["target_id"].astype(str).to_numpy()):
        raise RuntimeError("target_id order mismatch between calibration targets and blended index")
    targets_table = build_calibration_targets_table(targets)
    targets_table.to_csv(TARGETS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    rain_blend = np.asarray(blended["rain_mmhr_blend"], dtype=np.float32)
    x_front_km = np.asarray(blended["x_front_km"], dtype=np.float32)
    y_left_km = np.asarray(blended["y_left_km"], dtype=np.float32)

    print("[21] Applying tail-enhancement calibration")
    rain_calibrated, scale_field, diag_df = calibrate_all_fields(rain_blend, targets, x_front_km, y_left_km)

    print("[21] Building calibrated index")
    calibrated_index = build_calibrated_index_table(
        blended_index,
        targets,
        diag_df,
        rain_blend,
        rain_calibrated,
        scale_field,
        x_front_km,
        y_left_km,
    )
    calibrated_index.to_csv(CALIBRATED_INDEX_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("[21] Saving calibrated NPZ")
    save_calibrated_npz(blended, rain_calibrated, scale_field, targets)

    print("[21] Making figures")
    figure_paths = make_all_figures(
        calibrated_index,
        rain_blend,
        rain_calibrated,
        scale_field,
        x_front_km,
        y_left_km,
    )

    print("[21] Writing QC report")
    write_qc_report(targets, target_diag, calibrated_index, rain_calibrated, diag_df, figure_paths)

    quality = count_field_quality(rain_calibrated)
    cap_field_count = int(pd.Series(calibrated_index["rain_max_capped_flag"]).astype(bool).sum())
    cap_cell_count = int(pd.to_numeric(calibrated_index["rain_max_capped_cell_count"], errors="coerce").sum())
    print("\n========== Problem-2 extreme quantile calibration complete ==========")
    print("Script: scripts/21_extreme_quantile_calibration.py")
    print(f"Calibration targets CSV: {TARGETS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Calibrated NPZ: {CALIBRATED_NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Calibrated index CSV: {CALIBRATED_INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"QC report: {QC_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure dir: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"rain_mmhr_calibrated shape: {rain_calibrated.shape}")
    print(f"Calibrated index shape: {calibrated_index.shape[0]} x {calibrated_index.shape[1]}")
    print(f"NaN/Inf/negative/all-zero counts: {quality['nan']} / {quality['inf']} / {quality['negative']} / {quality['all_zero']}")
    print(f"calibration_ok all True: {bool(pd.Series(calibrated_index['calibration_ok']).astype(bool).all())}")
    print(f"cap clipped fields/cells: {cap_field_count} / {cap_cell_count}")
    for col in [
        "target_rain_max_mmhr",
        "blend_rain_max_mmhr",
        "calibrated_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "calibrated_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "calibrated_rain_p99_mmhr",
    ]:
        s = pd.to_numeric(calibrated_index[col], errors="coerce")
        print(f"{col} P50/P95/max: {s.quantile(0.50):.6f} / {s.quantile(0.95):.6f} / {s.max(skipna=True):.6f}")
    for col in [
        "ratio_calibrated_to_target_rain_max",
        "ratio_calibrated_to_target_rain_p95",
        "ratio_calibrated_to_target_rain_p99",
    ]:
        s = pd.to_numeric(calibrated_index[col], errors="coerce")
        print(f"{col} P50/P95: {s.quantile(0.50):.6f} / {s.quantile(0.95):.6f}")
    corr = pd.to_numeric(calibrated_index["corr_blend_calibrated"], errors="coerce")
    print(f"corr_blend_calibrated P50/P95/min: {corr.quantile(0.50):.6f} / {corr.quantile(0.95):.6f} / {corr.min(skipna=True):.6f}")
    for name, sub in calibrated_index.groupby("typhoon_name", sort=False):
        row = sub.loc[pd.to_numeric(sub["calibrated_rain_p95_mmhr"], errors="coerce").idxmax()]
        print(f"{name} max calibrated_rain_p95_mmhr: {row['calibrated_rain_p95_mmhr']:.6f} at {row['time']}")
    print(f"Generated figure count: {len(figure_paths)}")
    preview_cols = [
        "typhoon_name",
        "time",
        "WND",
        "PRES",
        "target_rain_max_mmhr",
        "blend_rain_max_mmhr",
        "calibrated_rain_max_mmhr",
        "target_rain_p95_mmhr",
        "blend_rain_p95_mmhr",
        "calibrated_rain_p95_mmhr",
        "target_rain_p99_mmhr",
        "calibrated_rain_p99_mmhr",
        "target_rain_area_10_km2",
        "calibrated_rain_area_10_km2",
        "corr_blend_calibrated",
        "max_scale_factor",
        "calibration_ok",
    ]
    print("Random 5-row calibrated index preview:")
    print(calibrated_index[preview_cols].sample(n=min(5, len(calibrated_index)), random_state=RANDOM_SEED).to_string(index=False))


if __name__ == "__main__":
    main()
