#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate initial Problem-2 rainfall fields from Top-K historical templates.

This script performs only the initial storm-relative template generation:

    G_t^(0)(x', y') = sum_j w_j(t) * log(1 + R_sj(x', y'))
    R_t^(0)(x', y') = exp(G_t^(0)(x', y')) - 1

It does not run EOF/PCA, extreme quantile calibration, final georeferencing, or
pseudo-missing validation. All output metrics from generated fields use the
`initial_` prefix to keep them separate from later calibrated/final products.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RegularGridInterpolator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_INPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_halfhour_inputs_safe.csv"
TOPK_TABLE_PATH = PROJECT_ROOT / "data/processed/problem2_target_topk_similar_history.csv"
HISTORICAL_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"

INDEX_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_initial_fields_index.csv"
NPZ_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_initial_fields_topk_weighted.npz"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_initial_generation_qc_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_initial_generation"

GRID_SIZE = 201
GRID_EXTENT_KM = 1000.0
MIN_VALID_TEMPLATES = 5
CACHE_SIZE = 1000
MAKE_FIGURES = True
RANDOM_SEED = 2026
NAN_SKIP_THRESHOLD = 0.50
KM_PER_DEG = 111.32
EPS = 1e-12

HISTORY_METRIC_COLS = [
    "history_rain_max_mmhr",
    "history_rain_p95_mmhr",
    "history_rain_p99_mmhr",
    "history_rain_area_10_km2",
    "history_centroid_offset_km",
    "history_anisotropy",
]

WEIGHTED_HISTORY_OUTPUT_MAP = {
    "history_rain_max_mmhr": "topk_weighted_history_rain_max_mmhr",
    "history_rain_p95_mmhr": "topk_weighted_history_rain_p95_mmhr",
    "history_rain_p99_mmhr": "topk_weighted_history_rain_p99_mmhr",
    "history_rain_area_10_km2": "topk_weighted_history_rain_area_10_km2",
    "history_centroid_offset_km": "topk_weighted_history_centroid_offset_km",
    "history_anisotropy": "topk_weighted_history_anisotropy",
}


# =========================
# Basic helpers
# =========================


def resolve_project_path(path: object) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_lon_180(lon: object) -> float:
    value = pd.to_numeric(pd.Series([lon]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def format_time(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def stats_summary(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return {"min": np.nan, "mean": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "min": float(s.min(skipna=True)),
        "mean": float(s.mean(skipna=True)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max(skipna=True)),
    }


def stats_text(series: pd.Series) -> str:
    stats = stats_summary(series)
    return ", ".join(f"{k}={v:.6g}" if np.isfinite(v) else f"{k}=NA" for k, v in stats.items())


def iter_progress(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


# =========================
# Loaders
# =========================


def load_target_inputs() -> pd.DataFrame:
    df = pd.read_csv(TARGET_INPUT_PATH, encoding="utf-8-sig", low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    required = ["target_id", "typhoon_name", "time", "lat", "lon_180", "move_dir_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Target input is missing required columns: {missing}")
    return df


def load_topk_table() -> pd.DataFrame:
    df = pd.read_csv(TOPK_TABLE_PATH, encoding="utf-8-sig", low_memory=False)
    for col in ["target_time", "history_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    required = [
        "target_id",
        "history_sample_id",
        "history_tif_path",
        "history_lat",
        "history_lon_180",
        "history_move_dir_deg",
        "similarity_weight",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Top-K table is missing required columns: {missing}")
    return df


def load_historical_library() -> pd.DataFrame:
    df = pd.read_csv(HISTORICAL_LIBRARY_PATH, encoding="utf-8-sig", low_memory=False)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def supplement_topk_from_history(topk: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = topk.copy()
    hist_cols = [
        "sample_id",
        "tif_path",
        "lat",
        "lon_180",
        "move_dir_deg",
        "gpm_center_lon",
        "gpm_center_lat",
    ]
    available = [c for c in hist_cols if c in history.columns]
    if "sample_id" not in available:
        return out
    hist_small = history[available].drop_duplicates("sample_id")
    merged = out.merge(hist_small, left_on="history_sample_id", right_on="sample_id", how="left")

    fill_pairs = {
        "history_tif_path": "tif_path",
        "history_lat": "lat",
        "history_lon_180": "lon_180",
        "history_move_dir_deg": "move_dir_deg",
    }
    for left, right in fill_pairs.items():
        if right in merged.columns:
            merged[left] = merged[left].where(merged[left].notna(), merged[right])

    for col in ["gpm_center_lon", "gpm_center_lat"]:
        if col in merged.columns and f"history_{col}" not in merged.columns:
            merged[f"history_{col}"] = merged[col]

    drop_cols = [c for c in ["sample_id", "tif_path", "lat", "lon_180", "move_dir_deg"] if c in merged.columns]
    return merged.drop(columns=drop_cols)


# =========================
# Raster and relative grid
# =========================


def read_gpm_tif(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        transform = src.transform
        height, width = arr.shape

    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= np.isclose(arr, float(nodata))
    invalid |= np.isclose(arr, -9999.0)
    negative_count = int(np.count_nonzero(np.isfinite(arr) & (arr < 0.0) & ~invalid))
    arr[invalid] = np.nan
    arr[np.isfinite(arr) & (arr < 0.0)] = 0.0

    lon1d, lat1d = get_raster_lat_lon_1d(transform, height, width)
    meta = {
        "shape": [int(height), int(width)],
        "nodata": nodata,
        "negative_cell_count": negative_count,
        "all_missing": bool(np.count_nonzero(np.isfinite(arr)) == 0),
    }
    return arr, lat1d, lon1d, meta


def get_raster_lat_lon_1d(transform, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    cols = np.arange(width, dtype=float) + 0.5
    rows = np.arange(height, dtype=float) + 0.5
    lon1d = transform.c + transform.a * cols + transform.b * 0.5
    lat1d = transform.f + transform.e * rows + transform.d * 0.5
    return lon1d.astype(float), lat1d.astype(float)


def build_relative_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_front_km = np.linspace(-GRID_EXTENT_KM, GRID_EXTENT_KM, GRID_SIZE, dtype=np.float64)
    y_left_km = np.linspace(-GRID_EXTENT_KM, GRID_EXTENT_KM, GRID_SIZE, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_front_km, y_left_km)
    return x_front_km, y_left_km, x_grid, y_grid


def prepare_interpolator(
    rain: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lon: float,
) -> Tuple[RegularGridInterpolator, str]:
    lon = lon1d.astype(float).copy()
    convention = "180"
    if np.nanmax(lon) > 180.0 and center_lon >= 0:
        convention = "360"
    else:
        lon = np.array([normalize_lon_180(v) for v in lon], dtype=float)

    lat_order = np.argsort(lat1d)
    lon_order = np.argsort(lon)
    lat_sorted = lat1d[lat_order]
    lon_sorted = lon[lon_order]
    rain_sorted = rain[np.ix_(lat_order, lon_order)]

    interpolator = RegularGridInterpolator(
        (lat_sorted, lon_sorted),
        rain_sorted,
        bounds_error=False,
        fill_value=np.nan,
        method="linear",
    )
    return interpolator, convention


def sample_history_field_to_relative_grid(
    rain: np.ndarray,
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    center_lat: float,
    center_lon_180: float,
    move_dir_deg: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not all(np.isfinite([center_lat, center_lon_180, move_dir_deg])):
        raise ValueError("history center lat/lon or move_dir_deg is missing")

    interpolator, lon_convention = prepare_interpolator(rain, lat1d, lon1d, center_lon_180)
    alpha = math.radians(move_dir_deg % 360.0)
    x_east = x_grid * math.sin(alpha) - y_grid * math.cos(alpha)
    y_north = x_grid * math.cos(alpha) + y_grid * math.sin(alpha)
    lat_query = center_lat + y_north / KM_PER_DEG
    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) <= EPS:
        raise ValueError("center latitude too close to pole for lon conversion")
    lon_query = center_lon_180 + x_east / (KM_PER_DEG * cos_lat)
    if lon_convention == "360":
        lon_query = np.where(lon_query < 0.0, lon_query + 360.0, lon_query)
    else:
        lon_query = ((lon_query + 180.0) % 360.0) - 180.0

    points = np.column_stack([lat_query.ravel(), lon_query.ravel()])
    sampled = interpolator(points).reshape(x_grid.shape).astype("float32")
    sampled[np.isfinite(sampled) & (sampled < 0.0)] = 0.0
    nan_ratio = float(np.isnan(sampled).mean())
    return sampled, {"nan_ratio": nan_ratio, "lon_convention": lon_convention}


class TemplateCache:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, Tuple[Optional[np.ndarray], Dict[str, object]]] = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[Optional[np.ndarray], Dict[str, object]]]:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def put(self, key: str, value: Tuple[Optional[np.ndarray], Dict[str, object]]) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


def get_cached_history_template(
    row: pd.Series,
    cache: TemplateCache,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    qc_events: List[Dict[str, object]],
    counters: Dict[str, int],
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    sample_id = str(row.get("history_sample_id", ""))
    tif_path = str(row.get("history_tif_path", ""))
    move_dir = pd.to_numeric(pd.Series([row.get("history_move_dir_deg")]), errors="coerce").iloc[0]
    key = f"{sample_id}|{tif_path}|{move_dir:.6f}" if pd.notna(move_dir) else f"{sample_id}|{tif_path}|nan"
    cached = cache.get(key)
    if cached is not None:
        counters["cache_hits"] += 1
        return cached

    counters["cache_misses"] += 1
    status = {
        "history_sample_id": sample_id,
        "history_tif_path": tif_path,
        "ok": False,
        "reason": "",
        "nan_ratio": np.nan,
    }

    center_lat = pd.to_numeric(pd.Series([row.get("history_lat")]), errors="coerce").iloc[0]
    center_lon = pd.to_numeric(pd.Series([row.get("history_lon_180")]), errors="coerce").iloc[0]
    if pd.isna(center_lat) or pd.isna(center_lon):
        status["reason"] = "missing_history_center"
        counters["missing_center"] += 1
        qc_events.append(status.copy())
        cache.put(key, (None, status))
        return None, status
    if pd.isna(move_dir):
        status["reason"] = "missing_history_move_dir_deg"
        counters["missing_move_dir"] += 1
        qc_events.append(status.copy())
        cache.put(key, (None, status))
        return None, status

    path = resolve_project_path(tif_path)
    if not path.exists():
        status["reason"] = "history_tif_path_not_exists"
        counters["missing_tif"] += 1
        qc_events.append(status.copy())
        cache.put(key, (None, status))
        return None, status

    try:
        rain, lat1d, lon1d, raster_meta = read_gpm_tif(path)
        counters["read_attempts"] += 1
        if raster_meta["all_missing"]:
            status["reason"] = "tif_all_missing"
            counters["all_missing_tif"] += 1
            qc_events.append(status.copy())
            cache.put(key, (None, status))
            return None, status
        sampled, sample_meta = sample_history_field_to_relative_grid(
            rain,
            lat1d,
            lon1d,
            float(center_lat),
            float(center_lon),
            float(move_dir),
            x_grid,
            y_grid,
        )
        status["nan_ratio"] = sample_meta["nan_ratio"]
        if sample_meta["nan_ratio"] > NAN_SKIP_THRESHOLD:
            status["reason"] = "resampled_nan_ratio_too_high"
            counters["resample_nan_too_high"] += 1
            qc_events.append(status.copy())
            cache.put(key, (None, status))
            return None, status
        log_template = np.log1p(sampled).astype("float32")
        status["ok"] = True
        status["reason"] = "ok"
        counters["templates_ok"] += 1
        cache.put(key, (log_template, status))
        return log_template, status
    except Exception as exc:
        status["reason"] = f"read_or_resample_failed: {type(exc).__name__}: {exc}"
        counters["read_or_resample_failed"] += 1
        qc_events.append(status.copy())
        cache.put(key, (None, status))
        return None, status


# =========================
# Generation and metrics
# =========================


def compute_weighted_topk_history_metrics(topk_one: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    weights = pd.to_numeric(topk_one["similarity_weight"], errors="coerce").to_numpy(dtype=float)
    for input_col, output_col in WEIGHTED_HISTORY_OUTPUT_MAP.items():
        if input_col not in topk_one.columns:
            out[output_col] = np.nan
            continue
        values = pd.to_numeric(topk_one[input_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not valid.any():
            out[output_col] = np.nan
            continue
        w = weights[valid]
        out[output_col] = float(np.sum(w * values[valid]) / np.sum(w))
    return out


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
) -> Dict[str, float]:
    rain = np.asarray(rain_mmhr, dtype=float)
    valid = np.isfinite(rain)
    values = rain[valid]
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    cell_area = dx * dy
    out: Dict[str, float] = {
        "initial_rain_mean_mmhr": np.nan,
        "initial_rain_max_mmhr": np.nan,
        "initial_rain_p50_mmhr": np.nan,
        "initial_rain_p75_mmhr": np.nan,
        "initial_rain_p90_mmhr": np.nan,
        "initial_rain_p95_mmhr": np.nan,
        "initial_rain_p99_mmhr": np.nan,
        "initial_rain_sum_halfhour_mm": np.nan,
        "initial_rain_volume_proxy_mm_km2": np.nan,
        "initial_rain_area_1_km2": np.nan,
        "initial_rain_area_5_km2": np.nan,
        "initial_rain_area_10_km2": np.nan,
        "initial_rain_area_20_km2": np.nan,
        "initial_heavy_rain_fraction_10": np.nan,
        "initial_centroid_x_front_km": np.nan,
        "initial_centroid_y_left_km": np.nan,
        "initial_centroid_offset_km": np.nan,
        "initial_centroid_angle_deg": np.nan,
        "initial_asym_front_back_ratio": np.nan,
        "initial_asym_left_right_ratio": np.nan,
        "initial_quad_front_left_sum": np.nan,
        "initial_quad_front_right_sum": np.nan,
        "initial_quad_back_left_sum": np.nan,
        "initial_quad_back_right_sum": np.nan,
        "initial_quad_front_left_ratio": np.nan,
        "initial_quad_front_right_ratio": np.nan,
        "initial_quad_back_left_ratio": np.nan,
        "initial_quad_back_right_ratio": np.nan,
        "initial_anisotropy": np.nan,
        "initial_rain_radius_r50_km": np.nan,
        "initial_rain_radius_r80_km": np.nan,
        "initial_rain_radius_r90_km": np.nan,
        "initial_rain_band_width_km": np.nan,
    }
    if values.size == 0:
        return out

    out.update(
        {
            "initial_rain_mean_mmhr": float(np.nanmean(values)),
            "initial_rain_max_mmhr": float(np.nanmax(values)),
            "initial_rain_p50_mmhr": float(np.nanpercentile(values, 50)),
            "initial_rain_p75_mmhr": float(np.nanpercentile(values, 75)),
            "initial_rain_p90_mmhr": float(np.nanpercentile(values, 90)),
            "initial_rain_p95_mmhr": float(np.nanpercentile(values, 95)),
            "initial_rain_p99_mmhr": float(np.nanpercentile(values, 99)),
            "initial_rain_sum_halfhour_mm": float(np.nansum(rain * 0.5)),
            "initial_rain_volume_proxy_mm_km2": float(np.nansum(rain * 0.5 * cell_area)),
        }
    )
    for threshold in [1, 5, 10, 20]:
        out[f"initial_rain_area_{threshold}_km2"] = float(np.count_nonzero(valid & (rain >= threshold)) * cell_area)
    out["initial_heavy_rain_fraction_10"] = float(np.count_nonzero(valid & (rain >= 10.0)) / np.count_nonzero(valid))

    weights = np.where(np.isfinite(rain) & (rain > 0.0), rain * 0.5, 0.0)
    total = float(np.sum(weights))
    if total <= EPS:
        return out

    cx = float(np.sum(x_grid * weights) / total)
    cy = float(np.sum(y_grid * weights) / total)
    out["initial_centroid_x_front_km"] = cx
    out["initial_centroid_y_left_km"] = cy
    out["initial_centroid_offset_km"] = float(math.hypot(cx, cy))
    out["initial_centroid_angle_deg"] = float((math.degrees(math.atan2(cy, cx)) + 360.0) % 360.0)

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
            "initial_asym_front_back_ratio": safe_div(front_sum, back_sum),
            "initial_asym_left_right_ratio": safe_div(left_sum, right_sum),
            "initial_quad_front_left_sum": fl_sum,
            "initial_quad_front_right_sum": fr_sum,
            "initial_quad_back_left_sum": bl_sum,
            "initial_quad_back_right_sum": br_sum,
            "initial_quad_front_left_ratio": safe_div(fl_sum, total),
            "initial_quad_front_right_ratio": safe_div(fr_sum, total),
            "initial_quad_back_left_ratio": safe_div(bl_sum, total),
            "initial_quad_back_right_ratio": safe_div(br_sum, total),
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
            out["initial_anisotropy"] = float((lam1 - lam2) / (lam1 + lam2))

    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    r50 = weighted_radius(radius, weights, 0.50)
    r80 = weighted_radius(radius, weights, 0.80)
    r90 = weighted_radius(radius, weights, 0.90)
    out["initial_rain_radius_r50_km"] = r50
    out["initial_rain_radius_r80_km"] = r80
    out["initial_rain_radius_r90_km"] = r90
    out["initial_rain_band_width_km"] = float(r90 - r50) if np.isfinite(r90) and np.isfinite(r50) else np.nan
    return out


def generate_initial_field_for_target(
    target_row: pd.Series,
    topk_one: pd.DataFrame,
    cache: TemplateCache,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    qc_events: List[Dict[str, object]],
    counters: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    weighted_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    valid_weight_sum_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    included_weights: List[float] = []
    skipped = 0

    for _, hist_row in topk_one.sort_values("rank").iterrows():
        weight = pd.to_numeric(pd.Series([hist_row.get("similarity_weight")]), errors="coerce").iloc[0]
        if pd.isna(weight) or float(weight) <= 0.0:
            skipped += 1
            counters["nonpositive_weight"] += 1
            continue
        log_template, status = get_cached_history_template(hist_row, cache, x_grid, y_grid, qc_events, counters)
        if log_template is None:
            skipped += 1
            continue
        valid = np.isfinite(log_template)
        if not valid.any():
            skipped += 1
            counters["template_no_valid_pixels"] += 1
            continue
        w = float(weight)
        weighted_sum[valid] += w * log_template[valid]
        valid_weight_sum_grid[valid] += w
        included_weights.append(w)

    valid_template_count = len(included_weights)
    before_renorm = float(np.sum(included_weights))
    log_initial = np.divide(
        weighted_sum,
        valid_weight_sum_grid,
        out=np.zeros_like(weighted_sum, dtype=np.float64),
        where=valid_weight_sum_grid > EPS,
    )
    rain_initial = np.expm1(log_initial)
    rain_initial[~np.isfinite(rain_initial)] = 0.0
    rain_initial[rain_initial < 0.0] = 0.0
    log_initial[~np.isfinite(log_initial)] = 0.0
    log_initial[log_initial < 0.0] = 0.0

    low_valid = valid_template_count < MIN_VALID_TEMPLATES
    status = {
        "topk_count": int(len(topk_one)),
        "valid_template_count": int(valid_template_count),
        "skipped_template_count": int(skipped),
        "valid_weight_sum_before_renorm": before_renorm,
        "low_valid_template_count": bool(low_valid),
        "generation_ok": bool(valid_template_count > 0),
    }
    if low_valid:
        qc_events.append(
            {
                "target_id": target_row.get("target_id"),
                "reason": "low_valid_template_count",
                "valid_template_count": int(valid_template_count),
            }
        )
    return rain_initial.astype("float32"), log_initial.astype("float32"), status


def build_initial_fields(
    target: pd.DataFrame,
    topk: pd.DataFrame,
    history: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, object]]:
    x_front_km, y_left_km, x_grid, y_grid = build_relative_grid()
    topk_full = supplement_topk_from_history(topk, history)
    topk_by_target = {tid: sub.copy() for tid, sub in topk_full.groupby("target_id", sort=False)}

    rain_fields = np.zeros((len(target), GRID_SIZE, GRID_SIZE), dtype="float32")
    log_fields = np.zeros((len(target), GRID_SIZE, GRID_SIZE), dtype="float32")
    index_rows: List[Dict[str, object]] = []
    qc_events: List[Dict[str, object]] = []
    counters = {
        "cache_hits": 0,
        "cache_misses": 0,
        "read_attempts": 0,
        "templates_ok": 0,
        "missing_center": 0,
        "missing_move_dir": 0,
        "missing_tif": 0,
        "all_missing_tif": 0,
        "resample_nan_too_high": 0,
        "read_or_resample_failed": 0,
        "template_no_valid_pixels": 0,
        "nonpositive_weight": 0,
    }
    cache = TemplateCache(CACHE_SIZE)

    iterator = iter_progress(target.iterrows(), total=len(target), desc="Generate initial fields")
    for field_index, (_, target_row) in enumerate(iterator):
        target_id = str(target_row["target_id"])
        topk_one = topk_by_target.get(target_id)
        if topk_one is None or topk_one.empty:
            qc_events.append({"target_id": target_id, "reason": "missing_topk_rows"})
            topk_one = pd.DataFrame(columns=topk_full.columns)
        rain_initial, log_initial, generation_status = generate_initial_field_for_target(
            target_row,
            topk_one,
            cache,
            x_grid,
            y_grid,
            qc_events,
            counters,
        )
        rain_fields[field_index] = rain_initial
        log_fields[field_index] = log_initial

        metrics = compute_rainfall_metrics_on_relative_grid(rain_initial, x_front_km, y_left_km, x_grid, y_grid)
        topk_metrics = compute_weighted_topk_history_metrics(topk_one) if not topk_one.empty else {
            v: np.nan for v in WEIGHTED_HISTORY_OUTPUT_MAP.values()
        }
        row = {
            "field_index": int(field_index),
            "target_id": target_id,
            "typhoon_name": target_row.get("typhoon_name"),
            "time": format_time(target_row.get("time")),
            "lat": target_row.get("lat"),
            "lon_180": target_row.get("lon_180"),
            "WND": target_row.get("WND"),
            "PRES": target_row.get("PRES"),
            "intensity": target_row.get("intensity"),
            "move_speed_kmh": target_row.get("move_speed_kmh"),
            "move_dir_deg": target_row.get("move_dir_deg"),
            "signed_coast_dist_km": target_row.get("signed_coast_dist_km"),
            "is_land": target_row.get("is_land"),
            "life_progress": target_row.get("life_progress"),
        }
        row.update(generation_status)
        row.update(metrics)
        row.update(topk_metrics)
        row["ratio_initial_to_topk_rain_max"] = safe_div(
            row["initial_rain_max_mmhr"], row["topk_weighted_history_rain_max_mmhr"]
        )
        row["ratio_initial_to_topk_rain_p95"] = safe_div(
            row["initial_rain_p95_mmhr"], row["topk_weighted_history_rain_p95_mmhr"]
        )
        row["ratio_initial_to_topk_rain_area_10"] = safe_div(
            row["initial_rain_area_10_km2"], row["topk_weighted_history_rain_area_10_km2"]
        )
        row["field_nan_count"] = int(np.count_nonzero(~np.isfinite(rain_initial)))
        row["field_inf_count"] = int(np.count_nonzero(np.isinf(rain_initial)))
        row["field_negative_count"] = int(np.count_nonzero(np.isfinite(rain_initial) & (rain_initial < 0.0)))
        row["field_all_zero"] = bool(np.all(np.nan_to_num(rain_initial, nan=0.0) == 0.0))
        if row["field_nan_count"] or row["field_inf_count"] or row["field_negative_count"] or row["field_all_zero"]:
            qc_events.append(
                {
                    "target_id": target_id,
                    "reason": "generated_field_quality_issue",
                    "nan": row["field_nan_count"],
                    "inf": row["field_inf_count"],
                    "negative": row["field_negative_count"],
                    "all_zero": row["field_all_zero"],
                }
            )
        index_rows.append(row)

    index_df = pd.DataFrame(index_rows)
    diagnostics = {
        "x_front_km": x_front_km,
        "y_left_km": y_left_km,
        "counters": counters,
        "qc_events": qc_events,
        "cache_final_size": len(cache),
        "unique_history_tif_path": int(topk_full["history_tif_path"].nunique()) if "history_tif_path" in topk_full else 0,
        "topk_rows": int(len(topk_full)),
    }
    return rain_fields, log_fields, index_df, diagnostics


def save_npz(rain_fields: np.ndarray, log_fields: np.ndarray, target: pd.DataFrame, x_front_km: np.ndarray, y_left_km: np.ndarray) -> None:
    np.savez_compressed(
        NPZ_OUTPUT_PATH,
        rain_mmhr_initial=rain_fields.astype("float32"),
        log_rain_initial=log_fields.astype("float32"),
        target_id=target["target_id"].astype(str).to_numpy(),
        typhoon_name=target["typhoon_name"].astype(str).to_numpy(),
        time=target["time"].map(format_time).astype(str).to_numpy(),
        lat=pd.to_numeric(target["lat"], errors="coerce").to_numpy(dtype="float32"),
        lon_180=pd.to_numeric(target["lon_180"], errors="coerce").to_numpy(dtype="float32"),
        move_dir_deg=pd.to_numeric(target["move_dir_deg"], errors="coerce").to_numpy(dtype="float32"),
        x_front_km=x_front_km.astype("float32"),
        y_left_km=y_left_km.astype("float32"),
    )


def save_index_csv(index_df: pd.DataFrame) -> None:
    index_df.to_csv(INDEX_OUTPUT_PATH, index=False, encoding="utf-8-sig")


# =========================
# Figures
# =========================


def safe_name(name: str) -> str:
    return str(name).replace("-", "_").replace(" ", "_")


def plot_field(
    ax,
    field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    title: str,
    vmax: Optional[float] = None,
):
    im = ax.imshow(
        field,
        origin="lower",
        extent=[x_front_km[0], x_front_km[-1], y_left_km[0], y_left_km[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax.axvline(0, color="white", linewidth=0.6, alpha=0.8)
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("x_front_km")
    ax.set_ylabel("y_left_km")
    ax.set_title(title, fontsize=9)
    return im


def representative_indices(sub_index: pd.DataFrame) -> List[int]:
    picks: List[int] = []
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        idx = (pd.to_numeric(sub_index["life_progress"], errors="coerce") - p).abs().idxmin()
        picks.append(int(sub_index.loc[idx, "field_index"]))
    max_p95_idx = int(sub_index.loc[pd.to_numeric(sub_index["initial_rain_p95_mmhr"], errors="coerce").idxmax(), "field_index"])
    if max_p95_idx not in picks:
        picks.append(max_p95_idx)
    return picks


def make_representative_figures(
    rain_fields: np.ndarray,
    index_df: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        picks = representative_indices(sub)
        n = len(picks)
        ncols = 3
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), constrained_layout=True)
        axes_arr = np.atleast_1d(axes).ravel()
        vmax = float(np.nanpercentile(rain_fields[picks], 99))
        vmax = max(vmax, 1.0)
        for ax, field_idx in zip(axes_arr, picks):
            row = index_df.loc[index_df["field_index"].eq(field_idx)].iloc[0]
            title = (
                f"{row['typhoon_name']} {row['time']}\n"
                f"WND={row['WND']:.1f}, PRES={row['PRES']:.1f}, "
                f"p95={row['initial_rain_p95_mmhr']:.2f}, max={row['initial_rain_max_mmhr']:.2f}"
            )
            im = plot_field(ax, rain_fields[field_idx], x_front_km, y_left_km, title, vmax=vmax)
        for ax in axes_arr[len(picks):]:
            ax.axis("off")
        fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.85, label="rain mm/hr")
        path = FIGURE_DIR / f"{safe_name(name)}_initial_representative_fields.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_cumulative_and_max_figures(
    rain_fields: np.ndarray,
    index_df: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        indices = sub["field_index"].astype(int).to_numpy()
        cumulative = np.sum(rain_fields[indices] * 0.5, axis=0)
        max_field = np.max(rain_fields[indices], axis=0)
        for kind, field, label in [
            ("cumulative", cumulative, "cumulative half-hour mm"),
            ("max", max_field, "max rain mm/hr"),
        ]:
            fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
            vmax = float(np.nanpercentile(field, 99))
            vmax = max(vmax, 1.0)
            im = plot_field(
                ax,
                field,
                x_front_km,
                y_left_km,
                f"{name} initial {kind} storm-relative field",
                vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label=label)
            filename = f"{safe_name(name)}_initial_{kind}_storm_relative.png"
            path = FIGURE_DIR / filename
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def make_timeseries_figures(index_df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
        axes[0].plot(sub["time"], sub["initial_rain_max_mmhr"], label="initial_rain_max_mmhr")
        axes[0].plot(sub["time"], sub["initial_rain_p95_mmhr"], label="initial_rain_p95_mmhr")
        axes[0].set_ylabel("mm/hr")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_title(f"{name} initial generation time series")

        axes[1].plot(sub["time"], sub["initial_rain_area_10_km2"], color="tab:green")
        axes[1].set_ylabel("area >=10 km2")

        axes[2].plot(sub["time"], sub["WND"], label="WND", color="tab:red")
        ax2 = axes[2].twinx()
        ax2.plot(sub["time"], sub["PRES"], label="PRES", color="tab:blue")
        axes[2].set_ylabel("WND")
        ax2.set_ylabel("PRES")
        axes[2].set_xlabel("time")

        path = FIGURE_DIR / f"{safe_name(name)}_initial_timeseries.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_figures(
    rain_fields: np.ndarray,
    index_df: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if not MAKE_FIGURES:
        return paths
    paths.extend(make_representative_figures(rain_fields, index_df, x_front_km, y_left_km))
    paths.extend(make_cumulative_and_max_figures(rain_fields, index_df, x_front_km, y_left_km))
    paths.extend(make_timeseries_figures(index_df))
    return paths


# =========================
# QC
# =========================


def per_typhoon_qc(index_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        max_rain_row = sub.loc[pd.to_numeric(sub["initial_rain_max_mmhr"], errors="coerce").idxmax()]
        max_p95_row = sub.loc[pd.to_numeric(sub["initial_rain_p95_mmhr"], errors="coerce").idxmax()]
        max_area_row = sub.loc[pd.to_numeric(sub["initial_rain_area_10_km2"], errors="coerce").idxmax()]
        total_proxy = float(pd.to_numeric(sub["initial_rain_volume_proxy_mm_km2"], errors="coerce").sum(skipna=True))
        duration_proxy = float((pd.to_numeric(sub["initial_rain_area_10_km2"], errors="coerce") > 0).sum() * 0.5)
        lines.extend(
            [
                f"### {name}",
                f"- 时刻数: {len(sub)}",
                f"- initial_rain_max_mmhr 最大值: {max_rain_row['initial_rain_max_mmhr']:.6g} at {max_rain_row['time']}",
                f"- initial_rain_p95_mmhr 最大值: {max_p95_row['initial_rain_p95_mmhr']:.6g} at {max_p95_row['time']}",
                f"- initial_rain_area_10_km2 最大值: {max_area_row['initial_rain_area_10_km2']:.6g} at {max_area_row['time']}",
                f"- 累计降水总量 proxy: {total_proxy:.6g}",
                f"- 强降水持续时间 proxy: {duration_proxy:.6g} hours",
            ]
        )
    return lines


def time_continuity_qc(index_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        lines.append(f"### {name}")
        for col in ["initial_rain_max_mmhr", "initial_rain_p95_mmhr", "initial_rain_area_10_km2"]:
            diff = pd.to_numeric(sub[col], errors="coerce").diff().abs()
            lines.append(
                f"- diff({col}): P95={float(diff.quantile(0.95)):.6g}, max={float(diff.max(skipna=True)):.6g}"
            )
    return lines


def write_qc_report(
    target: pd.DataFrame,
    topk: pd.DataFrame,
    index_df: pd.DataFrame,
    rain_fields: np.ndarray,
    log_fields: np.ndarray,
    diagnostics: Mapping[str, object],
    figure_paths: Sequence[Path],
) -> None:
    counters = diagnostics["counters"]
    qc_events = diagnostics["qc_events"]
    target_counts = target.groupby("typhoon_name")["time"].agg(["size", "min", "max"])
    nan_count = int(np.count_nonzero(~np.isfinite(rain_fields)))
    inf_count = int(np.count_nonzero(np.isinf(rain_fields)))
    negative_count = int(np.count_nonzero(np.isfinite(rain_fields) & (rain_fields < 0.0)))
    all_zero_count = int(index_df["field_all_zero"].sum())
    topk_actual = int(topk.groupby("target_id").size().median()) if len(topk) else 0
    smooth_warning = (
        "初始模板场存在极端降水平滑倾向，需在后续极端分位数校准步骤中修正。"
        if float(pd.to_numeric(index_df["ratio_initial_to_topk_rain_max"], errors="coerce").median(skipna=True)) < 0.7
        else "ratio_initial_to_topk_rain_max 中位数未低于 0.7。"
    )

    figure_summary = pd.Series([p.name for p in figure_paths]).to_string(index=False) if figure_paths else "(none)"
    event_preview = pd.DataFrame(qc_events).head(30).to_string(index=False) if qc_events else "(none)"

    lines = [
        "# Problem 2 Initial Rainfall Generation QC Report",
        "",
        "## 1. 输入文件与输出文件",
        f"- target input: `{TARGET_INPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Top-K 表: `{TOPK_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 历史库: `{HISTORICAL_LIBRARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- NPZ 输出: `{NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- index CSV 输出: `{INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- figures 输出目录: `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. 运行参数",
        f"- GRID_SIZE: {GRID_SIZE}",
        f"- GRID_EXTENT_KM: {GRID_EXTENT_KM}",
        f"- TOPK 实际值: {topk_actual}",
        f"- MIN_VALID_TEMPLATES: {MIN_VALID_TEMPLATES}",
        f"- CACHE_SIZE: {CACHE_SIZE}",
        f"- MAKE_FIGURES: {MAKE_FIGURES}",
        f"- NAN_SKIP_THRESHOLD: {NAN_SKIP_THRESHOLD}",
        "",
        "## 3. 目标样本统计",
        f"- 总目标时刻数: {len(target)}",
        "```",
        target_counts.to_string(),
        "```",
        f"- NPZ rain_mmhr_initial shape: {list(rain_fields.shape)}",
        f"- NPZ log_rain_initial shape: {list(log_fields.shape)}",
        f"- index CSV 行列数: {index_df.shape[0]} x {index_df.shape[1]}",
        "",
        "## 4. 模板读取与重采样统计",
        f"- Top-K 总行数: {len(topk)}",
        f"- 唯一 history_tif_path 数: {diagnostics['unique_history_tif_path']}",
        f"- 成功读取/重采样模板数: {counters['templates_ok']}",
        f"- 读取尝试数: {counters['read_attempts']}",
        f"- 读取失败或重采样失败数: {counters['read_or_resample_failed']}",
        f"- tif 不存在数: {counters['missing_tif']}",
        f"- tif 全缺测数: {counters['all_missing_tif']}",
        f"- 历史中心缺失数: {counters['missing_center']}",
        f"- history_move_dir_deg 缺失数: {counters['missing_move_dir']}",
        f"- 重采样 NaN 比例过高数: {counters['resample_nan_too_high']}",
        f"- cache hits/misses/final_size: {counters['cache_hits']} / {counters['cache_misses']} / {diagnostics['cache_final_size']}",
        f"- 平均 valid_template_count: {float(index_df['valid_template_count'].mean()):.6f}",
        f"- 最小 valid_template_count: {int(index_df['valid_template_count'].min())}",
        f"- low_valid_template_count 数量: {int(index_df['low_valid_template_count'].sum())}",
        f"- 是否存在 generation_ok=False: {bool((~index_df['generation_ok'].astype(bool)).any())}",
        "",
        "## 5. 生成场基本检查",
        f"- NaN 数量: {nan_count}",
        f"- Inf 数量: {inf_count}",
        f"- 负值数量: {negative_count}",
        f"- 全零场数量: {all_zero_count}",
        f"- rain_max 分布: {stats_text(index_df['initial_rain_max_mmhr'])}",
        f"- rain_p95 分布: {stats_text(index_df['initial_rain_p95_mmhr'])}",
        f"- rain_area_10_km2 分布: {stats_text(index_df['initial_rain_area_10_km2'])}",
        f"- centroid_offset 分布: {stats_text(index_df['initial_centroid_offset_km'])}",
        f"- anisotropy 分布: {stats_text(index_df['initial_anisotropy'])}",
        "",
        "## 6. 分台风统计",
        *per_typhoon_qc(index_df),
        "",
        "## 7. 平滑风险检查",
        f"- ratio_initial_to_topk_rain_max: {stats_text(index_df['ratio_initial_to_topk_rain_max'])}",
        f"- ratio_initial_to_topk_rain_p95: {stats_text(index_df['ratio_initial_to_topk_rain_p95'])}",
        f"- ratio_initial_to_topk_rain_area_10: {stats_text(index_df['ratio_initial_to_topk_rain_area_10'])}",
        f"- 判断: {smooth_warning}",
        "",
        "## 8. 时间连续性初检",
        *time_continuity_qc(index_df),
        "",
        "## 9. 图件",
        f"- 图件数量: {len(figure_paths)}",
        "```",
        figure_summary,
        "```",
        "",
        "## 10. 异常与日志样例",
        f"- 记录事件总数: {len(qc_events)}",
        "```",
        event_preview,
        "```",
        "",
        "## 11. 防泄漏声明",
        "本步骤使用 18 号 Top-K 检索结果读取历史 GPM 降水模板，生成 KONG-REY 和 MAN-YI 的初始降水场。"
        "目标台风输入不包含任何真实 GPM 降水信息；rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等历史降水字段"
        "仅用于历史模板指标对比和后续校准，不参与目标台风输入或相似度检索。",
    ]
    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def main() -> None:
    np.random.seed(RANDOM_SEED)
    INDEX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NPZ_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("[19] Loading target inputs")
    target = load_target_inputs()
    print("[19] Loading Top-K table")
    topk = load_topk_table()
    print("[19] Loading historical library for supplementation/checks")
    history = load_historical_library()

    print("[19] Building storm-relative initial fields")
    rain_fields, log_fields, index_df, diagnostics = build_initial_fields(target, topk, history)

    x_front_km = diagnostics["x_front_km"]
    y_left_km = diagnostics["y_left_km"]
    print("[19] Saving NPZ")
    save_npz(rain_fields, log_fields, target, x_front_km, y_left_km)
    print("[19] Saving index CSV")
    save_index_csv(index_df)
    print("[19] Making figures")
    figure_paths = make_figures(rain_fields, index_df, x_front_km, y_left_km)
    print("[19] Writing QC report")
    write_qc_report(target, topk, index_df, rain_fields, log_fields, diagnostics, figure_paths)

    valid_mean = float(index_df["valid_template_count"].mean())
    valid_min = int(index_df["valid_template_count"].min())
    generation_all_ok = bool(index_df["generation_ok"].astype(bool).all())
    nan_count = int(np.count_nonzero(~np.isfinite(rain_fields)))
    inf_count = int(np.count_nonzero(np.isinf(rain_fields)))
    negative_count = int(np.count_nonzero(np.isfinite(rain_fields) & (rain_fields < 0.0)))
    all_zero_count = int(index_df["field_all_zero"].sum())
    fig_files = list(FIGURE_DIR.glob("*.png"))

    print("\n========== Problem-2 initial rainfall generation complete ==========")
    print("Script: scripts/19_generate_initial_rainfall_fields_from_topk.py")
    print(f"NPZ: {NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Index CSV: {INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"QC report: {QC_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure dir: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"rain_mmhr_initial shape: {rain_fields.shape}")
    print(f"Index shape: {index_df.shape[0]} x {index_df.shape[1]}")
    print(index_df.groupby("typhoon_name").size().to_string())
    print(f"valid_template_count mean/min: {valid_mean:.6f} / {valid_min}")
    print(f"generation_ok all True: {generation_all_ok}")
    print(f"NaN/Inf/negative/all-zero counts: {nan_count} / {inf_count} / {negative_count} / {all_zero_count}")
    for col in ["initial_rain_max_mmhr", "initial_rain_p95_mmhr", "initial_rain_area_10_km2"]:
        s = pd.to_numeric(index_df[col], errors="coerce")
        print(f"{col} P50/P95/max: {s.quantile(0.50):.6f} / {s.quantile(0.95):.6f} / {s.max():.6f}")
    ratio = pd.to_numeric(index_df["ratio_initial_to_topk_rain_max"], errors="coerce")
    print(f"ratio_initial_to_topk_rain_max P50/P95: {ratio.quantile(0.50):.6f} / {ratio.quantile(0.95):.6f}")
    print(f"Figure count: {len(fig_files)}")
    sample_cols = [
        "typhoon_name",
        "time",
        "lat",
        "lon_180",
        "WND",
        "PRES",
        "valid_template_count",
        "initial_rain_max_mmhr",
        "initial_rain_p95_mmhr",
        "initial_rain_area_10_km2",
        "initial_centroid_offset_km",
        "initial_anisotropy",
        "ratio_initial_to_topk_rain_max",
        "generation_ok",
    ]
    print("\nRandom 5 index key rows:")
    print(index_df[sample_cols].sample(min(5, len(index_df)), random_state=RANDOM_SEED).to_string(index=False))


if __name__ == "__main__":
    main()
