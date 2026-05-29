#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 EOF/PCA large-scale structure diagnosis and low-rank correction.

This step builds an EOF/PCA basis from historical storm-relative GPM rainfall
templates, projects the step-19 target initial fields onto that basis, and
creates a beta-blended structure-corrected field. It does not run extreme
quantile calibration, final georeferencing, or pseudo-missing validation.
"""

from __future__ import annotations

import gc
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import IncrementalPCA

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

INITIAL_NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_generated_initial_fields_topk_weighted.npz"
INITIAL_INDEX_PATH = PROJECT_ROOT / "data/processed/problem2_generated_initial_fields_index.csv"
TOPK_TABLE_PATH = PROJECT_ROOT / "data/processed/problem2_target_topk_similar_history.csv"
HISTORICAL_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"
TARGET_INPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_halfhour_inputs_safe.csv"

MODEL_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_eof_pca_model.npz"
COEFF_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_eof_coefficients.csv"
BLENDED_NPZ_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_pca_blended_fields.npz"
BLENDED_INDEX_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_generated_pca_blended_fields_index.csv"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_eof_pca_correction_qc_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_eof_pca_correction"

PCA_TRAINING_SOURCE = "unique_topk_history_templates"
MAX_PCA_TEMPLATES = 5000
N_COMPONENTS = 20
BATCH_SIZE = 128
BETA_BLEND = 0.3
GRID_SIZE = 201
GRID_EXTENT_KM = 1000.0
RANDOM_SEED = 2026
MAKE_FIGURES = True

NAN_SKIP_THRESHOLD = 0.50
KM_PER_DEG = 111.32
EPS = 1e-12
TARGET_TYPHOON_NAMES = {"KONGREY", "KONG-REY", "KONG_REY", "MAN-YI", "MAN YI", "MANYI", "MAN_YI"}


# =========================
# Basic helpers
# =========================


def resolve_project_path(path: object) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_name(value: object) -> str:
    text = str(value).upper().strip()
    return text.replace(" ", "").replace("-", "").replace("_", "")


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


def safe_name(name: object) -> str:
    return str(name).replace("-", "_").replace(" ", "_")


def iter_progress(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def stats_summary(values: object) -> Dict[str, float]:
    s = pd.to_numeric(pd.Series(values).ravel() if isinstance(values, np.ndarray) else values, errors="coerce")
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


def quantile_text(values: object, quantiles: Sequence[float] = (0.50, 0.95)) -> str:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return ", ".join(f"P{int(q * 100)}=NA" for q in quantiles)
    return ", ".join(f"P{int(q * 100)}={float(s.quantile(q)):.6g}" for q in quantiles)


def as_str_array(values: object) -> np.ndarray:
    return np.asarray(values).astype(str)


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


# =========================
# Loaders
# =========================


def load_initial_npz() -> Dict[str, np.ndarray]:
    if not INITIAL_NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing step-19 initial NPZ: {INITIAL_NPZ_PATH}")
    with np.load(INITIAL_NPZ_PATH, allow_pickle=True) as z:
        required = [
            "rain_mmhr_initial",
            "log_rain_initial",
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
            raise RuntimeError(f"Initial NPZ is missing required arrays: {missing}")
        out = {k: z[k] for k in required}
    rain = np.asarray(out["rain_mmhr_initial"], dtype=np.float32)
    log_rain = np.asarray(out["log_rain_initial"], dtype=np.float32)
    if rain.shape != log_rain.shape:
        raise RuntimeError(f"Initial rain/log shapes differ: {rain.shape} vs {log_rain.shape}")
    if rain.ndim != 3 or rain.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"Unexpected initial field shape: {rain.shape}")
    out["rain_mmhr_initial"] = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    out["log_rain_initial"] = np.nan_to_num(log_rain, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    for key in ["target_id", "typhoon_name", "time"]:
        out[key] = as_str_array(out[key])
    for key in ["lat", "lon_180", "move_dir_deg", "x_front_km", "y_left_km"]:
        out[key] = np.asarray(out[key], dtype=np.float32)
    return out


def load_initial_index() -> pd.DataFrame:
    if not INITIAL_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing step-19 initial index: {INITIAL_INDEX_PATH}")
    df = pd.read_csv(INITIAL_INDEX_PATH, encoding="utf-8-sig", low_memory=False)
    required = ["field_index", "target_id", "typhoon_name", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Initial index is missing required columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("field_index").reset_index(drop=True)
    if not np.array_equal(df["field_index"].to_numpy(dtype=int), np.arange(len(df), dtype=int)):
        raise RuntimeError("Initial index field_index is not a contiguous 0-based sequence")
    return df


def load_topk_table() -> pd.DataFrame:
    if not TOPK_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing Top-K table: {TOPK_TABLE_PATH}")
    df = pd.read_csv(TOPK_TABLE_PATH, encoding="utf-8-sig", low_memory=False)
    for col in ["target_time", "history_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    required = [
        "history_sample_id",
        "history_tif_path",
        "history_lat",
        "history_lon_180",
        "history_move_dir_deg",
        "rank",
        "similarity_weight",
    ]
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


def validate_initial_order(initial: Mapping[str, np.ndarray], index_df: pd.DataFrame) -> None:
    n = int(initial["rain_mmhr_initial"].shape[0])
    if n != len(index_df):
        raise RuntimeError(f"NPZ field count {n} does not match initial index rows {len(index_df)}")
    npz_ids = as_str_array(initial["target_id"])
    idx_ids = index_df["target_id"].astype(str).to_numpy()
    if not np.array_equal(npz_ids, idx_ids):
        mismatch = np.where(npz_ids != idx_ids)[0][:10].tolist()
        raise RuntimeError(f"target_id order mismatch between NPZ and index. First mismatches: {mismatch}")


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


# =========================
# PCA training
# =========================


def _fill_from_history(topk: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = topk.copy()
    if history.empty or "sample_id" not in history.columns:
        return out
    hist_cols = [
        "sample_id",
        "event_uid",
        "typhoon_name",
        "time",
        "tif_path",
        "lat",
        "lon_180",
        "move_dir_deg",
        "WND",
        "PRES",
        "intensity",
        "move_speed_kmh",
        "signed_coast_dist_km",
        "is_land",
        "life_progress",
        "rain_max_mmhr",
        "rain_p95_mmhr",
        "rain_p99_mmhr",
        "rain_area_10_km2",
        "centroid_offset_km",
        "anisotropy",
        "rain_radius_r50_km",
        "rain_radius_r80_km",
        "rain_radius_r90_km",
        "rain_band_width_km",
        "target_excluded_flag",
    ]
    hist_cols = [c for c in hist_cols if c in history.columns]
    hist_small = history[hist_cols].drop_duplicates("sample_id")
    merged = out.merge(hist_small, left_on="history_sample_id", right_on="sample_id", how="left", suffixes=("", "_lib"))
    fill_pairs = {
        "history_event_uid": "event_uid",
        "history_typhoon_name": "typhoon_name",
        "history_time": "time",
        "history_tif_path": "tif_path",
        "history_lat": "lat",
        "history_lon_180": "lon_180",
        "history_move_dir_deg": "move_dir_deg",
        "history_WND": "WND",
        "history_PRES": "PRES",
        "history_intensity": "intensity",
        "history_move_speed_kmh": "move_speed_kmh",
        "history_signed_coast_dist_km": "signed_coast_dist_km",
        "history_is_land": "is_land",
        "history_life_progress": "life_progress",
        "history_rain_max_mmhr": "rain_max_mmhr",
        "history_rain_p95_mmhr": "rain_p95_mmhr",
        "history_rain_p99_mmhr": "rain_p99_mmhr",
        "history_rain_area_10_km2": "rain_area_10_km2",
        "history_centroid_offset_km": "centroid_offset_km",
        "history_anisotropy": "anisotropy",
        "history_rain_radius_r50_km": "rain_radius_r50_km",
        "history_rain_radius_r80_km": "rain_radius_r80_km",
        "history_rain_radius_r90_km": "rain_radius_r90_km",
        "history_rain_band_width_km": "rain_band_width_km",
        "history_target_excluded_flag": "target_excluded_flag",
    }
    for left, right in fill_pairs.items():
        if right in merged.columns:
            if left not in merged.columns:
                merged[left] = merged[right]
            else:
                merged[left] = merged[left].where(merged[left].notna(), merged[right])
    drop_cols = [c for c in hist_cols if c != "sample_id" and c in merged.columns]
    drop_cols.append("sample_id")
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


def select_pca_training_templates(topk: pd.DataFrame, history: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if PCA_TRAINING_SOURCE != "unique_topk_history_templates":
        raise RuntimeError(f"Unsupported PCA_TRAINING_SOURCE: {PCA_TRAINING_SOURCE}")

    topk_full = _fill_from_history(topk, history)
    unique_count = int(topk_full["history_sample_id"].nunique(dropna=True))
    if unique_count < N_COMPONENTS:
        raise RuntimeError(f"Top-K unique historical templates are fewer than N_COMPONENTS: {unique_count}")

    work = topk_full.copy()
    for col in [
        "rank",
        "similarity_weight",
        "history_WND",
        "history_PRES",
        "history_rain_max_mmhr",
        "history_rain_p95_mmhr",
        "history_rain_p99_mmhr",
        "history_rain_area_10_km2",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_spec = {
        "history_tif_path": ("history_tif_path", "first"),
        "history_event_uid": ("history_event_uid", "first") if "history_event_uid" in work.columns else ("history_sample_id", "first"),
        "history_typhoon_name": ("history_typhoon_name", "first") if "history_typhoon_name" in work.columns else ("history_sample_id", "first"),
        "history_time": ("history_time", "first") if "history_time" in work.columns else ("history_sample_id", "first"),
        "history_lat": ("history_lat", "first"),
        "history_lon_180": ("history_lon_180", "first"),
        "history_move_dir_deg": ("history_move_dir_deg", "first"),
        "rank_min": ("rank", "min"),
        "rank_mean": ("rank", "mean"),
        "similarity_weight_max": ("similarity_weight", "max"),
        "similarity_weight_mean": ("similarity_weight", "mean"),
        "similarity_weight_sum": ("similarity_weight", "sum"),
        "topk_occurrence_count": ("history_sample_id", "size"),
    }
    optional_first_cols = [
        "history_WND",
        "history_PRES",
        "history_intensity",
        "history_move_speed_kmh",
        "history_signed_coast_dist_km",
        "history_is_land",
        "history_life_progress",
        "history_target_excluded_flag",
    ]
    for col in optional_first_cols:
        if col in work.columns:
            agg_spec[col] = (col, "first")
    optional_max_cols = [
        "history_rain_max_mmhr",
        "history_rain_p95_mmhr",
        "history_rain_p99_mmhr",
        "history_rain_area_10_km2",
        "history_centroid_offset_km",
        "history_anisotropy",
        "history_rain_radius_r50_km",
        "history_rain_radius_r80_km",
        "history_rain_radius_r90_km",
        "history_rain_band_width_km",
    ]
    for col in optional_max_cols:
        if col in work.columns:
            agg_spec[col] = (col, "max")

    unique = work.groupby("history_sample_id", as_index=False, sort=False).agg(**agg_spec)
    unique["history_time"] = pd.to_datetime(unique["history_time"], errors="coerce")

    target_mask = unique["history_typhoon_name"].map(normalize_name).isin({"KONGREY", "MANYI"})
    target_like_count = int(target_mask.sum())
    unique = unique.loc[~target_mask].copy()
    if len(unique) < N_COMPONENTS:
        raise RuntimeError(
            f"PCA candidates after target-name exclusion are fewer than N_COMPONENTS: {len(unique)}"
        )

    rain_intensity = pd.to_numeric(unique.get("history_rain_p95_mmhr", np.nan), errors="coerce").fillna(0.0)
    if "history_rain_p99_mmhr" in unique.columns:
        rain_intensity = np.maximum(rain_intensity, pd.to_numeric(unique["history_rain_p99_mmhr"], errors="coerce").fillna(0.0))
    if "history_rain_max_mmhr" in unique.columns:
        rain_intensity = np.maximum(rain_intensity, pd.to_numeric(unique["history_rain_max_mmhr"], errors="coerce").fillna(0.0) / 5.0)
    unique["_rank_score"] = 1.0 / pd.to_numeric(unique["rank_min"], errors="coerce").clip(lower=1.0)
    unique["_weight_score"] = pd.to_numeric(unique["similarity_weight_max"], errors="coerce").fillna(0.0)
    unique["_rain_score"] = pd.Series(rain_intensity).rank(pct=True).to_numpy()
    unique["_score"] = (
        0.45 * unique["_rank_score"].rank(pct=True)
        + 0.35 * unique["_weight_score"].rank(pct=True)
        + 0.20 * unique["_rain_score"]
    )

    if len(unique) > MAX_PCA_TEMPLATES:
        selected_parts: List[pd.DataFrame] = []
        event_groups = list(unique.groupby("history_event_uid", sort=False))
        event_quota = max(1, MAX_PCA_TEMPLATES // max(len(event_groups), 1))
        for _, sub in event_groups:
            selected_parts.append(sub.sort_values("_score", ascending=False).head(event_quota))
        selected = pd.concat(selected_parts, ignore_index=False).drop_duplicates("history_sample_id")
        if len(selected) < MAX_PCA_TEMPLATES:
            remainder = unique.loc[~unique["history_sample_id"].isin(selected["history_sample_id"])]
            selected = pd.concat(
                [selected, remainder.sort_values("_score", ascending=False).head(MAX_PCA_TEMPLATES - len(selected))],
                ignore_index=False,
            )
        selected = selected.sort_values("_score", ascending=False).head(MAX_PCA_TEMPLATES).copy()
    else:
        selected = unique.sort_values(["rank_min", "similarity_weight_max"], ascending=[True, False]).copy()

    selected = selected.drop(columns=[c for c in ["_rank_score", "_weight_score", "_rain_score"] if c in selected.columns])
    diagnostics = {
        "topk_unique_history_templates": unique_count,
        "target_like_history_templates_excluded": target_like_count,
        "selected_before_resampling": int(len(selected)),
    }
    return selected.reset_index(drop=True), diagnostics


def build_pca_training_matrix(
    templates: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    n_candidates = len(templates)
    n_features = GRID_SIZE * GRID_SIZE
    matrix = np.empty((n_candidates, n_features), dtype=np.float32)
    metadata_rows: List[Dict[str, object]] = []
    skip_events: List[Dict[str, object]] = []
    counters: Counter = Counter()

    iterator = iter_progress(templates.iterrows(), total=n_candidates, desc="Build PCA templates")
    out_i = 0
    for _, row in iterator:
        sample_id = str(row.get("history_sample_id", ""))
        tif_path = row.get("history_tif_path")
        status = {
            "history_sample_id": sample_id,
            "history_tif_path": str(tif_path),
            "reason": "",
            "nan_ratio": np.nan,
        }
        center_lat = pd.to_numeric(pd.Series([row.get("history_lat")]), errors="coerce").iloc[0]
        center_lon = pd.to_numeric(pd.Series([row.get("history_lon_180")]), errors="coerce").iloc[0]
        move_dir = pd.to_numeric(pd.Series([row.get("history_move_dir_deg")]), errors="coerce").iloc[0]
        if pd.isna(center_lat) or pd.isna(center_lon):
            status["reason"] = "missing_history_center"
            counters["missing_history_center"] += 1
            skip_events.append(status)
            continue
        if pd.isna(move_dir):
            status["reason"] = "missing_history_move_dir_deg"
            counters["missing_history_move_dir_deg"] += 1
            skip_events.append(status)
            continue
        path = resolve_project_path(tif_path)
        if not path.exists():
            status["reason"] = "history_tif_path_not_exists"
            counters["history_tif_path_not_exists"] += 1
            skip_events.append(status)
            continue

        try:
            rain, lat1d, lon1d, raster_meta = read_gpm_tif(path)
            counters["read_attempts"] += 1
            if raster_meta["all_missing"]:
                status["reason"] = "tif_all_missing"
                counters["tif_all_missing"] += 1
                skip_events.append(status)
                continue
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
                counters["resampled_nan_ratio_too_high"] += 1
                skip_events.append(status)
                continue
            log_template = np.log1p(sampled).astype("float32")
            log_template = np.nan_to_num(log_template, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(log_template).all():
                status["reason"] = "log_template_nonfinite"
                counters["log_template_nonfinite"] += 1
                skip_events.append(status)
                continue
            matrix[out_i, :] = log_template.reshape(-1)
            meta = row.to_dict()
            meta["resampled_nan_ratio"] = sample_meta["nan_ratio"]
            meta["resolved_tif_path"] = str(path)
            metadata_rows.append(meta)
            counters["templates_ok"] += 1
            out_i += 1
        except Exception as exc:
            status["reason"] = f"read_or_resample_failed: {type(exc).__name__}: {exc}"
            counters["read_or_resample_failed"] += 1
            skip_events.append(status)

    if out_i < N_COMPONENTS:
        raise RuntimeError(f"PCA training samples after resampling are fewer than N_COMPONENTS: {out_i}")
    matrix = matrix[:out_i].copy()
    metadata = pd.DataFrame(metadata_rows)
    diagnostics = {
        "counters": dict(counters),
        "skip_events": skip_events,
        "n_candidates": n_candidates,
        "n_training_samples": int(out_i),
    }
    return matrix, metadata, diagnostics


def fit_eof_pca_model(training_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    if training_matrix.ndim != 2:
        raise RuntimeError(f"Training matrix must be 2-D, got {training_matrix.shape}")
    if training_matrix.shape[0] < N_COMPONENTS:
        raise RuntimeError(f"Training samples {training_matrix.shape[0]} < N_COMPONENTS {N_COMPONENTS}")
    if not np.isfinite(training_matrix).all():
        raise RuntimeError("PCA training matrix contains NaN or Inf")

    pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=max(BATCH_SIZE, N_COMPONENTS))
    pca.fit(training_matrix)

    mean_flat = np.asarray(pca.mean_, dtype=np.float64)
    components = np.asarray(pca.components_, dtype=np.float64)
    explained_variance = np.asarray(pca.explained_variance_, dtype=np.float64)
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(explained_variance_ratio)
    if not np.isfinite(mean_flat).all() or not np.isfinite(components).all() or not np.isfinite(explained_variance_ratio).all():
        raise RuntimeError("PCA training result contains NaN or Inf")

    return {
        "mean_log_field": mean_flat.reshape(GRID_SIZE, GRID_SIZE).astype("float32"),
        "eof_components": components.reshape(N_COMPONENTS, GRID_SIZE, GRID_SIZE).astype("float32"),
        "explained_variance": explained_variance.astype("float64"),
        "explained_variance_ratio": explained_variance_ratio.astype("float64"),
        "cumulative_explained_variance_ratio": cumulative.astype("float64"),
        "n_components": np.array(N_COMPONENTS, dtype=np.int32),
    }


def save_eof_pca_model(
    model: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    training_meta: pd.DataFrame,
) -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_OUTPUT_PATH,
        mean_log_field=np.asarray(model["mean_log_field"], dtype=np.float32),
        eof_components=np.asarray(model["eof_components"], dtype=np.float32),
        explained_variance=np.asarray(model["explained_variance"], dtype=np.float64),
        explained_variance_ratio=np.asarray(model["explained_variance_ratio"], dtype=np.float64),
        cumulative_explained_variance_ratio=np.asarray(model["cumulative_explained_variance_ratio"], dtype=np.float64),
        n_components=np.array(N_COMPONENTS, dtype=np.int32),
        beta_blend=np.array(BETA_BLEND, dtype=np.float32),
        x_front_km=x_front_km.astype("float32"),
        y_left_km=y_left_km.astype("float32"),
        training_sample_ids=training_meta["history_sample_id"].astype(str).to_numpy(dtype="U"),
        training_tif_paths=training_meta["history_tif_path"].astype(str).to_numpy(dtype="U"),
        training_typhoon_names=training_meta["history_typhoon_name"].astype(str).to_numpy(dtype="U"),
        training_times=training_meta["history_time"].map(format_time).astype(str).to_numpy(dtype="U"),
    )


# =========================
# Target projection and fields
# =========================


def project_target_initial_fields(log_initial: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    n = log_initial.shape[0]
    flat = log_initial.reshape(n, -1)
    mean_flat = np.asarray(model["mean_log_field"], dtype=np.float32).reshape(-1)
    components = np.asarray(model["eof_components"], dtype=np.float32).reshape(N_COMPONENTS, -1)
    coefficients = np.empty((n, N_COMPONENTS), dtype=np.float32)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x = np.nan_to_num(flat[start:end].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        coefficients[start:end] = (x - mean_flat) @ components.T
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Target EOF coefficients contain NaN or Inf")
    return coefficients


def reconstruct_eof_fields(
    log_initial: np.ndarray,
    coefficients: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    n = log_initial.shape[0]
    mean_flat = np.asarray(model["mean_log_field"], dtype=np.float32).reshape(-1)
    components = np.asarray(model["eof_components"], dtype=np.float32).reshape(N_COMPONENTS, -1)
    log_initial_flat = log_initial.reshape(n, -1)
    rain_eof = np.empty_like(log_initial, dtype=np.float32)
    log_eof = np.empty_like(log_initial, dtype=np.float32)
    rmse = np.empty(n, dtype=np.float32)
    corr = np.empty(n, dtype=np.float32)
    energy_ratio = np.empty(n, dtype=np.float32)
    initial_norm = np.empty(n, dtype=np.float32)
    residual_norm = np.empty(n, dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        coef = coefficients[start:end].astype(np.float32, copy=False)
        recon = mean_flat + coef @ components
        recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        x = np.nan_to_num(log_initial_flat[start:end].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        resid = x - recon
        rmse[start:end] = np.sqrt(np.mean(resid * resid, axis=1)).astype(np.float32)
        centered = x - mean_flat
        init_norm_batch = np.linalg.norm(centered, axis=1)
        resid_norm_batch = np.linalg.norm(resid, axis=1)
        initial_norm[start:end] = init_norm_batch.astype(np.float32)
        residual_norm[start:end] = resid_norm_batch.astype(np.float32)
        energy_ratio[start:end] = np.where(
            init_norm_batch > EPS,
            1.0 - (resid_norm_batch * resid_norm_batch) / (init_norm_batch * init_norm_batch),
            np.nan,
        ).astype(np.float32)
        for j in range(end - start):
            corr[start + j] = corr_flat(x[j], recon[j])
        rain = np.expm1(recon).astype(np.float32)
        rain = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0)
        rain[rain < 0.0] = 0.0
        rain_eof[start:end] = rain.reshape(end - start, GRID_SIZE, GRID_SIZE)
        log_eof[start:end] = np.log1p(rain_eof[start:end]).astype(np.float32)

    if not np.isfinite(rain_eof).all() or not np.isfinite(log_eof).all():
        raise RuntimeError("EOF reconstructed fields contain NaN or Inf after cleaning")
    diagnostics = {
        "eof_reconstruction_rmse_log": rmse,
        "eof_reconstruction_corr_log": corr,
        "eof_reconstruction_energy_ratio": energy_ratio,
        "initial_log_norm": initial_norm,
        "residual_log_norm": residual_norm,
    }
    return rain_eof, log_eof, diagnostics


def blend_initial_and_eof_fields(
    rain_initial: np.ndarray,
    rain_eof: np.ndarray,
    beta: float = BETA_BLEND,
) -> Tuple[np.ndarray, np.ndarray]:
    if rain_initial.shape != rain_eof.shape:
        raise RuntimeError(f"Initial and EOF field shapes differ: {rain_initial.shape} vs {rain_eof.shape}")
    blend = (1.0 - beta) * rain_initial.astype(np.float32) + beta * rain_eof.astype(np.float32)
    blend = np.nan_to_num(blend, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    blend[blend < 0.0] = 0.0
    log_blend = np.log1p(blend).astype(np.float32)
    if not np.isfinite(blend).all() or not np.isfinite(log_blend).all():
        raise RuntimeError("Blended fields contain NaN or Inf after cleaning")
    return blend, log_blend


# =========================
# Metrics and tables
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
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    cell_area = dx * dy
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


def build_coefficients_table(
    initial_index: pd.DataFrame,
    coefficients: np.ndarray,
    recon_diag: Mapping[str, np.ndarray],
) -> pd.DataFrame:
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
        "life_progress",
    ]
    out = initial_index[[c for c in id_cols if c in initial_index.columns]].copy()
    out["time"] = out["time"].map(format_time)
    for k in range(coefficients.shape[1]):
        out[f"eof_coef_{k + 1:02d}"] = coefficients[:, k].astype(float)
    for key, values in recon_diag.items():
        out[key] = np.asarray(values, dtype=float)

    out["_time_dt"] = pd.to_datetime(out["time"], errors="coerce")
    for k in range(min(5, coefficients.shape[1])):
        col = f"eof_coef_{k + 1:02d}"
        diff_col = f"{col}_diff_abs"
        out[diff_col] = np.nan
        for _, idx in out.sort_values("_time_dt").groupby("typhoon_name", sort=False).groups.items():
            sorted_idx = out.loc[idx].sort_values("_time_dt").index
            out.loc[sorted_idx, diff_col] = pd.to_numeric(out.loc[sorted_idx, col], errors="coerce").diff().abs()
    return out.drop(columns=["_time_dt"])


def build_blended_index_table(
    initial_index: pd.DataFrame,
    rain_initial: np.ndarray,
    rain_eof: np.ndarray,
    rain_blend: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    cumulative_evr: float,
) -> pd.DataFrame:
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
    rows: List[Dict[str, object]] = []
    iterator = iter_progress(range(len(initial_index)), total=len(initial_index), desc="Build blend index")
    for i in iterator:
        src = initial_index.iloc[i]
        row: Dict[str, object] = {c: src.get(c, np.nan) for c in id_cols if c in initial_index.columns}
        row["time"] = format_time(row.get("time"))
        row["beta_blend"] = BETA_BLEND
        row["n_components"] = N_COMPONENTS
        row["cumulative_explained_variance_ratio"] = cumulative_evr

        for c in initial_index.columns:
            if c.startswith("initial_"):
                row[c] = src[c]

        eof_metrics = compute_rainfall_metrics_on_relative_grid(
            rain_eof[i], x_front_km, y_left_km, x_grid, y_grid, prefix="eof"
        )
        blend_metrics = compute_rainfall_metrics_on_relative_grid(
            rain_blend[i], x_front_km, y_left_km, x_grid, y_grid, prefix="blend"
        )
        row.update(eof_metrics)
        row.update(blend_metrics)

        row["delta_blend_minus_initial_rain_max"] = row["blend_rain_max_mmhr"] - row.get("initial_rain_max_mmhr", np.nan)
        row["delta_blend_minus_initial_rain_p95"] = row["blend_rain_p95_mmhr"] - row.get("initial_rain_p95_mmhr", np.nan)
        row["delta_blend_minus_initial_area_10"] = row["blend_rain_area_10_km2"] - row.get("initial_rain_area_10_km2", np.nan)
        row["ratio_blend_to_initial_rain_max"] = safe_div(row["blend_rain_max_mmhr"], row.get("initial_rain_max_mmhr", np.nan))
        row["ratio_blend_to_initial_rain_p95"] = safe_div(row["blend_rain_p95_mmhr"], row.get("initial_rain_p95_mmhr", np.nan))
        row["ratio_blend_to_initial_area_10"] = safe_div(row["blend_rain_area_10_km2"], row.get("initial_rain_area_10_km2", np.nan))
        row["corr_initial_blend"] = corr_flat(rain_initial[i], rain_blend[i])
        row["rmse_initial_blend"] = float(np.sqrt(np.mean((rain_initial[i].astype(float) - rain_blend[i].astype(float)) ** 2)))

        row["eof_field_nan_count"] = int(np.count_nonzero(np.isnan(rain_eof[i])))
        row["eof_field_inf_count"] = int(np.count_nonzero(np.isinf(rain_eof[i])))
        row["eof_field_negative_count"] = int(np.count_nonzero(np.isfinite(rain_eof[i]) & (rain_eof[i] < 0.0)))
        row["eof_field_all_zero"] = bool(np.all(np.nan_to_num(rain_eof[i], nan=0.0) == 0.0))
        row["blend_field_nan_count"] = int(np.count_nonzero(np.isnan(rain_blend[i])))
        row["blend_field_inf_count"] = int(np.count_nonzero(np.isinf(rain_blend[i])))
        row["blend_field_negative_count"] = int(np.count_nonzero(np.isfinite(rain_blend[i]) & (rain_blend[i] < 0.0)))
        row["blend_field_all_zero"] = bool(np.all(np.nan_to_num(rain_blend[i], nan=0.0) == 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def save_blended_npz(
    initial: Mapping[str, np.ndarray],
    rain_eof: np.ndarray,
    log_eof: np.ndarray,
    rain_blend: np.ndarray,
    log_blend: np.ndarray,
    coefficients: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> None:
    BLENDED_NPZ_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BLENDED_NPZ_OUTPUT_PATH,
        rain_mmhr_initial=np.asarray(initial["rain_mmhr_initial"], dtype=np.float32),
        log_rain_initial=np.asarray(initial["log_rain_initial"], dtype=np.float32),
        rain_mmhr_eof=np.asarray(rain_eof, dtype=np.float32),
        log_rain_eof=np.asarray(log_eof, dtype=np.float32),
        rain_mmhr_blend=np.asarray(rain_blend, dtype=np.float32),
        log_rain_blend=np.asarray(log_blend, dtype=np.float32),
        eof_coefficients=np.asarray(coefficients, dtype=np.float32),
        target_id=as_str_array(initial["target_id"]).astype("U"),
        typhoon_name=as_str_array(initial["typhoon_name"]).astype("U"),
        time=as_str_array(initial["time"]).astype("U"),
        lat=np.asarray(initial["lat"], dtype=np.float32),
        lon_180=np.asarray(initial["lon_180"], dtype=np.float32),
        move_dir_deg=np.asarray(initial["move_dir_deg"], dtype=np.float32),
        x_front_km=x_front_km.astype("float32"),
        y_left_km=y_left_km.astype("float32"),
        beta_blend=np.array(BETA_BLEND, dtype=np.float32),
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


def make_explained_variance_figure(model: Mapping[str, np.ndarray]) -> Path:
    evr = np.asarray(model["explained_variance_ratio"], dtype=float)
    cum = np.asarray(model["cumulative_explained_variance_ratio"], dtype=float)
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(x, evr, color="tab:blue", alpha=0.75, label="explained variance ratio")
    ax.plot(x, cum, color="tab:red", marker="o", linewidth=1.4, label="cumulative")
    ax.set_xlabel("component number")
    ax.set_ylabel("ratio")
    ax.set_xticks(x)
    ax.set_ylim(0, max(1.0, float(cum[-1]) * 1.08))
    ax.set_title(f"EOF/PCA explained variance (cum={cum[-1]:.3f})")
    ax.legend(loc="best", fontsize=8)
    path = FIGURE_DIR / "eof_explained_variance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_eof_mode_figures(
    model: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    n_modes: int = 6,
) -> List[Path]:
    paths: List[Path] = []
    comps = np.asarray(model["eof_components"], dtype=float)
    evr = np.asarray(model["explained_variance_ratio"], dtype=float)
    for k in range(min(n_modes, comps.shape[0])):
        comp = comps[k]
        vmax = float(np.nanpercentile(np.abs(comp), 99))
        vmax = max(vmax, EPS)
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = _plot_field(
            ax,
            comp,
            x_front_km,
            y_left_km,
            f"EOF {k + 1:02d} (EVR={evr[k]:.4f})",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label="component loading")
        path = FIGURE_DIR / f"eof_mode_{k + 1:02d}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_mean_field_figures(
    model: Mapping[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    mean_log = np.asarray(model["mean_log_field"], dtype=float)
    mean_rain = np.expm1(mean_log)
    mean_rain[~np.isfinite(mean_rain)] = 0.0
    mean_rain[mean_rain < 0.0] = 0.0
    paths: List[Path] = []
    for filename, field, label, title in [
        ("mean_log_field.png", mean_log, "log(1 + rain)", "Historical mean log field"),
        ("mean_rain_field.png", mean_rain, "rain mm/hr", "Historical mean rain field"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        vmax = float(np.nanpercentile(field, 99))
        vmax = max(vmax, 1e-3)
        im = _plot_field(ax, field, x_front_km, y_left_km, title, vmax=vmax)
        fig.colorbar(im, ax=ax, label=label)
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_coefficient_timeseries_figures(coeff_df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    for name, sub in coeff_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
        for k in range(1, min(5, N_COMPONENTS) + 1):
            ax.plot(sub["time_dt"], sub[f"eof_coef_{k:02d}"], linewidth=1.2, label=f"EOF{k}")
        ax.set_title(f"{name} EOF coefficient time series")
        ax.set_xlabel("time")
        ax.set_ylabel("coefficient")
        ax.legend(loc="best", fontsize=8, ncol=3)
        path = FIGURE_DIR / f"{safe_name(name)}_eof_coefficients_timeseries.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def _representative_compare_indices(index_df: pd.DataFrame, name: object) -> List[int]:
    sub = index_df.loc[index_df["typhoon_name"].eq(name)].copy()
    sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
    sub = sub.sort_values("time_dt")
    picks = [int(sub.iloc[0]["field_index"])]
    max_p95_idx = int(sub.loc[pd.to_numeric(sub["initial_rain_p95_mmhr"], errors="coerce").idxmax(), "field_index"])
    picks.append(max_p95_idx)
    picks.append(int(sub.iloc[-1]["field_index"]))
    out: List[int] = []
    for p in picks:
        if p not in out:
            out.append(p)
    return out


def make_compare_figures(
    initial_index: pd.DataFrame,
    rain_initial: np.ndarray,
    rain_eof: np.ndarray,
    rain_blend: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name in initial_index["typhoon_name"].dropna().drop_duplicates():
        picks = _representative_compare_indices(initial_index, name)
        for field_idx in picks:
            row = initial_index.loc[initial_index["field_index"].eq(field_idx)].iloc[0]
            fields = [
                ("initial", rain_initial[field_idx]),
                ("EOF reconstruction", rain_eof[field_idx]),
                ("blended", rain_blend[field_idx]),
                ("blended - initial", rain_blend[field_idx] - rain_initial[field_idx]),
            ]
            vmax = float(np.nanpercentile(np.stack([rain_initial[field_idx], rain_eof[field_idx], rain_blend[field_idx]]), 99))
            vmax = max(vmax, 1.0)
            diff_abs = float(np.nanpercentile(np.abs(fields[-1][1]), 99))
            diff_abs = max(diff_abs, 0.1)
            fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), constrained_layout=True)
            for ax, (title, field) in zip(axes[:3], fields[:3]):
                im = _plot_field(ax, field, x_front_km, y_left_km, title, vmax=vmax)
                fig.colorbar(im, ax=ax, shrink=0.78)
            im = _plot_field(
                axes[3],
                fields[-1][1],
                x_front_km,
                y_left_km,
                fields[-1][0],
                cmap="RdBu_r",
                vmin=-diff_abs,
                vmax=diff_abs,
            )
            fig.colorbar(im, ax=axes[3], shrink=0.78)
            fig.suptitle(f"{name} {format_time(row['time'])} beta={BETA_BLEND}", fontsize=11)
            stamp = pd.Timestamp(row["time"]).strftime("%Y%m%d_%H%M")
            path = FIGURE_DIR / f"{safe_name(name)}_pca_blend_compare_{stamp}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def make_cumulative_and_max_figures(
    rain_blend: np.ndarray,
    index_df: pd.DataFrame,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        indices = sub["field_index"].astype(int).to_numpy()
        cumulative = np.sum(rain_blend[indices] * 0.5, axis=0)
        max_field = np.max(rain_blend[indices], axis=0)
        for kind, field, label, filename in [
            ("cumulative", cumulative, "cumulative half-hour mm", f"{safe_name(name)}_blend_cumulative_storm_relative.png"),
            ("max", max_field, "max rain mm/hr", f"{safe_name(name)}_blend_max_storm_relative.png"),
        ]:
            fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
            vmax = float(np.nanpercentile(field, 99))
            vmax = max(vmax, 1.0)
            im = _plot_field(ax, field, x_front_km, y_left_km, f"{name} blended {kind} storm-relative field", vmax=vmax)
            fig.colorbar(im, ax=ax, label=label)
            path = FIGURE_DIR / filename
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def make_blend_timeseries_figures(index_df: pd.DataFrame) -> List[Path]:
    paths: List[Path] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True, constrained_layout=True)
        axes[0].plot(sub["time_dt"], sub["blend_rain_max_mmhr"], label="blend_rain_max_mmhr", color="tab:red")
        axes[0].plot(sub["time_dt"], sub["blend_rain_p95_mmhr"], label="blend_rain_p95_mmhr", color="tab:blue")
        axes[0].set_ylabel("mm/hr")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].set_title(f"{name} blended metrics time series")

        axes[1].plot(sub["time_dt"], sub["blend_rain_area_10_km2"], color="tab:green")
        axes[1].set_ylabel("area >=10 mm/hr km2")

        axes[2].plot(sub["time_dt"], sub["WND"], color="tab:orange", label="WND")
        ax2 = axes[2].twinx()
        ax2.plot(sub["time_dt"], sub["PRES"], color="tab:purple", label="PRES")
        axes[2].set_ylabel("WND")
        ax2.set_ylabel("PRES")
        axes[2].set_xlabel("time")
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

        path = FIGURE_DIR / f"{safe_name(name)}_blend_metrics_timeseries.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_all_figures(
    model: Mapping[str, np.ndarray],
    coeff_df: pd.DataFrame,
    initial_index: pd.DataFrame,
    blended_index: pd.DataFrame,
    rain_initial: np.ndarray,
    rain_eof: np.ndarray,
    rain_blend: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not MAKE_FIGURES:
        return []
    paths: List[Path] = []
    paths.append(make_explained_variance_figure(model))
    paths.extend(make_eof_mode_figures(model, x_front_km, y_left_km))
    paths.extend(make_mean_field_figures(model, x_front_km, y_left_km))
    paths.extend(make_coefficient_timeseries_figures(coeff_df))
    paths.extend(make_compare_figures(initial_index, rain_initial, rain_eof, rain_blend, x_front_km, y_left_km))
    paths.extend(make_cumulative_and_max_figures(rain_blend, blended_index, x_front_km, y_left_km))
    paths.extend(make_blend_timeseries_figures(blended_index))
    return paths


# =========================
# QC report
# =========================


def _series_describe_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    available = [c for c in cols if c in df.columns]
    if not available:
        return "(none)"
    desc = df[available].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.5, 0.95]).T
    return desc.to_string(float_format=lambda x: f"{x:.6g}")


def _evr_table(model: Mapping[str, np.ndarray]) -> str:
    evr = np.asarray(model["explained_variance_ratio"], dtype=float)
    cum = np.asarray(model["cumulative_explained_variance_ratio"], dtype=float)
    table = pd.DataFrame(
        {
            "component": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance_ratio": cum,
        }
    )
    return table.to_string(index=False, float_format=lambda x: f"{x:.6g}")


def _mode_description(component: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> str:
    comp = np.asarray(component, dtype=float)
    weights = np.abs(comp)
    total = float(np.nansum(weights))
    if total <= EPS:
        return "信号很弱，空间结构不明显"
    cx = float(np.nansum(x_grid * weights) / total)
    cy = float(np.nansum(y_grid * weights) / total)
    front = float(np.nansum(weights[x_grid > 0.0]))
    back = float(np.nansum(weights[x_grid < 0.0]))
    left = float(np.nansum(weights[y_grid > 0.0]))
    right = float(np.nansum(weights[y_grid < 0.0]))
    fb = "前侧" if front >= back else "后侧"
    lr = "左侧" if left >= right else "右侧"
    return f"主要载荷偏向{fb}/{lr}，绝对载荷质心约 ({cx:.1f}, {cy:.1f}) km"


def _per_typhoon_qc(index_df: pd.DataFrame, coeff_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for name, sub in index_df.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        max_rain_row = sub.loc[pd.to_numeric(sub["blend_rain_max_mmhr"], errors="coerce").idxmax()]
        max_p95_row = sub.loc[pd.to_numeric(sub["blend_rain_p95_mmhr"], errors="coerce").idxmax()]
        max_area_row = sub.loc[pd.to_numeric(sub["blend_rain_area_10_km2"], errors="coerce").idxmax()]
        total_proxy = float(pd.to_numeric(sub["blend_rain_volume_proxy_mm_km2"], errors="coerce").sum(skipna=True))
        duration_proxy = float((pd.to_numeric(sub["blend_rain_area_10_km2"], errors="coerce") > 0).sum() * 0.5)
        csub = coeff_df.loc[coeff_df["typhoon_name"].eq(name)].copy()
        coef_lines: List[str] = []
        for k in range(1, min(5, N_COMPONENTS) + 1):
            c = pd.to_numeric(csub[f"eof_coef_{k:02d}"], errors="coerce")
            coef_lines.append(
                f"EOF{k}: mean={float(c.mean(skipna=True)):.6g}, std={float(c.std(skipna=True)):.6g}, max_abs={float(c.abs().max(skipna=True)):.6g}"
            )
        lines.extend(
            [
                f"### {name}",
                f"- 时刻数: {len(sub)}",
                f"- blend_rain_max_mmhr 最大值: {max_rain_row['blend_rain_max_mmhr']:.6g} at {max_rain_row['time']}",
                f"- blend_rain_p95_mmhr 最大值: {max_p95_row['blend_rain_p95_mmhr']:.6g} at {max_p95_row['time']}",
                f"- blend_rain_area_10_km2 最大值: {max_area_row['blend_rain_area_10_km2']:.6g} at {max_area_row['time']}",
                f"- blended 累计降水 proxy: {total_proxy:.6g}",
                f"- 强降水持续时间 proxy: {duration_proxy:.6g} hours",
                f"- 前 5 个 EOF 系数: {'; '.join(coef_lines)}",
            ]
        )
    return lines


def _time_continuity_qc(index_df: pd.DataFrame, coeff_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    merged = index_df[["field_index", "typhoon_name", "time", "blend_rain_max_mmhr", "blend_rain_p95_mmhr", "blend_rain_area_10_km2"]].merge(
        coeff_df[["field_index", "eof_coef_01", "eof_coef_02"]],
        on="field_index",
        how="left",
    )
    for name, sub in merged.groupby("typhoon_name", sort=False):
        sub = sub.copy()
        sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub.sort_values("time_dt")
        lines.append(f"### {name}")
        for col in [
            "blend_rain_max_mmhr",
            "blend_rain_p95_mmhr",
            "blend_rain_area_10_km2",
            "eof_coef_01",
            "eof_coef_02",
        ]:
            diff = pd.to_numeric(sub[col], errors="coerce").diff().abs()
            lines.append(
                f"- diff({col}): P95={float(diff.quantile(0.95)):.6g}, max={float(diff.max(skipna=True)):.6g}"
            )
    return lines


def write_qc_report(
    topk: pd.DataFrame,
    selected_templates: pd.DataFrame,
    training_meta: pd.DataFrame,
    selection_diag: Mapping[str, object],
    training_diag: Mapping[str, object],
    model: Mapping[str, np.ndarray],
    coeff_df: pd.DataFrame,
    blended_index: pd.DataFrame,
    rain_initial: np.ndarray,
    rain_eof: np.ndarray,
    rain_blend: np.ndarray,
    figure_paths: Sequence[Path],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> None:
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mean_log = np.asarray(model["mean_log_field"], dtype=float)
    evr = np.asarray(model["explained_variance_ratio"], dtype=float)
    cum = np.asarray(model["cumulative_explained_variance_ratio"], dtype=float)
    eof_quality = count_field_quality(rain_eof)
    blend_quality = count_field_quality(rain_blend)
    initial_quality = count_field_quality(rain_initial)
    top10_typhoons = training_meta["history_typhoon_name"].value_counts().head(10).to_string()
    skip_counts = pd.Series(training_diag.get("counters", {}), dtype=object).to_string() if training_diag.get("counters") else "(none)"
    skip_preview = pd.DataFrame(training_diag.get("skip_events", [])).head(30).to_string(index=False) if training_diag.get("skip_events") else "(none)"
    figure_summary = pd.Series([p.name for p in figure_paths]).to_string(index=False) if figure_paths else "(none)"
    target_like_training = bool(training_meta["history_typhoon_name"].map(normalize_name).isin({"KONGREY", "MANYI"}).any())
    target_excluded_flag_true = (
        bool(pd.Series(training_meta["history_target_excluded_flag"]).astype(str).str.lower().isin(["true", "1"]).any())
        if "history_target_excluded_flag" in training_meta.columns
        else False
    )
    paper_mode_lines = [
        f"- EOF1: {_mode_description(np.asarray(model['eof_components'])[0], x_grid, y_grid)}；可解释为主要大尺度雨带背景模态之一。",
        f"- EOF2: {_mode_description(np.asarray(model['eof_components'])[1], x_grid, y_grid)}；可用于描述相对运动方向上的非对称变化。",
        f"- EOF3: {_mode_description(np.asarray(model['eof_components'])[2], x_grid, y_grid)}；可用于描述雨带横向偏移或局地增强/减弱结构。",
    ]
    for name, sub in coeff_df.groupby("typhoon_name", sort=False):
        c1 = pd.to_numeric(sub["eof_coef_01"], errors="coerce")
        c2 = pd.to_numeric(sub["eof_coef_02"], errors="coerce")
        paper_mode_lines.append(
            f"- {name} 的 EOF1/EOF2 系数标准差分别为 {float(c1.std(skipna=True)):.3g}/{float(c2.std(skipna=True)):.3g}，可在论文中结合路径阶段讨论结构演变。"
        )
    paper_mode_lines.append("- PCA-blended 场保持 19 号 Top-K 初始场主体，同时引入历史 EOF 低秩约束，可作为 21 号极端分位数校准的结构底图。")
    paper_mode_lines.append("- 本步骤不负责极端峰值闭合；若 blended 后极端值偏弱，应在 21 号 R95/R99/Rmax 分位数校准中处理。")

    lines = [
        "# Problem 2 EOF/PCA Structure Correction QC Report",
        "",
        "## 1. 输入输出文件",
        f"- 19 号 initial NPZ: `{INITIAL_NPZ_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 19 号 index: `{INITIAL_INDEX_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Top-K 表: `{TOPK_TABLE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- EOF/PCA model 输出: `{MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- coefficients CSV 输出: `{COEFF_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- blended NPZ 输出: `{BLENDED_NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- blended index CSV 输出: `{BLENDED_INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- figures 目录: `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. 运行参数",
        f"- PCA_TRAINING_SOURCE: {PCA_TRAINING_SOURCE}",
        f"- MAX_PCA_TEMPLATES: {MAX_PCA_TEMPLATES}",
        f"- 实际训练样本数: {len(training_meta)}",
        f"- N_COMPONENTS: {N_COMPONENTS}",
        f"- BATCH_SIZE: {BATCH_SIZE}",
        f"- BETA_BLEND: {BETA_BLEND}",
        f"- GRID_SIZE: {GRID_SIZE}",
        f"- GRID_EXTENT_KM: {GRID_EXTENT_KM}",
        f"- RANDOM_SEED: {RANDOM_SEED}",
        "",
        "## 3. PCA 训练数据统计",
        f"- Top-K 唯一历史模板数: {selection_diag.get('topk_unique_history_templates')}",
        f"- Top-K 目标同名历史模板排除数: {selection_diag.get('target_like_history_templates_excluded')}",
        f"- 实际进入 PCA 候选样本数: {len(selected_templates)}",
        f"- 成功读取和重采样数量: {len(training_meta)}",
        f"- 跳过数量: {len(training_diag.get('skip_events', []))}",
        "- 跳过原因计数:",
        "```",
        skip_counts,
        "```",
        f"- 训练样本涉及历史台风事件数: {training_meta['history_event_uid'].nunique(dropna=True)}",
        "- 训练样本 typhoon_name Top 10:",
        "```",
        top10_typhoons,
        "```",
        "- 训练样本 WND/PRES/rain_p95 摘要:",
        "```",
        _series_describe_table(training_meta, ["history_WND", "history_PRES", "history_rain_p95_mmhr"]),
        "```",
        f"- 是否存在目标台风样本: {target_like_training or target_excluded_flag_true} (必须为 False)",
        "",
        "## 4. PCA 模型统计",
        "```",
        _evr_table(model),
        "```",
        f"- 前 5 个 EOF 的解释率: {', '.join(f'{v:.6g}' for v in evr[:5])}",
        f"- 前 10 个 EOF 的累计解释率: {cum[min(9, len(cum)-1)]:.6g}",
        f"- 全部 N_COMPONENTS 累计解释率: {cum[-1]:.6g}",
        f"- mean_log_field min/mean/max: {float(np.nanmin(mean_log)):.6g} / {float(np.nanmean(mean_log)):.6g} / {float(np.nanmax(mean_log)):.6g}",
        "",
        "## 5. 目标投影与重构统计",
        f"- 目标样本数: {len(coeff_df)}",
        f"- eof_coefficients shape: {len(coeff_df)} x {N_COMPONENTS}",
        f"- rain_mmhr_eof shape: {list(rain_eof.shape)}",
        f"- rain_mmhr_blend shape: {list(rain_blend.shape)}",
        f"- EOF 重构 log RMSE 分布: {stats_text(coeff_df['eof_reconstruction_rmse_log'])}",
        f"- EOF 重构 log Corr 分布: {stats_text(coeff_df['eof_reconstruction_corr_log'])}",
        f"- initial NaN/Inf/负值/全零场: {initial_quality['nan']} / {initial_quality['inf']} / {initial_quality['negative']} / {initial_quality['all_zero']}",
        f"- EOF NaN/Inf/负值/全零场: {eof_quality['nan']} / {eof_quality['inf']} / {eof_quality['negative']} / {eof_quality['all_zero']}",
        f"- blend NaN/Inf/负值/全零场: {blend_quality['nan']} / {blend_quality['inf']} / {blend_quality['negative']} / {blend_quality['all_zero']}",
        "",
        "## 6. PCA-blended 场基本检查",
        f"- blend_rain_max_mmhr 分布: {stats_text(blended_index['blend_rain_max_mmhr'])}",
        f"- blend_rain_p95_mmhr 分布: {stats_text(blended_index['blend_rain_p95_mmhr'])}",
        f"- blend_rain_area_10_km2 分布: {stats_text(blended_index['blend_rain_area_10_km2'])}",
        f"- blend_centroid_offset_km 分布: {stats_text(blended_index['blend_centroid_offset_km'])}",
        f"- blend_anisotropy 分布: {stats_text(blended_index['blend_anisotropy'])}",
        f"- corr_initial_blend 分布: {stats_text(blended_index['corr_initial_blend'])}",
        f"- rmse_initial_blend 分布: {stats_text(blended_index['rmse_initial_blend'])}",
        "",
        "## 7. 分台风统计",
        *_per_typhoon_qc(blended_index, coeff_df),
        "",
        "## 8. 与 19 号初始场的对比",
        f"- ratio_blend_to_initial_rain_max: {quantile_text(blended_index['ratio_blend_to_initial_rain_max'])}",
        f"- ratio_blend_to_initial_rain_p95: {quantile_text(blended_index['ratio_blend_to_initial_rain_p95'])}",
        f"- ratio_blend_to_initial_area_10: {quantile_text(blended_index['ratio_blend_to_initial_area_10'])}",
        f"- delta_blend_minus_initial_rain_max: mean={float(pd.to_numeric(blended_index['delta_blend_minus_initial_rain_max'], errors='coerce').mean(skipna=True)):.6g}, P95={float(pd.to_numeric(blended_index['delta_blend_minus_initial_rain_max'], errors='coerce').quantile(0.95)):.6g}",
        f"- delta_blend_minus_initial_rain_p95: mean={float(pd.to_numeric(blended_index['delta_blend_minus_initial_rain_p95'], errors='coerce').mean(skipna=True)):.6g}, P95={float(pd.to_numeric(blended_index['delta_blend_minus_initial_rain_p95'], errors='coerce').quantile(0.95)):.6g}",
        "",
        "本步骤只做结构修正，不负责把极端值校准到历史分位数。若 PCA-blended 后极端峰值仍偏低，极端降水偏弱问题将在 21 号极端分位数校准中处理。",
        "",
        "## 9. 时间连续性检查",
        *_time_continuity_qc(blended_index, coeff_df),
        "",
        "## 10. 防泄漏声明",
        "本步骤的 EOF/PCA 训练只使用历史 GPM 降水模板。目标台风 KONG-REY 和 MAN-YI 的真实 GPM 降水不存在且未被读取。目标台风 EOF 系数由 19 号初始生成场投影得到，不使用任何目标真实降水观测。rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等历史降水指标未参与目标台风安全输入构造和 Top-K 检索，仅用于历史训练样本诊断与后续校准评价。",
        "",
        "## 11. 图件",
        f"- 图件数量: {len(figure_paths)}",
        "```",
        figure_summary,
        "```",
        "",
        "## 12. 异常与日志样例",
        "```",
        skip_preview,
        "```",
        "",
        "## 13. 论文可写结论",
        *paper_mode_lines,
    ]
    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def main() -> None:
    np.random.seed(RANDOM_SEED)
    for path in [MODEL_OUTPUT_PATH, COEFF_OUTPUT_PATH, BLENDED_NPZ_OUTPUT_PATH, BLENDED_INDEX_OUTPUT_PATH, QC_REPORT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("[20] Loading step-19 index, Top-K table, and historical library")
    initial_index = load_initial_index()
    topk = load_topk_table()
    history = load_historical_library()

    print("[20] Selecting PCA training templates")
    selected_templates, selection_diag = select_pca_training_templates(topk, history)
    x_front_km, y_left_km, x_grid, y_grid = build_relative_grid()

    print("[20] Building PCA training matrix")
    training_matrix, training_meta, training_diag = build_pca_training_matrix(selected_templates, x_grid, y_grid)
    print(f"[20] Fitting EOF/PCA model on {training_matrix.shape[0]} samples x {training_matrix.shape[1]} grid cells")
    model = fit_eof_pca_model(training_matrix)
    print("[20] Saving EOF/PCA model")
    save_eof_pca_model(model, x_front_km, y_left_km, training_meta)
    del training_matrix
    gc.collect()

    print("[20] Loading step-19 initial fields")
    initial = load_initial_npz()
    validate_initial_order(initial, initial_index)
    rain_initial = np.asarray(initial["rain_mmhr_initial"], dtype=np.float32)
    log_initial = np.asarray(initial["log_rain_initial"], dtype=np.float32)

    print("[20] Projecting target fields into EOF/PCA latent space")
    coefficients = project_target_initial_fields(log_initial, model)
    print("[20] Reconstructing EOF fields")
    rain_eof, log_eof, recon_diag = reconstruct_eof_fields(log_initial, coefficients, model)
    print("[20] Blending initial and EOF fields")
    rain_blend, log_blend = blend_initial_and_eof_fields(rain_initial, rain_eof, BETA_BLEND)

    print("[20] Building coefficients table")
    coeff_df = build_coefficients_table(initial_index, coefficients, recon_diag)
    coeff_df.to_csv(COEFF_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("[20] Building blended index table")
    cumulative_evr = float(np.asarray(model["cumulative_explained_variance_ratio"])[-1])
    blended_index = build_blended_index_table(
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
    blended_index.to_csv(BLENDED_INDEX_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("[20] Saving blended NPZ")
    save_blended_npz(initial, rain_eof, log_eof, rain_blend, log_blend, coefficients, x_front_km, y_left_km)

    print("[20] Making figures")
    figure_paths = make_all_figures(
        model,
        coeff_df,
        initial_index,
        blended_index,
        rain_initial,
        rain_eof,
        rain_blend,
        x_front_km,
        y_left_km,
    )

    print("[20] Writing QC report")
    write_qc_report(
        topk,
        selected_templates,
        training_meta,
        selection_diag,
        training_diag,
        model,
        coeff_df,
        blended_index,
        rain_initial,
        rain_eof,
        rain_blend,
        figure_paths,
        x_grid,
        y_grid,
    )

    eof_quality = count_field_quality(rain_eof)
    blend_quality = count_field_quality(rain_blend)
    evr = np.asarray(model["explained_variance_ratio"], dtype=float)
    print("\n========== Problem-2 EOF/PCA structure correction complete ==========")
    print("Script: scripts/20_eof_pca_structure_correction.py")
    print(f"EOF/PCA model: {MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"EOF coefficients CSV: {COEFF_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Blended NPZ: {BLENDED_NPZ_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Blended index CSV: {BLENDED_INDEX_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"QC report: {QC_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure dir: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"PCA actual training samples: {len(training_meta)}")
    print(f"N_COMPONENTS: {N_COMPONENTS}")
    print(f"First 5 EOF explained_variance_ratio: {', '.join(f'{v:.8f}' for v in evr[:5])}")
    print(f"Cumulative explained_variance_ratio ({N_COMPONENTS}): {float(model['cumulative_explained_variance_ratio'][-1]):.8f}")
    print(f"eof_coefficients shape: {coefficients.shape}")
    print(f"rain_mmhr_eof shape: {rain_eof.shape}")
    print(f"rain_mmhr_blend shape: {rain_blend.shape}")
    print(f"beta_blend: {BETA_BLEND}")
    print(f"EOF NaN/Inf/negative/all-zero counts: {eof_quality['nan']} / {eof_quality['inf']} / {eof_quality['negative']} / {eof_quality['all_zero']}")
    print(f"Blend NaN/Inf/negative/all-zero counts: {blend_quality['nan']} / {blend_quality['inf']} / {blend_quality['negative']} / {blend_quality['all_zero']}")
    for col in ["blend_rain_max_mmhr", "blend_rain_p95_mmhr", "blend_rain_area_10_km2"]:
        s = pd.to_numeric(blended_index[col], errors="coerce")
        print(f"{col} P50/P95/max: {s.quantile(0.50):.6f} / {s.quantile(0.95):.6f} / {s.max(skipna=True):.6f}")
    for col in ["corr_initial_blend", "ratio_blend_to_initial_rain_max"]:
        s = pd.to_numeric(blended_index[col], errors="coerce")
        print(f"{col} P50/P95: {s.quantile(0.50):.6f} / {s.quantile(0.95):.6f}")
    for name, sub in blended_index.groupby("typhoon_name", sort=False):
        row = sub.loc[pd.to_numeric(sub["blend_rain_p95_mmhr"], errors="coerce").idxmax()]
        print(f"{name} max blend_rain_p95_mmhr: {row['blend_rain_p95_mmhr']:.6f} at {row['time']}")
    print(f"Generated figure count: {len(figure_paths)}")
    preview_cols = [
        "typhoon_name",
        "time",
        "WND",
        "PRES",
        "beta_blend",
        "initial_rain_max_mmhr",
        "eof_rain_max_mmhr",
        "blend_rain_max_mmhr",
        "initial_rain_p95_mmhr",
        "eof_rain_p95_mmhr",
        "blend_rain_p95_mmhr",
        "blend_rain_area_10_km2",
        "blend_centroid_offset_km",
        "blend_anisotropy",
        "ratio_blend_to_initial_rain_max",
    ]
    print("Random 5-row blended index preview:")
    print(blended_index[preview_cols].sample(n=min(5, len(blended_index)), random_state=RANDOM_SEED).to_string(index=False))


if __name__ == "__main__":
    main()
