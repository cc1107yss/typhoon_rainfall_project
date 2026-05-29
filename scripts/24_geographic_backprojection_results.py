#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 geographic backprojection for 2024 KONG-REY and MAN-YI.

This step only maps the step-21 calibrated storm-relative rainfall fields back
to fixed lon-lat grids. It does not redo Top-K retrieval, EOF/PCA correction,
extreme calibration, or pseudo-missing validation.
"""

from __future__ import annotations

import math
import os
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - optional runtime dependency
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_IMPORT_OK = True
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None
    CARTOPY_IMPORT_OK = False

try:  # pragma: no cover - progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CALIBRATED_NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields.npz"
CALIBRATED_INDEX_PATH = PROJECT_ROOT / "data/processed/problem2_generated_calibrated_fields_index.csv"
FINAL_TIMESERIES_PATH = PROJECT_ROOT / "data/processed/problem2_final_timeseries_metrics.csv"
FINAL_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_final_typhoon_metrics_summary.csv"
FINAL_KEY_TIMES_PATH = PROJECT_ROOT / "data/processed/problem2_final_key_times.csv"

GEOGRAPHIC_NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_final_geographic_fields.npz"
GEOGRAPHIC_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_final_geographic_summary.csv"
GEOGRAPHIC_KEY_LOCATIONS_PATH = PROJECT_ROOT / "data/processed/problem2_final_geographic_key_locations.csv"
GEOGRAPHIC_TIMESERIES_PATH = PROJECT_ROOT / "data/processed/problem2_final_geographic_timeseries_metrics.csv"
GEOGRAPHIC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_geographic_backprojection_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_geographic_results"
RUN_LOG_PATH = PROJECT_ROOT / "outputs/problem2_geographic_backprojection_run_log.txt"

EXPECTED_TOTAL_TIMES = 974
EXPECTED_TYPHOON_COUNTS = {"KONG-REY": 421, "MAN-YI": 553}
HALFHOUR_HOURS = 0.5
AREA10_THRESHOLD = 10.0
AREA20_THRESHOLD = 20.0
GRID_MODE = "per_typhoon"
GEO_MARGIN_DEG = 8.0
GEO_RES_DEG = 0.1
SAVE_TIMESLICE_GEO_FIELDS = False
EARTH_KM_PER_DEG = 111.32
EPS = 1e-12

FIGURE_SPECS = [
    ("geo_cumulative_rain_mm", "cumulative rainfall", "mm", "YlGnBu", "final_cumulative_rain_geographic"),
    ("geo_duration10_h", "duration >=10 mm/hr", "h", "plasma", "final_duration10_geographic"),
    ("geo_max_rain_mmhr", "maximum half-hour rain rate", "mm/hr", "viridis", "final_max_rain_geographic"),
    ("geo_duration20_h", "duration >=20 mm/hr", "h", "magma", "final_duration20_geographic"),
]


# =========================
# Helpers
# =========================


def resolve_project_path(path: object = PROJECT_ROOT) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def safe_name(value: object) -> str:
    return str(value).replace("-", "_").replace(" ", "_")


def format_time(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def add_issue(issues: List[str], message: str) -> None:
    issues.append(message)
    print(f"[issue] {message}")


def iter_progress(iterable, total: Optional[int] = None, desc: str = ""):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def normalize_lon_180(lon: object) -> object:
    arr = np.asarray(lon, dtype=np.float64)
    out = ((arr + 180.0) % 360.0) - 180.0
    out = np.where(np.isclose(out, -180.0) & (arr > 0.0), 180.0, out)
    if np.isscalar(lon):
        return float(out)
    return out


def lon_diff_deg(lon: np.ndarray, center_lon: float) -> np.ndarray:
    return np.asarray(normalize_lon_180(lon - center_lon), dtype=np.float64)


def distance_km(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lat1_rad = math.radians(float(lat1))
    lat2_rad = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    dlat = lat2_rad - lat1_rad
    dlon = np.deg2rad(lon_diff_deg(np.asarray(lon2, dtype=np.float64), float(lon1)))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def area_of_mask(mask: np.ndarray, cell_area_km2_1d: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise RuntimeError(f"Mask must be 2-D, got shape {valid.shape}")
    row_counts = np.count_nonzero(valid, axis=1).astype(np.float64)
    return float(np.sum(row_counts * cell_area_km2_1d))


def weighted_sum_field(field: np.ndarray, cell_area_km2_1d: np.ndarray) -> float:
    arr = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(arr * cell_area_km2_1d[:, None]))


def finite_positive_values(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    vals = arr[np.isfinite(arr) & (arr > 0.0)]
    return vals


def nonzero_percentile(field: np.ndarray, percentile: float) -> float:
    vals = finite_positive_values(field)
    if len(vals) == 0:
        return 0.0
    return float(np.percentile(vals, percentile))


def nonzero_mean(field: np.ndarray) -> float:
    vals = finite_positive_values(field)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def argmax_lon_lat(field: np.ndarray, lon_1d: np.ndarray, lat_1d: np.ndarray) -> Tuple[float, float, float, int, int]:
    arr = np.asarray(field, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan, np.nan, np.nan, -1, -1
    arr_clean = np.where(np.isfinite(arr), arr, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(arr_clean)), arr_clean.shape)
    return float(arr_clean[iy, ix]), float(lon_1d[ix]), float(lat_1d[iy]), int(iy), int(ix)


def quantile_location(
    field: np.ndarray,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    percentile: float,
) -> Tuple[float, float, float]:
    arr = np.asarray(field, dtype=np.float64)
    mask = np.isfinite(arr) & (arr > 0.0)
    if not mask.any():
        value, lon, lat, _, _ = argmax_lon_lat(arr, lon_1d, lat_1d)
        return value, lon, lat
    qval = float(np.percentile(arr[mask], percentile))
    distance = np.where(mask, np.abs(arr - qval), np.inf)
    iy, ix = np.unravel_index(int(np.argmin(distance)), arr.shape)
    return float(arr[iy, ix]), float(lon_1d[ix]), float(lat_1d[iy])


def write_run_log(issues: Sequence[str], status: str, extra: Optional[str] = None) -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Problem-2 Geographic Backprojection Run Log",
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


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# =========================
# Loaders
# =========================


def load_calibrated_npz(issues: List[str]) -> Dict[str, np.ndarray]:
    if not CALIBRATED_NPZ_PATH.exists():
        add_issue(issues, f"calibrated NPZ 缺失: {CALIBRATED_NPZ_PATH}")
        raise FileNotFoundError(f"Missing calibrated NPZ: {CALIBRATED_NPZ_PATH}")

    required = [
        "rain_mmhr_calibrated",
        "target_id",
        "typhoon_name",
        "time",
        "lat",
        "lon_180",
        "move_dir_deg",
        "x_front_km",
        "y_left_km",
    ]
    with np.load(CALIBRATED_NPZ_PATH, allow_pickle=True) as z:
        missing = [key for key in required if key not in z.files]
        if missing:
            add_issue(issues, f"calibrated NPZ 缺少必要数组: {missing}")
            raise RuntimeError(f"Calibrated NPZ is missing required arrays: {missing}")
        out = {key: z[key] for key in required}

    rain = np.asarray(out["rain_mmhr_calibrated"], dtype=np.float32)
    if rain.ndim != 3:
        add_issue(issues, f"calibrated 场 shape 异常: {rain.shape}")
        raise RuntimeError(f"Unexpected calibrated rain shape: {rain.shape}")
    if rain.shape[0] != EXPECTED_TOTAL_TIMES:
        add_issue(issues, f"calibrated 场时刻数异常: {rain.shape[0]}，期望 {EXPECTED_TOTAL_TIMES}")
        raise RuntimeError(f"Unexpected calibrated time count: {rain.shape[0]}")
    if rain.shape[1] != len(out["y_left_km"]) or rain.shape[2] != len(out["x_front_km"]):
        add_issue(
            issues,
            f"calibrated 场空间维度与 x/y 轴不匹配: rain={rain.shape}, "
            f"y={len(out['y_left_km'])}, x={len(out['x_front_km'])}",
        )
        raise RuntimeError("Calibrated field shape does not match storm-relative axes.")

    out["rain_mmhr_calibrated"] = np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for key in ["target_id", "typhoon_name", "time"]:
        out[key] = np.asarray(out[key]).astype(str)
    for key in ["lat", "lon_180", "move_dir_deg", "x_front_km", "y_left_km"]:
        out[key] = np.asarray(out[key], dtype=np.float64)

    if not np.isfinite(out["lat"]).all() or not np.isfinite(out["lon_180"]).all():
        add_issue(issues, "lat/lon_180 存在缺失或非有限值")
        raise RuntimeError("Missing or non-finite lat/lon in calibrated NPZ.")
    if not np.isfinite(out["move_dir_deg"]).all():
        add_issue(issues, "move_dir_deg 存在缺失或非有限值")
        raise RuntimeError("Missing or non-finite move_dir_deg in calibrated NPZ.")
    return out


def load_calibrated_index(issues: List[str]) -> pd.DataFrame:
    if not CALIBRATED_INDEX_PATH.exists():
        add_issue(issues, f"calibrated index 缺失: {CALIBRATED_INDEX_PATH}")
        raise FileNotFoundError(f"Missing calibrated index: {CALIBRATED_INDEX_PATH}")
    df = pd.read_csv(CALIBRATED_INDEX_PATH)
    if len(df) != EXPECTED_TOTAL_TIMES:
        add_issue(issues, f"calibrated index 行数异常: {len(df)}，期望 {EXPECTED_TOTAL_TIMES}")
        raise RuntimeError(f"Unexpected calibrated index row count: {len(df)}")
    return df


def load_final_timeseries(issues: List[str]) -> pd.DataFrame:
    if not FINAL_TIMESERIES_PATH.exists():
        add_issue(issues, f"final timeseries 缺失: {FINAL_TIMESERIES_PATH}")
        raise FileNotFoundError(f"Missing final timeseries: {FINAL_TIMESERIES_PATH}")
    df = pd.read_csv(FINAL_TIMESERIES_PATH)
    if len(df) != EXPECTED_TOTAL_TIMES:
        add_issue(issues, f"final timeseries 行数异常: {len(df)}，期望 {EXPECTED_TOTAL_TIMES}")
        raise RuntimeError(f"Unexpected final timeseries row count: {len(df)}")
    required = ["field_index", "target_id", "typhoon_name", "time", "lat", "lon_180", "WND", "PRES", "move_dir_deg"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        add_issue(issues, f"final timeseries 缺少必要字段: {missing}")
        raise RuntimeError(f"Final timeseries is missing required columns: {missing}")
    return df


def load_final_summary(issues: List[str]) -> pd.DataFrame:
    if not FINAL_SUMMARY_PATH.exists():
        add_issue(issues, f"final summary 缺失: {FINAL_SUMMARY_PATH}")
        raise FileNotFoundError(f"Missing final summary: {FINAL_SUMMARY_PATH}")
    df = pd.read_csv(FINAL_SUMMARY_PATH)
    if len(df) != len(EXPECTED_TYPHOON_COUNTS):
        add_issue(issues, f"final summary 行数异常: {len(df)}")
        raise RuntimeError(f"Unexpected final summary row count: {len(df)}")
    return df


def load_final_key_times(issues: List[str]) -> pd.DataFrame:
    if not FINAL_KEY_TIMES_PATH.exists():
        add_issue(issues, f"final key times 缺失: {FINAL_KEY_TIMES_PATH}")
        raise FileNotFoundError(f"Missing final key times: {FINAL_KEY_TIMES_PATH}")
    df = pd.read_csv(FINAL_KEY_TIMES_PATH)
    if len(df) == 0:
        add_issue(issues, "final key times 表为空")
        raise RuntimeError("Final key times table is empty.")
    return df


def build_canonical_timeseries(
    calibrated: Mapping[str, np.ndarray],
    final_timeseries: pd.DataFrame,
    issues: List[str],
) -> pd.DataFrame:
    n = len(calibrated["time"])
    base = pd.DataFrame(
        {
            "field_index": np.arange(n, dtype=int),
            "target_id": np.asarray(calibrated["target_id"]).astype(str),
            "typhoon_name": np.asarray(calibrated["typhoon_name"]).astype(str),
            "time": np.asarray(calibrated["time"]).astype(str),
            "lat": np.asarray(calibrated["lat"], dtype=float),
            "lon_180": np.asarray(normalize_lon_180(calibrated["lon_180"]), dtype=float),
            "move_dir_deg": np.asarray(calibrated["move_dir_deg"], dtype=float),
        }
    )

    final_sorted = final_timeseries.sort_values("field_index").reset_index(drop=True).copy()
    if len(final_sorted) != n:
        add_issue(issues, f"final timeseries 排序后行数与 NPZ 不一致: {len(final_sorted)} vs {n}")
        raise RuntimeError("Final timeseries and NPZ row counts differ.")
    if not np.array_equal(base["target_id"].to_numpy(), final_sorted["target_id"].astype(str).to_numpy()):
        add_issue(issues, "NPZ target_id 顺序与 final_timeseries field_index 顺序不一致")
        raise RuntimeError("NPZ order and final_timeseries order mismatch.")

    supplement_cols = [col for col in final_sorted.columns if col not in base.columns]
    if supplement_cols:
        base = pd.concat([base, final_sorted[supplement_cols].reset_index(drop=True)], axis=1)

    counts = base["typhoon_name"].value_counts().to_dict()
    for name, expected in EXPECTED_TYPHOON_COUNTS.items():
        actual = int(counts.get(name, 0))
        if actual != expected:
            add_issue(issues, f"{name} 时刻数异常: {actual}，期望 {expected}")
            raise RuntimeError(f"Unexpected row count for {name}: {actual}")
    unexpected = sorted(set(counts) - set(EXPECTED_TYPHOON_COUNTS))
    if unexpected:
        add_issue(issues, f"发现非目标台风名称: {unexpected}")
        raise RuntimeError(f"Unexpected typhoon names: {unexpected}")

    for col in ["lat", "lon_180", "move_dir_deg", "WND", "PRES"]:
        if col not in base.columns or not np.isfinite(pd.to_numeric(base[col], errors="coerce")).all():
            add_issue(issues, f"{col} 缺失或存在非有限值")
            raise RuntimeError(f"Missing or non-finite {col}.")
    return base


def validate_input_consistency(
    calibrated: Mapping[str, np.ndarray],
    calibrated_index: pd.DataFrame,
    canonical_timeseries: pd.DataFrame,
    issues: List[str],
) -> Dict[str, object]:
    rain = np.asarray(calibrated["rain_mmhr_calibrated"])
    if rain.shape[0] != len(calibrated_index) or rain.shape[0] != len(canonical_timeseries):
        add_issue(
            issues,
            f"NPZ shape 与表格行数不一致: NPZ={rain.shape[0]}, index={len(calibrated_index)}, "
            f"final={len(canonical_timeseries)}",
        )
        raise RuntimeError("NPZ time dimension and table row counts mismatch.")
    if "field_index" in calibrated_index.columns and "target_id" in calibrated_index.columns:
        index_sorted = calibrated_index.sort_values("field_index").reset_index(drop=True)
        if not np.array_equal(
            np.asarray(calibrated["target_id"]).astype(str),
            index_sorted["target_id"].astype(str).to_numpy(),
        ):
            add_issue(issues, "calibrated NPZ 与 calibrated index 的 target_id 顺序不一致")
            raise RuntimeError("target_id order mismatch between calibrated NPZ and index.")

    quality = {
        "input_nan": int(np.count_nonzero(np.isnan(rain))),
        "input_inf": int(np.count_nonzero(np.isinf(rain))),
        "input_negative": int(np.count_nonzero(np.isfinite(rain) & (rain < 0.0))),
        "input_all_zero_fields": int(np.count_nonzero(np.all(np.nan_to_num(rain, nan=0.0, posinf=0.0, neginf=0.0) == 0.0, axis=(1, 2)))),
    }
    if quality["input_nan"] > 0 or quality["input_inf"] > 0 or quality["input_negative"] > 0:
        add_issue(issues, f"rain_mmhr_calibrated 存在 NaN/Inf/负值: {quality}")
        raise RuntimeError("Calibrated rainfall has invalid values.")
    if quality["input_all_zero_fields"] > 0:
        add_issue(issues, f"rain_mmhr_calibrated 存在全零场数量: {quality['input_all_zero_fields']}")
        raise RuntimeError("Calibrated rainfall has all-zero fields.")
    return quality


# =========================
# Geographic grid and sampling
# =========================


def build_geo_grid_for_typhoon(sub: pd.DataFrame) -> Dict[str, np.ndarray]:
    if GRID_MODE != "per_typhoon":
        raise RuntimeError(f"Unsupported GRID_MODE for this script: {GRID_MODE}")

    lon = np.asarray(normalize_lon_180(pd.to_numeric(sub["lon_180"], errors="coerce").to_numpy()), dtype=np.float64)
    lat = pd.to_numeric(sub["lat"], errors="coerce").to_numpy(dtype=np.float64)
    if len(lon) == 0 or len(lat) == 0 or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise RuntimeError("Cannot build geographic grid from missing lon/lat.")

    simple_span = float(np.nanmax(lon) - np.nanmin(lon))
    if simple_span > 180.0:
        lon_min, lon_max = -180.0, 180.0
    else:
        lon_min = math.floor(float(np.nanmin(lon)) - GEO_MARGIN_DEG)
        lon_max = math.ceil(float(np.nanmax(lon)) + GEO_MARGIN_DEG)
        lon_min = max(-180.0, lon_min)
        lon_max = min(180.0, lon_max)

    lat_min = math.floor(float(np.nanmin(lat)) - GEO_MARGIN_DEG)
    lat_max = math.ceil(float(np.nanmax(lat)) + GEO_MARGIN_DEG)
    lat_min = max(-90.0, lat_min)
    lat_max = min(90.0, lat_max)
    if not (lon_min < lon_max and lat_min < lat_max):
        raise RuntimeError(f"Invalid geographic grid bounds: lon=({lon_min},{lon_max}), lat=({lat_min},{lat_max})")

    geo_lon_1d = np.round(np.arange(lon_min, lon_max + GEO_RES_DEG * 0.5, GEO_RES_DEG), 6)
    geo_lat_1d = np.round(np.arange(lat_min, lat_max + GEO_RES_DEG * 0.5, GEO_RES_DEG), 6)
    geo_lon_1d = np.asarray(normalize_lon_180(geo_lon_1d), dtype=np.float64)
    geo_lat_1d = np.clip(geo_lat_1d, -90.0, 90.0).astype(np.float64)
    geo_lon2d, geo_lat2d = np.meshgrid(geo_lon_1d, geo_lat_1d)
    if geo_lon2d.size == 0 or geo_lat2d.size == 0:
        raise RuntimeError("Geographic grid is empty.")
    return {
        "geo_lon_1d": geo_lon_1d,
        "geo_lat_1d": geo_lat_1d,
        "geo_lon2d": geo_lon2d,
        "geo_lat2d": geo_lat2d,
        "lon_min": float(geo_lon_1d.min()),
        "lon_max": float(geo_lon_1d.max()),
        "lat_min": float(geo_lat_1d.min()),
        "lat_max": float(geo_lat_1d.max()),
    }


def compute_geo_cell_area(geo_lat_1d: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(geo_lat_1d, dtype=np.float64))
    area = EARTH_KM_PER_DEG * GEO_RES_DEG * EARTH_KM_PER_DEG * GEO_RES_DEG * np.cos(lat_rad)
    return np.maximum(area, 0.0)


def sample_storm_relative_to_geo_grid(
    rain_field: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    center_lon: float,
    center_lat: float,
    move_dir_deg: float,
    geo_lon2d: np.ndarray,
    geo_lat2d: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    alpha = math.radians(float(move_dir_deg))
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    center_lat_rad = math.radians(float(center_lat))

    x_east_km = EARTH_KM_PER_DEG * math.cos(center_lat_rad) * lon_diff_deg(geo_lon2d, float(center_lon))
    y_north_km = EARTH_KM_PER_DEG * (geo_lat2d - float(center_lat))
    x_front = x_east_km * sin_a + y_north_km * cos_a
    y_left = -x_east_km * cos_a + y_north_km * sin_a

    xmin = float(np.min(x_front_km))
    xmax = float(np.max(x_front_km))
    ymin = float(np.min(y_left_km))
    ymax = float(np.max(y_left_km))
    in_bounds = (x_front >= xmin) & (x_front <= xmax) & (y_left >= ymin) & (y_left <= ymax)

    out = np.zeros(geo_lon2d.shape, dtype=np.float32)
    qc = {"nan": 0, "inf": 0, "negative": 0, "sampled_points": int(np.count_nonzero(in_bounds))}
    if qc["sampled_points"] == 0:
        return out, qc

    interpolator = RegularGridInterpolator(
        (y_left_km, x_front_km),
        np.asarray(rain_field, dtype=np.float32),
        bounds_error=False,
        fill_value=0.0,
    )
    points = np.column_stack([y_left[in_bounds].ravel(), x_front[in_bounds].ravel()])
    sampled = interpolator(points)
    qc["nan"] = int(np.count_nonzero(np.isnan(sampled)))
    qc["inf"] = int(np.count_nonzero(np.isinf(sampled)))
    qc["negative"] = int(np.count_nonzero(np.isfinite(sampled) & (sampled < 0.0)))
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    sampled = np.maximum(sampled, 0.0).astype(np.float32)
    out[in_bounds] = sampled
    return out, qc


# =========================
# Backprojection metrics
# =========================


def backproject_one_typhoon(
    name: str,
    sub: pd.DataFrame,
    rain_calibrated: np.ndarray,
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    issues: List[str],
) -> Dict[str, object]:
    sub = sub.copy()
    sub["time_dt"] = pd.to_datetime(sub["time"], errors="coerce")
    if sub["time_dt"].isna().any():
        add_issue(issues, f"{name} 存在无法解析的 time")
        raise RuntimeError(f"Cannot parse times for {name}.")
    sub = sub.sort_values("time_dt").reset_index(drop=True)

    grid = build_geo_grid_for_typhoon(sub)
    geo_lon_1d = grid["geo_lon_1d"]
    geo_lat_1d = grid["geo_lat_1d"]
    geo_lon2d = grid["geo_lon2d"]
    geo_lat2d = grid["geo_lat2d"]
    cell_area = compute_geo_cell_area(geo_lat_1d)
    shape = geo_lon2d.shape

    cumulative = np.zeros(shape, dtype=np.float64)
    max_rain = np.zeros(shape, dtype=np.float32)
    duration10 = np.zeros(shape, dtype=np.float32)
    duration20 = np.zeros(shape, dtype=np.float32)
    max_rain_time_index = np.full(shape, -1, dtype=np.int32)

    timeseries_rows: List[Dict[str, object]] = []
    qc = {
        "nan": 0,
        "inf": 0,
        "negative": 0,
        "all_zero_timeslices": 0,
        "sampled_points_total": 0,
        "n_times": int(len(sub)),
    }
    global_max = {"value": -np.inf, "lon": np.nan, "lat": np.nan, "time": "NA"}

    for local_i, row in iter_progress(list(sub.iterrows()), total=len(sub), desc=f"Backproject {name}"):
        field_index = int(row["field_index"])
        r_geo, step_qc = sample_storm_relative_to_geo_grid(
            rain_calibrated[field_index],
            x_front_km,
            y_left_km,
            float(row["lon_180"]),
            float(row["lat"]),
            float(row["move_dir_deg"]),
            geo_lon2d,
            geo_lat2d,
        )
        for key in ["nan", "inf", "negative", "sampled_points"]:
            qc_key = "sampled_points_total" if key == "sampled_points" else key
            qc[qc_key] += int(step_qc[key])

        if not np.any(r_geo > 0.0):
            qc["all_zero_timeslices"] += 1

        cumulative += r_geo.astype(np.float64) * HALFHOUR_HOURS
        improved = r_geo > max_rain
        max_rain[improved] = r_geo[improved]
        max_rain_time_index[improved] = local_i
        duration10 += (r_geo >= AREA10_THRESHOLD).astype(np.float32) * HALFHOUR_HOURS
        duration20 += (r_geo >= AREA20_THRESHOLD).astype(np.float32) * HALFHOUR_HOURS

        step_max, step_lon, step_lat, _, _ = argmax_lon_lat(r_geo, geo_lon_1d, geo_lat_1d)
        if np.isfinite(step_max) and step_max > float(global_max["value"]):
            global_max = {
                "value": float(step_max),
                "lon": float(step_lon),
                "lat": float(step_lat),
                "time": format_time(row["time"]),
            }

        nonzero = r_geo[np.isfinite(r_geo) & (r_geo > 0.0)]
        p95_nonzero = float(np.percentile(nonzero, 95)) if len(nonzero) else 0.0
        area10 = area_of_mask(r_geo >= AREA10_THRESHOLD, cell_area)
        area20 = area_of_mask(r_geo >= AREA20_THRESHOLD, cell_area)
        volume_proxy = weighted_sum_field(r_geo * HALFHOUR_HOURS, cell_area)
        timeseries_rows.append(
            {
                "typhoon_name": name,
                "time": format_time(row["time"]),
                "field_index": field_index,
                "lat": float(row["lat"]),
                "lon_180": float(row["lon_180"]),
                "WND": float(row["WND"]) if pd.notna(row.get("WND", np.nan)) else np.nan,
                "PRES": float(row["PRES"]) if pd.notna(row.get("PRES", np.nan)) else np.nan,
                "move_dir_deg": float(row["move_dir_deg"]),
                "geo_rain_max_mmhr": float(step_max),
                "geo_rain_p95_mmhr_nonzero": p95_nonzero,
                "geo_area_10_km2": area10,
                "geo_area_20_km2": area20,
                "geo_volume_proxy_mm_km2": volume_proxy,
                "geo_max_rain_lon": float(step_lon),
                "geo_max_rain_lat": float(step_lat),
            }
        )

    result_all_zero = bool((np.nanmax(cumulative) <= 0.0) and (np.nanmax(max_rain) <= 0.0))
    if result_all_zero:
        add_issue(issues, f"{name} 反投影后地理结果全零")
    qc["result_all_zero"] = int(result_all_zero)

    return {
        "typhoon_name": name,
        "grid": grid,
        "cell_area_km2_1d": cell_area,
        "geo_cumulative_rain_mm": cumulative.astype(np.float32),
        "geo_max_rain_mmhr": max_rain.astype(np.float32),
        "geo_duration10_h": duration10.astype(np.float32),
        "geo_duration20_h": duration20.astype(np.float32),
        "geo_max_rain_time_index": max_rain_time_index,
        "global_max_rain": global_max,
        "timeseries_metrics": pd.DataFrame(timeseries_rows),
        "track": sub,
        "qc": qc,
        "area_time_10_km2_h": float(np.sum(pd.to_numeric(pd.DataFrame(timeseries_rows)["geo_area_10_km2"], errors="coerce")) * HALFHOUR_HOURS),
        "area_time_20_km2_h": float(np.sum(pd.to_numeric(pd.DataFrame(timeseries_rows)["geo_area_20_km2"], errors="coerce")) * HALFHOUR_HOURS),
    }


def find_nearest_track_point(lon: float, lat: float, track: pd.DataFrame) -> Dict[str, object]:
    d = distance_km(
        lon,
        lat,
        pd.to_numeric(track["lon_180"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(track["lat"], errors="coerce").to_numpy(dtype=np.float64),
    )
    idx = int(np.nanargmin(d))
    row = track.iloc[idx]
    return {
        "nearest_track_time": format_time(row["time"]),
        "nearest_track_distance_km": float(d[idx]),
    }


def compute_geo_summary(result: Mapping[str, object]) -> Dict[str, object]:
    name = str(result["typhoon_name"])
    grid = result["grid"]
    lon_1d = np.asarray(grid["geo_lon_1d"], dtype=np.float64)
    lat_1d = np.asarray(grid["geo_lat_1d"], dtype=np.float64)
    cell_area = np.asarray(result["cell_area_km2_1d"], dtype=np.float64)
    cumulative = np.asarray(result["geo_cumulative_rain_mm"], dtype=np.float64)
    max_rain = np.asarray(result["geo_max_rain_mmhr"], dtype=np.float64)
    duration10 = np.asarray(result["geo_duration10_h"], dtype=np.float64)
    duration20 = np.asarray(result["geo_duration20_h"], dtype=np.float64)
    track = result["track"].copy()
    track["time_dt"] = pd.to_datetime(track["time"], errors="coerce")

    cum_max, cum_lon, cum_lat, _, _ = argmax_lon_lat(cumulative, lon_1d, lat_1d)
    dur10_max, dur10_lon, dur10_lat, _, _ = argmax_lon_lat(duration10, lon_1d, lat_1d)
    dur20_max, dur20_lon, dur20_lat, _, _ = argmax_lon_lat(duration20, lon_1d, lat_1d)
    global_max = result["global_max_rain"]

    row = {
        "typhoon_name": name,
        "geo_lon_min": float(np.min(lon_1d)),
        "geo_lon_max": float(np.max(lon_1d)),
        "geo_lat_min": float(np.min(lat_1d)),
        "geo_lat_max": float(np.max(lat_1d)),
        "geo_res_deg": GEO_RES_DEG,
        "n_geo_lon": int(len(lon_1d)),
        "n_geo_lat": int(len(lat_1d)),
        "n_times": int(len(track)),
        "start_time": format_time(track["time_dt"].iloc[0]),
        "end_time": format_time(track["time_dt"].iloc[-1]),
        "geo_max_cumulative_rain_mm": float(cum_max),
        "geo_max_cumulative_rain_lon": float(cum_lon),
        "geo_max_cumulative_rain_lat": float(cum_lat),
        "geo_mean_cumulative_rain_mm_nonzero": nonzero_mean(cumulative),
        "geo_p95_cumulative_rain_mm": nonzero_percentile(cumulative, 95),
        "geo_p99_cumulative_rain_mm": nonzero_percentile(cumulative, 99),
        "geo_max_rain_mmhr": float(global_max["value"]),
        "geo_max_rain_lon": float(global_max["lon"]),
        "geo_max_rain_lat": float(global_max["lat"]),
        "geo_max_rain_time": str(global_max["time"]),
        "geo_max_duration10_h": float(dur10_max),
        "geo_max_duration10_lon": float(dur10_lon),
        "geo_max_duration10_lat": float(dur10_lat),
        "geo_max_duration20_h": float(dur20_max),
        "geo_max_duration20_lon": float(dur20_lon),
        "geo_max_duration20_lat": float(dur20_lat),
        "geo_area_cumulative_rain_ge_50mm_km2": area_of_mask(cumulative >= 50.0, cell_area),
        "geo_area_cumulative_rain_ge_100mm_km2": area_of_mask(cumulative >= 100.0, cell_area),
        "geo_area_cumulative_rain_ge_200mm_km2": area_of_mask(cumulative >= 200.0, cell_area),
        "geo_area_duration10_ge_1h_km2": area_of_mask(duration10 >= 1.0, cell_area),
        "geo_area_duration10_ge_3h_km2": area_of_mask(duration10 >= 3.0, cell_area),
        "geo_area_duration10_ge_6h_km2": area_of_mask(duration10 >= 6.0, cell_area),
        "geo_area_duration20_ge_1h_km2": area_of_mask(duration20 >= 1.0, cell_area),
        "geo_area_duration20_ge_3h_km2": area_of_mask(duration20 >= 3.0, cell_area),
        "geo_area_duration20_ge_6h_km2": area_of_mask(duration20 >= 6.0, cell_area),
        "geo_area_time_10_km2_h": float(result["area_time_10_km2_h"]),
        "geo_area_time_20_km2_h": float(result["area_time_20_km2_h"]),
        "track_lon_min": float(numeric(track, "lon_180").min()),
        "track_lon_max": float(numeric(track, "lon_180").max()),
        "track_lat_min": float(numeric(track, "lat").min()),
        "track_lat_max": float(numeric(track, "lat").max()),
        "WND_max": float(numeric(track, "WND").max()),
        "PRES_min": float(numeric(track, "PRES").min()),
        "geo_qc_nan_count": int(result["qc"]["nan"]),
        "geo_qc_inf_count": int(result["qc"]["inf"]),
        "geo_qc_negative_count": int(result["qc"]["negative"]),
        "geo_qc_all_zero_timeslices": int(result["qc"]["all_zero_timeslices"]),
        "geo_qc_result_all_zero": bool(result["qc"]["result_all_zero"]),
    }
    return row


def compute_geo_key_locations(result: Mapping[str, object]) -> pd.DataFrame:
    name = str(result["typhoon_name"])
    grid = result["grid"]
    lon_1d = np.asarray(grid["geo_lon_1d"], dtype=np.float64)
    lat_1d = np.asarray(grid["geo_lat_1d"], dtype=np.float64)
    cumulative = np.asarray(result["geo_cumulative_rain_mm"], dtype=np.float64)
    max_rain = np.asarray(result["geo_max_rain_mmhr"], dtype=np.float64)
    duration10 = np.asarray(result["geo_duration10_h"], dtype=np.float64)
    duration20 = np.asarray(result["geo_duration20_h"], dtype=np.float64)
    max_time_index = np.asarray(result["geo_max_rain_time_index"], dtype=np.int32)
    track = result["track"].reset_index(drop=True)

    rows: List[Dict[str, object]] = []

    def append_row(
        key_type: str,
        lon: float,
        lat: float,
        value: float,
        unit: str,
        related_time: str,
        explanation: str,
    ) -> None:
        nearest = find_nearest_track_point(lon, lat, track)
        rows.append(
            {
                "typhoon_name": name,
                "key_type": key_type,
                "lon": float(lon),
                "lat": float(lat),
                "value": float(value),
                "unit": unit,
                "related_time": related_time,
                "nearest_track_time": nearest["nearest_track_time"],
                "nearest_track_distance_km": nearest["nearest_track_distance_km"],
                "explanation": explanation,
            }
        )

    value, lon, lat, _, _ = argmax_lon_lat(cumulative, lon_1d, lat_1d)
    append_row("max_cumulative_rain", lon, lat, value, "mm", "NA", "Maximum accumulated rainfall on the fixed geographic grid.")

    value, lon, lat, iy, ix = argmax_lon_lat(max_rain, lon_1d, lat_1d)
    time_idx = int(max_time_index[iy, ix]) if iy >= 0 and ix >= 0 else -1
    related_time = format_time(track.iloc[time_idx]["time"]) if 0 <= time_idx < len(track) else "NA"
    append_row("max_rain_mmhr", lon, lat, value, "mm/hr", related_time, "Maximum half-hour rain rate after geographic backprojection.")

    value, lon, lat, _, _ = argmax_lon_lat(duration10, lon_1d, lat_1d)
    append_row("max_duration10", lon, lat, value, "h", "NA", "Longest duration with rain rate >=10 mm/hr.")

    value, lon, lat, _, _ = argmax_lon_lat(duration20, lon_1d, lat_1d)
    append_row("max_duration20", lon, lat, value, "h", "NA", "Longest duration with rain rate >=20 mm/hr.")

    value, lon, lat = quantile_location(cumulative, lon_1d, lat_1d, 99)
    append_row("cumulative_rain_p99_location", lon, lat, value, "mm", "NA", "Representative location closest to the nonzero P99 accumulated rainfall.")

    value, lon, lat = quantile_location(duration10, lon_1d, lat_1d, 99)
    append_row("duration10_p99_location", lon, lat, value, "h", "NA", "Representative location closest to the nonzero P99 duration >=10 mm/hr.")

    return pd.DataFrame(rows)


def compute_geo_timeseries_metrics(results: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    frames = [result["timeseries_metrics"] for result in results.values()]
    return pd.concat(frames, ignore_index=True)


# =========================
# Saving
# =========================


def save_geographic_npz(results: Mapping[str, Mapping[str, object]], issues: List[str]) -> None:
    payload: Dict[str, np.ndarray] = {
        "GRID_MODE": np.asarray(GRID_MODE),
        "GEO_RES_DEG": np.asarray(GEO_RES_DEG, dtype=np.float32),
        "GEO_MARGIN_DEG": np.asarray(GEO_MARGIN_DEG, dtype=np.float32),
        "SAVE_TIMESLICE_GEO_FIELDS": np.asarray(SAVE_TIMESLICE_GEO_FIELDS),
    }
    for name, result in results.items():
        prefix = safe_name(name)
        grid = result["grid"]
        track = result["track"]
        payload[f"{prefix}_geo_lon_1d"] = np.asarray(grid["geo_lon_1d"], dtype=np.float32)
        payload[f"{prefix}_geo_lat_1d"] = np.asarray(grid["geo_lat_1d"], dtype=np.float32)
        payload[f"{prefix}_geo_cumulative_rain_mm"] = np.asarray(result["geo_cumulative_rain_mm"], dtype=np.float32)
        payload[f"{prefix}_geo_max_rain_mmhr"] = np.asarray(result["geo_max_rain_mmhr"], dtype=np.float32)
        payload[f"{prefix}_geo_duration10_h"] = np.asarray(result["geo_duration10_h"], dtype=np.float32)
        payload[f"{prefix}_geo_duration20_h"] = np.asarray(result["geo_duration20_h"], dtype=np.float32)
        payload[f"{prefix}_track_lon"] = pd.to_numeric(track["lon_180"], errors="coerce").to_numpy(dtype=np.float32)
        payload[f"{prefix}_track_lat"] = pd.to_numeric(track["lat"], errors="coerce").to_numpy(dtype=np.float32)
        payload[f"{prefix}_track_time"] = track["time"].astype(str).to_numpy()
        payload[f"{prefix}_track_WND"] = pd.to_numeric(track["WND"], errors="coerce").to_numpy(dtype=np.float32)
        payload[f"{prefix}_track_PRES"] = pd.to_numeric(track["PRES"], errors="coerce").to_numpy(dtype=np.float32)
    try:
        GEOGRAPHIC_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GEOGRAPHIC_NPZ_PATH, **payload)
    except Exception as exc:
        add_issue(issues, f"地理 NPZ 输出文件写入失败: {exc}")
        raise


def save_geographic_tables(
    summary_df: pd.DataFrame,
    key_locations_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    issues: List[str],
) -> None:
    try:
        GEOGRAPHIC_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(GEOGRAPHIC_SUMMARY_PATH, index=False)
        key_locations_df.to_csv(GEOGRAPHIC_KEY_LOCATIONS_PATH, index=False)
        timeseries_df.to_csv(GEOGRAPHIC_TIMESERIES_PATH, index=False)
    except Exception as exc:
        add_issue(issues, f"地理 CSV 输出文件写入失败: {exc}")
        raise


# =========================
# Figures
# =========================


def cartopy_local_features_available() -> bool:
    if not CARTOPY_IMPORT_OK:
        return False
    try:
        import cartopy

        roots = [
            Path(str(cartopy.config.get("data_dir", ""))),
            Path(str(cartopy.config.get("pre_existing_data_dir", ""))),
            Path(str(cartopy.config.get("repo_data_dir", ""))),
        ]
        needed = [
            "ne_50m_coastline.shp",
            "ne_50m_land.shp",
            "ne_50m_ocean.shp",
            "ne_50m_admin_0_boundary_lines_land.shp",
        ]
        available = set()
        for root in roots:
            if root.exists():
                for shp in root.rglob("*.shp"):
                    available.add(shp.name)
        return all(name in available for name in needed)
    except Exception:
        return False


def setup_geo_axes(fig, extent: Sequence[float], use_cartopy: bool):
    if use_cartopy and CARTOPY_IMPORT_OK:
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        try:
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.92", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="aliceblue", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":", zorder=3)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, linestyle="--", alpha=0.45)
        gl.top_labels = False
        gl.right_labels = False
        return ax, proj
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(True, linewidth=0.35, linestyle="--", alpha=0.35)
    return ax, None


def plot_track(ax, track: pd.DataFrame, transform=None) -> None:
    lon = pd.to_numeric(track["lon_180"], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(track["lat"], errors="coerce").to_numpy(dtype=np.float64)
    wnd = pd.to_numeric(track["WND"], errors="coerce").to_numpy(dtype=np.float64)
    kwargs = {"transform": transform} if transform is not None else {}
    ax.plot(lon, lat, color="black", linewidth=1.3, alpha=0.85, zorder=5, **kwargs)
    if np.isfinite(wnd).any() and float(np.nanmax(wnd)) > float(np.nanmin(wnd)):
        sizes = 12.0 + 44.0 * (wnd - np.nanmin(wnd)) / max(float(np.nanmax(wnd) - np.nanmin(wnd)), EPS)
    else:
        sizes = np.full_like(lon, 22.0)
    sc = ax.scatter(lon, lat, c=wnd, s=sizes, cmap="coolwarm", edgecolor="black", linewidth=0.25, zorder=6, **kwargs)
    ax.scatter([lon[0]], [lat[0]], marker="s", s=70, color="#2CA02C", edgecolor="black", linewidth=0.5, zorder=7, **kwargs)
    ax.scatter([lon[-1]], [lat[-1]], marker="X", s=90, color="#D62728", edgecolor="black", linewidth=0.5, zorder=7, **kwargs)
    return sc


def make_geo_map_figure(
    name: str,
    result: Mapping[str, object],
    field_key: str,
    title_suffix: str,
    unit: str,
    cmap: str,
    output_suffix: str,
    use_cartopy: bool,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grid = result["grid"]
    lon_1d = np.asarray(grid["geo_lon_1d"], dtype=np.float64)
    lat_1d = np.asarray(grid["geo_lat_1d"], dtype=np.float64)
    field = np.asarray(result[field_key], dtype=np.float64)
    track = result["track"]
    extent = [float(lon_1d.min()), float(lon_1d.max()), float(lat_1d.min()), float(lat_1d.max())]
    vmax = float(np.nanpercentile(field[field > 0.0], 99.2)) if np.any(field > 0.0) else 1.0
    vmax = max(vmax, 1.0)

    fig = plt.figure(figsize=(9.5, 7.2))
    ax, proj = setup_geo_axes(fig, extent, use_cartopy)
    mesh_kwargs = {"transform": proj} if proj is not None else {}
    im = ax.pcolormesh(lon_1d, lat_1d, field, cmap=cmap, shading="auto", vmin=0.0, vmax=vmax, zorder=2, **mesh_kwargs)
    sc = plot_track(ax, track, transform=proj)
    ax.set_title(f"{name} geographic {title_suffix}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label(unit)
    try:
        cbar2 = fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.09)
        cbar2.set_label("WND")
    except Exception:
        pass
    path = FIGURE_DIR / f"{safe_name(name)}_{output_suffix}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def make_geo_compare_figures(results: Mapping[str, Mapping[str, object]], use_cartopy: bool) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    compare_specs = [
        ("geo_cumulative_rain_mm", "geographic cumulative rainfall", "mm", "YlGnBu", "problem2_geo_cumulative_compare.png"),
        ("geo_duration10_h", "geographic duration >=10 mm/hr", "h", "plasma", "problem2_geo_duration10_compare.png"),
    ]
    for field_key, title, unit, cmap, filename in compare_specs:
        fields = [np.asarray(results[name][field_key], dtype=np.float64) for name in EXPECTED_TYPHOON_COUNTS]
        positive = np.concatenate([f[f > 0.0].ravel() for f in fields if np.any(f > 0.0)])
        vmax = max(float(np.percentile(positive, 99.2)), 1.0) if len(positive) else 1.0

        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
        last_im = None
        for ax, name in zip(axes, EXPECTED_TYPHOON_COUNTS):
            result = results[name]
            grid = result["grid"]
            lon_1d = np.asarray(grid["geo_lon_1d"], dtype=np.float64)
            lat_1d = np.asarray(grid["geo_lat_1d"], dtype=np.float64)
            field = np.asarray(result[field_key], dtype=np.float64)
            ax.set_xlim(float(lon_1d.min()), float(lon_1d.max()))
            ax.set_ylim(float(lat_1d.min()), float(lat_1d.max()))
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            ax.grid(True, linewidth=0.35, linestyle="--", alpha=0.35)
            last_im = ax.pcolormesh(lon_1d, lat_1d, field, cmap=cmap, shading="auto", vmin=0.0, vmax=vmax)
            plot_track(ax, result["track"], transform=None)
            ax.set_title(f"{name} {title}")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, label=unit)
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_all_figures(results: Mapping[str, Mapping[str, object]], issues: List[str], use_cartopy: bool) -> List[Path]:
    paths: List[Path] = []
    for name, result in results.items():
        for field_key, title_suffix, unit, cmap, output_suffix in FIGURE_SPECS:
            try:
                paths.append(make_geo_map_figure(name, result, field_key, title_suffix, unit, cmap, output_suffix, use_cartopy))
            except Exception as exc:
                add_issue(issues, f"图件生成失败 {name} {field_key}: {exc}")
                raise
    try:
        paths.extend(make_geo_compare_figures(results, use_cartopy))
    except Exception as exc:
        add_issue(issues, f"对比图件生成失败: {exc}")
        raise
    return paths


# =========================
# Report
# =========================


def conclusion_lines(summary_df: pd.DataFrame) -> List[str]:
    rows = {str(row["typhoon_name"]): row for _, row in summary_df.iterrows()}
    kong = rows["KONG-REY"]
    manyi = rows["MAN-YI"]
    duration_winner = "KONG-REY" if float(kong["geo_max_duration10_h"]) >= float(manyi["geo_max_duration10_h"]) else "MAN-YI"
    area_winner = (
        "KONG-REY"
        if float(kong["geo_area_cumulative_rain_ge_100mm_km2"]) >= float(manyi["geo_area_cumulative_rain_ge_100mm_km2"])
        else "MAN-YI"
    )
    return [
        (
            f"KONG-REY 地理累计降水高值中心位于约 "
            f"{float(kong['geo_max_cumulative_rain_lon']):.2f}E, {float(kong['geo_max_cumulative_rain_lat']):.2f}N，"
            f"最大累计降水约 {float(kong['geo_max_cumulative_rain_mm']):.1f} mm。"
        ),
        (
            f"MAN-YI 地理累计降水高值中心位于约 "
            f"{float(manyi['geo_max_cumulative_rain_lon']):.2f}E, {float(manyi['geo_max_cumulative_rain_lat']):.2f}N，"
            f"最大累计降水约 {float(manyi['geo_max_cumulative_rain_mm']):.1f} mm。"
        ),
        f"{duration_winner} 的 duration10 地理最大持续时间更长，说明其固定地理落区上的强降水停留特征更突出。",
        f"{area_winner} 的累计降水超过 100 mm 的地理面积更大，反映其可能影响范围更广。",
        "地理图补足了 storm-relative 图不能表示固定经纬度落区的不足。",
        "最终问题二结果应同时展示 storm-relative 结构图和 geographic 落区图，两者回答的问题不同，不能互相替代。",
    ]


def write_geo_report(
    calibrated: Mapping[str, np.ndarray],
    calibrated_index: pd.DataFrame,
    final_timeseries: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_key_times: pd.DataFrame,
    results: Mapping[str, Mapping[str, object]],
    summary_df: pd.DataFrame,
    key_locations_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    figure_paths: Sequence[Path],
    input_quality: Mapping[str, object],
    issues: Sequence[str],
    use_cartopy: bool,
) -> None:
    rain = np.asarray(calibrated["rain_mmhr_calibrated"])
    x_front = np.asarray(calibrated["x_front_km"], dtype=np.float64)
    y_left = np.asarray(calibrated["y_left_km"], dtype=np.float64)
    lines: List[str] = [
        "# Problem 2 Geographic Backprojection Report",
        "",
        "## 1. 输入输出文件",
        f"- calibrated NPZ: `{rel_path(CALIBRATED_NPZ_PATH)}`",
        f"- calibrated index: `{rel_path(CALIBRATED_INDEX_PATH)}`",
        f"- final timeseries: `{rel_path(FINAL_TIMESERIES_PATH)}`",
        f"- final summary: `{rel_path(FINAL_SUMMARY_PATH)}`",
        f"- final key times: `{rel_path(FINAL_KEY_TIMES_PATH)}`",
        f"- geographic NPZ: `{rel_path(GEOGRAPHIC_NPZ_PATH)}`",
        f"- geographic summary CSV: `{rel_path(GEOGRAPHIC_SUMMARY_PATH)}`",
        f"- geographic key locations CSV: `{rel_path(GEOGRAPHIC_KEY_LOCATIONS_PATH)}`",
        f"- geographic timeseries metrics CSV: `{rel_path(GEOGRAPHIC_TIMESERIES_PATH)}`",
        f"- figures directory: `{rel_path(FIGURE_DIR)}`",
        "",
        "## 2. 运行参数",
        f"- GRID_MODE: `{GRID_MODE}`",
        f"- GEO_RES_DEG: `{GEO_RES_DEG}`",
        f"- GEO_MARGIN_DEG: `{GEO_MARGIN_DEG}`",
        f"- SAVE_TIMESLICE_GEO_FIELDS: `{SAVE_TIMESLICE_GEO_FIELDS}`",
        f"- 是否使用 cartopy: `{use_cartopy}`",
        "- 插值方法: `scipy.interpolate.RegularGridInterpolator`, target-grid reverse sampling, out-of-bounds filled with 0.0",
        f"- storm-relative x_front_km range: {float(np.min(x_front)):.1f} to {float(np.max(x_front)):.1f} km",
        f"- storm-relative y_left_km range: {float(np.min(y_left)):.1f} to {float(np.max(y_left)):.1f} km",
        "",
        "## 3. 数据完整性检查",
        f"- calibrated 场 shape: `{rain.shape}`",
        f"- calibrated index shape: `{calibrated_index.shape}`",
        f"- final timeseries shape: `{final_timeseries.shape}`",
        f"- final summary shape: `{final_summary.shape}`",
        f"- final key times shape: `{final_key_times.shape}`",
        f"- KONG-REY 时刻数: {EXPECTED_TYPHOON_COUNTS['KONG-REY']}",
        f"- MAN-YI 时刻数: {EXPECTED_TYPHOON_COUNTS['MAN-YI']}",
        f"- 输入场 NaN/Inf/负值/全零检查: `{dict(input_quality)}`",
    ]

    for name, result in results.items():
        grid = result["grid"]
        qc = result["qc"]
        lines.extend(
            [
                f"- {name} 经纬度网格范围: lon {float(grid['lon_min']):.2f} to {float(grid['lon_max']):.2f}, "
                f"lat {float(grid['lat_min']):.2f} to {float(grid['lat_max']):.2f}, "
                f"shape ({len(grid['geo_lat_1d'])}, {len(grid['geo_lon_1d'])})",
                f"- {name} 反投影后 NaN/Inf/负值/全零检查: `{qc}`",
            ]
        )
    if issues:
        lines.extend(["", "### 记录的问题", *[f"- {item}" for item in issues]])
    else:
        lines.extend(["", "### 记录的问题", "- None"])

    lines.extend(["", "## 4. 分台风地理结果摘要"])
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"### {row['typhoon_name']}",
                f"- 地理累计降水最大值: {float(row['geo_max_cumulative_rain_mm']):.3f} mm "
                f"at ({float(row['geo_max_cumulative_rain_lon']):.3f}, {float(row['geo_max_cumulative_rain_lat']):.3f})",
                f"- 地理最大半小时雨强: {float(row['geo_max_rain_mmhr']):.3f} mm/hr "
                f"at ({float(row['geo_max_rain_lon']):.3f}, {float(row['geo_max_rain_lat']):.3f}), "
                f"time {row['geo_max_rain_time']}",
                f"- 最大 duration10: {float(row['geo_max_duration10_h']):.3f} h "
                f"at ({float(row['geo_max_duration10_lon']):.3f}, {float(row['geo_max_duration10_lat']):.3f})",
                f"- 最大 duration20: {float(row['geo_max_duration20_h']):.3f} h "
                f"at ({float(row['geo_max_duration20_lon']):.3f}, {float(row['geo_max_duration20_lat']):.3f})",
                f"- 累计降水 >=50/100/200 mm 面积: "
                f"{float(row['geo_area_cumulative_rain_ge_50mm_km2']):.1f} / "
                f"{float(row['geo_area_cumulative_rain_ge_100mm_km2']):.1f} / "
                f"{float(row['geo_area_cumulative_rain_ge_200mm_km2']):.1f} km2",
                f"- duration10 >=1/3/6 h 面积: "
                f"{float(row['geo_area_duration10_ge_1h_km2']):.1f} / "
                f"{float(row['geo_area_duration10_ge_3h_km2']):.1f} / "
                f"{float(row['geo_area_duration10_ge_6h_km2']):.1f} km2",
                f"- area_time_10 / area_time_20: "
                f"{float(row['geo_area_time_10_km2_h']):.1f} / {float(row['geo_area_time_20_km2_h']):.1f} km2 h",
            ]
        )

    lines.extend(
        [
            "",
            "## 5. 与 storm-relative 结果关系说明",
            "- storm-relative 图用于解释降水相对台风中心和移动方向的结构。",
            "- geographic 图用于展示固定地理经纬度上的可能降水落区。",
            "- 两者回答的问题不同，不能互相替代。",
            "- 论文中应同时保留两类图：storm-relative 结构图负责机理解释，geographic 落区图负责真实空间影响表达。",
            "",
            "## 6. 输出检查",
            f"- geographic summary shape: `{summary_df.shape}`",
            f"- geographic key locations shape: `{key_locations_df.shape}`",
            f"- geographic timeseries shape: `{timeseries_df.shape}`",
            f"- generated figure count: {len(figure_paths)}",
        ]
    )
    lines.extend([f"- `{rel_path(path)}`" for path in figure_paths])

    lines.extend(["", "## 7. 论文可写结论"])
    lines.extend([f"- {item}" for item in conclusion_lines(summary_df)])

    try:
        GEOGRAPHIC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEOGRAPHIC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"QC/report output write failed: {exc}") from exc


# =========================
# Main
# =========================


def main() -> None:
    issues: List[str] = []
    try:
        calibrated = load_calibrated_npz(issues)
        calibrated_index = load_calibrated_index(issues)
        final_timeseries = load_final_timeseries(issues)
        final_summary = load_final_summary(issues)
        final_key_times = load_final_key_times(issues)
        canonical_timeseries = build_canonical_timeseries(calibrated, final_timeseries, issues)
        input_quality = validate_input_consistency(calibrated, calibrated_index, canonical_timeseries, issues)

        use_cartopy = cartopy_local_features_available()
        print(f"[info] cartopy import ok: {CARTOPY_IMPORT_OK}; using cartopy map base: {use_cartopy}")

        rain = np.asarray(calibrated["rain_mmhr_calibrated"], dtype=np.float32)
        x_front = np.asarray(calibrated["x_front_km"], dtype=np.float64)
        y_left = np.asarray(calibrated["y_left_km"], dtype=np.float64)

        results: Dict[str, Mapping[str, object]] = {}
        for name in EXPECTED_TYPHOON_COUNTS:
            sub = canonical_timeseries.loc[canonical_timeseries["typhoon_name"].eq(name)].copy()
            results[name] = backproject_one_typhoon(name, sub, rain, x_front, y_left, issues)

        summary_df = pd.DataFrame([compute_geo_summary(result) for result in results.values()])
        key_locations_df = pd.concat([compute_geo_key_locations(result) for result in results.values()], ignore_index=True)
        geo_timeseries_df = compute_geo_timeseries_metrics(results)

        save_geographic_npz(results, issues)
        save_geographic_tables(summary_df, key_locations_df, geo_timeseries_df, issues)
        figure_paths = make_all_figures(results, issues, use_cartopy)
        write_geo_report(
            calibrated,
            calibrated_index,
            final_timeseries,
            final_summary,
            final_key_times,
            results,
            summary_df,
            key_locations_df,
            geo_timeseries_df,
            figure_paths,
            input_quality,
            issues,
            use_cartopy,
        )

        write_run_log(issues, "success")
        print("[done] Geographic backprojection completed.")
        print(f"[done] NPZ: {GEOGRAPHIC_NPZ_PATH}")
        print(f"[done] Summary: {GEOGRAPHIC_SUMMARY_PATH}")
        print(f"[done] Key locations: {GEOGRAPHIC_KEY_LOCATIONS_PATH}")
        print(f"[done] Timeseries: {GEOGRAPHIC_TIMESERIES_PATH}")
        print(f"[done] Report: {GEOGRAPHIC_REPORT_PATH}")
        print(f"[done] Figures: {FIGURE_DIR} ({len(figure_paths)} files)")
    except Exception as exc:
        tb = traceback.format_exc()
        add_issue(issues, f"不可恢复错误: {exc}")
        write_run_log(issues, "failed", tb)
        raise


if __name__ == "__main__":
    main()
