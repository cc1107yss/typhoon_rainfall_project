#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Problem-2 historical half-hour sample library.

The output table is intentionally a historical sample library: rain_*,
centroid_*, quad_*, radius and anisotropy metrics are labels / template
diagnostics computed from historical GPM fields. They must not be used as
input features for the 2024 KONG-REY or MAN-YI target cases.
"""

from __future__ import annotations

import argparse
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is available in this project env.
    def tqdm(iterable=None, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPM_ROOT = PROJECT_ROOT / "data" / "raw" / "GPM_3IMERGHHE.07"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_CSV = DEFAULT_PROCESSED_DIR / "problem2_historical_halfhour_sample_library.csv"
DEFAULT_QC_REPORT = PROJECT_ROOT / "outputs" / "problem2_historical_library_qc_report.md"

TRACK_MODEL_CANDIDATES = [
    DEFAULT_PROCESSED_DIR / "env_added" / "gpm_track_model_features_interp_env.csv",
    DEFAULT_PROCESSED_DIR / "gpm_track_model_features_motion.csv",
    DEFAULT_PROCESSED_DIR / "gpm_track_model_features_interp.csv",
    DEFAULT_PROCESSED_DIR / "gpm_track_interpolated_features_clean.csv",
    DEFAULT_PROCESSED_DIR / "gpm_track_interpolated_features_all.csv",
]
TRACK_FEATURE_PATH = DEFAULT_PROCESSED_DIR / "typhoon_track_features.csv"
EVENT_MAPPING_PATH = DEFAULT_PROCESSED_DIR / "gpm_track_event_mapping.csv"

NODATA_DEFAULT = -9999.0
KM_PER_DEG = 111.32
CELL_SIZE_FALLBACK_DEG = 0.1
CENTER_MATCH_OK_KM = 80.0
ABNORMAL_RAIN_MMHR = 1000.0
EPS = 1e-12

TARGET_NAME_NORMS = {"KONGREY", "MANYI"}
TARGET_WINDOWS = [
    ("KONG-REY", pd.Timestamp("2024-10-24"), pd.Timestamp("2024-11-03")),
    ("MAN-YI", pd.Timestamp("2024-11-08"), pd.Timestamp("2024-11-21")),
]

FILENAME_PATTERN = re.compile(
    r"3IMERG\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6}).*?"
    r"_center_(?P<center_lon>[-+]?\d+(?:\.\d+)?)E_(?P<center_lat>[-+]?\d+(?:\.\d+)?)N"
    r"_bbox_(?P<lon_min>[-+]?\d+(?:\.\d+)?)E_(?P<lon_max>[-+]?\d+(?:\.\d+)?)E_"
    r"(?P<lat_min>[-+]?\d+(?:\.\d+)?)N_(?P<lat_max>[-+]?\d+(?:\.\d+)?)N"
)

ENV_OPTIONAL_COLS = [
    "landfrac_100km",
    "landfrac_200km",
    "landfrac_300km",
    "landfrac_500km",
    "elev_mean_200km",
    "elev_max_200km",
    "terrain_std_200km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",
]

KEY_DISPLAY_COLS = [
    "event_uid",
    "typhoon_name",
    "time",
    "tif_path",
    "lat",
    "lon_180",
    "WND",
    "PRES",
    "move_speed_kmh",
    "move_dir_deg",
    "rain_max_mmhr",
    "rain_p95_mmhr",
    "rain_area_10_km2",
    "centroid_offset_km",
    "anisotropy",
]


@dataclass
class BuildStats:
    tif_total: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    failed_records: List[Dict[str, object]] = field(default_factory=list)
    excluded_target_records: List[Dict[str, object]] = field(default_factory=list)
    filename_parse_errors: List[Dict[str, object]] = field(default_factory=list)
    missing_tif_records: List[Dict[str, object]] = field(default_factory=list)
    center_mismatch_records: List[Dict[str, object]] = field(default_factory=list)
    all_missing_rain_records: List[Dict[str, object]] = field(default_factory=list)
    missing_motion_for_structure: int = 0
    negative_cell_count: int = 0
    abnormal_cell_count: int = 0
    env_missing_cols: List[str] = field(default_factory=list)
    input_model_file: Optional[str] = None
    track_fallback_used: bool = False


def parse_gpm_filename(tif_path: Path) -> Dict[str, object]:
    """Parse time, filename center and bbox metadata from a GPM GeoTIFF name."""
    match = FILENAME_PATTERN.search(tif_path.name)
    if match is None:
        raise ValueError(f"Cannot parse GPM filename: {tif_path.name}")

    info = match.groupdict()
    time = pd.to_datetime(info["date"] + info["start"], format="%Y%m%d%H%M%S")
    time_end = pd.to_datetime(info["date"] + info["end"], format="%Y%m%d%H%M%S")
    return {
        "time": time,
        "time_end": time_end,
        "gpm_center_lon": float(info["center_lon"]),
        "gpm_center_lat": float(info["center_lat"]),
        "bbox_lon_min": float(info["lon_min"]),
        "bbox_lon_max": float(info["lon_max"]),
        "bbox_lat_min": float(info["lat_min"]),
        "bbox_lat_max": float(info["lat_max"]),
    }


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def to_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def wrap_lon_180(lon: object) -> float:
    value = to_float(lon)
    if not np.isfinite(value):
        return np.nan
    return float(((value + 180.0) % 360.0) - 180.0)


def lon_180_to_360(lon_180: object) -> float:
    value = to_float(lon_180)
    if not np.isfinite(value):
        return np.nan
    return float(value if value >= 0 else value + 360.0)


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def first_value(row: pd.Series, candidates: Sequence[str], default=np.nan):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return value
    return default


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Supports numpy arrays."""
    radius = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    """Bearing from point 1 to point 2. 0 north, 90 east."""
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def coordinate_grids(transform: rasterio.Affine, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon_grid = transform.a * col_grid + transform.b * row_grid + transform.c
    lat_grid = transform.d * col_grid + transform.e * row_grid + transform.f
    return lon_grid, lat_grid


def read_gpm_tif(tif_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Read one GeoTIFF and return cleaned rain rate, lon grid, lat grid and metadata."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else NODATA_DEFAULT
        transform = src.transform
        crs = src.crs
        lon_grid, lat_grid = coordinate_grids(transform, arr.shape)

    rain = np.where(np.isfinite(arr) & (arr != nodata), arr, np.nan)
    negative_count = int(np.count_nonzero(np.isfinite(rain) & (rain < 0.0)))
    abnormal_count = int(np.count_nonzero(np.isfinite(rain) & (rain > ABNORMAL_RAIN_MMHR)))
    rain = np.where(np.isfinite(rain) & (rain < 0.0), 0.0, rain)
    rain = np.where(np.isfinite(rain) & (rain > ABNORMAL_RAIN_MMHR), np.nan, rain)

    pixel_size_deg = float(np.nanmean([abs(transform.a), abs(transform.e)]))
    meta = {
        "grid_nrow": int(arr.shape[0]),
        "grid_ncol": int(arr.shape[1]),
        "pixel_size_deg": pixel_size_deg,
        "crs": str(crs) if crs is not None else "",
        "negative_cell_count": negative_count,
        "abnormal_cell_count": abnormal_count,
    }
    return rain, lon_grid, lat_grid, meta


def compute_grid_area(lat_grid: np.ndarray, pixel_size_deg: float) -> np.ndarray:
    """Area of each grid cell in km2, using row-wise latitude weighting."""
    cell_size = pixel_size_deg if np.isfinite(pixel_size_deg) and pixel_size_deg > 0 else CELL_SIZE_FALLBACK_DEG
    return (KM_PER_DEG * cell_size) * (KM_PER_DEG * cell_size) * np.cos(np.radians(lat_grid))


def weighted_radius(dist_km: np.ndarray, weights: np.ndarray, valid: np.ndarray, q: float) -> float:
    positive = valid & np.isfinite(weights) & (weights > 0.0) & np.isfinite(dist_km)
    if int(np.count_nonzero(positive)) == 0:
        return np.nan
    d = dist_km[positive].ravel()
    w = weights[positive].ravel()
    order = np.argsort(d)
    d_sorted = d[order]
    w_sorted = w[order]
    cumsum = np.cumsum(w_sorted)
    if not np.isfinite(cumsum[-1]) or cumsum[-1] <= EPS:
        return np.nan
    idx = int(np.searchsorted(cumsum, q * cumsum[-1]))
    idx = min(idx, len(d_sorted) - 1)
    return float(d_sorted[idx])


def compute_rainfall_metrics(
    rain_mmhr: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    pixel_size_deg: float,
) -> Dict[str, object]:
    valid = np.isfinite(rain_mmhr)
    valid_n = int(np.count_nonzero(valid))
    total_n = int(rain_mmhr.size)
    out: Dict[str, object] = {
        "rain_valid_ratio": float(valid_n / total_n) if total_n else np.nan,
        "rain_mean_mmhr": np.nan,
        "rain_max_mmhr": np.nan,
        "rain_p50_mmhr": np.nan,
        "rain_p75_mmhr": np.nan,
        "rain_p90_mmhr": np.nan,
        "rain_p95_mmhr": np.nan,
        "rain_p99_mmhr": np.nan,
        "rain_sum_halfhour_mm": np.nan,
        "rain_area_1_km2": np.nan,
        "rain_area_5_km2": np.nan,
        "rain_area_10_km2": np.nan,
        "rain_area_20_km2": np.nan,
        "heavy_rain_fraction_10": np.nan,
    }
    if valid_n == 0:
        return out

    values = rain_mmhr[valid]
    rain_halfhour_mm = rain_mmhr * 0.5
    cell_area = compute_grid_area(lat_grid, pixel_size_deg)
    out.update({
        "rain_mean_mmhr": float(np.nanmean(values)),
        "rain_max_mmhr": float(np.nanmax(values)),
        "rain_p50_mmhr": float(np.nanpercentile(values, 50)),
        "rain_p75_mmhr": float(np.nanpercentile(values, 75)),
        "rain_p90_mmhr": float(np.nanpercentile(values, 90)),
        "rain_p95_mmhr": float(np.nanpercentile(values, 95)),
        "rain_p99_mmhr": float(np.nanpercentile(values, 99)),
        "rain_sum_halfhour_mm": float(np.nansum(rain_halfhour_mm[valid])),
    })
    for threshold in [1, 5, 10, 20]:
        mask = valid & (rain_mmhr >= float(threshold))
        out[f"rain_area_{threshold}_km2"] = float(np.nansum(cell_area[mask]))
    out["heavy_rain_fraction_10"] = float(np.count_nonzero(valid & (rain_mmhr >= 10.0)) / valid_n)
    return out


def compute_spatial_structure_metrics(
    rain_mmhr: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    center_lon: float,
    center_lat: float,
    move_dir_deg: float,
) -> Dict[str, object]:
    names = [
        "rain_centroid_lon",
        "rain_centroid_lat",
        "centroid_offset_km",
        "centroid_angle_deg",
        "asym_front_back_ratio",
        "asym_left_right_ratio",
        "quad_front_left_sum",
        "quad_front_right_sum",
        "quad_back_left_sum",
        "quad_back_right_sum",
        "quad_front_left_ratio",
        "quad_front_right_ratio",
        "quad_back_left_ratio",
        "quad_back_right_ratio",
        "anisotropy",
        "rain_radius_r50_km",
        "rain_radius_r80_km",
        "rain_radius_r90_km",
        "rain_band_width_km",
    ]
    out: Dict[str, object] = {name: np.nan for name in names}
    out["rel_grid_available"] = False

    if not (np.isfinite(center_lon) and np.isfinite(center_lat) and np.isfinite(move_dir_deg)):
        return out

    valid = np.isfinite(rain_mmhr)
    weights = np.where(valid, rain_mmhr * 0.5, 0.0)
    total = float(np.sum(weights))
    out["rel_grid_available"] = True
    if total <= EPS:
        return out

    dx_deg = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    x_east_km = dx_deg * KM_PER_DEG * math.cos(math.radians(center_lat))
    y_north_km = (lat_grid - center_lat) * KM_PER_DEG
    theta = math.radians(move_dir_deg % 360.0)

    x_front = x_east_km * math.sin(theta) + y_north_km * math.cos(theta)
    y_left = -x_east_km * math.cos(theta) + y_north_km * math.sin(theta)

    centroid_dx_deg = float(np.sum(dx_deg * weights) / total)
    centroid_lat = float(np.sum(lat_grid * weights) / total)
    centroid_lon = wrap_lon_180(center_lon + centroid_dx_deg)
    out["rain_centroid_lon"] = centroid_lon
    out["rain_centroid_lat"] = centroid_lat
    out["centroid_offset_km"] = float(haversine_km(center_lon, center_lat, centroid_lon, centroid_lat))
    out["centroid_angle_deg"] = float(bearing_deg(center_lon, center_lat, centroid_lon, centroid_lat))

    front = x_front >= 0.0
    back = ~front
    left = y_left >= 0.0
    right = ~left

    front_sum = float(np.sum(weights[front]))
    back_sum = float(np.sum(weights[back]))
    left_sum = float(np.sum(weights[left]))
    right_sum = float(np.sum(weights[right]))
    fl_sum = float(np.sum(weights[front & left]))
    fr_sum = float(np.sum(weights[front & right]))
    bl_sum = float(np.sum(weights[back & left]))
    br_sum = float(np.sum(weights[back & right]))

    out.update({
        "asym_front_back_ratio": safe_div(front_sum, back_sum),
        "asym_left_right_ratio": safe_div(left_sum, right_sum),
        "quad_front_left_sum": fl_sum,
        "quad_front_right_sum": fr_sum,
        "quad_back_left_sum": bl_sum,
        "quad_back_right_sum": br_sum,
        "quad_front_left_ratio": safe_div(fl_sum, total),
        "quad_front_right_ratio": safe_div(fr_sum, total),
        "quad_back_left_ratio": safe_div(bl_sum, total),
        "quad_back_right_ratio": safe_div(br_sum, total),
    })

    positive = valid & (rain_mmhr > 0.0)
    if int(np.count_nonzero(positive)) >= 2:
        x = x_front[positive].ravel()
        y = y_left[positive].ravel()
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
            out["anisotropy"] = float((lam1 - lam2) / (lam1 + lam2))

    dist_km = np.sqrt(x_east_km * x_east_km + y_north_km * y_north_km)
    r50 = weighted_radius(dist_km, weights, valid, 0.50)
    r80 = weighted_radius(dist_km, weights, valid, 0.80)
    r90 = weighted_radius(dist_km, weights, valid, 0.90)
    out["rain_radius_r50_km"] = r50
    out["rain_radius_r80_km"] = r80
    out["rain_radius_r90_km"] = r90
    out["rain_band_width_km"] = float(r90 - r50) if np.isfinite(r90) and np.isfinite(r50) else np.nan
    return out


def find_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def read_csv_with_times(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in ["time", "time_end", "track_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["gpm_event_uid", "track_event_uid", "event_uid"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def interp_numeric(t_src: np.ndarray, y_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y_src)
    if int(valid.sum()) < 2:
        return np.full_like(t_new, np.nan, dtype=float)
    return np.interp(t_new, t_src[valid], y_src[valid])


def interp_lon_degree(t_src: np.ndarray, lon_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    valid = np.isfinite(lon_src)
    if int(valid.sum()) < 2:
        return np.full_like(t_new, np.nan, dtype=float)
    lon_unwrap = np.unwrap(np.deg2rad(lon_src[valid]))
    interp_rad = np.interp(t_new, t_src[valid], lon_unwrap)
    return ((np.rad2deg(interp_rad) + 180.0) % 360.0) - 180.0


def nearest_rows_by_time(track_one: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    track_times = track_one["track_time"].values.astype("datetime64[ns]")
    indices = []
    for t in times.values.astype("datetime64[ns]"):
        pos = np.searchsorted(track_times, t)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(track_times):
            candidates.append(pos)
        best = min(candidates, key=lambda i: abs((pd.Timestamp(t) - track_one.iloc[i]["track_time"]).total_seconds()))
        indices.append(track_one.index[best])
    return track_one.loc[indices].reset_index(drop=True)


def interpolate_track_to_gpm_times(gpm_df: pd.DataFrame, track_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback interpolation if the ready-made interpolated table is absent.
    In the current project this function is normally bypassed by the existing
    gpm_track_model_features_motion.csv table.
    """
    gpm = gpm_df.copy()
    track = track_df.copy()
    mapping = mapping_df.copy()
    if "event_uid" in gpm.columns:
        gpm = gpm.rename(columns={"event_uid": "gpm_event_uid"})
    if "event_uid" in track.columns:
        track = track.rename(columns={"event_uid": "track_event_uid", "time": "track_time"})

    event_map = (
        mapping.groupby(["gpm_event_uid", "track_event_uid"])
        .size()
        .reset_index(name="count")
        .sort_values(["gpm_event_uid", "count"], ascending=[True, False])
        .drop_duplicates("gpm_event_uid")
    )
    gpm_to_track = dict(zip(event_map["gpm_event_uid"].astype(str), event_map["track_event_uid"].astype(str)))

    parts = []
    for gpm_event_uid, gpm_one in gpm.groupby("gpm_event_uid"):
        gpm_one = gpm_one.sort_values("time").copy()
        track_event_uid = gpm_to_track.get(str(gpm_event_uid))
        if not track_event_uid:
            gpm_one["track_event_uid"] = np.nan
            gpm_one["interp_match_status"] = "no_event_mapping"
            parts.append(gpm_one)
            continue
        track_one = track[track["track_event_uid"].astype(str).eq(str(track_event_uid))].sort_values("track_time").copy()
        if len(track_one) < 2:
            gpm_one["track_event_uid"] = track_event_uid
            gpm_one["interp_match_status"] = "insufficient_track_points"
            parts.append(gpm_one)
            continue

        t_src = track_one["track_time"].astype("int64").to_numpy(dtype=np.float64) / 1e9
        t_new = gpm_one["time"].astype("int64").to_numpy(dtype=np.float64) / 1e9
        gpm_one["track_event_uid"] = track_event_uid
        gpm_one["track_lat"] = interp_numeric(t_src, pd.to_numeric(track_one["lat"], errors="coerce").to_numpy(), t_new)
        gpm_one["track_lon_180"] = interp_lon_degree(t_src, pd.to_numeric(track_one["lon_180"], errors="coerce").to_numpy(), t_new)
        for src, dst in [("wind", "track_wind"), ("pressure", "track_pressure"), ("WND", "track_wind"), ("PRES", "track_pressure")]:
            if src in track_one.columns and dst not in gpm_one.columns:
                gpm_one[dst] = interp_numeric(t_src, pd.to_numeric(track_one[src], errors="coerce").to_numpy(), t_new)
        for src in ["coast_dist_km", "signed_coast_dist_km", "wind_change_rate", "pressure_change_rate"]:
            if src in track_one.columns:
                gpm_one[f"track_{src}"] = interp_numeric(t_src, pd.to_numeric(track_one[src], errors="coerce").to_numpy(), t_new)
        nearest = nearest_rows_by_time(track_one, gpm_one["time"])
        for src in ["source_file", "typhoon_name", "intensity", "is_land"]:
            if src in nearest.columns:
                gpm_one[f"track_{src}"] = nearest[src].values
        gpm_one["interp_match_status"] = "fallback_interpolated"
        parts.append(gpm_one)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compute_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill/recompute half-hour motion features from interpolated centers."""
    out = df.copy()
    for col in ["dt_h", "move_distance_km", "move_speed_kmh", "move_dir_deg", "wind_change_rate", "pressure_change_rate"]:
        if col not in out.columns:
            out[col] = np.nan

    for event_uid, idx in out.groupby("event_uid", dropna=False).groups.items():
        sub = out.loc[idx].sort_values("time")
        sub_idx = sub.index
        dt = sub["time"].diff().dt.total_seconds() / 3600.0
        prev_lon = sub["lon_180"].shift(1)
        prev_lat = sub["lat"].shift(1)
        dist = haversine_km(prev_lon, prev_lat, sub["lon_180"], sub["lat"])
        direction = bearing_deg(prev_lon, prev_lat, sub["lon_180"], sub["lat"])
        speed = dist / dt
        wnd_rate = pd.to_numeric(sub["WND"], errors="coerce").diff() / dt
        pres_rate = pd.to_numeric(sub["PRES"], errors="coerce").diff() / dt

        for target, values in [
            ("dt_h", dt),
            ("move_distance_km", dist),
            ("move_speed_kmh", speed),
            ("move_dir_deg", direction),
            ("wind_change_rate", wnd_rate),
            ("pressure_change_rate", pres_rate),
        ]:
            existing = pd.to_numeric(out.loc[sub_idx, target], errors="coerce")
            filled = existing.where(existing.notna(), pd.Series(values, index=sub_idx))
            out.loc[sub_idx, target] = filled.values

        # First frames often lack a backward direction. Use the next center to estimate it.
        dirs = pd.to_numeric(out.loc[sub_idx, "move_dir_deg"], errors="coerce").copy()
        need = dirs.isna()
        if need.any() and len(sub) >= 2:
            next_lon = sub["lon_180"].shift(-1)
            next_lat = sub["lat"].shift(-1)
            forward_dir = pd.Series(bearing_deg(sub["lon_180"], sub["lat"], next_lon, next_lat), index=sub_idx)
            dirs = dirs.where(dirs.notna(), forward_dir)
            out.loc[sub_idx, "move_dir_deg"] = dirs.values

    return out


def resolve_tif_path(row: pd.Series, gpm_root: Path) -> Tuple[Path, bool]:
    existing = str(first_value(row, ["tif_path"], "")).strip()
    if existing and existing.lower() != "nan":
        p = Path(existing)
        candidates = [p]
        if not p.is_absolute():
            candidates.append(PROJECT_ROOT / p)
        for candidate in candidates:
            if candidate.exists():
                return candidate, True

    event = str(first_value(row, ["gpm_event_dir", "gpm_event_uid", "event_uid"], "")).strip()
    source = str(first_value(row, ["source_file"], "")).strip()
    candidates = []
    if event and source and event.lower() != "nan" and source.lower() != "nan":
        candidates.append(gpm_root / event / source)
    if source and source.lower() != "nan":
        candidates.append(gpm_root / source)
    for candidate in candidates:
        if candidate.exists():
            return candidate, True
    return (candidates[0] if candidates else Path(source)), False


def add_or_map_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "track_event_uid": "event_uid",
        "gpm_event_uid": "gpm_event_dir",
        "track_typhoon_name": "typhoon_name",
        "track_lat": "lat",
        "track_lon_180": "lon_180",
        "track_wind": "WND",
        "track_pressure": "PRES",
        "track_intensity": "intensity",
        "track_move_speed_kmh": "move_speed_kmh",
        "track_move_dir_deg": "move_dir_deg",
        "track_move_distance_km": "move_distance_km",
        "track_wind_change_rate": "wind_change_rate",
        "track_pressure_change_rate": "pressure_change_rate",
        "track_dt_h": "dt_h",
        "track_is_land": "is_land",
        "track_coast_dist_km": "coast_dist_km",
        "track_signed_coast_dist_km": "signed_coast_dist_km",
    }
    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "event_uid" not in out.columns and "gpm_event_dir" in out.columns:
        out["event_uid"] = out["gpm_event_dir"].astype(str)
    if "gpm_event_dir" not in out.columns and "event_uid" in out.columns:
        out["gpm_event_dir"] = out["event_uid"].astype(str)
    if "lon_180" not in out.columns:
        lon_col = first_existing(out, ["lon", "center_lon"])
        out["lon_180"] = out[lon_col].map(wrap_lon_180) if lon_col else np.nan
    if "lon" not in out.columns:
        out["lon"] = out["lon_180"].map(lon_180_to_360)
    if "WND" not in out.columns:
        wind_col = first_existing(out, ["wind", "track_wind", "WND"])
        out["WND"] = out[wind_col] if wind_col else np.nan
    if "PRES" not in out.columns:
        pres_col = first_existing(out, ["pressure", "track_pressure", "PRES"])
        out["PRES"] = out[pres_col] if pres_col else np.nan
    if "typhoon_name" not in out.columns:
        out["typhoon_name"] = "UNKNOWN"

    out["event_uid"] = out["event_uid"].astype(str)
    out["gpm_event_dir"] = out["gpm_event_dir"].astype(str)
    out["typhoon_name"] = out["typhoon_name"].fillna("UNKNOWN").astype(str).str.upper()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    out["day"] = out["time"].dt.day
    out["hour"] = out["time"].dt.hour + out["time"].dt.minute / 60.0
    out["season"] = ((out["month"] % 12) // 3 + 1).map({1: "winter", 2: "spring", 3: "summer", 4: "autumn"})
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)

    out = out.sort_values(["event_uid", "time", "source_file"]).reset_index(drop=True)
    out["life_time_index"] = out.groupby("event_uid").cumcount()
    counts = out.groupby("event_uid")["life_time_index"].transform("max")
    out["life_progress"] = np.where(counts > 0, out["life_time_index"] / counts, 0.0)
    return out


def mark_target_candidates(df: pd.DataFrame) -> pd.Series:
    name_hit = df["typhoon_name"].map(normalize_name).isin(TARGET_NAME_NORMS)
    time = pd.to_datetime(df["time"], errors="coerce")
    time_hit = pd.Series(False, index=df.index)
    for _, start, end_exclusive in TARGET_WINDOWS:
        time_hit = time_hit | ((time >= start) & (time < end_exclusive))
    return name_hit | time_hit


def load_input_table(stats: BuildStats, gpm_root: Path, input_table: Optional[Path] = None) -> pd.DataFrame:
    model_file = input_table if input_table is not None else find_existing_path(TRACK_MODEL_CANDIDATES)
    if model_file is not None:
        try:
            stats.input_model_file = str(model_file.relative_to(PROJECT_ROOT))
        except ValueError:
            stats.input_model_file = str(model_file)
        return read_csv_with_times(model_file)

    gpm_feature_path = DEFAULT_PROCESSED_DIR / "gpm_precip_features.csv"
    if not (gpm_feature_path.exists() and TRACK_FEATURE_PATH.exists() and EVENT_MAPPING_PATH.exists()):
        raise FileNotFoundError(
            "Missing ready-made interpolated table and fallback inputs. "
            "Expected gpm_track_model_features_motion.csv or gpm_precip_features.csv + "
            "typhoon_track_features.csv + gpm_track_event_mapping.csv."
        )

    stats.track_fallback_used = True
    gpm = read_csv_with_times(gpm_feature_path)
    track = read_csv_with_times(TRACK_FEATURE_PATH)
    mapping = read_csv_with_times(EVENT_MAPPING_PATH)
    return interpolate_track_to_gpm_times(gpm, track, mapping)


def format_time(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def build_historical_library(
    gpm_root: Path = DEFAULT_GPM_ROOT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    qc_report: Path = DEFAULT_QC_REPORT,
    max_rows: Optional[int] = None,
    input_table: Optional[Path] = None,
) -> Tuple[pd.DataFrame, BuildStats]:
    stats = BuildStats()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    qc_report.parent.mkdir(parents=True, exist_ok=True)

    stats.tif_total = len(list(gpm_root.glob("*/*.tif")))
    base = load_input_table(stats, gpm_root, input_table=input_table)
    base = add_or_map_base_columns(base)
    base = add_time_features(base)
    base = compute_motion_features(base)

    for col in ENV_OPTIONAL_COLS:
        source_col = first_existing(base, [col, f"track_{col}"])
        if source_col and col not in base.columns:
            base[col] = base[source_col]
        elif col not in base.columns:
            base[col] = np.nan
            stats.env_missing_cols.append(col)

    target_mask = mark_target_candidates(base)
    target_candidates = base.loc[target_mask].copy()
    for _, row in target_candidates.iterrows():
        tif_path, exists = resolve_tif_path(row, gpm_root)
        stats.excluded_target_records.append({
            "event_uid": row.get("event_uid", ""),
            "gpm_event_dir": row.get("gpm_event_dir", ""),
            "typhoon_name": row.get("typhoon_name", ""),
            "time": format_time(row.get("time")),
            "tif_path": relative_path(tif_path),
            "reason": "target_name_or_2024_target_time_window",
            "tif_path_exists": bool(exists),
        })
    base = base.loc[~target_mask].copy().reset_index(drop=True)
    if max_rows is not None:
        base = base.head(max_rows).copy()

    records: List[Dict[str, object]] = []
    stats.rows_seen = int(len(base))

    iterator = tqdm(base.iterrows(), total=len(base), desc="Build Problem-2 half-hour library")
    for _, row in iterator:
        sample_context = {
            "event_uid": row.get("event_uid", ""),
            "gpm_event_dir": row.get("gpm_event_dir", ""),
            "typhoon_name": row.get("typhoon_name", ""),
            "time": format_time(row.get("time")),
            "source_file": row.get("source_file", ""),
        }
        try:
            tif_path, tif_exists = resolve_tif_path(row, gpm_root)
            if not tif_exists:
                rec = {**sample_context, "tif_path": relative_path(tif_path), "error": "missing_tif"}
                stats.missing_tif_records.append(rec)
                stats.failed_records.append(rec)
                continue

            try:
                parsed = parse_gpm_filename(tif_path)
            except Exception as exc:
                rec = {**sample_context, "tif_path": relative_path(tif_path), "error": str(exc)}
                stats.filename_parse_errors.append(rec)
                stats.failed_records.append(rec)
                continue

            rain, lon_grid, lat_grid, tif_meta = read_gpm_tif(tif_path)
            stats.negative_cell_count += int(tif_meta["negative_cell_count"])
            stats.abnormal_cell_count += int(tif_meta["abnormal_cell_count"])

            gpm_center_lon = float(parsed["gpm_center_lon"])
            gpm_center_lat = float(parsed["gpm_center_lat"])
            center_lon = to_float(row.get("lon_180"))
            center_lat = to_float(row.get("lat"))
            move_dir = to_float(row.get("move_dir_deg"))
            center_match_distance = (
                float(haversine_km(gpm_center_lon, gpm_center_lat, center_lon, center_lat))
                if np.isfinite(center_lon) and np.isfinite(center_lat)
                else np.nan
            )
            center_match_ok = bool(np.isfinite(center_match_distance) and center_match_distance < CENTER_MATCH_OK_KM)
            if np.isfinite(center_match_distance) and not center_match_ok:
                stats.center_mismatch_records.append({
                    **sample_context,
                    "tif_path": relative_path(tif_path),
                    "center_match_distance_km": center_match_distance,
                })

            rain_metrics = compute_rainfall_metrics(
                rain,
                lon_grid,
                lat_grid,
                float(tif_meta["pixel_size_deg"]),
            )
            if not np.isfinite(rain_metrics.get("rain_valid_ratio", np.nan)) or rain_metrics["rain_valid_ratio"] <= 0:
                stats.all_missing_rain_records.append({
                    **sample_context,
                    "tif_path": relative_path(tif_path),
                    "error": "all_missing_rain",
                })

            spatial_metrics = compute_spatial_structure_metrics(
                rain,
                lon_grid,
                lat_grid,
                center_lon,
                center_lat,
                move_dir,
            )
            if not bool(spatial_metrics.get("rel_grid_available", False)):
                stats.missing_motion_for_structure += 1

            record = {
                "sample_id": f"{row.get('event_uid', '')}_{pd.Timestamp(row['time']).strftime('%Y%m%d%H%M%S')}",
                "event_uid": row.get("event_uid", ""),
                "gpm_event_dir": row.get("gpm_event_dir", ""),
                "typhoon_name": str(row.get("typhoon_name", "UNKNOWN")).upper(),
                "time": format_time(row.get("time")),
                "tif_path": relative_path(tif_path),
                "source_file": row.get("source_file", ""),
                "gpm_source_file": row.get("source_file", ""),
                "track_source_file": row.get("track_source_file", ""),
                **parsed,
                "grid_nrow": tif_meta["grid_nrow"],
                "grid_ncol": tif_meta["grid_ncol"],
                "pixel_size_deg": tif_meta["pixel_size_deg"],
                "lat": center_lat,
                "lon": to_float(row.get("lon")),
                "lon_180": center_lon,
                "WND": to_float(row.get("WND")),
                "PRES": to_float(row.get("PRES")),
                "intensity": row.get("intensity", np.nan),
                "move_speed_kmh": to_float(row.get("move_speed_kmh")),
                "move_dir_deg": move_dir,
                "move_distance_km": to_float(row.get("move_distance_km")),
                "wind_change_rate": to_float(row.get("wind_change_rate")),
                "pressure_change_rate": to_float(row.get("pressure_change_rate")),
                "dt_h": to_float(row.get("dt_h")),
                "is_land": row.get("is_land", np.nan),
                "coast_dist_km": to_float(row.get("coast_dist_km")),
                "signed_coast_dist_km": to_float(row.get("signed_coast_dist_km")),
                "landfrac_100km": to_float(row.get("landfrac_100km")),
                "landfrac_200km": to_float(row.get("landfrac_200km")),
                "landfrac_300km": to_float(row.get("landfrac_300km")),
                "landfrac_500km": to_float(row.get("landfrac_500km")),
                "elev_mean_200km": to_float(row.get("elev_mean_200km")),
                "elev_max_200km": to_float(row.get("elev_max_200km")),
                "terrain_std_200km": to_float(row.get("terrain_std_200km")),
                "terrain_mean_300km": to_float(row.get("terrain_mean_300km")),
                "terrain_max_300km": to_float(row.get("terrain_max_300km")),
                "terrain_std_300km": to_float(row.get("terrain_std_300km")),
                "year": int(row["year"]) if pd.notna(row.get("year")) else np.nan,
                "month": int(row["month"]) if pd.notna(row.get("month")) else np.nan,
                "day": int(row["day"]) if pd.notna(row.get("day")) else np.nan,
                "hour": to_float(row.get("hour")),
                "season": row.get("season", ""),
                "month_sin": to_float(row.get("month_sin")),
                "month_cos": to_float(row.get("month_cos")),
                "hour_sin": to_float(row.get("hour_sin")),
                "hour_cos": to_float(row.get("hour_cos")),
                "life_time_index": int(row["life_time_index"]) if pd.notna(row.get("life_time_index")) else np.nan,
                "life_progress": to_float(row.get("life_progress")),
                **rain_metrics,
                **spatial_metrics,
                "center_match_distance_km": center_match_distance,
                "center_match_ok": center_match_ok,
                "target_excluded_flag": False,
            }
            records.append(record)
        except Exception as exc:  # pragma: no cover - batch robustness.
            stats.failed_records.append({
                **sample_context,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=3),
            })

    library = pd.DataFrame(records)
    stats.rows_written = int(len(library))
    if len(library) > 0:
        library = library.sort_values(["event_uid", "time", "source_file"]).reset_index(drop=True)
        library.to_csv(output_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig")

    write_qc_report(library, stats, output_csv, qc_report)
    return library, stats


def describe_series(df: pd.DataFrame, col: str) -> Dict[str, object]:
    if col not in df.columns:
        return {"available": False}
    s = pd.to_numeric(df[col], errors="coerce")
    if int(s.notna().sum()) == 0:
        return {"available": True, "count": 0}
    return {
        "available": True,
        "count": int(s.notna().sum()),
        "missing_rate": float(s.isna().mean()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def markdown_stat_table(stats: Dict[str, object]) -> str:
    if not stats.get("available", False):
        return "| 指标 | 值 |\n|---|---:|\n| available | False |"
    lines = ["| 指标 | 值 |", "|---|---:|"]
    for key in ["count", "missing_rate", "mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"]:
        if key in stats:
            value = stats[key]
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            lines.append(f"| {key} | {text} |")
    return "\n".join(lines)


def fmt_pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2%}"


def write_qc_report(library: pd.DataFrame, stats: BuildStats, output_csv: Path, qc_report: Path) -> None:
    time_min = library["time"].min() if "time" in library.columns and len(library) else ""
    time_max = library["time"].max() if "time" in library.columns and len(library) else ""
    tif_exists_rate = float(library["tif_path"].map(lambda p: (PROJECT_ROOT / str(p)).exists()).mean()) if len(library) else np.nan
    final_target_all_false = (
        bool((library["target_excluded_flag"] == False).all())  # noqa: E712
        if "target_excluded_flag" in library.columns and len(library)
        else True
    )

    missing = {}
    for col in ["WND", "PRES", "move_speed_kmh", "move_dir_deg"]:
        missing[col] = float(pd.to_numeric(library[col], errors="coerce").isna().mean()) if col in library.columns and len(library) else np.nan

    center_stats = describe_series(library, "center_match_distance_km")
    center_mean = center_stats.get("mean", np.nan)
    center_p95 = center_stats.get("p95", np.nan)
    center_max = center_stats.get("max", np.nan)

    lines: List[str] = [
        "# 问题二历史半小时样本库 QC 报告",
        "",
        "## 数据泄漏说明",
        "",
        "本表中的 rain_*、rain_centroid_*、centroid_*、asym_*、quad_*、rain_radius_*、rain_band_width_km、anisotropy 等字段由历史 GPM 降水场计算得到，只能作为历史样本标签、模板校准目标、伪缺失验证评价指标或论文结果分析指标。它们不得作为 2024 年 KONG-REY 和 MAN-YI 的模型输入特征。",
        "",
        "## 总体统计",
        "",
        f"- tif 文件总数：{stats.tif_total}",
        f"- 待处理非目标半小时样本数：{stats.rows_seen}",
        f"- 成功入库样本数：{stats.rows_written}",
        f"- 失败样本数：{len(stats.failed_records)}",
        f"- 目标台风排除样本数：{len(stats.excluded_target_records)}",
        f"- 历史台风事件数：{library['event_uid'].nunique() if 'event_uid' in library.columns and len(library) else 0}",
        f"- 时间范围：{time_min} 至 {time_max}",
        f"- 字段数量：{library.shape[1] if len(library) else 0}",
        f"- 输入路径骨架表：{stats.input_model_file or 'fallback interpolation'}",
        "",
        "## 数据完整性",
        "",
        f"- tif_path 存在率：{fmt_pct(tif_exists_rate)}",
        f"- WND 缺失率：{fmt_pct(missing['WND'])}",
        f"- PRES 缺失率：{fmt_pct(missing['PRES'])}",
        f"- move_speed_kmh 缺失率：{fmt_pct(missing['move_speed_kmh'])}",
        f"- move_dir_deg 缺失率：{fmt_pct(missing['move_dir_deg'])}",
        f"- center_match_distance_km 均值 / P95 / 最大值：{center_mean:.3f} / {center_p95:.3f} / {center_max:.3f} km",
        f"- 方向性结构指标因中心或移动方向缺失而不可构造数量：{stats.missing_motion_for_structure}",
        f"- 负降水格点处理：{stats.negative_cell_count} 个负值格点被置为 0。",
        f"- 异常降水格点处理：{stats.abnormal_cell_count} 个 > {ABNORMAL_RAIN_MMHR:g} mm/hr 的格点被置为 NaN。",
        "",
        "### rain_valid_ratio 分布",
        "",
        markdown_stat_table(describe_series(library, "rain_valid_ratio")),
        "",
        "### center_match_distance_km 分布",
        "",
        markdown_stat_table(center_stats),
        "",
        "## 环境字段生成情况",
        "",
        "- 已复用路径表中的 is_land、coast_dist_km、signed_coast_dist_km。",
        f"- 未在现有中间表中找到、因此保留为空值的环境字段：{', '.join(stats.env_missing_cols) if stats.env_missing_cols else '无'}。",
        "- 未联网补充 landfrac/elev/terrain 外部数据。",
        "",
        "## 目标台风排除检查",
        "",
        f"- 是否发现 KONG-REY / MAN-YI 或其时间窗样本：{'是' if stats.excluded_target_records else '否'}",
        f"- 发现并排除数量：{len(stats.excluded_target_records)}",
        f"- 最终历史库中 target_excluded_flag 是否全为 False：{final_target_all_false}",
        "",
    ]

    if stats.excluded_target_records:
        lines.extend([
            "### 被排除路径",
            "",
            "| event_uid | gpm_event_dir | typhoon_name | time | tif_path |",
            "|---|---|---|---|---|",
        ])
        for rec in stats.excluded_target_records:
            lines.append(
                f"| {rec.get('event_uid', '')} | {rec.get('gpm_event_dir', '')} | "
                f"{rec.get('typhoon_name', '')} | {rec.get('time', '')} | {rec.get('tif_path', '')} |"
            )
        lines.append("")

    lines.extend([
        "## 降水指标合理性",
        "",
        "### rain_max_mmhr 分布",
        "",
        markdown_stat_table(describe_series(library, "rain_max_mmhr")),
        "",
        "### rain_p95_mmhr 分布",
        "",
        markdown_stat_table(describe_series(library, "rain_p95_mmhr")),
        "",
        "### rain_area_10_km2 分布",
        "",
        markdown_stat_table(describe_series(library, "rain_area_10_km2")),
        "",
        "### centroid_offset_km 分布",
        "",
        markdown_stat_table(describe_series(library, "centroid_offset_km")),
        "",
        "### anisotropy 分布",
        "",
        markdown_stat_table(describe_series(library, "anisotropy")),
        "",
        "## 空间结构口径",
        "",
        "- 台风相对坐标以 CMABST 插值中心为原点，move_dir_deg 为前进方向。",
        "- x_front > 0 表示移动前方，y_left > 0 表示移动方向左侧。",
        "- 面积使用每一行格点中心纬度加权：111.32 * 0.1 * 111.32 * 0.1 * cos(lat)。",
        "- rain_band_width_km 定义为 rain_radius_r90_km - rain_radius_r50_km。",
        "- 降水率强度指标保留 mm/hr 口径；rain_sum_halfhour_mm 和四象限 sum 使用 rain_mmhr * 0.5 的半小时累计口径。",
        "",
        "## 异常与失败记录",
        "",
        f"- tif 读取/解析/路径缺失等失败记录数：{len(stats.failed_records)}",
        f"- 文件名解析失败数：{len(stats.filename_parse_errors)}",
        f"- tif_path 缺失数：{len(stats.missing_tif_records)}",
        f"- 降水全缺测样本数：{len(stats.all_missing_rain_records)}",
        f"- center_match_distance_km >= {CENTER_MATCH_OK_KM:g} km 样本数：{len(stats.center_mismatch_records)}",
        "",
        "## 输出文件",
        "",
        f"- CSV：{relative_path(output_csv)}",
        f"- QC 报告：{relative_path(qc_report)}",
        f"- 行列数：{library.shape[0]} × {library.shape[1]}",
        "",
    ])

    if stats.failed_records:
        lines.extend([
            "### 失败样例（前 50 条）",
            "",
            "| event_uid | gpm_event_dir | typhoon_name | time | source_file | error |",
            "|---|---|---|---|---|---|",
        ])
        for rec in stats.failed_records[:50]:
            lines.append(
                f"| {rec.get('event_uid', '')} | {rec.get('gpm_event_dir', '')} | "
                f"{rec.get('typhoon_name', '')} | {rec.get('time', '')} | "
                f"{rec.get('source_file', '')} | {str(rec.get('error', '')).replace('|', '/')} |"
            )
        lines.append("")

    qc_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Problem-2 historical half-hour sample library.")
    parser.add_argument("--input-table", type=str, default=None,
                        help="Optional precomputed GPM-track model table. Defaults to env_added table if present.")
    parser.add_argument("--gpm-root", type=str, default=str(DEFAULT_GPM_ROOT))
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--qc-report", type=str, default=str(DEFAULT_QC_REPORT))
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: process first N non-target rows.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    library, stats = build_historical_library(
        gpm_root=Path(args.gpm_root),
        output_csv=Path(args.output_csv),
        qc_report=Path(args.qc_report),
        max_rows=args.max_rows,
        input_table=Path(args.input_table) if args.input_table else None,
    )
    print("\n========== Problem-2 historical half-hour library complete ==========")
    print(f"Script: {relative_path(Path(__file__))}")
    print(f"CSV: {relative_path(Path(args.output_csv))}")
    print(f"QC report: {relative_path(Path(args.qc_report))}")
    print(f"Rows x columns: {library.shape[0]} x {library.shape[1]}")
    if len(library):
        tif_exists_rate = library["tif_path"].map(lambda p: (PROJECT_ROOT / str(p)).exists()).mean()
        print(f"tif_path exists rate: {tif_exists_rate:.6f}")
        for col in ["WND", "PRES"]:
            print(f"{col} missing rate: {pd.to_numeric(library[col], errors='coerce').isna().mean():.6f}")
        print(f"center_match_distance_km P95: {pd.to_numeric(library['center_match_distance_km'], errors='coerce').quantile(0.95):.6f}")
        print("first 10 columns:", library.columns[:10].tolist())
        print("last 10 columns:", library.columns[-10:].tolist())
        sample_cols = [c for c in KEY_DISPLAY_COLS if c in library.columns]
        print("\nRandom 5 key rows:")
        print(library[sample_cols].sample(n=min(5, len(library)), random_state=42).to_string(index=False))
    print(f"Excluded target candidates: {len(stats.excluded_target_records)}")
    print(f"Failed samples: {len(stats.failed_records)}")


if __name__ == "__main__":
    main()
