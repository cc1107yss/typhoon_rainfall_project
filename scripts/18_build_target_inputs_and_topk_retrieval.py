#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Problem-2 target-safe half-hour inputs and Top-K historical retrieval.

The target table is leakage safe: it contains only target-track, intensity,
motion, time, and available land/coast/environment predictors. The Top-K
distance also uses only those safe predictors. Historical GPM-derived rain and
structure metrics are copied to the retrieval output only as labels/templates
for later calibration and pseudo-missing validation.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"
TARGET_HALFHOUR_SAFE_CANDIDATES = [
    PROJECT_ROOT / "data/processed/env_added/target_typhoon_inputs_2024_halfhour_leakage_safe_env.csv",
    PROJECT_ROOT / "data/processed/target_typhoon_inputs_2024_halfhour_leakage_safe.csv",
    PROJECT_ROOT / "data/processed/problem2_target_model_x_aligned.csv",
]
TARGET_TRACK_POINT_CANDIDATES = [
    PROJECT_ROOT / "data/processed/target_typhoon_inputs_2024_track_points_leakage_safe.csv",
    PROJECT_ROOT / "output/target_typhoon_tracks_2024_with_coast.csv",
    PROJECT_ROOT / "output/target_typhoon_tracks_2024.csv",
]

TARGET_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_halfhour_inputs_safe.csv"
TOPK_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_target_topk_similar_history.csv"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_topk_retrieval_qc_report.md"

TOPK = 20
DIVERSIFY_BY_EVENT = True
MAX_PER_HISTORY_EVENT = 3
TOPK_CANDIDATE_POOL = 1000

COMPONENT_WEIGHTS = {
    "track": 1.0,
    "intensity": 1.5,
    "motion": 1.2,
    "environment": 1.2,
    "time": 0.8,
    "life_progress": 0.8,
}

BASE_ENV_FEATURES = [
    "is_land",
    "signed_coast_dist_km",
    "coast_dist_km",
]

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

ENV_FEATURE_SET = os.environ.get("P2_ENV_FEATURE_SET", "env-full").strip().lower() or "env-full"

SAFE_FEATURE_COMPONENTS = {
    "track": ["lat", "lon_180"],
    "intensity": ["WND", "PRES", "intensity"],
    "motion": [
        "move_speed_kmh",
        "move_dir_sin",
        "move_dir_cos",
        "wind_change_rate",
        "pressure_change_rate",
    ],
    "environment": list(ENV_FEATURE_SETS.get(ENV_FEATURE_SET, ENV_FEATURE_SETS["env-full"])),
    "time": ["month_sin", "month_cos"],
    "life_progress": ["life_progress"],
}

ENVIRONMENT_FIELDS = list(ENV_FEATURE_SETS.get(ENV_FEATURE_SET, ENV_FEATURE_SETS["env-full"]))

TARGET_WINDOWS = {
    "KONGREY": {
        "display": "KONG-REY",
        "start": pd.Timestamp("2024-10-24 00:00:00"),
        "end_exclusive": pd.Timestamp("2024-11-03 00:00:00"),
    },
    "MANYI": {
        "display": "MAN-YI",
        "start": pd.Timestamp("2024-11-08 00:00:00"),
        "end_exclusive": pd.Timestamp("2024-11-21 00:00:00"),
    },
}
TARGET_NAME_SET = set(TARGET_WINDOWS)

LEAKAGE_PATTERNS = [
    re.compile(r"^rain_", re.IGNORECASE),
    re.compile(r"^rainband_", re.IGNORECASE),
    re.compile(r"^rainband10_", re.IGNORECASE),
    re.compile(r"^rain_centroid_", re.IGNORECASE),
    re.compile(r"^centroid_", re.IGNORECASE),
    re.compile(r"^quad_", re.IGNORECASE),
    re.compile(r"^asym_", re.IGNORECASE),
    re.compile(r"^anisotropy$", re.IGNORECASE),
    re.compile(r"^rain_radius_", re.IGNORECASE),
    re.compile(r"^rain_band_width_km$", re.IGNORECASE),
    re.compile(r"^gpm_center_", re.IGNORECASE),
    re.compile(r"^center_match_distance_km$", re.IGNORECASE),
    re.compile(r"^tif_path$", re.IGNORECASE),
]

HISTORICAL_RAIN_OUTPUT_FIELDS = [
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
    "rainband_width_km",
    "rainband_length_km",
    "rainband_aspect_ratio",
    "rainband_width10_km",
    "rainband_length10_km",
    "rainband_aspect_ratio10",
]

KM_PER_DEG = 111.32
EPS = 1e-12


def configure_environment_features(feature_set: object = "env-full") -> List[str]:
    """Configure the D_env features used by retrieval.

    D_env is the mean of squared standardized differences across the selected
    environment features. Inputs are normalized to unprefixed names upstream;
    target tables may still provide track_* aliases, which normalize_target_columns
    maps into these canonical names.
    """
    global ENV_FEATURE_SET, ENVIRONMENT_FIELDS
    if isinstance(feature_set, str):
        key = feature_set.strip().lower()
        if key not in ENV_FEATURE_SETS:
            raise ValueError(f"Unknown environment feature set: {feature_set}")
        features = list(ENV_FEATURE_SETS[key])
        ENV_FEATURE_SET = key
    else:
        features = [str(v) for v in feature_set]
        ENV_FEATURE_SET = "custom"
    SAFE_FEATURE_COMPONENTS["environment"] = features
    ENVIRONMENT_FIELDS = list(features)
    return features


configure_environment_features(ENV_FEATURE_SET)


# =========================
# Utilities
# =========================


def normalize_typhoon_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def display_target_name(value: object) -> str:
    name = normalize_typhoon_name(value)
    if name in TARGET_WINDOWS:
        return str(TARGET_WINDOWS[name]["display"])
    return str(value).upper()


def wrap_lon_180(lon: object) -> float:
    value = pd.to_numeric(pd.Series([lon]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def lon_180_to_lon_360(lon_180: object) -> float:
    value = pd.to_numeric(pd.Series([lon_180]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    value = float(value)
    return value if value >= 0.0 else value + 360.0


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return np.nan
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return float(2.0 * r * math.asin(min(1.0, math.sqrt(a))))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return np.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def read_csv_with_time(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in ["time", "target_time", "history_time", "time_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def find_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def is_leakage_feature(feature: str) -> bool:
    return any(pattern.search(feature) for pattern in LEAKAGE_PATTERNS)


def target_time_window_mask(df: pd.DataFrame, name_col: str = "typhoon_name") -> pd.Series:
    if "time" not in df.columns:
        return pd.Series(False, index=df.index)
    times = pd.to_datetime(df["time"], errors="coerce")
    names = df[name_col].map(normalize_typhoon_name) if name_col in df.columns else pd.Series("", index=df.index)
    mask = pd.Series(False, index=df.index)
    for norm_name, meta in TARGET_WINDOWS.items():
        in_window = (times >= meta["start"]) & (times < meta["end_exclusive"])
        mask = mask | (names.eq(norm_name) & in_window)
    any_target_window = (
        ((times >= TARGET_WINDOWS["KONGREY"]["start"]) & (times < TARGET_WINDOWS["KONGREY"]["end_exclusive"]))
        | ((times >= TARGET_WINDOWS["MANYI"]["start"]) & (times < TARGET_WINDOWS["MANYI"]["end_exclusive"]))
    )
    return mask | any_target_window


def format_time(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def value_for_csv(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return format_time(value)
    if pd.isna(value):
        return np.nan
    return value


# =========================
# Target input construction
# =========================


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    out["year"] = time.dt.year
    out["month"] = time.dt.month
    out["day"] = time.dt.day
    out["hour"] = time.dt.hour + time.dt.minute / 60.0 + time.dt.second / 3600.0
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    out["season"] = out["month"].map(season_map)
    month_rad = 2.0 * np.pi * (out["month"].astype(float) - 1.0) / 12.0
    hour_rad = 2.0 * np.pi * out["hour"].astype(float) / 24.0
    out["month_sin"] = np.sin(month_rad)
    out["month_cos"] = np.cos(month_rad)
    out["hour_sin"] = np.sin(hour_rad)
    out["hour_cos"] = np.cos(hour_rad)
    return out


def add_direction_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    angle = pd.to_numeric(out.get("move_dir_deg", np.nan), errors="coerce")
    rad = np.deg2rad(angle)
    out["move_dir_sin"] = np.sin(rad)
    out["move_dir_cos"] = np.cos(rad)
    return out


def compute_motion_features(df: pd.DataFrame, group_col: str = "typhoon_name") -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for col in [
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "dt_h",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    pieces = []
    for _, one in out.sort_values([group_col, "time"]).groupby(group_col, dropna=False):
        one = one.copy()
        lon = pd.to_numeric(one["lon_180"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(one["lat"], errors="coerce").to_numpy(dtype=float)
        wnd = pd.to_numeric(one["WND"], errors="coerce").to_numpy(dtype=float)
        pres = pd.to_numeric(one["PRES"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_datetime(one["time"], errors="coerce")
        dt_h = t.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
        dist = np.full(len(one), np.nan, dtype=float)
        speed = np.full(len(one), np.nan, dtype=float)
        direction = np.full(len(one), np.nan, dtype=float)
        wind_rate = np.full(len(one), np.nan, dtype=float)
        pres_rate = np.full(len(one), np.nan, dtype=float)
        for i in range(1, len(one)):
            if np.isfinite(dt_h[i]) and dt_h[i] > 0:
                dist[i] = haversine_km(lon[i - 1], lat[i - 1], lon[i], lat[i])
                speed[i] = dist[i] / dt_h[i] if np.isfinite(dist[i]) else np.nan
                direction[i] = bearing_deg(lon[i - 1], lat[i - 1], lon[i], lat[i])
                wind_rate[i] = (wnd[i] - wnd[i - 1]) / dt_h[i] if np.isfinite(wnd[i]) and np.isfinite(wnd[i - 1]) else np.nan
                pres_rate[i] = (pres[i] - pres[i - 1]) / dt_h[i] if np.isfinite(pres[i]) and np.isfinite(pres[i - 1]) else np.nan

        if len(one) > 1:
            dt_h[0] = dt_h[1] if np.isfinite(dt_h[1]) else 0.5
            dist[0] = dist[1]
            speed[0] = speed[1]
            direction[0] = direction[1]
            wind_rate[0] = wind_rate[1]
            pres_rate[0] = pres_rate[1]

        fill_map = {
            "dt_h": dt_h,
            "move_distance_km": dist,
            "move_speed_kmh": speed,
            "move_dir_deg": direction,
            "wind_change_rate": wind_rate,
            "pressure_change_rate": pres_rate,
        }
        for col, values in fill_map.items():
            one[col] = one[col].where(one[col].notna(), values)
        pieces.append(one)
    return pd.concat(pieces, ignore_index=True) if pieces else out


def interpolate_target_tracks_to_halfhour(track_points: pd.DataFrame) -> pd.DataFrame:
    """Fallback half-hour interpolation from target track points."""
    src = normalize_target_columns(track_points, source_is_halfhour=False)
    pieces = []
    for _, one in src.sort_values(["typhoon_name", "time"]).groupby("typhoon_name", dropna=False):
        one = one.dropna(subset=["time"]).copy()
        if one.empty:
            continue
        full_times = pd.date_range(one["time"].min(), one["time"].max(), freq="30min")
        base = pd.DataFrame({"time": full_times})
        base["typhoon_name"] = one["typhoon_name"].iloc[0]
        for col in ["event_uid", "source_file"]:
            if col in one.columns:
                base[col] = one[col].dropna().iloc[0] if one[col].notna().any() else np.nan
        t_src = one["time"].astype("int64").to_numpy(dtype=float) / 1e9
        t_new = base["time"].astype("int64").to_numpy(dtype=float) / 1e9
        for col in ["lat", "WND", "PRES", "coast_dist_km", "signed_coast_dist_km"]:
            if col in one.columns:
                y = pd.to_numeric(one[col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(y)
                base[col] = np.interp(t_new, t_src[valid], y[valid]) if int(valid.sum()) >= 2 else np.nan
        lon = pd.to_numeric(one["lon_180"], errors="coerce").to_numpy(dtype=float)
        valid_lon = np.isfinite(lon)
        if int(valid_lon.sum()) >= 2:
            lon_unwrap = np.unwrap(np.deg2rad(lon[valid_lon]))
            lon_interp = np.rad2deg(np.interp(t_new, t_src[valid_lon], lon_unwrap))
            base["lon_180"] = ((lon_interp + 180.0) % 360.0) - 180.0
        else:
            base["lon_180"] = np.nan
        if "intensity" in one.columns:
            base = pd.merge_asof(
                base.sort_values("time"),
                one[["time", "intensity"]].sort_values("time"),
                on="time",
                direction="nearest",
            )
        for col in ENVIRONMENT_FIELDS:
            if col not in base.columns:
                base[col] = np.nan
        if "is_land" in one.columns:
            env = pd.merge_asof(
                base[["time"]].sort_values("time"),
                one[["time", "is_land"]].sort_values("time"),
                on="time",
                direction="nearest",
            )
            base["is_land"] = env["is_land"].to_numpy()
        pieces.append(base)
    if not pieces:
        raise RuntimeError("No target track points available for interpolation.")
    return compute_motion_features(pd.concat(pieces, ignore_index=True))


def normalize_target_columns(df: pd.DataFrame, source_is_halfhour: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")

    rename_map = {
        "target_display_name": "typhoon_name",
        "track_lat": "lat",
        "track_lon_180": "lon_180",
        "track_wind": "WND",
        "track_pressure": "PRES",
        "track_intensity": "intensity",
        "track_move_distance_km": "move_distance_km",
        "track_move_speed_kmh": "move_speed_kmh",
        "track_move_dir_deg": "move_dir_deg",
        "track_wind_change_rate": "wind_change_rate",
        "track_pressure_change_rate": "pressure_change_rate",
        "track_dt_h": "dt_h",
        "track_is_land": "is_land",
        "track_coast_dist_km": "coast_dist_km",
        "track_signed_coast_dist_km": "signed_coast_dist_km",
        "pressure": "PRES",
        "wind": "WND",
    }
    existing_map = {k: v for k, v in rename_map.items() if k in out.columns and v not in out.columns}
    out = out.rename(columns=existing_map)
    if "typhoon_name" not in out.columns and "target_name_norm" in out.columns:
        out["typhoon_name"] = out["target_name_norm"].map(display_target_name)

    for col in ["lat", "lon_180", "WND", "PRES", "intensity"]:
        if col not in out.columns:
            out[col] = np.nan
    for col in ["move_distance_km", "move_speed_kmh", "move_dir_deg", "wind_change_rate", "pressure_change_rate", "dt_h"]:
        if col not in out.columns:
            out[col] = np.nan
    for col in ENVIRONMENT_FIELDS:
        if col not in out.columns:
            out[col] = np.nan

    out["target_name_norm"] = out["typhoon_name"].map(normalize_typhoon_name)
    out["typhoon_name"] = out["target_name_norm"].map(display_target_name)
    out["lon_180"] = out["lon_180"].map(wrap_lon_180)
    if "lon" not in out.columns:
        out["lon"] = out["lon_180"].map(lon_180_to_lon_360)
    else:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce").fillna(out["lon_180"].map(lon_180_to_lon_360))

    keep = out["target_name_norm"].isin(TARGET_NAME_SET)
    out = out.loc[keep].copy()
    out = out.loc[target_time_window_mask(out, "typhoon_name")].copy()

    numeric_cols = [
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
    ] + ENVIRONMENT_FIELDS
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if source_is_halfhour:
        out = compute_motion_features(out)
    out = add_time_features(out)
    out = add_direction_sin_cos(out)

    out = out.sort_values(["typhoon_name", "time"]).reset_index(drop=True)
    out["life_time_index"] = out.groupby("typhoon_name").cumcount()
    event_sizes = out.groupby("typhoon_name")["time"].transform("size")
    out["life_progress"] = np.where(event_sizes > 1, out["life_time_index"] / (event_sizes - 1), 0.0)
    out["target_time_window_flag"] = target_time_window_mask(out, "typhoon_name").to_numpy()
    out["safe_input_flag"] = True
    out["is_target"] = True
    out["target_id"] = out["typhoon_name"].astype(str) + "_" + out["time"].dt.strftime("%Y%m%d%H%M%S")
    return out


def load_or_build_target_inputs() -> Tuple[pd.DataFrame, Dict[str, object]]:
    halfhour_path = find_existing_path(TARGET_HALFHOUR_SAFE_CANDIDATES)
    report: Dict[str, object] = {
        "source_type": None,
        "source_path": None,
        "missing_environment_fields": [],
        "leakage_columns_in_source": [],
    }
    if halfhour_path is not None:
        raw = read_csv_with_time(halfhour_path)
        report["source_type"] = "existing_halfhour_safe_input"
        report["source_path"] = str(halfhour_path.relative_to(PROJECT_ROOT))
        report["source_shape"] = [int(raw.shape[0]), int(raw.shape[1])]
        report["leakage_columns_in_source"] = [c for c in raw.columns if is_leakage_feature(c)]
        target = normalize_target_columns(raw, source_is_halfhour=True)
    else:
        track_path = find_existing_path(TARGET_TRACK_POINT_CANDIDATES)
        if track_path is None:
            raise FileNotFoundError("No target half-hour safe table or track-point target table found.")
        raw = read_csv_with_time(track_path)
        report["source_type"] = "interpolated_from_track_points"
        report["source_path"] = str(track_path.relative_to(PROJECT_ROOT))
        report["source_shape"] = [int(raw.shape[0]), int(raw.shape[1])]
        report["leakage_columns_in_source"] = [c for c in raw.columns if is_leakage_feature(c)]
        target = interpolate_target_tracks_to_halfhour(raw)
        target = normalize_target_columns(target, source_is_halfhour=True)

    report["missing_environment_fields"] = [
        c for c in ENVIRONMENT_FIELDS if c not in target.columns or pd.to_numeric(target[c], errors="coerce").isna().all()
    ]

    base_cols = [
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
    ]
    time_cols = [
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
    ]
    helper_cols = ["move_dir_sin", "move_dir_cos", "target_time_window_flag", "safe_input_flag"]
    optional_meta = [
        c
        for c in ["typhoon_id", "storm_seq", "typhoon_code", "record_count", "cadence_hours"]
        if c in target.columns
    ]
    for col in base_cols + ENVIRONMENT_FIELDS + time_cols + helper_cols:
        if col not in target.columns:
            target[col] = np.nan
    output_cols = base_cols + optional_meta + ENVIRONMENT_FIELDS + time_cols + helper_cols
    target = target[output_cols].copy()
    target["time"] = pd.to_datetime(target["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    target.to_csv(TARGET_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return read_csv_with_time(TARGET_OUTPUT_PATH), report


# =========================
# Historical library
# =========================


def load_historical_library() -> Tuple[pd.DataFrame, Dict[str, object]]:
    history = read_csv_with_time(HISTORICAL_LIBRARY_PATH)
    original_rows = len(history)
    if "typhoon_name" not in history.columns:
        raise RuntimeError("Historical library must contain typhoon_name.")
    history["name_norm"] = history["typhoon_name"].map(normalize_typhoon_name)
    name_mask = history["name_norm"].isin(TARGET_NAME_SET)
    window_mask = target_time_window_mask(history, "typhoon_name")
    filtered = history.loc[~(name_mask | window_mask)].copy()
    if "move_dir_sin" not in filtered.columns or "move_dir_cos" not in filtered.columns:
        filtered = add_direction_sin_cos(filtered)
    report = {
        "original_rows": int(original_rows),
        "target_name_rows_found": int(name_mask.sum()),
        "target_window_rows_found": int(window_mask.sum()),
        "filtered_rows": int(len(filtered)),
        "removed_rows": int(original_rows - len(filtered)),
    }
    return filtered.reset_index(drop=True), report


# =========================
# Retrieval feature handling
# =========================


def select_safe_retrieval_features(history: pd.DataFrame, target: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    selected: Dict[str, List[str]] = {}
    skipped: Dict[str, str] = {}
    for component, features in SAFE_FEATURE_COMPONENTS.items():
        selected[component] = []
        for feature in features:
            if is_leakage_feature(feature):
                skipped[feature] = "blocked_by_leakage_pattern"
                continue
            if feature not in history.columns or feature not in target.columns:
                skipped[feature] = "missing_from_history_or_target"
                continue
            hist_missing = pd.to_numeric(history[feature], errors="coerce").isna().mean()
            targ_missing = pd.to_numeric(target[feature], errors="coerce").isna().mean()
            if hist_missing >= 1.0:
                skipped[feature] = "history_all_missing"
                continue
            if targ_missing >= 1.0:
                skipped[feature] = "target_all_missing"
                continue
            selected[component].append(feature)
    return selected, skipped


def interpolate_by_group_time(df: pd.DataFrame, feature: str, group_col: str) -> pd.Series:
    out = pd.to_numeric(df[feature], errors="coerce").copy()
    if "time" not in df.columns or group_col not in df.columns:
        return out
    tmp = df[[group_col, "time"]].copy()
    tmp[feature] = out
    pieces = []
    for _, one in tmp.sort_values([group_col, "time"]).groupby(group_col, dropna=False):
        s = pd.to_numeric(one[feature], errors="coerce")
        s = s.interpolate(method="linear", limit_direction="both")
        pieces.append(s)
    if not pieces:
        return out
    filled = pd.concat(pieces).sort_index()
    return filled.reindex(df.index)


def feature_fill_strategy(feature: str) -> str:
    if feature == "intensity":
        return "mode_then_history_median"
    if feature in {"move_speed_kmh", "move_dir_sin", "move_dir_cos", "wind_change_rate", "pressure_change_rate"}:
        return "same_event_time_interpolation_then_event_median_then_history_median"
    if feature == "is_land":
        return "mode_then_history_median"
    if any(token in feature for token in ["coast", "landfrac", "elev", "terrain"]):
        return "history_median_fill_for_available_environment_feature"
    return "history_median_fill"


def impute_one_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    history_reference: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
        if feature in {"move_speed_kmh", "move_dir_sin", "move_dir_cos", "wind_change_rate", "pressure_change_rate"}:
            out[feature] = interpolate_by_group_time(out, feature, group_col)
            med_by_group = out.groupby(group_col)[feature].transform("median") if group_col in out.columns else np.nan
            out[feature] = out[feature].fillna(med_by_group)
        if feature == "intensity":
            mode = out[feature].mode(dropna=True)
            if len(mode):
                out[feature] = out[feature].fillna(float(mode.iloc[0]))
        if feature == "is_land":
            mode = out[feature].mode(dropna=True)
            if len(mode):
                out[feature] = out[feature].fillna(float(mode.iloc[0]))
        hist_median = pd.to_numeric(history_reference[feature], errors="coerce").median(skipna=True)
        if not np.isfinite(hist_median):
            hist_median = 0.0
        out[feature] = out[feature].fillna(float(hist_median))
    return out


def impute_retrieval_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    selected_components: Mapping[str, Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [f for feats in selected_components.values() for f in feats]
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "component": next(comp for comp, feats in selected_components.items() if feature in feats),
                "history_missing_rate_before": float(pd.to_numeric(history[feature], errors="coerce").isna().mean()),
                "target_missing_rate_before": float(pd.to_numeric(target[feature], errors="coerce").isna().mean()),
                "strategy": feature_fill_strategy(feature),
            }
        )

    history_imp = impute_one_frame(history, features, history, "event_uid")
    target_imp = impute_one_frame(target, features, history_imp, "typhoon_name")

    for row in rows:
        feature = row["feature"]
        row["history_missing_rate_after"] = float(pd.to_numeric(history_imp[feature], errors="coerce").isna().mean())
        row["target_missing_rate_after"] = float(pd.to_numeric(target_imp[feature], errors="coerce").isna().mean())

    missing_after = {
        f: (
            float(pd.to_numeric(history_imp[f], errors="coerce").isna().mean()),
            float(pd.to_numeric(target_imp[f], errors="coerce").isna().mean()),
        )
        for f in features
    }
    bad = {f: rates for f, rates in missing_after.items() if rates[0] > 0 or rates[1] > 0}
    if bad:
        raise RuntimeError(f"Retrieval features still contain NaN after imputation: {bad}")

    return history_imp, target_imp, pd.DataFrame(rows)


def compute_standardization_params(history: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(history[feature], errors="coerce")
        mean = float(values.mean(skipna=True))
        std = float(values.std(skipna=True, ddof=0))
        if not np.isfinite(std) or std <= EPS:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        rows.append({"feature": feature, "mean": mean, "std": std})
    return pd.DataFrame(rows)


def standardize_values(df: pd.DataFrame, params: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    lookup = params.set_index("feature")
    cols = []
    for feature in features:
        mean = float(lookup.loc[feature, "mean"])
        std = float(lookup.loc[feature, "std"])
        vals = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        cols.append(((vals - mean) / std).astype(np.float32))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0), dtype=np.float32)


def component_feature_indices(selected_components: Mapping[str, Sequence[str]], features: Sequence[str]) -> Dict[str, List[int]]:
    index = {feature: i for i, feature in enumerate(features)}
    return {component: [index[f] for f in feats if f in index] for component, feats in selected_components.items()}


def compute_component_distances(
    target_z: np.ndarray,
    history_z: np.ndarray,
    component_indices: Mapping[str, Sequence[int]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    components: Dict[str, np.ndarray] = {}
    total = np.zeros(history_z.shape[0], dtype=np.float64)
    for component, indices in component_indices.items():
        if not indices:
            components[component] = np.zeros(history_z.shape[0], dtype=np.float64)
            continue
        diff = history_z[:, indices] - target_z[np.newaxis, indices]
        comp_dist = np.mean(diff * diff, axis=1, dtype=np.float64)
        components[component] = comp_dist
        total += COMPONENT_WEIGHTS[component] * comp_dist
    return total, components


def diversify_topk_by_event(
    sorted_indices: np.ndarray,
    distances: np.ndarray,
    event_ids: np.ndarray,
    k: int,
) -> List[int]:
    if not DIVERSIFY_BY_EVENT:
        return sorted_indices[:k].astype(int).tolist()

    selected: List[int] = []
    event_counts: Counter = Counter()
    selected_set = set()
    for idx in sorted_indices:
        event_id = str(event_ids[int(idx)])
        if event_counts[event_id] >= MAX_PER_HISTORY_EVENT:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        event_counts[event_id] += 1
        if len(selected) >= k:
            return selected

    if len(selected) < k:
        for idx in sorted_indices:
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            selected.append(idx_int)
            if len(selected) >= k:
                return selected
    return selected


def retrieve_topk_for_one_target(
    target_z: np.ndarray,
    history_z: np.ndarray,
    component_indices: Mapping[str, Sequence[int]],
    event_ids: np.ndarray,
    k: int,
) -> Tuple[List[int], np.ndarray, Dict[str, np.ndarray]]:
    distances, components = compute_component_distances(target_z, history_z, component_indices)
    pool_n = min(len(distances), max(TOPK_CANDIDATE_POOL, k * 10))
    candidate_idx = np.argpartition(distances, pool_n - 1)[:pool_n]
    candidate_idx = candidate_idx[np.argsort(distances[candidate_idx])]
    selected = diversify_topk_by_event(candidate_idx, distances, event_ids, k)
    if len(selected) < k:
        full_sorted = np.argsort(distances)
        selected = diversify_topk_by_event(full_sorted, distances, event_ids, k)
    selected_arr = np.array(selected, dtype=int)
    component_selected = {name: values[selected_arr] for name, values in components.items()}
    return selected, distances[selected_arr], component_selected


def compute_softmax_weights(distances: Sequence[float]) -> np.ndarray:
    dist = np.asarray(distances, dtype=float)
    if len(dist) == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(dist)):
        dist = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 0.0)
    if np.nanmax(dist) - np.nanmin(dist) <= EPS:
        return np.full(len(dist), 1.0 / len(dist), dtype=float)
    tau = float(np.nanmedian(dist))
    if not np.isfinite(tau) or tau <= EPS:
        tau = 1.0
    score = np.exp(-(dist - np.nanmin(dist)) / tau)
    denom = float(np.sum(score))
    return score / denom if denom > EPS else np.full(len(dist), 1.0 / len(dist), dtype=float)


# =========================
# Top-K table
# =========================


def build_topk_table(
    history: pd.DataFrame,
    target: pd.DataFrame,
    selected_components: Mapping[str, Sequence[str]],
    standardization_params: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    features = [f for feats in selected_components.values() for f in feats]
    history_z = standardize_values(history, standardization_params, features)
    target_z = standardize_values(target, standardization_params, features)
    indices_by_component = component_feature_indices(selected_components, features)
    event_ids = history["event_uid"].astype(str).to_numpy()

    records: List[Dict[str, object]] = []
    for i in range(len(target)):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[18] Retrieving target {i + 1}/{len(target)}")
        top_indices, distances, comp_values = retrieve_topk_for_one_target(
            target_z[i], history_z, indices_by_component, event_ids, TOPK
        )
        weights = compute_softmax_weights(distances)
        target_row = target.iloc[i]
        for rank_i, hist_idx in enumerate(top_indices, start=1):
            hist_row = history.iloc[int(hist_idx)]
            rec: Dict[str, object] = {
                "target_id": target_row["target_id"],
                "target_typhoon_name": target_row["typhoon_name"],
                "target_time": format_time(target_row["time"]),
                "target_lat": target_row["lat"],
                "target_lon_180": target_row["lon_180"],
                "target_WND": target_row["WND"],
                "target_PRES": target_row["PRES"],
                "target_intensity": target_row["intensity"],
                "target_move_speed_kmh": target_row["move_speed_kmh"],
                "target_move_dir_deg": target_row["move_dir_deg"],
                "target_signed_coast_dist_km": target_row["signed_coast_dist_km"],
                "target_is_land": target_row["is_land"],
                "target_month": target_row["month"],
                "target_life_progress": target_row["life_progress"],
                "history_sample_id": hist_row["sample_id"],
                "history_event_uid": hist_row["event_uid"],
                "history_typhoon_name": hist_row["typhoon_name"],
                "history_time": format_time(hist_row["time"]),
                "history_tif_path": hist_row["tif_path"],
                "history_lat": hist_row["lat"],
                "history_lon_180": hist_row["lon_180"],
                "history_WND": hist_row["WND"],
                "history_PRES": hist_row["PRES"],
                "history_intensity": hist_row["intensity"],
                "history_move_speed_kmh": hist_row["move_speed_kmh"],
                "history_move_dir_deg": hist_row["move_dir_deg"],
                "history_signed_coast_dist_km": hist_row["signed_coast_dist_km"],
                "history_is_land": hist_row["is_land"],
                "history_month": hist_row["month"],
                "history_life_progress": hist_row["life_progress"],
                "rank": int(rank_i),
                "similarity_distance": float(distances[rank_i - 1]),
                "similarity_weight": float(weights[rank_i - 1]),
                "distance_track": float(comp_values["track"][rank_i - 1]),
                "distance_intensity": float(comp_values["intensity"][rank_i - 1]),
                "distance_motion": float(comp_values["motion"][rank_i - 1]),
                "distance_environment": float(comp_values["environment"][rank_i - 1]),
                "distance_time": float(comp_values["time"][rank_i - 1]),
                "distance_life_progress": float(comp_values["life_progress"][rank_i - 1]),
            }
            for field in HISTORICAL_RAIN_OUTPUT_FIELDS:
                rec[f"history_{field}"] = value_for_csv(hist_row[field]) if field in hist_row.index else np.nan
            records.append(rec)

    topk = pd.DataFrame(records)
    topk.to_csv(TOPK_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    diagnostics = {
        "features": features,
        "component_feature_counts": {k: len(v) for k, v in selected_components.items()},
        "history_z_shape": [int(history_z.shape[0]), int(history_z.shape[1])],
        "target_z_shape": [int(target_z.shape[0]), int(target_z.shape[1])],
    }
    return topk, diagnostics


# =========================
# QC report
# =========================


def summary_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
    return {
        "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
        "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
        "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
        "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
        "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
    }


def stats_to_line(stats: Mapping[str, float]) -> str:
    return ", ".join(f"{k}={v:.6g}" if np.isfinite(v) else f"{k}=NA" for k, v in stats.items())


def top_counts_text(series: pd.Series, n: int = 10) -> str:
    counts = series.value_counts(dropna=False).head(n)
    if counts.empty:
        return "(none)"
    return "\n".join(f"- {idx}: {int(val)}" for idx, val in counts.items())


def describe_text(df: pd.DataFrame, cols: Sequence[str]) -> str:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return "(none)"
    return df[existing].describe().to_string()


def leakage_fields_used(selected_components: Mapping[str, Sequence[str]]) -> List[str]:
    return [f for feats in selected_components.values() for f in feats if is_leakage_feature(f)]


def write_qc_report(
    history: pd.DataFrame,
    target: pd.DataFrame,
    topk: pd.DataFrame,
    target_source_report: Mapping[str, object],
    history_report: Mapping[str, object],
    selected_components: Mapping[str, Sequence[str]],
    skipped_features: Mapping[str, str],
    impute_report: pd.DataFrame,
    retrieval_diagnostics: Mapping[str, object],
) -> None:
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    target_missing_cols = ["WND", "PRES", "move_speed_kmh", "move_dir_deg"]
    missing_lines = [
        f"- {c}: {pd.to_numeric(target[c], errors='coerce').isna().mean():.6f}"
        for c in target_missing_cols
        if c in target.columns
    ]
    selected_features = [f for feats in selected_components.values() for f in feats]
    banned_statement = [
        "rain_*",
        "centroid_*",
        "quad_*",
        "anisotropy",
        "rain_radius_*",
        "rain_band_width_km",
        "tif_path",
        "gpm_center_lon/gpm_center_lat",
        "center_match_distance_km",
    ]

    per_target_counts = target.groupby("typhoon_name")["time"].agg(["size", "min", "max"])
    rows_per_target = topk.groupby("target_id").size()
    weight_sums = topk.groupby("target_id")["similarity_weight"].sum()
    unique_events = topk.groupby("target_id")["history_event_uid"].nunique()
    max_event_share = (
        topk.groupby(["target_id", "history_event_uid"]).size().groupby("target_id").max() / TOPK
        if len(topk)
        else pd.Series(dtype=float)
    )

    lines: List[str] = [
        "# Problem 2 Top-K Retrieval QC Report",
        "",
        "## 1. 输入文件",
        f"- 历史库路径: `{HISTORICAL_LIBRARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- 目标路径/安全输入来源: `{target_source_report.get('source_path')}` ({target_source_report.get('source_type')})",
        f"- 目标安全输入输出: `{TARGET_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Top-K 输出: `{TOPK_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- QC 报告: `{QC_REPORT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. 目标输入表统计",
        f"- 总行数: {len(target)}",
        f"- 分台风行数和时间范围:\n```\n{per_target_counts.to_string()}\n```",
        "- WND/PRES/move_speed/move_dir 缺失率:",
        *missing_lines,
        f"- 使用字段列表: {', '.join(target.columns)}",
        f"- 被明确排除的泄漏字段列表: {', '.join(banned_statement)}",
        f"- 源文件中检测到的泄漏字段: {target_source_report.get('leakage_columns_in_source')}",
        f"- 未生成或全缺失环境字段: {target_source_report.get('missing_environment_fields')}",
        "",
        "## 3. 历史库过滤统计",
        f"- 原始历史库行数: {history_report['original_rows']}",
        f"- 过滤后历史库行数: {history_report['filtered_rows']}",
        f"- 是否发现 KONG-REY/MAN-YI 名称: {'是' if history_report['target_name_rows_found'] else '否'} ({history_report['target_name_rows_found']})",
        f"- 是否发现目标时间窗样本: {'是' if history_report['target_window_rows_found'] else '否'} ({history_report['target_window_rows_found']})",
        f"- 最终参与检索历史样本数: {len(history)}",
        "",
        "## 4. 检索特征与权重",
        f"- TOPK: {TOPK}",
        f"- DIVERSIFY_BY_EVENT: {DIVERSIFY_BY_EVENT}",
        f"- MAX_PER_HISTORY_EVENT: {MAX_PER_HISTORY_EVENT}",
        f"- ENV_FEATURE_SET: {ENV_FEATURE_SET}",
        f"- D_env 定义: selected environment features standardized by history mean/std, then averaged squared difference.",
        f"- 分量权重: {COMPONENT_WEIGHTS}",
        f"- 实际参与检索的特征列表: {selected_features}",
        f"- 按分量参与特征: {dict(selected_components)}",
        f"- 被跳过的候选特征及原因: {dict(skipped_features)}",
        f"- 检索特征中误入泄漏字段: {leakage_fields_used(selected_components)}",
        "- 每个特征缺失率和填补策略:",
        "```",
        impute_report.to_string(index=False),
        "```",
        "",
        "## 5. Top-K 结果统计",
        f"- 输出 Top-K 表行数: {len(topk)}",
        f"- 每个 target_id 是否都有 K 行: {bool((rows_per_target == TOPK).all())}",
        f"- 每个 target_id 权重和是否为 1: {bool(np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-6))}",
        f"- similarity_distance: {stats_to_line(summary_stats(topk['similarity_distance']))}",
        f"- 每个目标时刻唯一历史事件数均值: {float(unique_events.mean()):.6f}",
        f"- 每个目标时刻唯一历史事件数最小值: {int(unique_events.min())}",
        f"- 是否存在某个目标时刻 Top-K 被单一台风主导: {bool((max_event_share > 0.5).any())}",
        f"- history_tif_path 存在率: {float(topk['history_tif_path'].map(lambda p: (PROJECT_ROOT / str(p)).exists()).mean()):.6f}",
        "",
        "## 6. 分台风统计",
    ]

    for name in ["KONG-REY", "MAN-YI"]:
        sub_target = target[target["typhoon_name"].eq(name)]
        sub_topk = topk[topk["target_typhoon_name"].eq(name)]
        rank1 = sub_topk[sub_topk["rank"].eq(1)]
        lines.extend(
            [
                f"### {name}",
                f"- 目标时刻数: {len(sub_target)}",
                f"- Top-K 行数: {len(sub_topk)}",
                f"- 平均 similarity_distance: {float(sub_topk['similarity_distance'].mean()) if len(sub_topk) else np.nan:.6f}",
                "- rank=1 历史台风 Top 10:",
                top_counts_text(rank1["history_typhoon_name"] if len(rank1) else pd.Series(dtype=str)),
                "- Top-K 历史台风 Top 10:",
                top_counts_text(sub_topk["history_typhoon_name"] if len(sub_topk) else pd.Series(dtype=str)),
                "- 相似历史样本 WND/PRES/month 摘要:",
                "```",
                describe_text(sub_topk, ["history_WND", "history_PRES", "history_month"]),
                "```",
                "- 目标样本 WND/PRES/month 摘要:",
                "```",
                describe_text(sub_target, ["WND", "PRES", "month"]),
                "```",
            ]
        )

    lines.extend(
        [
            "## 7. 防泄漏声明",
            "本步骤 Top-K 检索只使用路径、强度、移动、时间、海陆和地形等安全输入特征；"
            "rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等由 GPM 降水计算得到的字段未参与距离计算，"
            "仅随历史样本保留供后续模板生成、极端校准和伪缺失验证使用。",
            "",
            "## 8. 其他说明",
            f"- 标准化矩阵维度: {retrieval_diagnostics}",
        ]
    )

    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def main() -> None:
    TARGET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOPK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[18] Loading historical library")
    history, history_report = load_historical_library()
    print("[18] Loading/building target safe inputs")
    target, target_source_report = load_or_build_target_inputs()

    print("[18] Selecting safe retrieval features")
    selected_components, skipped_features = select_safe_retrieval_features(history, target)
    selected_features = [f for feats in selected_components.values() for f in feats]
    if not selected_features:
        raise RuntimeError("No safe retrieval features are available.")
    leaked = leakage_fields_used(selected_components)
    if leaked:
        raise RuntimeError(f"Leakage fields selected for retrieval: {leaked}")

    print(f"[18] Imputing retrieval features: {len(selected_features)} features")
    history_imp, target_imp, impute_report = impute_retrieval_features(history, target, selected_components)
    standardization_params = compute_standardization_params(history_imp, selected_features)

    print(f"[18] Building Top-{TOPK} table: target={len(target_imp)}, history={len(history_imp)}")
    topk, retrieval_diagnostics = build_topk_table(
        history_imp, target_imp, selected_components, standardization_params
    )

    write_qc_report(
        history_imp,
        target_imp,
        topk,
        target_source_report,
        history_report,
        selected_components,
        skipped_features,
        impute_report,
        retrieval_diagnostics,
    )

    rows_per_target = topk.groupby("target_id").size()
    weight_sums = topk.groupby("target_id")["similarity_weight"].sum()
    unique_events = topk.groupby("target_id")["history_event_uid"].nunique()
    dist = pd.to_numeric(topk["similarity_distance"], errors="coerce")
    rank1_top = topk[topk["rank"].eq(1)]["history_typhoon_name"].value_counts().head(10)

    print("\n========== Problem-2 target inputs + Top-K retrieval complete ==========")
    print(f"Script: scripts/18_build_target_inputs_and_topk_retrieval.py")
    print(f"Target safe inputs: {TARGET_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Top-K table: {TOPK_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"QC report: {QC_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Target safe shape: {target_imp.shape[0]} x {target_imp.shape[1]}")
    for name, sub in target_imp.groupby("typhoon_name"):
        print(f"{name}: rows={len(sub)}, time={sub['time'].min()} to {sub['time'].max()}")
    print(f"Top-K shape: {topk.shape[0]} x {topk.shape[1]}")
    print(f"TOPK: {TOPK}")
    print(f"Each target_id has K rows: {bool((rows_per_target == TOPK).all())}")
    print(f"Each target_id weight sum == 1: {bool(np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-6))}")
    print(f"similarity_distance P50: {float(dist.quantile(0.50)):.6f}")
    print(f"similarity_distance P95: {float(dist.quantile(0.95)):.6f}")
    print(f"Top-K unique history events mean: {float(unique_events.mean()):.6f}")
    print(f"Top-K unique history events min: {int(unique_events.min())}")
    print("rank=1 history typhoon Top 10:")
    print(rank1_top.to_string())
    print(f"Retrieval leakage fields used: {leaked}")
    sample_cols = [
        "target_typhoon_name",
        "target_time",
        "target_lat",
        "target_lon_180",
        "target_WND",
        "target_PRES",
        "history_event_uid",
        "history_typhoon_name",
        "history_time",
        "history_tif_path",
        "rank",
        "similarity_distance",
        "similarity_weight",
        "history_rain_max_mmhr",
        "history_rain_p95_mmhr",
        "history_rain_area_10_km2",
    ]
    print("\nRandom 5 key rows:")
    print(topk[sample_cols].sample(min(5, len(topk)), random_state=20260427).to_string(index=False))


if __name__ == "__main__":
    main()
