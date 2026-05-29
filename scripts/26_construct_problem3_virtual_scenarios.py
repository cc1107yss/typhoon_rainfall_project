#!/usr/bin/env python3
"""Construct Problem-3 virtual typhoon scenario inputs.

This script only builds leakage-safe scenario input tables for the Problem-2
rainfall-field generator. It does not generate rainfall fields.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIMARY_TARGET_PATH = PROCESSED_DIR / "problem2_target_model_x_aligned.csv"
OUTPUT_INPUTS_PATH = PROCESSED_DIR / "problem3_scenario_inputs.csv"
OUTPUT_SUMMARY_PATH = PROCESSED_DIR / "problem3_scenario_summary.csv"
HISTORICAL_LIBRARY_PATH = PROCESSED_DIR / "problem2_historical_library_index.csv"

BASE_NAME = "KONG-REY"
BASE_NAME_NORM = "KONGREY"
KM_PER_DEG = 111.32
MIN_HISTORY_ROWS = 30
EPS = 1e-12


NAME_CANDIDATES = [
    "typhoon_name",
    "name",
    "storm_name",
    "name_norm",
    "target_name_norm",
    "target_display_name",
    "track_typhoon_name",
]
TIME_CANDIDATES = ["time", "datetime", "valid_time", "track_time"]
LAT_CANDIDATES = ["lat", "track_lat", "center_lat"]
LON_CANDIDATES = ["lon_180", "track_lon_180", "lon", "track_lon", "center_lon"]
WIND_CANDIDATES = ["WND", "wind", "max_wind", "track_wind"]
PRESSURE_CANDIDATES = ["PRES", "pressure", "track_pressure"]
MOVE_SPEED_CANDIDATES = ["move_speed_kmh", "track_move_speed_kmh"]
MOVE_DIR_CANDIDATES = ["move_dir_deg", "track_move_dir_deg"]
MOVE_DISTANCE_CANDIDATES = ["move_distance_km", "track_move_distance_km"]
DT_CANDIDATES = ["dt_h", "track_dt_h", "cadence_hours"]
WIND_RATE_CANDIDATES = ["wind_change_rate", "track_wind_change_rate"]
PRESSURE_RATE_CANDIDATES = ["pressure_change_rate", "track_pressure_change_rate"]
IS_LAND_CANDIDATES = ["is_land", "track_is_land", "track_is_land_interp"]
COAST_DIST_CANDIDATES = ["coast_dist_km", "track_coast_dist_km"]
SIGNED_COAST_DIST_CANDIDATES = ["signed_coast_dist_km", "track_signed_coast_dist_km"]
NEAR_COAST_CANDIDATES = ["near_coast_index", "coast_influence_exp"]
INTENSITY_CANDIDATES = ["intensity", "track_intensity"]


SCENARIO_NAMES = {
    "S0": "baseline_kong_rey",
    "S1": "intensity_enhanced",
    "S2": "near_coast_westward_path_shift",
    "S3": "slower_translation",
    "S4": "compound_high_impact",
}

SCENARIO_NOTES = {
    "S0": "baseline_unperturbed",
    "S1": "intensity_enhanced_with_pressure_adjustment",
    "S2": "westward_100km_path_shift_without_coast_recompute",
    "S3": "slower_translation_time_axis_stretched",
    "S4": "compound_high_impact_scenario",
}


@dataclass
class FieldMap:
    name: Optional[str] = None
    time: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None
    wind: Optional[str] = None
    pressure: Optional[str] = None
    move_speed: Optional[str] = None
    move_dir: Optional[str] = None
    move_distance: Optional[str] = None
    dt_h: Optional[str] = None
    wind_rate: Optional[str] = None
    pressure_rate: Optional[str] = None
    is_land: Optional[str] = None
    coast_dist: Optional[str] = None
    signed_coast_dist: Optional[str] = None
    near_coast: Optional[str] = None
    intensity: Optional[str] = None


@dataclass
class DerivedCalibration:
    pressure_reference: Optional[float] = None
    wind_z_intercept: Optional[float] = None
    wind_z_slope: Optional[float] = None
    pressure_deficit_z_intercept: Optional[float] = None
    pressure_deficit_z_slope: Optional[float] = None


@dataclass
class ScenarioConstraints:
    source: str
    history_rows: int = 0
    history_rows_after_month: int = 0
    history_rows_after_lat: int = 0
    history_rows_after_environment: int = 0
    wind_q50: Optional[float] = None
    wind_q75: Optional[float] = None
    wind_q95: Optional[float] = None
    pressure_q05: Optional[float] = None
    pressure_q25: Optional[float] = None
    pressure_q50: Optional[float] = None
    move_speed_q25: Optional[float] = None
    move_speed_q50: Optional[float] = None
    pressure_wind_slope: Optional[float] = None
    pressure_wind_intercept: Optional[float] = None
    gamma: float = 0.75
    wind_delta: Optional[float] = None
    pressure_adjustment_source: str = "fallback_2hpa_per_ms"


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    exact = set(columns)
    lower_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        match = lower_lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def detect_fields(df: pd.DataFrame) -> FieldMap:
    columns = list(df.columns)
    return FieldMap(
        name=first_existing(columns, NAME_CANDIDATES),
        time=first_existing(columns, TIME_CANDIDATES),
        lat=first_existing(columns, LAT_CANDIDATES),
        lon=first_existing(columns, LON_CANDIDATES),
        wind=first_existing(columns, WIND_CANDIDATES),
        pressure=first_existing(columns, PRESSURE_CANDIDATES),
        move_speed=first_existing(columns, MOVE_SPEED_CANDIDATES),
        move_dir=first_existing(columns, MOVE_DIR_CANDIDATES),
        move_distance=first_existing(columns, MOVE_DISTANCE_CANDIDATES),
        dt_h=first_existing(columns, DT_CANDIDATES),
        wind_rate=first_existing(columns, WIND_RATE_CANDIDATES),
        pressure_rate=first_existing(columns, PRESSURE_RATE_CANDIDATES),
        is_land=first_existing(columns, IS_LAND_CANDIDATES),
        coast_dist=first_existing(columns, COAST_DIST_CANDIDATES),
        signed_coast_dist=first_existing(columns, SIGNED_COAST_DIST_CANDIDATES),
        near_coast=first_existing(columns, NEAR_COAST_CANDIDATES),
        intensity=first_existing(columns, INTENSITY_CANDIDATES),
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    return df


def candidate_target_paths() -> list[Path]:
    paths = [
        PRIMARY_TARGET_PATH,
        PROCESSED_DIR / "target_typhoon_inputs_2024_halfhour_leakage_safe.csv",
        PROCESSED_DIR / "target_typhoon_inputs_2024_track_points_leakage_safe.csv",
        PROCESSED_DIR / "problem2_target_halfhour_inputs_safe.csv",
    ]
    glob_patterns = [
        "*target*typhoon*halfhour*safe*.csv",
        "*target*typhoon*input*safe*.csv",
        "*problem2*target*halfhour*safe*.csv",
        "*target*model*x*halfhour*.csv",
    ]
    for pattern in glob_patterns:
        paths.extend(sorted(PROCESSED_DIR.glob(pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def missing_required_fields(mapping: FieldMap) -> list[str]:
    required = {
        "name": mapping.name,
        "time": mapping.time,
        "lat": mapping.lat,
        "lon": mapping.lon,
        "wind": mapping.wind,
        "pressure": mapping.pressure,
    }
    return [key for key, value in required.items() if value is None]


def evaluate_candidate(df: pd.DataFrame, mapping: FieldMap) -> tuple[bool, str]:
    missing = missing_required_fields(mapping)
    if missing:
        return False, "missing required fields: " + ",".join(missing)

    names = df[mapping.name].map(normalize_name)
    sub = df.loc[names.eq(BASE_NAME_NORM)].copy()
    if sub.empty:
        return False, f"no {BASE_NAME} rows after name normalization"

    for label, col in [
        ("time", mapping.time),
        ("lat", mapping.lat),
        ("lon", mapping.lon),
        ("wind", mapping.wind),
        ("pressure", mapping.pressure),
    ]:
        if label == "time":
            values = pd.to_datetime(sub[col], errors="coerce")
        else:
            values = pd.to_numeric(sub[col], errors="coerce")
        if values.notna().sum() == 0:
            return False, f"{label} column has no usable values for {BASE_NAME}: {col}"
    return True, f"usable {BASE_NAME} rows: {len(sub)}"


def choose_target_input() -> tuple[pd.DataFrame, FieldMap, Path, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    for path in candidate_target_paths():
        if not path.exists():
            attempts.append({"path": str(path), "status": "missing", "reason": "file not found"})
            continue
        df = read_csv(path)
        mapping = detect_fields(df)
        usable, reason = evaluate_candidate(df, mapping)
        attempts.append({"path": str(path), "status": "usable" if usable else "not_usable", "reason": reason})
        if usable:
            return df, mapping, path, attempts

    details = "\n".join(f"- {row['path']}: {row['status']} ({row['reason']})" for row in attempts)
    raise FileNotFoundError(
        "No usable target typhoon input CSV was found. Expected "
        "data/processed/problem2_target_model_x_aligned.csv or a fallback "
        "target half-hour leakage-safe input with name, time, lat/lon, wind, "
        f"and pressure fields.\nChecked files:\n{details}"
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def wrap_lon_180(values: object) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return ((series + 180.0) % 360.0) - 180.0


def lon_180_to_360(values: object) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series.where(series >= 0.0, series + 360.0)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return np.nan
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return float(2.0 * radius * math.asin(min(1.0, math.sqrt(a))))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return np.nan
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def fit_linear(x: pd.Series, y: pd.Series) -> tuple[Optional[float], Optional[float]]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 2 or float(xx.loc[mask].std()) <= EPS:
        return None, None
    slope, intercept = np.polyfit(xx.loc[mask].to_numpy(dtype=float), yy.loc[mask].to_numpy(dtype=float), 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None, None
    return float(intercept), float(slope)


def infer_derived_calibration(base: pd.DataFrame) -> DerivedCalibration:
    calib = DerivedCalibration()
    if "pressure_deficit" in base.columns:
        pressure_ref = numeric(base["PRES"]) + numeric(base["pressure_deficit"])
        value = float(pressure_ref.median(skipna=True))
        if np.isfinite(value):
            calib.pressure_reference = value
    if "wind_z" in base.columns:
        intercept, slope = fit_linear(base["wind"], base["wind_z"])
        calib.wind_z_intercept = intercept
        calib.wind_z_slope = slope
    if "pressure_deficit_z" in base.columns and "pressure_deficit" in base.columns:
        intercept, slope = fit_linear(base["pressure_deficit"], base["pressure_deficit_z"])
        calib.pressure_deficit_z_intercept = intercept
        calib.pressure_deficit_z_slope = slope
    return calib


def season_from_month(month: pd.Series) -> pd.Series:
    mapping = {
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
    return month.map(mapping)


def refresh_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    out["year"] = time.dt.year
    out["month"] = time.dt.month
    out["day"] = time.dt.day
    out["hour"] = time.dt.hour + time.dt.minute / 60.0 + time.dt.second / 3600.0
    out["season"] = season_from_month(out["month"])
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"].astype(float) / 12.0)
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"].astype(float) / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"].astype(float) / 24.0)
    return out


def compute_motion_features(
    df: pd.DataFrame,
    overwrite_motion: bool,
    overwrite_rates: bool,
) -> pd.DataFrame:
    out = df.sort_values("time").reset_index(drop=True).copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    lon = numeric(out["lon_180"]).to_numpy(dtype=float)
    lat = numeric(out["lat"]).to_numpy(dtype=float)
    wind = numeric(out["wind"]).to_numpy(dtype=float)
    pressure = numeric(out["pressure"]).to_numpy(dtype=float)
    n = len(out)

    dt_h = np.full(n, np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)
    speed = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)
    wind_rate = np.full(n, np.nan, dtype=float)
    pressure_rate = np.full(n, np.nan, dtype=float)

    for i in range(1, n):
        delta_h = (time.iloc[i] - time.iloc[i - 1]).total_seconds() / 3600.0
        dt_h[i] = delta_h
        if np.isfinite(delta_h) and delta_h > 0.0:
            dist[i] = haversine_km(lon[i - 1], lat[i - 1], lon[i], lat[i])
            speed[i] = dist[i] / delta_h if np.isfinite(dist[i]) else np.nan
            direction[i] = bearing_deg(lon[i - 1], lat[i - 1], lon[i], lat[i])
            if np.isfinite(wind[i]) and np.isfinite(wind[i - 1]):
                wind_rate[i] = (wind[i] - wind[i - 1]) / delta_h
            if np.isfinite(pressure[i]) and np.isfinite(pressure[i - 1]):
                pressure_rate[i] = (pressure[i] - pressure[i - 1]) / delta_h

    if n > 1:
        dt_h[0] = dt_h[1] if np.isfinite(dt_h[1]) else 0.5
        dist[0] = dist[1]
        speed[0] = speed[1]
        direction[0] = direction[1]
        wind_rate[0] = wind_rate[1]
        pressure_rate[0] = pressure_rate[1]
    elif n == 1:
        dt_h[0] = 0.5
        dist[0] = 0.0
        speed[0] = 0.0
        direction[0] = 0.0
        wind_rate[0] = 0.0
        pressure_rate[0] = 0.0

    fill_or_set = {
        "dt_h": dt_h,
        "move_distance_km": dist,
        "move_speed_kmh": speed,
        "move_dir_deg": direction,
    }
    for col, values in fill_or_set.items():
        if col not in out.columns or overwrite_motion:
            out[col] = values
        else:
            out[col] = numeric(out[col]).where(numeric(out[col]).notna(), values)

    rate_values = {
        "wind_change_rate": wind_rate,
        "pressure_change_rate": pressure_rate,
    }
    for col, values in rate_values.items():
        if col not in out.columns or overwrite_rates:
            out[col] = values
        else:
            out[col] = numeric(out[col]).where(numeric(out[col]).notna(), values)

    return out


def refresh_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "signed_coast_dist_km" in out.columns:
        signed = numeric(out["signed_coast_dist_km"])
    else:
        signed = pd.Series(np.nan, index=out.index)
        out["signed_coast_dist_km"] = signed

    if "coast_dist_km" in out.columns:
        coast = numeric(out["coast_dist_km"])
    else:
        coast = signed.abs()
        out["coast_dist_km"] = coast

    if signed.notna().any():
        out["is_land"] = (signed < 0.0).astype(int)
        distance_for_index = signed.abs()
    else:
        if "is_land" not in out.columns:
            out["is_land"] = np.nan
        distance_for_index = coast.abs()

    out["near_coast_index"] = np.exp(-distance_for_index / 200.0)
    out["coast_influence_exp"] = out["near_coast_index"]
    if signed.notna().any():
        out["is_near_coast_100km"] = ((signed >= -100.0) & (signed <= 100.0)).astype(int)
        out["is_near_coast_200km"] = ((signed >= -200.0) & (signed <= 200.0)).astype(int)
        out["is_offshore_far_300km"] = (signed > 300.0).astype(int)
        out["is_inland_100km"] = (signed < -100.0).astype(int)
    return out


def refresh_derived_intensity_features(df: pd.DataFrame, calib: DerivedCalibration) -> pd.DataFrame:
    out = df.copy()
    if "pressure_deficit" in out.columns and calib.pressure_reference is not None:
        out["pressure_deficit"] = calib.pressure_reference - numeric(out["pressure"])
    if "wind_z" in out.columns and calib.wind_z_intercept is not None and calib.wind_z_slope is not None:
        out["wind_z"] = calib.wind_z_intercept + calib.wind_z_slope * numeric(out["wind"])
    if (
        "pressure_deficit_z" in out.columns
        and "pressure_deficit" in out.columns
        and calib.pressure_deficit_z_intercept is not None
        and calib.pressure_deficit_z_slope is not None
    ):
        out["pressure_deficit_z"] = (
            calib.pressure_deficit_z_intercept
            + calib.pressure_deficit_z_slope * numeric(out["pressure_deficit"])
        )
    if {"intensity_index", "wind_z", "pressure_deficit_z"}.issubset(out.columns):
        out["intensity_index"] = 0.5 * numeric(out["wind_z"]) + 0.5 * numeric(out["pressure_deficit_z"])

    wind_rate = numeric(out["wind_change_rate"]) if "wind_change_rate" in out.columns else pd.Series(np.nan, index=out.index)
    pressure_rate = (
        numeric(out["pressure_change_rate"]) if "pressure_change_rate" in out.columns else pd.Series(np.nan, index=out.index)
    )
    out["is_intensifying_wind"] = (wind_rate > 0.0).astype(int)
    out["is_weakening_wind"] = (wind_rate < 0.0).astype(int)
    out["is_intensifying_pressure"] = (pressure_rate < 0.0).astype(int)
    out["is_weakening_pressure"] = (pressure_rate > 0.0).astype(int)
    return out


def sync_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["WND"] = numeric(out["wind"])
    out["PRES"] = numeric(out["pressure"])
    out["lon_180"] = wrap_lon_180(out["lon_180"]).to_numpy()
    out["lon"] = lon_180_to_360(out["lon_180"]).to_numpy()

    aliases = {
        "track_lat": "lat",
        "center_lat": "lat",
        "track_lon_180": "lon_180",
        "center_lon": "lon",
        "track_wind": "wind",
        "track_pressure": "pressure",
        "track_move_distance_km": "move_distance_km",
        "track_move_speed_kmh": "move_speed_kmh",
        "track_move_dir_deg": "move_dir_deg",
        "track_wind_change_rate": "wind_change_rate",
        "track_pressure_change_rate": "pressure_change_rate",
        "track_dt_h": "dt_h",
        "track_is_land": "is_land",
        "track_coast_dist_km": "coast_dist_km",
        "track_signed_coast_dist_km": "signed_coast_dist_km",
    }
    for alias, source in aliases.items():
        if alias in out.columns and source in out.columns:
            out[alias] = out[source]

    move_rad = np.deg2rad(numeric(out["move_dir_deg"]))
    out["move_dir_sin"] = np.sin(move_rad)
    out["move_dir_cos"] = np.cos(move_rad)
    if "track_move_dir_sin" in out.columns:
        out["track_move_dir_sin"] = out["move_dir_sin"]
    if "track_move_dir_cos" in out.columns:
        out["track_move_dir_cos"] = out["move_dir_cos"]
    return out


def refresh_life_progress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("time").reset_index(drop=True).copy()
    out["life_time_index"] = np.arange(len(out), dtype=int)
    if len(out) > 1:
        out["life_progress"] = out["life_time_index"] / float(len(out) - 1)
    else:
        out["life_progress"] = 0.0
    return out


def refresh_all_features(
    df: pd.DataFrame,
    calib: DerivedCalibration,
    overwrite_motion: bool,
    overwrite_rates: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = compute_motion_features(out, overwrite_motion=overwrite_motion, overwrite_rates=overwrite_rates)
    out = refresh_time_features(out)
    out = refresh_environment_features(out)
    out = refresh_derived_intensity_features(out, calib)
    out = sync_alias_columns(out)
    out = refresh_life_progress(out)
    return out


def standardize_target_input(df: pd.DataFrame, mapping: FieldMap) -> pd.DataFrame:
    out = df.copy()
    names = out[mapping.name].map(normalize_name)
    out = out.loc[names.eq(BASE_NAME_NORM)].copy()
    out["time"] = pd.to_datetime(out[mapping.time], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No usable {BASE_NAME} rows remain after parsing time.")

    if "base_target_id" not in out.columns and "target_id" in out.columns:
        out["base_target_id"] = out["target_id"]

    out["base_typhoon"] = BASE_NAME
    out["typhoon_name"] = BASE_NAME
    out["target_name_norm"] = BASE_NAME_NORM
    out["lat"] = numeric(out[mapping.lat])
    out["lon_180"] = wrap_lon_180(out[mapping.lon]).to_numpy()
    out["lon"] = lon_180_to_360(out["lon_180"]).to_numpy()
    out["wind"] = numeric(out[mapping.wind])
    out["pressure"] = numeric(out[mapping.pressure])
    out["WND"] = out["wind"]
    out["PRES"] = out["pressure"]

    if mapping.move_speed:
        out["move_speed_kmh"] = numeric(out[mapping.move_speed])
    if mapping.move_dir:
        out["move_dir_deg"] = numeric(out[mapping.move_dir])
    if mapping.move_distance:
        out["move_distance_km"] = numeric(out[mapping.move_distance])
    if mapping.dt_h:
        out["dt_h"] = numeric(out[mapping.dt_h])
    if mapping.wind_rate:
        out["wind_change_rate"] = numeric(out[mapping.wind_rate])
    if mapping.pressure_rate:
        out["pressure_change_rate"] = numeric(out[mapping.pressure_rate])
    if mapping.is_land:
        out["is_land"] = numeric(out[mapping.is_land])
    if mapping.coast_dist:
        out["coast_dist_km"] = numeric(out[mapping.coast_dist])
    if mapping.signed_coast_dist:
        out["signed_coast_dist_km"] = numeric(out[mapping.signed_coast_dist])
    if mapping.near_coast:
        out["near_coast_index"] = numeric(out[mapping.near_coast])
    if mapping.intensity:
        out["intensity"] = numeric(out[mapping.intensity])

    required = ["lat", "lon_180", "wind", "pressure"]
    missing = [col for col in required if numeric(out[col]).isna().any()]
    if missing:
        raise RuntimeError(f"Standardized {BASE_NAME} input has missing values in {missing}.")

    calib = infer_derived_calibration(out)
    out = refresh_all_features(out, calib, overwrite_motion=False, overwrite_rates=False)
    return out


def finite_quantile(values: pd.Series, q: float) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    value = float(clean.quantile(q))
    return value if np.isfinite(value) else None


def clip_gamma(value: object) -> float:
    try:
        gamma = float(value)
    except (TypeError, ValueError):
        return 0.75
    if not np.isfinite(gamma) or gamma <= 0.0:
        return 0.75
    return float(np.clip(gamma, 0.6, 0.9))


def build_history_constraints(base: pd.DataFrame) -> ScenarioConstraints:
    if not HISTORICAL_LIBRARY_PATH.exists():
        return ScenarioConstraints(source="fallback_rule")

    hist = read_csv(HISTORICAL_LIBRARY_PATH)
    mapping = detect_fields(hist)
    required = [mapping.time, mapping.lat, mapping.wind, mapping.pressure, mapping.move_speed]
    if any(col is None for col in required):
        return ScenarioConstraints(source="fallback_rule", history_rows=len(hist))

    hist["time"] = pd.to_datetime(hist[mapping.time], errors="coerce")
    hist["_month"] = hist["time"].dt.month
    hist["_lat"] = numeric(hist[mapping.lat])
    hist["_wind"] = numeric(hist[mapping.wind])
    hist["_pressure"] = numeric(hist[mapping.pressure])
    hist["_move_speed"] = numeric(hist[mapping.move_speed])

    month_mask = hist["_month"].isin([9, 10, 11])
    h_month = hist.loc[month_mask].copy()
    if h_month.empty:
        return ScenarioConstraints(source="fallback_rule", history_rows=len(hist))

    lat_min = float(numeric(base["lat"]).min() - 5.0)
    lat_max = float(numeric(base["lat"]).max() + 5.0)
    h_lat = h_month.loc[h_month["_lat"].between(lat_min, lat_max)].copy()
    if len(h_lat) < MIN_HISTORY_ROWS:
        h_selected = h_month.copy()
        h_env = h_selected.copy()
    else:
        h_selected = h_lat.copy()
        h_env = h_selected.copy()
        if mapping.signed_coast_dist and "signed_coast_dist_km" in base.columns:
            base_signed = numeric(base["signed_coast_dist_km"]).abs()
            hist_signed = numeric(h_selected[mapping.signed_coast_dist]).abs()
            base_near = bool(base_signed.median(skipna=True) <= 200.0)
            env_mask = hist_signed <= 200.0 if base_near else hist_signed > 200.0
            h_tmp = h_selected.loc[env_mask.fillna(False)].copy()
            if len(h_tmp) >= MIN_HISTORY_ROWS:
                h_env = h_tmp
        elif mapping.is_land and "is_land" in base.columns:
            base_land = int(round(float(numeric(base["is_land"]).median(skipna=True))))
            hist_land = numeric(h_selected[mapping.is_land]).round()
            h_tmp = h_selected.loc[hist_land.eq(base_land)].copy()
            if len(h_tmp) >= MIN_HISTORY_ROWS:
                h_env = h_tmp

    if len(h_env) < MIN_HISTORY_ROWS:
        h_env = h_month.copy()

    wind_q50 = finite_quantile(h_env["_wind"], 0.50)
    wind_q75 = finite_quantile(h_env["_wind"], 0.75)
    wind_q95 = finite_quantile(h_env["_wind"], 0.95)
    pressure_q05 = finite_quantile(h_env["_pressure"], 0.05)
    pressure_q25 = finite_quantile(h_env["_pressure"], 0.25)
    pressure_q50 = finite_quantile(h_env["_pressure"], 0.50)
    speed_q25 = finite_quantile(h_env["_move_speed"], 0.25)
    speed_q50 = finite_quantile(h_env["_move_speed"], 0.50)

    needed = [wind_q50, wind_q75, wind_q95, pressure_q05, pressure_q25, pressure_q50, speed_q25, speed_q50]
    if any(value is None for value in needed):
        return ScenarioConstraints(
            source="fallback_rule",
            history_rows=len(hist),
            history_rows_after_month=len(h_month),
            history_rows_after_lat=len(h_lat),
            history_rows_after_environment=len(h_env),
        )

    gamma = clip_gamma(speed_q25 / speed_q50 if speed_q50 and speed_q50 > EPS else np.nan)
    wind_delta = max(0.0, float(wind_q75 - wind_q50))
    constraints = ScenarioConstraints(
        source="historical_library",
        history_rows=len(hist),
        history_rows_after_month=len(h_month),
        history_rows_after_lat=len(h_lat),
        history_rows_after_environment=len(h_env),
        wind_q50=wind_q50,
        wind_q75=wind_q75,
        wind_q95=wind_q95,
        pressure_q05=pressure_q05,
        pressure_q25=pressure_q25,
        pressure_q50=pressure_q50,
        move_speed_q25=speed_q25,
        move_speed_q50=speed_q50,
        gamma=gamma,
        wind_delta=wind_delta,
    )

    fit = h_env[["_wind", "_pressure"]].dropna()
    if len(fit) >= MIN_HISTORY_ROWS and float(fit["_wind"].std()) > EPS:
        slope, intercept = np.polyfit(fit["_wind"].to_numpy(dtype=float), fit["_pressure"].to_numpy(dtype=float), 1)
        if np.isfinite(slope) and np.isfinite(intercept) and slope < 0.0:
            constraints.pressure_wind_slope = float(slope)
            constraints.pressure_wind_intercept = float(intercept)
            constraints.pressure_adjustment_source = "historical_pressure_wind_fit"
    return constraints


def apply_intensity_enhancement(df: pd.DataFrame, constraints: ScenarioConstraints, calib: DerivedCalibration) -> pd.DataFrame:
    out = df.copy()
    wind_old = numeric(out["wind"]).to_numpy(dtype=float)
    pressure_old = numeric(out["pressure"]).to_numpy(dtype=float)

    if constraints.source == "historical_library" and constraints.wind_delta is not None and constraints.wind_q95 is not None:
        wind_candidate = np.minimum(wind_old + float(constraints.wind_delta), float(constraints.wind_q95))
        wind_new = np.maximum(wind_old, wind_candidate)
        if float(np.nanmean(wind_new)) <= float(np.nanmean(wind_old)):
            cap = float(np.nanmax(wind_old) * 1.25)
            wind_new = np.minimum(wind_old * 1.15, cap)
    else:
        cap = float(np.nanmax(wind_old) * 1.25)
        wind_new = np.minimum(wind_old * 1.15, cap)

    delta = wind_new - wind_old
    if constraints.pressure_wind_slope is not None and constraints.pressure_wind_slope < 0.0:
        pressure_new = pressure_old + float(constraints.pressure_wind_slope) * delta
        pressure_new = np.maximum(pressure_new, 880.0)
    else:
        pressure_new = np.maximum(pressure_old - 2.0 * delta, 880.0)

    out["wind"] = wind_new
    out["WND"] = wind_new
    out["pressure"] = pressure_new
    out["PRES"] = pressure_new
    out = refresh_all_features(out, calib, overwrite_motion=False, overwrite_rates=True)
    return out


def apply_westward_path_shift(df: pd.DataFrame, calib: DerivedCalibration) -> pd.DataFrame:
    out = df.copy()
    lat = numeric(out["lat"]).to_numpy(dtype=float)
    lon = numeric(out["lon_180"]).to_numpy(dtype=float)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = np.where(np.abs(cos_lat) < 0.1, np.sign(cos_lat) * 0.1, cos_lat)
    lon_shift = 100.0 / (KM_PER_DEG * cos_lat)
    out["lon_180"] = wrap_lon_180(lon - lon_shift).to_numpy()
    out["lon"] = lon_180_to_360(out["lon_180"]).to_numpy()
    out = refresh_all_features(out, calib, overwrite_motion=True, overwrite_rates=False)
    return out


def cumulative_path_distance(df: pd.DataFrame) -> np.ndarray:
    lon = numeric(df["lon_180"]).to_numpy(dtype=float)
    lat = numeric(df["lat"]).to_numpy(dtype=float)
    dist = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        value = haversine_km(lon[i - 1], lat[i - 1], lon[i], lat[i])
        dist[i] = value if np.isfinite(value) else 0.0
    path = np.cumsum(dist)
    if len(path) and float(path[-1]) <= EPS:
        path = np.arange(len(df), dtype=float)
    return path


def interp_numeric_by_path(path: np.ndarray, values: pd.Series, new_path: np.ndarray) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(path) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return np.full(len(new_path), np.nan, dtype=float)
    if int(mask.sum()) == 1:
        return np.full(len(new_path), float(y[mask][0]), dtype=float)
    x = path[mask]
    yy = y[mask]
    unique_x, unique_idx = np.unique(x, return_index=True)
    if len(unique_x) == 1:
        return np.full(len(new_path), float(yy[unique_idx[0]]), dtype=float)
    return np.interp(new_path, unique_x, yy[unique_idx])


def interp_lon_by_path(path: np.ndarray, values: pd.Series, new_path: np.ndarray) -> np.ndarray:
    lon = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(path) & np.isfinite(lon)
    if int(mask.sum()) == 0:
        return np.full(len(new_path), np.nan, dtype=float)
    if int(mask.sum()) == 1:
        return np.full(len(new_path), float(lon[mask][0]), dtype=float)
    x = path[mask]
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon[mask])))
    unique_x, unique_idx = np.unique(x, return_index=True)
    if len(unique_x) == 1:
        return np.full(len(new_path), float(lon[mask][unique_idx[0]]), dtype=float)
    lon_interp = np.interp(new_path, unique_x, lon_unwrapped[unique_idx])
    return (((lon_interp + 180.0) % 360.0) - 180.0).astype(float)


def nearest_path_indices(path: np.ndarray, new_path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.array([], dtype=int)
    idx_right = np.searchsorted(path, new_path, side="left")
    idx_right = np.clip(idx_right, 0, len(path) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(path) - 1)
    choose_left = np.abs(new_path - path[idx_left]) <= np.abs(path[idx_right] - new_path)
    return np.where(choose_left, idx_left, idx_right).astype(int)


def stretch_time_axis(df: pd.DataFrame, gamma: float, calib: DerivedCalibration) -> pd.DataFrame:
    source = df.sort_values("time").reset_index(drop=True).copy()
    gamma = clip_gamma(gamma)
    time = pd.to_datetime(source["time"], errors="coerce")
    duration_old_h = (time.iloc[-1] - time.iloc[0]).total_seconds() / 3600.0
    if not np.isfinite(duration_old_h) or duration_old_h <= 0.0:
        raise RuntimeError("Cannot stretch a target track with non-positive duration.")

    duration_new_h = duration_old_h / gamma
    n_steps = int(math.ceil(duration_new_h / 0.5))
    elapsed_h = np.arange(n_steps + 1, dtype=float) * 0.5
    new_time = time.iloc[0] + pd.to_timedelta(elapsed_h, unit="h")

    path = cumulative_path_distance(source)
    total_path = float(path[-1]) if len(path) else 0.0
    if total_path <= EPS:
        progress_path = np.linspace(0.0, float(max(len(source) - 1, 0)), len(source))
        path = progress_path
        total_path = float(path[-1]) if len(path) else 0.0

    old_elapsed_h = (time - time.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    mapped_old_elapsed_h = np.clip(elapsed_h * gamma, 0.0, duration_old_h)
    valid_progress = np.isfinite(old_elapsed_h) & np.isfinite(path)
    if int(valid_progress.sum()) < 2:
        new_path = np.clip(elapsed_h / duration_new_h, 0.0, 1.0) * total_path
    else:
        new_path = np.interp(mapped_old_elapsed_h, old_elapsed_h[valid_progress], path[valid_progress])

    nearest_idx = nearest_path_indices(path, new_path)
    out = source.iloc[nearest_idx].copy().reset_index(drop=True)
    out["time"] = new_time

    continuous_cols = [
        "lat",
        "wind",
        "WND",
        "pressure",
        "PRES",
        "coast_dist_km",
        "signed_coast_dist_km",
        "near_coast_index",
        "coast_influence_exp",
        "center_lat",
        "track_lat",
        "track_wind",
        "track_pressure",
        "track_coast_dist_km",
        "track_signed_coast_dist_km",
    ]
    for col in continuous_cols:
        if col in source.columns:
            out[col] = interp_numeric_by_path(path, source[col], new_path)

    lon_cols = ["lon_180", "track_lon_180"]
    for col in lon_cols:
        if col in source.columns:
            out[col] = interp_lon_by_path(path, source[col], new_path)
    if "lon_180" in out.columns:
        out["lon"] = lon_180_to_360(out["lon_180"]).to_numpy()
    if "center_lon" in out.columns:
        out["center_lon"] = out["lon"]

    out = refresh_all_features(out, calib, overwrite_motion=True, overwrite_rates=True)
    return out


def annotate_scenario(
    df: pd.DataFrame,
    scenario_id: str,
    constraints: ScenarioConstraints,
    coast_recomputed: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["scenario_id"] = scenario_id
    out["scenario_name"] = SCENARIO_NAMES[scenario_id]
    out["base_typhoon"] = BASE_NAME
    out["scenario_note"] = SCENARIO_NOTES[scenario_id]
    out["constraint_source"] = constraints.source
    out["coast_recomputed"] = bool(coast_recomputed)
    out["scenario_time_index"] = np.arange(len(out), dtype=int)
    if "base_target_id" not in out.columns and "target_id" in out.columns:
        out["base_target_id"] = out["target_id"]
    out["target_id"] = scenario_id + "_" + pd.to_datetime(out["time"]).dt.strftime("%Y%m%d%H%M%S")
    return out


def leakage_columns(columns: Iterable[str]) -> list[str]:
    patterns = [
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
    bad = []
    for col in columns:
        if any(pattern.search(col) for pattern in patterns):
            bad.append(col)
    return bad


def drop_forbidden_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    owd_cols = [col for col in df.columns if col.upper() == "OWD"]
    leak_cols = leakage_columns(df.columns)
    to_drop = sorted(set(owd_cols + leak_cols))
    out = df.drop(columns=to_drop, errors="ignore").copy()
    return out, owd_cols, leak_cols


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario_id in ["S0", "S1", "S2", "S3", "S4"]:
        sub = df.loc[df["scenario_id"] == scenario_id].sort_values("time")
        if sub.empty:
            continue
        start = pd.to_datetime(sub["time"]).min()
        end = pd.to_datetime(sub["time"]).max()
        records.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": sub["scenario_name"].iloc[0],
                "n_rows": int(len(sub)),
                "start_time": start,
                "end_time": end,
                "duration_h": float((end - start).total_seconds() / 3600.0),
                "min_lat": float(numeric(sub["lat"]).min()),
                "max_lat": float(numeric(sub["lat"]).max()),
                "min_lon": float(numeric(sub["lon_180"]).min()),
                "max_lon": float(numeric(sub["lon_180"]).max()),
                "mean_wind": float(numeric(sub["wind"]).mean()),
                "max_wind": float(numeric(sub["wind"]).max()),
                "min_pressure": float(numeric(sub["pressure"]).min()),
                "mean_move_speed_kmh": float(numeric(sub["move_speed_kmh"]).mean()),
                "max_near_coast_index": float(numeric(sub["near_coast_index"]).max()),
                "constraint_source": sub["constraint_source"].iloc[0],
                "coast_recomputed": bool(sub["coast_recomputed"].iloc[0]),
                "scenario_note": sub["scenario_note"].iloc[0],
            }
        )
    return pd.DataFrame(records)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "scenario_id",
        "scenario_name",
        "base_typhoon",
        "target_id",
        "base_target_id",
        "typhoon_name",
        "target_name_norm",
        "time",
        "lat",
        "lon",
        "lon_180",
        "wind",
        "pressure",
        "WND",
        "PRES",
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "is_land",
        "coast_dist_km",
        "signed_coast_dist_km",
        "near_coast_index",
        "constraint_source",
        "coast_recomputed",
        "scenario_note",
    ]
    ordered = [col for col in first_cols if col in df.columns]
    ordered.extend([col for col in df.columns if col not in ordered])
    return df[ordered].copy()


def assert_hard_rules(df: pd.DataFrame, summary: pd.DataFrame, base_rows: int) -> None:
    assert OUTPUT_INPUTS_PATH.exists(), f"Missing output file: {OUTPUT_INPUTS_PATH}"
    assert OUTPUT_SUMMARY_PATH.exists(), f"Missing output file: {OUTPUT_SUMMARY_PATH}"

    scenario_ids = set(df["scenario_id"].dropna().astype(str).unique())
    expected = {"S0", "S1", "S2", "S3", "S4"}
    assert expected.issubset(scenario_ids), f"scenario_id missing values: {sorted(expected - scenario_ids)}"

    rows_by_id = df.groupby("scenario_id").size().to_dict()
    assert int(rows_by_id.get("S0", -1)) == int(base_rows), (
        f"S0 row count {rows_by_id.get('S0')} does not match base KONG-REY rows {base_rows}"
    )

    stats = summary.set_index("scenario_id")
    assert stats.loc["S1", "mean_wind"] > stats.loc["S0", "mean_wind"], "S1 mean wind is not higher than S0"
    assert stats.loc["S1", "min_pressure"] <= stats.loc["S0", "min_pressure"], "S1 min pressure is not <= S0"
    assert numeric(df.loc[df["scenario_id"] == "S2", "lon_180"]).mean() < numeric(
        df.loc[df["scenario_id"] == "S0", "lon_180"]
    ).mean(), "S2 mean longitude is not west of S0"
    assert stats.loc["S3", "duration_h"] > stats.loc["S0", "duration_h"], "S3 duration is not longer than S0"
    assert stats.loc["S4", "mean_wind"] > stats.loc["S0", "mean_wind"], "S4 mean wind is not higher than S0"
    assert stats.loc["S4", "duration_h"] > stats.loc["S0", "duration_h"], "S4 duration is not longer than S0"
    assert numeric(df.loc[df["scenario_id"] == "S4", "lon_180"]).mean() < numeric(
        df.loc[df["scenario_id"] == "S0", "lon_180"]
    ).mean(), "S4 mean longitude is not west of S0"

    owd_cols = [col for col in df.columns if col.upper() == "OWD"]
    assert not owd_cols, f"Forbidden OWD columns remain: {owd_cols}"
    leak_cols = leakage_columns(df.columns)
    assert not leak_cols, f"Leakage columns remain: {leak_cols}"

    key_fields = [
        "scenario_id",
        "time",
        "lat",
        "lon_180",
        "wind",
        "pressure",
        "move_speed_kmh",
        "move_dir_deg",
    ]
    missing_cols = [col for col in key_fields if col not in df.columns]
    assert not missing_cols, f"Missing key fields: {missing_cols}"
    nan_counts = df[key_fields].isna().sum()
    bad_nan = nan_counts[nan_counts > 0].to_dict()
    assert not bad_nan, f"Key fields contain NaN: {bad_nan}"


def print_mapping(mapping: FieldMap) -> None:
    print("Detected field mapping:")
    for field in mapping.__dataclass_fields__:
        print(f"  {field}: {getattr(mapping, field)}")


def print_constraints(constraints: ScenarioConstraints) -> None:
    print("Historical constraint source:", constraints.source)
    print("  history rows:", constraints.history_rows)
    print("  H rows after month / lat / environment:", constraints.history_rows_after_month, constraints.history_rows_after_lat, constraints.history_rows_after_environment)
    print("  wind Q50/Q75/Q95:", constraints.wind_q50, constraints.wind_q75, constraints.wind_q95)
    print("  pressure Q05/Q25/Q50:", constraints.pressure_q05, constraints.pressure_q25, constraints.pressure_q50)
    print("  move speed Q25/Q50 and gamma:", constraints.move_speed_q25, constraints.move_speed_q50, constraints.gamma)
    print("  pressure adjustment:", constraints.pressure_adjustment_source)


def main() -> None:
    print("Constructing Problem-3 virtual scenario inputs...")

    raw, mapping, input_path, attempts = choose_target_input()
    print("Input candidate check:")
    for row in attempts:
        print(f"  {row['status']}: {row['path']} ({row['reason']})")
    print("Using input file:", input_path)
    print_mapping(mapping)

    base = standardize_target_input(raw, mapping)
    base_rows = len(base)
    calib = infer_derived_calibration(base)
    print(f"Base {BASE_NAME} rows:", base_rows)
    print("Base time span:", base["time"].min(), "to", base["time"].max())

    constraints = build_history_constraints(base)
    print_constraints(constraints)

    s0 = annotate_scenario(base, "S0", constraints, coast_recomputed=False)
    s1_base = apply_intensity_enhancement(base, constraints, calib)
    s1 = annotate_scenario(s1_base, "S1", constraints, coast_recomputed=False)
    s2_base = apply_westward_path_shift(base, calib)
    s2 = annotate_scenario(s2_base, "S2", constraints, coast_recomputed=False)
    s3_base = stretch_time_axis(base, constraints.gamma, calib)
    s3 = annotate_scenario(s3_base, "S3", constraints, coast_recomputed=False)
    s4_step1 = apply_intensity_enhancement(base, constraints, calib)
    s4_step2 = apply_westward_path_shift(s4_step1, calib)
    s4_base = stretch_time_axis(s4_step2, constraints.gamma, calib)
    s4 = annotate_scenario(s4_base, "S4", constraints, coast_recomputed=False)

    scenario_df = pd.concat([s0, s1, s2, s3, s4], ignore_index=True, sort=False)
    scenario_order = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
    scenario_df["_scenario_order"] = scenario_df["scenario_id"].map(scenario_order)
    scenario_df = scenario_df.sort_values(["_scenario_order", "time"]).drop(columns=["_scenario_order"]).reset_index(drop=True)

    scenario_df, owd_cols, leak_cols = drop_forbidden_columns(scenario_df)
    if owd_cols:
        print("Removed OWD columns:", owd_cols)
    else:
        print("Removed OWD columns: []")
    if leak_cols:
        print("Removed leakage columns:", leak_cols)
    else:
        print("Removed leakage columns: []")

    scenario_df = reorder_columns(scenario_df)
    summary_df = make_summary(scenario_df)

    OUTPUT_INPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(OUTPUT_INPUTS_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    assert_hard_rules(scenario_df, summary_df, base_rows)

    print("Scenario rows:")
    print(scenario_df.groupby("scenario_id").size().to_string())
    print("Scenario summary:")
    print(
        summary_df[
            [
                "scenario_id",
                "n_rows",
                "duration_h",
                "mean_wind",
                "max_wind",
                "min_pressure",
                "mean_move_speed_kmh",
                "max_near_coast_index",
            ]
        ].to_string(index=False)
    )
    print("Output:", OUTPUT_INPUTS_PATH)
    print("Output:", OUTPUT_SUMMARY_PATH)
    print("problem3_scenario_inputs.csv shape:", scenario_df.shape)
    print("problem3_scenario_summary.csv shape:", summary_df.shape)
    print("Hard rule checks: PASS")


if __name__ == "__main__":
    main()
