#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem 3 step 1: virtual scenario design and input audit.

This script builds only leakage-safe scenario input tables. It does not train
models, generate rainfall fields, or modify Problem-2 outputs.
"""

from __future__ import annotations

import importlib.util
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "problem3"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "problem3"

TARGET_SAFE_INPUT_PATH = PROCESSED_DIR / "problem2_env" / "problem2_target_halfhour_inputs_safe_env.csv"
TARGET_ENV_SOURCE_PATH = PROCESSED_DIR / "env_added" / "target_typhoon_inputs_2024_halfhour_leakage_safe_env.csv"
HISTORY_PATH = PROCESSED_DIR / "problem2_env" / "problem2_historical_halfhour_sample_library_env.csv"
SCRIPT27_PATH = PROJECT_ROOT / "scripts" / "27_add_landfrac_terrain_features.py"
NATURAL_EARTH_LAND_PATH = PROJECT_ROOT / "data" / "external" / "naturalearth" / "ne_50m_land" / "ne_50m_land.shp"
DEM_PATH = PROJECT_ROOT / "data" / "external" / "etopo2022" / "ETOPO_2022_v1_30s_N90W180_bed.tif"
ENV_CACHE_DIR = PROCESSED_DIR / "env_cache"

SCENARIO_INPUT_PATH = PROCESSED_DIR / "problem3_scenario_inputs.csv"
SCENARIO_SUMMARY_PATH = PROCESSED_DIR / "problem3_scenario_summary.csv"
DESIGN_TABLE_PATH = TABLE_DIR / "problem3_scenario_design_table.csv"
VALIDITY_AUDIT_PATH = TABLE_DIR / "problem3_scenario_validity_audit.csv"
REPORT_PATH = REPORT_DIR / "problem3_step1_scenario_design_report.md"

BASE_TYPHOON = "KONG-REY"
BASE_TYPHOON_NORM = "KONGREY"
KM_PER_DEG = 111.32
MIN_COMPARABLE_ROWS = 500
ENV_COLS = [
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_std_300km",
    "terrain_max_300km",
]

BASE_REQUIRED_COLS = [
    "target_id",
    "typhoon_name",
    "target_name_norm",
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
]

SAFE_EXTRA_COLS = [
    "year",
    "month",
    "day",
    "hour",
    "season",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "move_dir_sin",
    "move_dir_cos",
    "life_time_index",
    "life_progress",
    "target_time_window_flag",
    "safe_input_flag",
]

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

SCENARIO_META = {
    "S0": {
        "scenario_name": "baseline_kong_rey",
        "description": "KONG-REY Problem-2 leakage-safe half-hour input without perturbation.",
        "changed": "none",
        "recomputed": "none; baseline Problem-2 inputs retained",
    },
    "S1": {
        "scenario_name": "intensity_enhanced",
        "description": "Only WND is increased under historical support and PRES is lowered by the historical WND-PRES fit.",
        "changed": "WND; PRES",
        "recomputed": "wind_change_rate; pressure_change_rate; time/life aliases",
    },
    "S2": {
        "scenario_name": "near_coast_westward_path_shift",
        "description": "The whole track is shifted 100 km westward to represent a closer-to-coast path.",
        "changed": "lon; lon_180",
        "recomputed": "move_distance_km; move_speed_kmh; move_dir_deg; coast_dist_km; signed_coast_dist_km; is_land; landfrac/terrain; derived time/life aliases",
    },
    "S3": {
        "scenario_name": "nearshore_segment_landfall_shift",
        "description": "A smooth local northward displacement is applied near the closest-coast segment to shift the nearshore/landfall point.",
        "changed": "lat",
        "recomputed": "move_distance_km; move_speed_kmh; move_dir_deg; coast_dist_km; signed_coast_dist_km; is_land; landfrac/terrain; derived time/life aliases",
    },
    "S4": {
        "scenario_name": "slower_translation",
        "description": "The path geometry is preserved while the time axis is stretched and half-hour samples are regenerated.",
        "changed": "time axis; lat/lon sampling along the same path",
        "recomputed": "move_distance_km; move_speed_kmh; move_dir_deg; wind_change_rate; pressure_change_rate; coast_dist_km; signed_coast_dist_km; is_land; landfrac/terrain; time/life aliases",
    },
    "S5": {
        "scenario_name": "compound_moderate_high_risk",
        "description": "A compound scenario combines half of the S1 intensity increment, half of the S2 westward shift, and half of the S4 slowdown effect.",
        "changed": "WND; PRES; lon; lon_180; time axis",
        "recomputed": "move_distance_km; move_speed_kmh; move_dir_deg; wind_change_rate; pressure_change_rate; coast_dist_km; signed_coast_dist_km; is_land; landfrac/terrain; time/life aliases",
    },
}


@dataclass
class HistoricalContext:
    sample: pd.DataFrame
    filter_note: str
    total_valid_rows: int
    strict_rows: int
    relaxed_month_rows: int
    broad_wnp_rows: int
    wind_q01: float
    wind_q05: float
    wind_q50: float
    wind_q75: float
    wind_q95: float
    wind_q99: float
    wind_min: float
    wind_max: float
    pres_q01: float
    pres_q05: float
    pres_q50: float
    pres_min: float
    pres_max: float
    speed_q01: float
    speed_q05: float
    speed_q25: float
    speed_q50: float
    speed_q75: float
    speed_q95: float
    speed_q99: float
    speed_min: float
    speed_max: float
    coast_min: float
    coast_max: float
    pressure_wind_intercept: float | None
    pressure_wind_slope: float | None
    full_wind_increment: float
    full_west_shift_km: float
    local_shift_km: float
    speed_gamma_full: float
    speed_gamma_compound: float


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def warn(msg: str, warning_messages: list[str]) -> None:
    warning_messages.append(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    return df


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def lon_to_180(values: object) -> pd.Series:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return ((s + 180.0) % 360.0) - 180.0


def lon_to_360(values: object) -> pd.Series:
    s = lon_to_180(values)
    return s.where(s >= 0.0, s + 360.0)


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


def leakage_columns(columns: Iterable[str]) -> list[str]:
    bad = []
    for col in columns:
        if str(col).upper() == "OWD":
            bad.append(str(col))
            continue
        if any(pattern.search(str(col)) for pattern in LEAKAGE_PATTERNS):
            bad.append(str(col))
    return bad


def finite_quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def percentile_of_score(values: pd.Series, score: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0 or not np.isfinite(score):
        return np.nan
    return float(100.0 * np.searchsorted(np.sort(arr), score, side="right") / arr.size)


def require_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{rel(path)} is missing required columns: {missing}")


def load_base_input(warning_messages: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    base = read_csv(TARGET_SAFE_INPUT_PATH)
    require_columns(base, BASE_REQUIRED_COLS, TARGET_SAFE_INPUT_PATH)
    base["time"] = pd.to_datetime(base["time"], errors="coerce")
    name_col = "target_name_norm" if "target_name_norm" in base.columns else "typhoon_name"
    sub = base.loc[base[name_col].map(normalize_name).eq(BASE_TYPHOON_NORM)].copy()
    sub = sub.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f"No {BASE_TYPHOON} rows found in {rel(TARGET_SAFE_INPUT_PATH)}")

    env_read_paths: list[Path] = [TARGET_SAFE_INPUT_PATH]
    if TARGET_ENV_SOURCE_PATH.exists():
        env_src = read_csv(TARGET_ENV_SOURCE_PATH)
        env_read_paths.append(TARGET_ENV_SOURCE_PATH)
        env_src["time"] = pd.to_datetime(env_src["time"], errors="coerce")
        env_name = "target_name_norm" if "target_name_norm" in env_src.columns else None
        if env_name is not None and set(ENV_COLS).issubset(env_src.columns):
            env_sub = env_src.loc[env_src[env_name].map(normalize_name).eq(BASE_TYPHOON_NORM)].copy()
            keep = ["time", env_name] + ENV_COLS
            env_sub = env_sub[keep].drop_duplicates(subset=["time", env_name])
            sub = sub.merge(
                env_sub.rename(columns={env_name: "_env_name_norm"}),
                left_on=["time", "target_name_norm"],
                right_on=["time", "_env_name_norm"],
                how="left",
            )
            sub = sub.drop(columns=["_env_name_norm"], errors="ignore")
        else:
            warn(
                f"{rel(TARGET_ENV_SOURCE_PATH)} lacks target_name_norm or complete ENV_COLS; baseline env columns will be recomputed if needed.",
                warning_messages,
            )
    else:
        warn(f"Optional env source missing: {rel(TARGET_ENV_SOURCE_PATH)}", warning_messages)

    for col in ENV_COLS:
        if col not in sub.columns:
            sub[col] = np.nan

    for col in ["lat", "lon_180", "WND", "PRES", "move_speed_kmh", "move_dir_deg", "coast_dist_km", "signed_coast_dist_km"]:
        sub[col] = numeric(sub[col])
    sub["lon_180"] = lon_to_180(sub["lon_180"]).to_numpy()
    sub["lon"] = lon_to_360(sub["lon_180"]).to_numpy()
    sub["base_typhoon"] = BASE_TYPHOON
    sub["base_target_id"] = sub["target_id"].astype(str)
    return refresh_all_features(sub, recompute_motion=False, recompute_rates=False), env_read_paths


def load_historical_context(base: pd.DataFrame) -> HistoricalContext:
    usecols = [
        "typhoon_name",
        "time",
        "lat",
        "lon_180",
        "WND",
        "PRES",
        "move_speed_kmh",
        "coast_dist_km",
        "signed_coast_dist_km",
        "is_land",
        "landfrac_500km",
    ]
    history_all = read_csv(HISTORY_PATH)
    missing = [col for col in usecols if col not in history_all.columns]
    if missing:
        raise ValueError(f"{rel(HISTORY_PATH)} is missing historical constraint columns: {missing}")
    hist = history_all[usecols].copy()
    hist["time"] = pd.to_datetime(hist["time"], errors="coerce")
    hist["month"] = hist["time"].dt.month
    for col in ["lat", "lon_180", "WND", "PRES", "move_speed_kmh", "coast_dist_km", "signed_coast_dist_km"]:
        hist[col] = numeric(hist[col])
    hist["lon_180"] = lon_to_180(hist["lon_180"]).to_numpy()
    hist["_name_norm"] = hist["typhoon_name"].map(normalize_name)
    valid = hist.dropna(subset=["time", "lat", "lon_180", "WND", "PRES", "move_speed_kmh"]).copy()
    valid = valid.loc[~valid["_name_norm"].isin({BASE_TYPHOON_NORM, "MANYI"})].copy()

    base_lat_min = float(base["lat"].min() - 5.0)
    base_lat_max = float(base["lat"].max() + 5.0)
    base_lon_min = float(base["lon_180"].min() - 15.0)
    base_lon_max = float(base["lon_180"].max() + 15.0)
    strict_mask = (
        valid["month"].isin([9, 10, 11])
        & valid["lat"].between(base_lat_min, base_lat_max)
        & valid["lon_180"].between(base_lon_min, base_lon_max)
    )
    relaxed_mask = (
        valid["month"].isin([8, 9, 10, 11])
        & valid["lat"].between(base_lat_min, base_lat_max)
        & valid["lon_180"].between(base_lon_min, base_lon_max)
    )
    broad_mask = valid["month"].isin([8, 9, 10, 11]) & valid["lat"].between(0.0, 45.0) & valid["lon_180"].between(100.0, 180.0)

    strict = valid.loc[strict_mask].copy()
    relaxed = valid.loc[relaxed_mask].copy()
    broad = valid.loc[broad_mask].copy()
    if len(strict) >= MIN_COMPARABLE_ROWS:
        selected = strict
        filter_note = (
            "strict: months 9-11, latitude within KONG-REY range +/-5 deg, "
            "longitude within KONG-REY range +/-15 deg"
        )
    elif len(relaxed) >= MIN_COMPARABLE_ROWS:
        selected = relaxed
        filter_note = (
            "relaxed-month: months 8-11, latitude within KONG-REY range +/-5 deg, "
            "longitude within KONG-REY range +/-15 deg"
        )
    elif len(broad) >= MIN_COMPARABLE_ROWS:
        selected = broad
        filter_note = "broad WNP: months 8-11, lat 0-45N, lon 100-180E"
    else:
        selected = valid
        filter_note = "fallback: all valid historical rows because WNP seasonal filters were too small"

    fit = selected[["WND", "PRES"]].dropna()
    intercept: float | None = None
    slope: float | None = None
    if len(fit) >= 2 and float(fit["WND"].std()) > 1e-9:
        slope_fit, intercept_fit = np.polyfit(fit["WND"].to_numpy(dtype=float), fit["PRES"].to_numpy(dtype=float), 1)
        if np.isfinite(slope_fit) and np.isfinite(intercept_fit) and slope_fit < 0.0:
            slope = float(slope_fit)
            intercept = float(intercept_fit)

    base_mean_speed = float(base["move_speed_kmh"].mean())
    speed_q50 = finite_quantile(selected["move_speed_kmh"], 0.50)
    gamma_full = float(np.clip(speed_q50 / base_mean_speed, 0.65, 0.80)) if base_mean_speed > 1e-9 else 0.75
    gamma_compound = float(1.0 - 0.5 * (1.0 - gamma_full))
    wind_q75 = finite_quantile(selected["WND"], 0.75)
    wind_q95 = finite_quantile(selected["WND"], 0.95)
    full_increment = float(np.clip(0.5 * (wind_q95 - wind_q75), 4.0, 8.0))

    return HistoricalContext(
        sample=selected.reset_index(drop=True),
        filter_note=filter_note,
        total_valid_rows=int(len(valid)),
        strict_rows=int(len(strict)),
        relaxed_month_rows=int(len(relaxed)),
        broad_wnp_rows=int(len(broad)),
        wind_q01=finite_quantile(selected["WND"], 0.01),
        wind_q05=finite_quantile(selected["WND"], 0.05),
        wind_q50=finite_quantile(selected["WND"], 0.50),
        wind_q75=wind_q75,
        wind_q95=wind_q95,
        wind_q99=finite_quantile(selected["WND"], 0.99),
        wind_min=float(selected["WND"].min(skipna=True)),
        wind_max=float(selected["WND"].max(skipna=True)),
        pres_q01=finite_quantile(selected["PRES"], 0.01),
        pres_q05=finite_quantile(selected["PRES"], 0.05),
        pres_q50=finite_quantile(selected["PRES"], 0.50),
        pres_min=float(selected["PRES"].min(skipna=True)),
        pres_max=float(selected["PRES"].max(skipna=True)),
        speed_q01=finite_quantile(selected["move_speed_kmh"], 0.01),
        speed_q05=finite_quantile(selected["move_speed_kmh"], 0.05),
        speed_q25=finite_quantile(selected["move_speed_kmh"], 0.25),
        speed_q50=speed_q50,
        speed_q75=finite_quantile(selected["move_speed_kmh"], 0.75),
        speed_q95=finite_quantile(selected["move_speed_kmh"], 0.95),
        speed_q99=finite_quantile(selected["move_speed_kmh"], 0.99),
        speed_min=float(selected["move_speed_kmh"].min(skipna=True)),
        speed_max=float(selected["move_speed_kmh"].max(skipna=True)),
        coast_min=float(selected["coast_dist_km"].min(skipna=True)),
        coast_max=float(selected["coast_dist_km"].max(skipna=True)),
        pressure_wind_intercept=intercept,
        pressure_wind_slope=slope,
        full_wind_increment=full_increment,
        full_west_shift_km=100.0,
        local_shift_km=75.0,
        speed_gamma_full=gamma_full,
        speed_gamma_compound=gamma_compound,
    )


class CoastlineSampler:
    def __init__(self, shp_path: Path) -> None:
        if not shp_path.exists():
            raise FileNotFoundError(f"Natural Earth land shapefile not found: {shp_path}")
        geoms = []
        reader = shapefile.Reader(str(shp_path))
        for shp in reader.shapes():
            try:
                geoms.append(shape(shp.__geo_interface__))
            except Exception:
                continue
        if not geoms:
            raise RuntimeError(f"No valid land polygons found in {shp_path}")
        self.land = unary_union(geoms)
        self.boundary = self.land.boundary
        self.prepared = prep(self.land)

    def signed_distance(self, lon: float, lat: float) -> tuple[float, float, int]:
        if not all(np.isfinite([lon, lat])):
            return np.nan, np.nan, 0
        lon180 = float(lon_to_180([lon]).iloc[0])
        point = Point(lon180, float(lat))
        is_land = bool(self.prepared.contains(point) or self.land.touches(point))
        nearest = nearest_points(point, self.boundary)[1]
        distance = haversine_km(lon180, float(lat), float(nearest.x), float(nearest.y))
        signed = -distance if is_land else distance
        return distance, signed, int(is_land)


class EnvironmentSampler:
    def __init__(self) -> None:
        if not SCRIPT27_PATH.exists():
            raise FileNotFoundError(f"Environment script missing: {SCRIPT27_PATH}")
        spec = importlib.util.spec_from_file_location("problem3_env_tools", SCRIPT27_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {SCRIPT27_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.landmask = module.build_or_load_landmask(
            shp_path=NATURAL_EARTH_LAND_PATH,
            cache_dir=ENV_CACHE_DIR,
            res_deg=0.05,
        )
        self.terrain = module.TerrainSampler(DEM_PATH)
        self.cache: dict[tuple[float, float], dict[str, float]] = {}

    def close(self) -> None:
        self.terrain.close()

    def compute(self, lat: float, lon: float) -> dict[str, float]:
        if not np.isfinite(lat) or not np.isfinite(lon):
            return {col: np.nan for col in ENV_COLS}
        lon180 = float(lon_to_180([lon]).iloc[0])
        key = (round(round(float(lat) / 0.02) * 0.02, 5), round(round(lon180 / 0.02) * 0.02, 5))
        if key not in self.cache:
            result = self.module.compute_env_one_center(
                lat=key[0],
                lon=key[1],
                landmask=self.landmask,
                terrain=self.terrain,
                sample_step_km=50,
            )
            self.cache[key] = {col: float(result.get(col, np.nan)) for col in ENV_COLS}
        return self.cache[key]


def compute_motion(df: pd.DataFrame, overwrite_motion: bool = True, overwrite_rates: bool = True) -> pd.DataFrame:
    out = df.sort_values("time").reset_index(drop=True).copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    lon = numeric(out["lon_180"]).to_numpy(dtype=float)
    lat = numeric(out["lat"]).to_numpy(dtype=float)
    wind = numeric(out["WND"]).to_numpy(dtype=float)
    pres = numeric(out["PRES"]).to_numpy(dtype=float)
    n = len(out)
    dt_h = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    wind_rate = np.full(n, np.nan)
    pres_rate = np.full(n, np.nan)
    for i in range(1, n):
        delta = (time.iloc[i] - time.iloc[i - 1]).total_seconds() / 3600.0
        dt_h[i] = delta
        if np.isfinite(delta) and delta > 0.0:
            dist[i] = haversine_km(lon[i - 1], lat[i - 1], lon[i], lat[i])
            speed[i] = dist[i] / delta if np.isfinite(dist[i]) else np.nan
            direction[i] = bearing_deg(lon[i - 1], lat[i - 1], lon[i], lat[i])
            if np.isfinite(wind[i]) and np.isfinite(wind[i - 1]):
                wind_rate[i] = (wind[i] - wind[i - 1]) / delta
            if np.isfinite(pres[i]) and np.isfinite(pres[i - 1]):
                pres_rate[i] = (pres[i] - pres[i - 1]) / delta
    if n > 1:
        for arr in [dt_h, dist, speed, direction, wind_rate, pres_rate]:
            arr[0] = arr[1]
    elif n == 1:
        dt_h[0] = 0.5
        dist[0] = 0.0
        speed[0] = 0.0
        direction[0] = 0.0
        wind_rate[0] = 0.0
        pres_rate[0] = 0.0

    values = {
        "dt_h": dt_h,
        "move_distance_km": dist,
        "move_speed_kmh": speed,
        "move_dir_deg": direction,
        "wind_change_rate": wind_rate,
        "pressure_change_rate": pres_rate,
    }
    for col, arr in values.items():
        overwrite = overwrite_motion if col in {"dt_h", "move_distance_km", "move_speed_kmh", "move_dir_deg"} else overwrite_rates
        if overwrite or col not in out.columns:
            out[col] = arr
        else:
            current = numeric(out[col])
            out[col] = current.where(current.notna(), arr)
    return out


def refresh_all_features(df: pd.DataFrame, recompute_motion: bool = True, recompute_rates: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["lon_180"] = lon_to_180(out["lon_180"]).to_numpy()
    out["lon"] = lon_to_360(out["lon_180"]).to_numpy()
    out = compute_motion(out, overwrite_motion=recompute_motion, overwrite_rates=recompute_rates)
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
    move_rad = np.deg2rad(numeric(out["move_dir_deg"]))
    out["move_dir_sin"] = np.sin(move_rad)
    out["move_dir_cos"] = np.cos(move_rad)
    out["life_time_index"] = np.arange(len(out), dtype=int)
    out["life_progress"] = out["life_time_index"] / float(len(out) - 1) if len(out) > 1 else 0.0
    signed = numeric(out["signed_coast_dist_km"]) if "signed_coast_dist_km" in out.columns else pd.Series(np.nan, index=out.index)
    out["near_coast_index"] = np.exp(-signed.abs() / 200.0)
    out["coast_influence_exp"] = out["near_coast_index"]
    out["is_near_coast_100km"] = ((signed >= -100.0) & (signed <= 100.0)).astype(int)
    out["is_near_coast_200km"] = ((signed >= -200.0) & (signed <= 200.0)).astype(int)
    out["is_offshore_far_300km"] = (signed > 300.0).astype(int)
    out["is_inland_100km"] = (signed < -100.0).astype(int)
    return out


def recompute_spatial_environment(
    df: pd.DataFrame,
    coastline: CoastlineSampler,
    env_sampler: EnvironmentSampler,
    recompute_coast: bool,
    recompute_env: bool,
) -> pd.DataFrame:
    out = df.copy()
    if recompute_coast:
        coast_records = [coastline.signed_distance(lon, lat) for lon, lat in out[["lon_180", "lat"]].itertuples(index=False, name=None)]
        coast = pd.DataFrame(coast_records, columns=["coast_dist_km", "signed_coast_dist_km", "is_land"])
        for col in coast.columns:
            out[col] = coast[col].to_numpy()
    if recompute_env or out[ENV_COLS].isna().any().any():
        env_records = [env_sampler.compute(lat, lon) for lat, lon in out[["lat", "lon_180"]].itertuples(index=False, name=None)]
        env = pd.DataFrame(env_records)
        for col in ENV_COLS:
            out[col] = env[col].to_numpy()
    out = refresh_all_features(out, recompute_motion=False, recompute_rates=False)
    return out


def apply_intensity(df: pd.DataFrame, ctx: HistoricalContext, fraction: float) -> pd.DataFrame:
    out = df.copy()
    wind_old = numeric(out["WND"]).to_numpy(dtype=float)
    pres_old = numeric(out["PRES"]).to_numpy(dtype=float)
    delta_target = ctx.full_wind_increment * fraction
    wind_new = np.minimum(wind_old + delta_target, ctx.wind_q99)
    wind_new = np.maximum(wind_old, wind_new)
    delta = wind_new - wind_old
    if ctx.pressure_wind_slope is not None and ctx.pressure_wind_slope < 0.0:
        pres_new = pres_old + ctx.pressure_wind_slope * delta
    else:
        pres_new = pres_old - 2.0 * delta
    pres_new = np.maximum(pres_new, ctx.pres_q01)
    out["WND"] = wind_new
    out["PRES"] = pres_new
    out = refresh_all_features(out, recompute_motion=False, recompute_rates=True)
    return out


def shift_west(df: pd.DataFrame, shift_km: float, weights: np.ndarray | None = None) -> pd.DataFrame:
    out = df.copy()
    lat = numeric(out["lat"]).to_numpy(dtype=float)
    lon = numeric(out["lon_180"]).to_numpy(dtype=float)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = np.where(np.abs(cos_lat) < 0.1, 0.1, cos_lat)
    weight = np.ones(len(out), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    lon_shift = shift_km * weight / (KM_PER_DEG * cos_lat)
    out["lon_180"] = lon_to_180(lon - lon_shift).to_numpy()
    out["lon"] = lon_to_360(out["lon_180"]).to_numpy()
    return refresh_all_features(out, recompute_motion=True, recompute_rates=False)


def shift_nearshore_segment(df: pd.DataFrame, shift_km: float) -> pd.DataFrame:
    out = df.copy()
    time = pd.to_datetime(out["time"], errors="coerce")
    signed_abs = numeric(out["signed_coast_dist_km"]).abs()
    center_idx = int(signed_abs.idxmin())
    center_time = time.loc[center_idx]
    hours = (time - center_time).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    sigma_h = 24.0
    weights = np.exp(-0.5 * (hours / sigma_h) ** 2)
    out["lat"] = numeric(out["lat"]).to_numpy(dtype=float) + (shift_km / KM_PER_DEG) * weights
    return refresh_all_features(out, recompute_motion=True, recompute_rates=False)


def cumulative_path_distance(df: pd.DataFrame) -> np.ndarray:
    lon = numeric(df["lon_180"]).to_numpy(dtype=float)
    lat = numeric(df["lat"]).to_numpy(dtype=float)
    step = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        dist = haversine_km(lon[i - 1], lat[i - 1], lon[i], lat[i])
        step[i] = dist if np.isfinite(dist) else 0.0
    path = np.cumsum(step)
    if len(path) and path[-1] <= 1e-9:
        path = np.arange(len(df), dtype=float)
    return path


def interp_numeric(path: np.ndarray, values: pd.Series, new_path: np.ndarray) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(path) & np.isfinite(y)
    if mask.sum() == 0:
        return np.full(len(new_path), np.nan)
    if mask.sum() == 1:
        return np.full(len(new_path), float(y[mask][0]))
    unique_x, unique_idx = np.unique(path[mask], return_index=True)
    if len(unique_x) == 1:
        return np.full(len(new_path), float(y[mask][unique_idx[0]]))
    return np.interp(new_path, unique_x, y[mask][unique_idx])


def interp_lon(path: np.ndarray, values: pd.Series, new_path: np.ndarray) -> np.ndarray:
    lon = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(path) & np.isfinite(lon)
    if mask.sum() == 0:
        return np.full(len(new_path), np.nan)
    if mask.sum() == 1:
        return np.full(len(new_path), float(lon[mask][0]))
    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon[mask])))
    unique_x, unique_idx = np.unique(path[mask], return_index=True)
    if len(unique_x) == 1:
        return np.full(len(new_path), float(lon[mask][unique_idx[0]]))
    return lon_to_180(np.interp(new_path, unique_x, lon_unwrapped[unique_idx])).to_numpy()


def nearest_indices(path: np.ndarray, new_path: np.ndarray) -> np.ndarray:
    idx_right = np.searchsorted(path, new_path, side="left")
    idx_right = np.clip(idx_right, 0, len(path) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(path) - 1)
    choose_left = np.abs(new_path - path[idx_left]) <= np.abs(path[idx_right] - new_path)
    return np.where(choose_left, idx_left, idx_right)


def stretch_time_axis(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    source = df.sort_values("time").reset_index(drop=True).copy()
    time = pd.to_datetime(source["time"], errors="coerce")
    old_duration_h = (time.iloc[-1] - time.iloc[0]).total_seconds() / 3600.0
    if not np.isfinite(old_duration_h) or old_duration_h <= 0.0:
        raise RuntimeError("Cannot slow a track with non-positive duration.")
    new_duration_h = old_duration_h / gamma
    elapsed_h = np.arange(0.0, math.ceil(new_duration_h / 0.5) * 0.5 + 0.001, 0.5)
    new_time = time.iloc[0] + pd.to_timedelta(elapsed_h, unit="h")
    path = cumulative_path_distance(source)
    old_elapsed_h = (time - time.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    mapped_elapsed = np.clip(elapsed_h * gamma, 0.0, old_duration_h)
    new_path = np.interp(mapped_elapsed, old_elapsed_h, path)
    nearest = nearest_indices(path, new_path)
    out = source.iloc[nearest].copy().reset_index(drop=True)
    out["time"] = new_time
    for col in ["lat", "WND", "PRES"]:
        out[col] = interp_numeric(path, source[col], new_path)
    if "intensity" in source.columns:
        out["intensity"] = np.rint(interp_numeric(path, source["intensity"], new_path))
    out["lon_180"] = interp_lon(path, source["lon_180"], new_path)
    out["lon"] = lon_to_360(out["lon_180"]).to_numpy()
    return refresh_all_features(out, recompute_motion=True, recompute_rates=True)


def annotate(df: pd.DataFrame, scenario_id: str, construction_note: str) -> pd.DataFrame:
    meta = SCENARIO_META[scenario_id]
    out = df.copy()
    out["scenario_id"] = scenario_id
    out["scenario_name"] = meta["scenario_name"]
    out["base_typhoon"] = BASE_TYPHOON
    out["typhoon_name"] = f"{BASE_TYPHOON}_{scenario_id}"
    out["target_name_norm"] = f"{BASE_TYPHOON_NORM}_{scenario_id}"
    out["scenario_time_index"] = np.arange(len(out), dtype=int)
    out["target_id"] = scenario_id + "_" + pd.to_datetime(out["time"]).dt.strftime("%Y%m%d%H%M%S")
    out["scenario_description"] = meta["description"]
    out["changed_variables"] = meta["changed"]
    out["recomputed_variables"] = meta["recomputed"]
    out["construction_note"] = construction_note
    return out


def build_scenarios(base: pd.DataFrame, ctx: HistoricalContext, warning_messages: list[str]) -> pd.DataFrame:
    coastline = CoastlineSampler(NATURAL_EARTH_LAND_PATH)
    env_sampler = EnvironmentSampler()
    try:
        base_env_missing = base[ENV_COLS].isna().any().any()
        s0_base = recompute_spatial_environment(base, coastline, env_sampler, recompute_coast=False, recompute_env=base_env_missing)
        s1_base = apply_intensity(s0_base, ctx, fraction=1.0)
        s1_base = recompute_spatial_environment(s1_base, coastline, env_sampler, recompute_coast=False, recompute_env=False)

        s2_base = shift_west(s0_base, ctx.full_west_shift_km)
        s2_base = recompute_spatial_environment(s2_base, coastline, env_sampler, recompute_coast=True, recompute_env=True)

        s3_base = shift_nearshore_segment(s0_base, ctx.local_shift_km)
        s3_base = recompute_spatial_environment(s3_base, coastline, env_sampler, recompute_coast=True, recompute_env=True)

        s4_base = stretch_time_axis(s0_base, ctx.speed_gamma_full)
        s4_base = recompute_spatial_environment(s4_base, coastline, env_sampler, recompute_coast=True, recompute_env=True)

        s5_base = apply_intensity(s0_base, ctx, fraction=0.5)
        s5_base = shift_west(s5_base, ctx.full_west_shift_km * 0.5)
        s5_base = stretch_time_axis(s5_base, ctx.speed_gamma_compound)
        s5_base = recompute_spatial_environment(s5_base, coastline, env_sampler, recompute_coast=True, recompute_env=True)
    finally:
        env_sampler.close()

    if base_env_missing:
        warn("Baseline landfrac/terrain columns were missing or incomplete and were recomputed from local environmental assets.", warning_messages)

    scenario_frames = [
        annotate(s0_base, "S0", "baseline retained from Problem-2 KONG-REY input"),
        annotate(s1_base, "S1", f"full WND increment={ctx.full_wind_increment:.3f} m/s; cap WND at historical P99={ctx.wind_q99:.3f}; pressure floor historical P01={ctx.pres_q01:.3f}"),
        annotate(s2_base, "S2", f"uniform westward shift={ctx.full_west_shift_km:.1f} km; spatial variables recomputed"),
        annotate(s3_base, "S3", f"Gaussian nearshore northward shift peak={ctx.local_shift_km:.1f} km, sigma=24 h; spatial variables recomputed"),
        annotate(s4_base, "S4", f"time-axis speed multiplier gamma={ctx.speed_gamma_full:.3f}; half-hour inputs regenerated"),
        annotate(s5_base, "S5", f"compound: 0.5*S1 intensity increment, 0.5*S2 shift, gamma={ctx.speed_gamma_compound:.3f}"),
    ]
    out = pd.concat(scenario_frames, ignore_index=True, sort=False)
    order = {sid: i for i, sid in enumerate(["S0", "S1", "S2", "S3", "S4", "S5"])}
    out["_scenario_order"] = out["scenario_id"].map(order)
    out = out.sort_values(["_scenario_order", "time", "scenario_time_index"]).drop(columns=["_scenario_order"]).reset_index(drop=True)
    return out


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [
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
        "coast_dist_km",
        "signed_coast_dist_km",
        "landfrac_200km",
        "landfrac_500km",
        "terrain_mean_300km",
        "terrain_std_300km",
        "terrain_max_300km",
        "near_coast_index",
        "coast_influence_exp",
        "is_near_coast_100km",
        "is_near_coast_200km",
        "is_offshore_far_300km",
        "is_inland_100km",
        "year",
        "month",
        "day",
        "hour",
        "season",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "move_dir_sin",
        "move_dir_cos",
        "life_time_index",
        "life_progress",
        "scenario_time_index",
        "scenario_description",
        "changed_variables",
        "recomputed_variables",
        "construction_note",
    ]
    first.extend([col for col in SAFE_EXTRA_COLS if col in df.columns and col not in first])
    ordered = [col for col in first if col in df.columns]
    out = df[ordered].copy()
    bad = leakage_columns(out.columns)
    if bad:
        raise RuntimeError(f"Forbidden leakage/OWD columns would be written: {bad}")
    return out


def make_validity_audit(df: pd.DataFrame, ctx: HistoricalContext) -> pd.DataFrame:
    hist = ctx.sample
    records = []
    support_cols = {
        "WND": (ctx.wind_min, ctx.wind_max),
        "PRES": (ctx.pres_min, ctx.pres_max),
        "move_speed_kmh": (ctx.speed_min, ctx.speed_max),
        "coast_dist_km": (ctx.coast_min, ctx.coast_max),
    }
    for scenario_id, sub in df.groupby("scenario_id", sort=False):
        checks = []
        variable_counts: dict[str, int] = {}
        for col, (lower, upper) in support_cols.items():
            vals = numeric(sub[col])
            check = vals.lt(lower) | vals.gt(upper)
            count = int(check.fillna(False).sum())
            if count:
                variable_counts[col] = count
            checks.append(check)
        any_out = np.logical_or.reduce([check.fillna(False).to_numpy() for check in checks])
        out_count = int(any_out.sum())
        ratio = float(out_count / len(sub)) if len(sub) else np.nan
        detail = "; ".join(f"{col}={count}" for col, count in variable_counts.items()) if variable_counts else "none"
        if out_count == 0:
            level = "within_historical_min_max"
            note = "All audited values fall within the comparable historical min-max envelope."
        elif set(variable_counts).issubset({"move_speed_kmh"}):
            level = "limited_speed_tail_support"
            note = "Out-of-support values are inherited from the observed KONG-REY translation-speed tail; WND, PRES, and coast distance remain within historical min-max."
        elif set(variable_counts).issubset({"move_speed_kmh", "coast_dist_km"}) and ratio <= 0.05:
            level = "limited_speed_or_coast_tail_support"
            note = "Out-of-support values are limited to the observed high-speed tail and/or near-zero coast distances below the historical sample minimum."
        elif ratio <= 0.02:
            level = "minor_out_of_support"
            note = "A small fraction of audited values falls outside comparable historical min-max."
        else:
            level = "warning_out_of_support"
            note = "More than 2% of audited values falls outside comparable historical min-max; inspect before using in Step 2."
        records.append(
            {
                "scenario_id": scenario_id,
                "n_timesteps": int(len(sub)),
                "WND_min_percentile": percentile_of_score(hist["WND"], float(numeric(sub["WND"]).min(skipna=True))),
                "WND_max_percentile": percentile_of_score(hist["WND"], float(numeric(sub["WND"]).max(skipna=True))),
                "PRES_min_percentile": percentile_of_score(hist["PRES"], float(numeric(sub["PRES"]).min(skipna=True))),
                "PRES_max_percentile": percentile_of_score(hist["PRES"], float(numeric(sub["PRES"]).max(skipna=True))),
                "move_speed_min_percentile": percentile_of_score(hist["move_speed_kmh"], float(numeric(sub["move_speed_kmh"]).min(skipna=True))),
                "move_speed_max_percentile": percentile_of_score(hist["move_speed_kmh"], float(numeric(sub["move_speed_kmh"]).max(skipna=True))),
                "coast_dist_min_percentile": percentile_of_score(hist["coast_dist_km"], float(numeric(sub["coast_dist_km"]).min(skipna=True))),
                "coast_dist_max_percentile": percentile_of_score(hist["coast_dist_km"], float(numeric(sub["coast_dist_km"]).max(skipna=True))),
                "out_of_support_count": out_count,
                "out_of_support_ratio": ratio,
                "out_of_support_variables": detail,
                "validity_level": level,
                "validity_note": note,
            }
        )
    return pd.DataFrame(records)


def make_summary(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    audit_note = audit.set_index("scenario_id")["validity_note"].to_dict()
    records = []
    for scenario_id, sub in df.groupby("scenario_id", sort=False):
        time = pd.to_datetime(sub["time"], errors="coerce")
        records.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": sub["scenario_name"].iloc[0],
                "base_typhoon": BASE_TYPHOON,
                "n_timesteps": int(len(sub)),
                "start_time": time.min(),
                "end_time": time.max(),
                "min_lat": float(numeric(sub["lat"]).min(skipna=True)),
                "max_lat": float(numeric(sub["lat"]).max(skipna=True)),
                "min_lon": float(numeric(sub["lon_180"]).min(skipna=True)),
                "max_lon": float(numeric(sub["lon_180"]).max(skipna=True)),
                "max_WND": float(numeric(sub["WND"]).max(skipna=True)),
                "min_PRES": float(numeric(sub["PRES"]).min(skipna=True)),
                "mean_move_speed_kmh": float(numeric(sub["move_speed_kmh"]).mean(skipna=True)),
                "min_coast_dist_km": float(numeric(sub["coast_dist_km"]).min(skipna=True)),
                "max_landfrac_500km": float(numeric(sub["landfrac_500km"]).max(skipna=True)) if "landfrac_500km" in sub.columns else np.nan,
                "scenario_description": sub["scenario_description"].iloc[0],
                "changed_variables": sub["changed_variables"].iloc[0],
                "recomputed_variables": sub["recomputed_variables"].iloc[0],
                "validity_note": audit_note.get(scenario_id, ""),
            }
        )
    return pd.DataFrame(records)


def make_design_table(ctx: HistoricalContext) -> pd.DataFrame:
    rows = [
        {
            "scenario_id": "S0",
            "scenario_name": SCENARIO_META["S0"]["scenario_name"],
            "control_factor": "baseline",
            "mathematical_operation": "x_S0(t) = x_KONG-REY(t)",
            "historical_constraint": "Observed 2024 KONG-REY target input; no perturbation.",
            "physical_meaning": "Control group for later rainfall generation.",
            "expected_effect": "Reproduce the Problem-2 KONG-REY input state.",
            "paper_usage": "Baseline comparison in Problem 3.",
        },
        {
            "scenario_id": "S1",
            "scenario_name": SCENARIO_META["S1"]["scenario_name"],
            "control_factor": "intensity",
            "mathematical_operation": f"WND'=min(WND+{ctx.full_wind_increment:.2f}, historical WND P99={ctx.wind_q99:.2f}); PRES'=max(PRES+b*dWND, PRES P01={ctx.pres_q01:.2f}), b={ctx.pressure_wind_slope:.3f}" if ctx.pressure_wind_slope is not None else f"WND'=min(WND+{ctx.full_wind_increment:.2f}, historical WND P99={ctx.wind_q99:.2f}); PRES'=max(PRES-2*dWND, PRES P01={ctx.pres_q01:.2f})",
            "historical_constraint": "WND cap and PRES floor from comparable historical sample; WND-PRES relationship estimated by linear fit.",
            "physical_meaning": "A stronger typhoon with dynamically consistent lower central pressure.",
            "expected_effect": "Higher analog intensity while keeping path and motion unchanged.",
            "paper_usage": "Single-factor intensity sensitivity scenario.",
        },
        {
            "scenario_id": "S2",
            "scenario_name": SCENARIO_META["S2"]["scenario_name"],
            "control_factor": "path-coast proximity",
            "mathematical_operation": f"lon'=lon-{ctx.full_west_shift_km:.0f}/(111.32*cos(lat)); spatial covariates recomputed.",
            "historical_constraint": "Shift chosen to keep the path inside WNP comparable longitude/coast-distance support.",
            "physical_meaning": "A track closer to the China coast without changing intensity.",
            "expected_effect": "Greater land/coast interaction for the same storm intensity.",
            "paper_usage": "Single-factor path sensitivity scenario.",
        },
        {
            "scenario_id": "S3",
            "scenario_name": SCENARIO_META["S3"]["scenario_name"],
            "control_factor": "nearshore/landfall segment",
            "mathematical_operation": f"lat'=lat+({ctx.local_shift_km:.0f}/111.32)*exp[-0.5*((t-t0)/24h)^2], t0 at minimum coast distance.",
            "historical_constraint": "Local displacement remains inside the comparable KONG-REY latitude band.",
            "physical_meaning": "A shifted nearshore or landfall segment while preserving most of the track.",
            "expected_effect": "Changes where the highest coastal exposure occurs.",
            "paper_usage": "Landfall-location sensitivity scenario.",
        },
        {
            "scenario_id": "S4",
            "scenario_name": SCENARIO_META["S4"]["scenario_name"],
            "control_factor": "translation speed",
            "mathematical_operation": f"elapsed_old=elapsed_new*gamma, gamma={ctx.speed_gamma_full:.3f}; interpolate path/intensity to half-hour times.",
            "historical_constraint": "Gamma selected from historical median move speed divided by KONG-REY mean move speed and clipped to [0.65,0.80].",
            "physical_meaning": "A slower-moving typhoon along the same path.",
            "expected_effect": "Longer exposure duration without directly amplifying rainfall.",
            "paper_usage": "Single-factor movement-speed sensitivity scenario.",
        },
        {
            "scenario_id": "S5",
            "scenario_name": SCENARIO_META["S5"]["scenario_name"],
            "control_factor": "compound high risk",
            "mathematical_operation": f"0.5*S1 intensity increment + 0.5*S2 westward shift + gamma={ctx.speed_gamma_compound:.3f}.",
            "historical_constraint": "Uses moderate rather than extreme components from S1/S2/S4.",
            "physical_meaning": "A plausible compound future-risk case.",
            "expected_effect": "Jointly stronger, closer, and slower storm input for later rainfall-field generation.",
            "paper_usage": "Main illustrative virtual typhoon scenario in Problem 3 Step 2.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = pd.to_datetime(work[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in work.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in work.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    ctx: HistoricalContext,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    read_paths: list[Path],
    warning_messages: list[str],
) -> None:
    out_support = int(audit["out_of_support_count"].sum()) if not audit.empty else 0
    rows_text = markdown_table(summary[["scenario_id", "n_timesteps", "start_time", "end_time", "max_WND", "min_PRES", "mean_move_speed_kmh", "min_coast_dist_km"]])
    audit_text = markdown_table(audit[["scenario_id", "out_of_support_count", "out_of_support_ratio", "out_of_support_variables", "validity_level"]])
    lines = [
        "# Problem 3 Step 1 Scenario Design Report",
        "",
        "## 1. Inputs read",
    ]
    for path in read_paths:
        lines.append(f"- `{rel(path)}`")
    lines.extend(
        [
            f"- `{rel(NATURAL_EARTH_LAND_PATH)}` for is_land and signed coast distance recalculation",
            f"- `{rel(DEM_PATH)}` plus `{rel(SCRIPT27_PATH)}` for land fraction and terrain recalculation",
            "",
            "## 2. Why KONG-REY is S0",
            "KONG-REY is used as the main baseline because the current Problem-2 chain already has complete half-hour target inputs and final calibrated/geographic rainfall outputs for it. It also has a near-coast segment and a strong intensity evolution, so it is suitable as the main text case for virtual typhoon perturbation. MAN-YI can remain a later robustness comparison.",
            "",
            "## 3. Scenario rules",
            "- S0: unperturbed KONG-REY Problem-2 input.",
            f"- S1: WND is increased by up to {ctx.full_wind_increment:.2f} m/s and capped at historical WND P99={ctx.wind_q99:.2f}; PRES is lowered synchronously using the historical WND-PRES fit.",
            f"- S2: the whole path is shifted {ctx.full_west_shift_km:.0f} km westward; motion, coast distance, land/terrain variables are recomputed.",
            f"- S3: only the closest-coast segment is shifted smoothly northward with a Gaussian peak of {ctx.local_shift_km:.0f} km and sigma 24 h; motion and environmental variables are recomputed.",
            f"- S4: the time axis is stretched with gamma={ctx.speed_gamma_full:.3f}; half-hour input times are regenerated and motion/intensity rates are recomputed.",
            f"- S5: uses half of the S1 intensity increment, half of the S2 westward shift, and a moderate slowdown gamma={ctx.speed_gamma_compound:.3f}.",
            "",
            "## 4. Directly perturbed vs recomputed variables",
            "Direct perturbations are limited to WND/PRES for S1, lon for S2, lat for S3, time-axis/path sampling for S4, and moderate WND/PRES/lon/time-axis changes for S5. For path and time-axis scenarios, move_speed_kmh, move_dir_deg, wind_change_rate, pressure_change_rate, is_land, coast_dist_km, signed_coast_dist_km, landfrac_200km, landfrac_500km, terrain_mean_300km, terrain_std_300km, and terrain_max_300km are recomputed as applicable.",
            "",
            "## 5. WND and PRES handling",
            f"The comparable historical WND-PRES linear fit uses {len(ctx.sample)} rows. The fitted slope is {ctx.pressure_wind_slope if ctx.pressure_wind_slope is not None else 'unavailable'}, so pressure decreases when WND increases. The script caps enhanced WND at historical P99 and floors PRES at historical P01 to avoid unbounded intensification.",
            "",
            "## 6. OWD and leakage exclusions",
            "OWD is absent from the required input set and is not used. The output scenario input table is checked before writing so that OWD, rain_*, centroid_*, anisotropy, asym_*, quad_*, r50/r80/r90, rainband, major/minor axis, orientation, rain_gini, and rain_entropy columns are not written.",
            "",
            "## 7. Historical comparable sample",
            f"- Source: `{rel(HISTORY_PATH)}`",
            f"- Valid historical rows before regional/seasonal filtering: {ctx.total_valid_rows}",
            f"- Strict sample rows: {ctx.strict_rows}",
            f"- Relaxed-month rows: {ctx.relaxed_month_rows}",
            f"- Broad WNP rows: {ctx.broad_wnp_rows}",
            f"- Final sample rule: {ctx.filter_note}",
            f"- Final sample rows: {len(ctx.sample)}",
            "",
            "## 8. Scenario summary",
            rows_text,
            "",
            "## 9. Historical support audit",
            audit_text,
            f"- Total out-of-support audited time steps: {out_support}",
            "",
            "## 10. Next step",
            "Problem 3 Step 2 should feed S0-S5 rows from `data/processed/problem3_scenario_inputs.csv` into the established Problem-2 path-intensity-environment rainfall generation chain, then generate scenario rainfall fields and compute extreme precipitation indicators such as cumulative rainfall, P95/P99, heavy-rain area, and duration.",
        ]
    )
    if warning_messages:
        lines.extend(["", "## 11. Warnings"])
        for msg in warning_messages:
            lines.append(f"- {msg}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hard_checks(df: pd.DataFrame) -> None:
    required = [
        "scenario_id",
        "scenario_name",
        "base_typhoon",
        "time",
        "lat",
        "lon",
        "WND",
        "PRES",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
        "is_land",
        "coast_dist_km",
        "signed_coast_dist_km",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Scenario input missing required fields: {missing}")
    bad = leakage_columns(df.columns)
    if bad:
        raise RuntimeError(f"Scenario input contains forbidden columns: {bad}")
    ids = set(df["scenario_id"].astype(str).unique())
    expected = {"S0", "S1", "S2", "S3", "S4", "S5"}
    if ids != expected:
        raise RuntimeError(f"Scenario IDs mismatch: expected {sorted(expected)}, got {sorted(ids)}")
    for col in ["lat", "lon", "WND", "PRES", "move_speed_kmh", "move_dir_deg", "coast_dist_km"]:
        if numeric(df[col]).isna().any():
            raise RuntimeError(f"Scenario input has NaN in required numeric column: {col}")
    if (numeric(df["WND"]) < 0).any():
        raise RuntimeError("Scenario WND contains negative values.")
    if (numeric(df["coast_dist_km"]) < 0).any():
        raise RuntimeError("Scenario coast_dist_km contains negative values.")


def main() -> None:
    warning_messages: list[str] = []
    print("Problem 3 Step 1 scenario design/audit started.")
    for out_dir in [PROCESSED_DIR, TABLE_DIR, REPORT_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)

    base, read_paths = load_base_input(warning_messages)
    ctx = load_historical_context(base)
    read_paths.append(HISTORY_PATH)

    print(f"Base typhoon: {BASE_TYPHOON}, rows={len(base)}")
    print(f"Historical comparable sample rows: {len(ctx.sample)} ({ctx.filter_note})")
    print(f"S1 full WND increment: {ctx.full_wind_increment:.3f}; WND cap P99={ctx.wind_q99:.3f}")
    print(f"S4 speed gamma: {ctx.speed_gamma_full:.3f}; S5 gamma: {ctx.speed_gamma_compound:.3f}")

    scenarios = build_scenarios(base, ctx, warning_messages)
    scenario_inputs = select_output_columns(scenarios)
    hard_checks(scenario_inputs)
    audit = make_validity_audit(scenario_inputs, ctx)
    summary = make_summary(scenario_inputs, audit)
    design = make_design_table(ctx)

    scenario_inputs.to_csv(SCENARIO_INPUT_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SCENARIO_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    design.to_csv(DESIGN_TABLE_PATH, index=False, encoding="utf-8-sig")
    audit.to_csv(VALIDITY_AUDIT_PATH, index=False, encoding="utf-8-sig")
    write_report(ctx, summary, audit, read_paths, warning_messages)

    rows = scenario_inputs.groupby("scenario_id").size()
    out_support = int(audit["out_of_support_count"].sum())
    status = "WARNING" if warning_messages or out_support > 0 else "SUCCESS"

    print("\nOutput files:")
    for path in [SCENARIO_INPUT_PATH, SCENARIO_SUMMARY_PATH, DESIGN_TABLE_PATH, VALIDITY_AUDIT_PATH, REPORT_PATH]:
        print(f"- {rel(path)}")
    print("\nScenario row counts:")
    print(rows.to_string())
    print(f"\nHistorical comparable sample count: {len(ctx.sample)}")
    print(f"Out-of-support exists: {'YES' if out_support > 0 else 'NO'} (count={out_support})")
    if warning_messages:
        print("Warnings:")
        for msg in warning_messages:
            print(f"- {msg}")
    print(status)


if __name__ == "__main__":
    main()
