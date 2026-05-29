#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 pseudo-missing validation on historical typhoon GPM events.

For each selected historical event, this script hides its GPM rainfall from the
generation stage, retrieves Top-K analogs from the remaining historical library,
generates initial / EOF-blended / calibrated storm-relative rainfall fields,
and evaluates the generated fields against the hidden truth GPM field.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

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
EOF_MODEL_PATH = PROJECT_ROOT / "data/processed/problem2_eof_pca_model.npz"

VALIDATION_EVENTS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_events.csv"
TIMESLICE_METRICS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_timeslice_metrics.csv"
EVENT_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_event_summary.csv"
MODEL_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_model_summary.csv"
GENERATED_FIELDS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_generated_fields.npz"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_pseudo_validation_qc_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_pseudo_validation"

N_VALIDATION_EVENTS = 8
MIN_TIMES_PER_EVENT = 40
MAX_TIMES_PER_EVENT = 80
TOPK = 20
TOPK_CANDIDATE_POOL = 1000
MAX_PER_HISTORY_EVENT = 3
MIN_VALID_TEMPLATES = 5
BETA_BLEND = 0.3
USE_EOF_MODEL = True
USE_EXTREME_CALIBRATION = True
RANDOM_SEED = 2026
MAKE_FIGURES = True

GRID_SIZE = 201
GRID_EXTENT_KM = 1000.0
CACHE_SIZE = 1500
NAN_SKIP_THRESHOLD = 0.50
KM_PER_DEG = 111.32
EPS = 1e-12

MODEL_VERSIONS = ["initial", "blend", "calibrated"]
TARGET_EXCLUDED_NAMES = {"KONGREY", "MANYI"}

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


def configure_environment_features(feature_set: object = "env-full") -> List[str]:
    """Configure the D_env features used in pseudo-missing retrieval."""
    global ENV_FEATURE_SET
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
    return features


configure_environment_features(ENV_FEATURE_SET)

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

HISTORY_RAIN_OUTPUT_FIELDS = [
    "rain_max_mmhr",
    "rain_p95_mmhr",
    "rain_p99_mmhr",
    "rain_area_10_km2",
    "rain_area_20_km2",
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

# Step-21 calibration constants.
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
GLOBAL_RAIN_MAX_CAP_MMHR = 120.0
PER_TARGET_CAP_FACTOR = 1.25
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


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def safe_name(value: object) -> str:
    text = str(value).replace("-", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def format_time(value: object) -> object:
    if pd.isna(value):
        return np.nan
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def rel_abs_error(pred: float, truth: float) -> float:
    return safe_div(abs(float(pred) - float(truth)), abs(float(truth)))


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


def is_leakage_feature(feature: str) -> bool:
    return any(pattern.search(feature) for pattern in LEAKAGE_PATTERNS)


def cell_area_from_grid(x_front_km: np.ndarray, y_left_km: np.ndarray) -> float:
    dx = float(abs(x_front_km[1] - x_front_km[0])) if len(x_front_km) > 1 else 1.0
    dy = float(abs(y_left_km[1] - y_left_km[0])) if len(y_left_km) > 1 else 1.0
    return dx * dy


# =========================
# Loaders and event selection
# =========================


def load_historical_library() -> pd.DataFrame:
    if not HISTORICAL_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Missing historical library: {HISTORICAL_LIBRARY_PATH}")
    df = pd.read_csv(HISTORICAL_LIBRARY_PATH, encoding="utf-8-sig", low_memory=False)
    if "time" not in df.columns:
        raise RuntimeError("Historical library must contain time.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    required = [
        "event_uid",
        "typhoon_name",
        "time",
        "tif_path",
        "lat",
        "lon_180",
        "WND",
        "PRES",
        "intensity",
        "move_dir_deg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Historical library is missing required columns: {missing}")
    if "move_dir_sin" not in df.columns or "move_dir_cos" not in df.columns:
        df = add_direction_sin_cos(df)
    return df.sort_values(["event_uid", "time"]).reset_index(drop=True)


def load_eof_pca_model() -> Dict[str, np.ndarray]:
    if not EOF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing EOF/PCA model: {EOF_MODEL_PATH}")
    with np.load(EOF_MODEL_PATH, allow_pickle=True) as z:
        required = ["mean_log_field", "eof_components", "x_front_km", "y_left_km", "n_components"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"EOF/PCA model is missing arrays: {missing}")
        model = {k: z[k] for k in z.files}
    mean = np.asarray(model["mean_log_field"], dtype=np.float32)
    comps = np.asarray(model["eof_components"], dtype=np.float32)
    if mean.shape != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"Unexpected EOF mean shape: {mean.shape}")
    if comps.ndim != 3 or comps.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"Unexpected EOF component shape: {comps.shape}")
    model["mean_log_field"] = mean
    model["eof_components"] = comps
    return model


def select_validation_events(history: pd.DataFrame) -> pd.DataFrame:
    def first_non_null(series: pd.Series) -> object:
        s = series.dropna()
        return s.iloc[0] if len(s) else np.nan

    hist = history.copy()
    hist["name_norm"] = hist["typhoon_name"].map(normalize_name)
    hist["tif_exists"] = hist["tif_path"].map(lambda p: resolve_project_path(p).exists())
    ok_time = hist["time"].notna()
    ok_tif = hist["tif_path"].notna() & hist["tif_exists"]
    ok_center = pd.to_numeric(hist["lat"], errors="coerce").notna() & pd.to_numeric(hist["lon_180"], errors="coerce").notna()
    ok_move = pd.to_numeric(hist["move_dir_deg"], errors="coerce").notna()
    ok_rain = pd.to_numeric(hist.get("rain_p95_mmhr"), errors="coerce").notna()
    usable = hist.loc[ok_time & ok_tif & ok_center & ok_move & ok_rain].copy()

    grouped = usable.groupby("event_uid", dropna=False)
    summary = grouped.agg(
        typhoon_name=("typhoon_name", first_non_null),
        name_norm=("name_norm", first_non_null),
        year=("time", lambda s: int(pd.to_datetime(s).dt.year.mode().iloc[0]) if len(s) else np.nan),
        start_time=("time", "min"),
        end_time=("time", "max"),
        n_total_times=("time", "size"),
        WND_max=("WND", "max"),
        PRES_min=("PRES", "min"),
        rain_max_mmhr_max=("rain_max_mmhr", "max"),
        rain_p95_mmhr_max=("rain_p95_mmhr", "max"),
        mean_signed_coast_dist_km=("signed_coast_dist_km", "mean"),
        land_time_fraction=("is_land", "mean"),
    ).reset_index()

    summary = summary.loc[summary["n_total_times"] >= MIN_TIMES_PER_EVENT].copy()
    summary = summary.loc[~summary["name_norm"].isin(TARGET_EXCLUDED_NAMES)].copy()
    if summary.empty:
        raise RuntimeError("No validation event candidates after filtering.")

    non_nameless = summary.loc[summary["name_norm"] != "NAMELESS"].copy()
    candidate = non_nameless if len(non_nameless) >= N_VALIDATION_EVENTS else summary.copy()
    if len(candidate) < N_VALIDATION_EVENTS:
        raise RuntimeError(
            f"Validation event candidates are insufficient: {len(candidate)} < {N_VALIDATION_EVENTS}"
        )

    chosen: List[pd.Series] = []
    chosen_ids: set = set()

    def add_pick(sub: pd.DataFrame, reason: str, sort_cols: Sequence[str], ascending: Sequence[bool]) -> None:
        nonlocal chosen, chosen_ids
        if len(chosen) >= N_VALIDATION_EVENTS or sub.empty:
            return
        sub = sub.loc[~sub["event_uid"].astype(str).isin(chosen_ids)].copy()
        if sub.empty:
            return
        row = sub.sort_values(list(sort_cols), ascending=list(ascending)).iloc[0].copy()
        row["selection_reason"] = reason
        chosen.append(row)
        chosen_ids.add(str(row["event_uid"]))

    add_pick(candidate, "strongest_wind", ["WND_max", "rain_p95_mmhr_max"], [False, False])
    add_pick(candidate, "highest_p95_rain", ["rain_p95_mmhr_max", "WND_max"], [False, False])
    add_pick(candidate, "most_land_or_landfall_influenced", ["land_time_fraction", "rain_p95_mmhr_max"], [False, False])
    add_pick(
        candidate.loc[pd.to_numeric(candidate["mean_signed_coast_dist_km"], errors="coerce") <= 100.0],
        "near_coast_or_landfall",
        ["rain_p95_mmhr_max", "WND_max"],
        [False, False],
    )
    add_pick(
        candidate.loc[
            (pd.to_numeric(candidate["land_time_fraction"], errors="coerce") <= 0.05)
            & (pd.to_numeric(candidate["mean_signed_coast_dist_km"], errors="coerce") >= 250.0)
        ],
        "open_ocean",
        ["rain_p95_mmhr_max", "WND_max"],
        [False, False],
    )
    add_pick(
        candidate.loc[pd.to_numeric(candidate["WND_max"], errors="coerce") <= 25.0],
        "weak_intensity",
        ["rain_p95_mmhr_max", "n_total_times"],
        [False, False],
    )

    year_counts: Counter = Counter(int(r["year"]) for r in chosen if pd.notna(r["year"]))
    score_df = candidate.loc[~candidate["event_uid"].astype(str).isin(chosen_ids)].copy()
    for _, row in score_df.iterrows():
        if len(chosen) >= N_VALIDATION_EVENTS:
            break
        cand = score_df.loc[~score_df["event_uid"].astype(str).isin(chosen_ids)].copy()
        if cand.empty:
            break
        wnd = pd.to_numeric(cand["WND_max"], errors="coerce")
        rain = pd.to_numeric(cand["rain_p95_mmhr_max"], errors="coerce")
        land = pd.to_numeric(cand["land_time_fraction"], errors="coerce")
        coast = pd.to_numeric(cand["mean_signed_coast_dist_km"], errors="coerce")
        years = pd.to_numeric(cand["year"], errors="coerce")
        wnd_score = (wnd - wnd.min()) / (wnd.max() - wnd.min() + EPS)
        rain_score = (rain - rain.min()) / (rain.max() - rain.min() + EPS)
        land_score = (land - land.min()) / (land.max() - land.min() + EPS)
        ocean_score = (coast - coast.min()) / (coast.max() - coast.min() + EPS)
        diversity_penalty = years.map(lambda y: 0.12 * year_counts[int(y)] if np.isfinite(y) else 0.0)
        cand["selection_score"] = 0.35 * wnd_score + 0.35 * rain_score + 0.15 * land_score + 0.15 * ocean_score - diversity_penalty
        picked = cand.sort_values("selection_score", ascending=False).iloc[0].copy()
        picked["selection_reason"] = "diversity_fill"
        chosen.append(picked)
        chosen_ids.add(str(picked["event_uid"]))
        if pd.notna(picked["year"]):
            year_counts[int(picked["year"])] += 1

    events = pd.DataFrame(chosen).head(N_VALIDATION_EVENTS).copy()
    if len(events) < N_VALIDATION_EVENTS:
        raise RuntimeError(f"Could only select {len(events)} validation events.")
    events = events.rename(columns={"event_uid": "validation_event_uid"})
    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce").map(format_time)
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce").map(format_time)
    ordered = [
        "validation_event_uid",
        "typhoon_name",
        "year",
        "start_time",
        "end_time",
        "n_total_times",
        "WND_max",
        "PRES_min",
        "rain_max_mmhr_max",
        "rain_p95_mmhr_max",
        "mean_signed_coast_dist_km",
        "land_time_fraction",
        "selection_reason",
    ]
    return events[ordered].reset_index(drop=True)


def select_validation_times_for_event(event_df: pd.DataFrame) -> pd.DataFrame:
    event_df = event_df.sort_values("time").reset_index(drop=True)
    n = len(event_df)
    if n <= MAX_TIMES_PER_EVENT:
        out = event_df.copy()
        out["field_index_within_event"] = np.arange(len(out), dtype=int)
        return out

    picks: set = set()
    for frac in [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0]:
        picks.add(int(round(frac * (n - 1))))

    for col in ["WND", "rain_p95_mmhr", "rain_p99_mmhr", "rain_max_mmhr"]:
        if col in event_df.columns and pd.to_numeric(event_df[col], errors="coerce").notna().any():
            picks.add(int(pd.to_numeric(event_df[col], errors="coerce").idxmax()))
    if "wind_change_rate" in event_df.columns:
        s = pd.to_numeric(event_df["wind_change_rate"], errors="coerce").abs()
        if s.notna().any():
            picks.add(int(s.idxmax()))

    evenly = np.linspace(0, n - 1, MAX_TIMES_PER_EVENT, dtype=int)
    for idx in evenly:
        picks.add(int(idx))
        if len(picks) >= MAX_TIMES_PER_EVENT:
            break

    selected = sorted(picks)
    if len(selected) > MAX_TIMES_PER_EVENT:
        key_set = set()
        for idx in selected:
            if idx in {0, n - 1}:
                key_set.add(idx)
        for col in ["WND", "rain_p95_mmhr", "rain_p99_mmhr", "rain_max_mmhr"]:
            if col in event_df.columns and pd.to_numeric(event_df[col], errors="coerce").notna().any():
                key_set.add(int(pd.to_numeric(event_df[col], errors="coerce").idxmax()))
        remaining = [i for i in selected if i not in key_set]
        need = MAX_TIMES_PER_EVENT - len(key_set)
        if need > 0:
            keep = np.linspace(0, len(remaining) - 1, need, dtype=int)
            key_set.update(remaining[i] for i in keep)
        selected = sorted(key_set)[:MAX_TIMES_PER_EVENT]

    out = event_df.iloc[selected].copy().reset_index(drop=True)
    out["field_index_within_event"] = np.arange(len(out), dtype=int)
    return out


# =========================
# Retrieval feature handling
# =========================


def add_direction_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    deg = pd.to_numeric(out.get("move_dir_deg"), errors="coerce")
    rad = np.deg2rad(deg)
    out["move_dir_sin"] = np.sin(rad)
    out["move_dir_cos"] = np.cos(rad)
    return out


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
    if not any(selected.values()):
        raise RuntimeError("No safe retrieval features are available.")
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
    return pd.concat(pieces).sort_index().reindex(df.index)


def feature_fill_strategy(feature: str) -> str:
    if feature in {"move_speed_kmh", "move_dir_sin", "move_dir_cos", "wind_change_rate", "pressure_change_rate"}:
        return "same_event_time_interpolation_then_event_median_then_training_median"
    if feature in {"intensity", "is_land"}:
        return "mode_then_training_median"
    if any(token in feature for token in ["coast", "landfrac", "terrain", "elev"]):
        return "training_median_fill_for_available_environment_feature"
    return "training_median_fill"


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
            if group_col in out.columns:
                out[feature] = out[feature].fillna(out.groupby(group_col)[feature].transform("median"))
        if feature in {"intensity", "is_land"}:
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
    target_imp = impute_one_frame(target, features, history_imp, "event_uid")

    for row in rows:
        feature = row["feature"]
        row["history_missing_rate_after"] = float(pd.to_numeric(history_imp[feature], errors="coerce").isna().mean())
        row["target_missing_rate_after"] = float(pd.to_numeric(target_imp[feature], errors="coerce").isna().mean())

    bad = {}
    for feature in features:
        h_rate = float(pd.to_numeric(history_imp[feature], errors="coerce").isna().mean())
        t_rate = float(pd.to_numeric(target_imp[feature], errors="coerce").isna().mean())
        if h_rate > 0 or t_rate > 0:
            bad[feature] = (h_rate, t_rate)
    if bad:
        raise RuntimeError(f"Retrieval features still contain NaN after imputation: {bad}")
    return history_imp, target_imp, pd.DataFrame(rows)


def compute_standardization_params(history: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(history[feature], errors="coerce")
        mean = float(values.mean(skipna=True))
        std = float(values.std(skipna=True, ddof=0))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std <= EPS:
            std = 1.0
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
            values = np.zeros(history_z.shape[0], dtype=np.float64)
        else:
            diff = history_z[:, indices] - target_z[np.newaxis, indices]
            values = np.mean(diff * diff, axis=1, dtype=np.float64)
        components[component] = values
        total += COMPONENT_WEIGHTS[component] * values
    return total, components


def diversify_topk_by_event(
    sorted_indices: np.ndarray,
    event_ids: np.ndarray,
    k: int,
) -> List[int]:
    selected: List[int] = []
    counts: Counter = Counter()
    seen: set = set()
    for idx in sorted_indices:
        idx_int = int(idx)
        event_id = str(event_ids[idx_int])
        if counts[event_id] >= MAX_PER_HISTORY_EVENT:
            continue
        selected.append(idx_int)
        seen.add(idx_int)
        counts[event_id] += 1
        if len(selected) >= k:
            return selected
    for idx in sorted_indices:
        idx_int = int(idx)
        if idx_int in seen:
            continue
        selected.append(idx_int)
        if len(selected) >= k:
            return selected
    return selected


def compute_softmax_weights(distances: Sequence[float]) -> np.ndarray:
    dist = np.asarray(distances, dtype=float)
    if len(dist) == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(dist)):
        finite = np.isfinite(dist)
        fill = float(np.nanmax(dist[finite])) if finite.any() else 0.0
        dist = np.where(finite, dist, fill)
    if np.nanmax(dist) - np.nanmin(dist) <= EPS:
        return np.full(len(dist), 1.0 / len(dist), dtype=float)
    tau = float(np.nanmedian(dist))
    if not np.isfinite(tau) or tau <= EPS:
        tau = 1.0
    score = np.exp(-(dist - np.nanmin(dist)) / tau)
    denom = float(np.sum(score))
    return score / denom if denom > EPS else np.full(len(dist), 1.0 / len(dist), dtype=float)


def retrieve_topk_for_validation_times(
    train_history: pd.DataFrame,
    validation_times: pd.DataFrame,
    selected_components: Mapping[str, Sequence[str]],
    standardization_params: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    features = [f for feats in selected_components.values() for f in feats]
    history_z = standardize_values(train_history, standardization_params, features)
    target_z = standardize_values(validation_times, standardization_params, features)
    indices_by_component = component_feature_indices(selected_components, features)
    event_ids = train_history["event_uid"].astype(str).to_numpy()

    records: List[Dict[str, object]] = []
    self_match_count = 0
    for i in range(len(validation_times)):
        target_row = validation_times.iloc[i]
        distances, comp_values = compute_component_distances(target_z[i], history_z, indices_by_component)
        pool_n = min(len(distances), max(TOPK_CANDIDATE_POOL, TOPK * 10))
        if pool_n <= 0:
            raise RuntimeError("Training history is empty during retrieval.")
        candidate_idx = np.argpartition(distances, pool_n - 1)[:pool_n]
        candidate_idx = candidate_idx[np.argsort(distances[candidate_idx])]
        selected = diversify_topk_by_event(candidate_idx, event_ids, TOPK)
        if len(selected) < TOPK:
            selected = diversify_topk_by_event(np.argsort(distances), event_ids, TOPK)
        selected_arr = np.asarray(selected, dtype=int)
        selected_dist = distances[selected_arr]
        weights = compute_softmax_weights(selected_dist)
        target_id = str(target_row["sample_id"])
        validation_event_uid = str(target_row["event_uid"])
        for rank_i, hist_idx in enumerate(selected_arr, start=1):
            hist_row = train_history.iloc[int(hist_idx)]
            if str(hist_row["event_uid"]) == validation_event_uid:
                self_match_count += 1
            rec: Dict[str, object] = {
                "target_id": target_id,
                "validation_event_uid": validation_event_uid,
                "target_typhoon_name": target_row.get("typhoon_name"),
                "target_time": format_time(target_row.get("time")),
                "history_sample_id": hist_row.get("sample_id"),
                "history_event_uid": hist_row.get("event_uid"),
                "history_typhoon_name": hist_row.get("typhoon_name"),
                "history_time": format_time(hist_row.get("time")),
                "history_tif_path": hist_row.get("tif_path"),
                "history_lat": hist_row.get("lat"),
                "history_lon_180": hist_row.get("lon_180"),
                "history_move_dir_deg": hist_row.get("move_dir_deg"),
                "rank": int(rank_i),
                "similarity_distance": float(selected_dist[rank_i - 1]),
                "similarity_weight": float(weights[rank_i - 1]),
                "distance_track": float(comp_values["track"][hist_idx]),
                "distance_intensity": float(comp_values["intensity"][hist_idx]),
                "distance_motion": float(comp_values["motion"][hist_idx]),
                "distance_environment": float(comp_values["environment"][hist_idx]),
                "distance_time": float(comp_values["time"][hist_idx]),
                "distance_life_progress": float(comp_values["life_progress"][hist_idx]),
            }
            for field in HISTORY_RAIN_OUTPUT_FIELDS:
                rec[f"history_{field}"] = hist_row[field] if field in hist_row.index else np.nan
            records.append(rec)

    topk = pd.DataFrame(records)
    diagnostics = {
        "features": features,
        "component_feature_counts": {k: len(v) for k, v in selected_components.items()},
        "history_z_shape": [int(history_z.shape[0]), int(history_z.shape[1])],
        "target_z_shape": [int(target_z.shape[0]), int(target_z.shape[1])],
        "self_match_count": int(self_match_count),
        "topk_short_targets": int((topk.groupby("target_id").size() < TOPK).sum()) if len(topk) else len(validation_times),
    }
    return topk, diagnostics


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


def normalize_lon_180(lon: object) -> float:
    value = pd.to_numeric(pd.Series([lon]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    return float(((float(value) + 180.0) % 360.0) - 180.0)


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
        raise ValueError("center lat/lon or move_dir_deg is missing")

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
    return sampled, {"nan_ratio": float(np.isnan(sampled).mean()), "lon_convention": lon_convention}


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
    counters: Counter,
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
        sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        sampled[sampled < 0.0] = 0.0
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
# Generation
# =========================


def generate_initial_field(
    target_row: pd.Series,
    topk_one: pd.DataFrame,
    cache: TemplateCache,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    qc_events: List[Dict[str, object]],
    counters: Counter,
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
        log_template, _ = get_cached_history_template(hist_row, cache, x_grid, y_grid, qc_events, counters)
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
    log_initial = np.divide(
        weighted_sum,
        valid_weight_sum_grid,
        out=np.zeros_like(weighted_sum, dtype=np.float64),
        where=valid_weight_sum_grid > EPS,
    )
    rain_initial = np.expm1(log_initial)
    rain_initial = np.nan_to_num(rain_initial, nan=0.0, posinf=0.0, neginf=0.0)
    rain_initial[rain_initial < 0.0] = 0.0
    log_initial = np.log1p(rain_initial)
    if not np.isfinite(rain_initial).all() or np.any(rain_initial < 0.0):
        raise RuntimeError("Initial generated field contains NaN/Inf/negative values.")
    status = {
        "topk_count": int(len(topk_one)),
        "valid_template_count": int(valid_template_count),
        "skipped_template_count": int(skipped),
        "valid_weight_sum_before_renorm": float(np.sum(included_weights)) if included_weights else 0.0,
        "low_valid_template_count": bool(valid_template_count < MIN_VALID_TEMPLATES),
        "generation_ok": bool(valid_template_count >= MIN_VALID_TEMPLATES),
    }
    if valid_template_count < MIN_VALID_TEMPLATES:
        qc_events.append(
            {
                "target_id": target_row.get("sample_id"),
                "reason": "low_valid_template_count",
                "valid_template_count": int(valid_template_count),
            }
        )
    return rain_initial.astype("float32"), log_initial.astype("float32"), status


def apply_eof_blend(
    rain_initial: np.ndarray,
    log_initial: np.ndarray,
    model: Mapping[str, np.ndarray],
    beta: float = BETA_BLEND,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if not USE_EOF_MODEL:
        return rain_initial.copy(), log_initial.copy(), {"eof_reconstruction_rmse_log": np.nan, "eof_reconstruction_corr_log": np.nan}
    try:
        mean_flat = np.asarray(model["mean_log_field"], dtype=np.float32).reshape(-1)
        components = np.asarray(model["eof_components"], dtype=np.float32).reshape(np.asarray(model["eof_components"]).shape[0], -1)
        x = np.nan_to_num(log_initial.reshape(1, -1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        coef = (x - mean_flat) @ components.T
        recon = mean_flat + coef @ components
        recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0).reshape(GRID_SIZE, GRID_SIZE)
        rain_eof = np.expm1(recon).astype(np.float32)
        rain_eof = np.nan_to_num(rain_eof, nan=0.0, posinf=0.0, neginf=0.0)
        rain_eof[rain_eof < 0.0] = 0.0
        rain_blend = (1.0 - beta) * rain_initial.astype(np.float32) + beta * rain_eof
        rain_blend = np.nan_to_num(rain_blend, nan=0.0, posinf=0.0, neginf=0.0)
        rain_blend[rain_blend < 0.0] = 0.0
        log_blend = np.log1p(rain_blend).astype(np.float32)
        resid = x.reshape(-1) - recon.reshape(-1)
        diag = {
            "eof_reconstruction_rmse_log": float(np.sqrt(np.mean(resid * resid))),
            "eof_reconstruction_corr_log": corr_flat(x.reshape(-1), recon.reshape(-1)),
        }
        return rain_blend.astype("float32"), log_blend, diag
    except Exception as exc:
        raise RuntimeError(f"EOF blend failed: {type(exc).__name__}: {exc}") from exc


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


def weighted_metric(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    w = weights[valid]
    return float(np.sum(values[valid] * w) / np.sum(w))


def compute_weighted_calibration_targets(topk_one: pd.DataFrame) -> Dict[str, float]:
    weights = pd.to_numeric(topk_one["similarity_weight"], errors="coerce").to_numpy(dtype=float)
    out: Dict[str, float] = {
        "target_rain_max_mmhr": np.nan,
        "target_rain_p95_mmhr": np.nan,
        "target_rain_p99_mmhr": np.nan,
        "target_rain_area_10_km2": np.nan,
        "target_rain_area_20_km2": np.nan,
        "target_cap_mmhr": GLOBAL_RAIN_MAX_CAP_MMHR,
    }
    mapping = {
        "target_rain_max_mmhr": "history_rain_max_mmhr",
        "target_rain_p95_mmhr": "history_rain_p95_mmhr",
        "target_rain_p99_mmhr": "history_rain_p99_mmhr",
        "target_rain_area_10_km2": "history_rain_area_10_km2",
        "target_rain_area_20_km2": "history_rain_area_20_km2",
    }
    for target_col, source_col in mapping.items():
        values = pd.to_numeric(topk_one.get(source_col), errors="coerce").to_numpy(dtype=float)
        out[target_col] = weighted_metric(winsorize_topk_values(values), weights)
    if np.isfinite(out["target_rain_p95_mmhr"]) and np.isfinite(out["target_rain_p99_mmhr"]):
        out["target_rain_p99_mmhr"] = max(out["target_rain_p99_mmhr"], out["target_rain_p95_mmhr"])
    if np.isfinite(out["target_rain_max_mmhr"]) and np.isfinite(out["target_rain_p99_mmhr"]):
        out["target_rain_max_mmhr"] = max(out["target_rain_max_mmhr"], out["target_rain_p99_mmhr"])
    max_values = pd.to_numeric(topk_one.get("history_rain_max_mmhr"), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(max_values).any():
        out["target_cap_mmhr"] = float(min(GLOBAL_RAIN_MAX_CAP_MMHR, PER_TARGET_CAP_FACTOR * np.nanmax(max_values)))
    return out


def finite_target(value: object, fallback: float) -> Tuple[float, bool]:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v) or not np.isfinite(v):
        return float(fallback), True
    return float(v), False


def piecewise_tail_scale_map(
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
    flat = out.ravel()
    if current < lower_ok:
        needed = int(math.ceil((lower_ok - current) / cell_area_km2))
        candidates = np.where((flat >= AREA10_LOWER_CANDIDATE) & (flat < AREA10_THRESHOLD))[0]
        if len(candidates) < needed:
            candidates = np.where((flat >= AREA10_EXPANDED_LOWER_CANDIDATE) & (flat < AREA10_THRESHOLD))[0]
        if len(candidates) > 0 and needed > 0:
            n = min(needed, len(candidates), max_adjust_cells)
            order = candidates[np.argsort(flat[candidates])[-n:]]
            flat[order] = AREA10_THRESHOLD + 0.05
            diag["area10_action"] = "boosted_near_threshold"
            diag["area10_adjusted_cells"] = int(n)
    elif current > upper_ok:
        target_cells = max(0, int(math.floor(upper_ok / cell_area_km2)))
        current_cells = int(np.count_nonzero(flat >= AREA10_THRESHOLD))
        reduce_n = min(max(0, current_cells - target_cells), max_adjust_cells)
        candidates = np.where((flat >= AREA10_THRESHOLD) & (flat < max(AREA20_THRESHOLD, float(np.nanquantile(flat, 0.99)))))[0]
        if len(candidates) > 0 and reduce_n > 0:
            n = min(reduce_n, len(candidates))
            order = candidates[np.argsort(flat[candidates])[:n]]
            flat[order] = np.minimum(AREA10_THRESHOLD - 0.01, flat[order] * 0.97).astype(np.float32)
            diag["area10_action"] = "reduced_near_threshold"
            diag["area10_adjusted_cells"] = int(n)
    out = flat.reshape(out.shape)
    diag["area10_after"] = float(np.count_nonzero(out >= AREA10_THRESHOLD) * cell_area_km2)
    return out, diag


def apply_physical_caps(rain: np.ndarray, cap_mmhr: float) -> Tuple[np.ndarray, Dict[str, object]]:
    cap = float(cap_mmhr) if np.isfinite(cap_mmhr) and cap_mmhr > 0.0 else GLOBAL_RAIN_MAX_CAP_MMHR
    cap = min(GLOBAL_RAIN_MAX_CAP_MMHR, cap)
    before_max = float(np.nanmax(rain)) if np.isfinite(rain).any() else np.nan
    capped = rain.astype(np.float32, copy=True)
    mask = np.isfinite(capped) & (capped > cap)
    if np.count_nonzero(mask):
        capped[mask] = cap
    return capped, {
        "target_cap_mmhr": cap,
        "rain_max_before_cap": before_max,
        "rain_max_capped_cell_count": int(np.count_nonzero(mask)),
    }


def calibrate_one_field_tail_enhancement(
    rain_blend: np.ndarray,
    target_row: Mapping[str, object],
    cell_area_km2: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not USE_EXTREME_CALIBRATION:
        return rain_blend.astype(np.float32, copy=True), {"calibration_ok": True, "calibration_issue": "disabled"}
    r = np.nan_to_num(np.asarray(rain_blend, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r[r < 0.0] = 0.0
    diag: Dict[str, object] = {
        "calibration_ok": True,
        "calibration_issue": "",
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
    if q95 <= EPS or q99 <= EPS or rmax <= EPS:
        diag["calibration_ok"] = False
        diag["calibration_issue"] = "blend_tail_quantile_too_small"
        return r, diag
    t95, miss95 = finite_target(target_row.get("target_rain_p95_mmhr"), q95)
    t99, miss99 = finite_target(target_row.get("target_rain_p99_mmhr"), q99)
    tmax, missmax = finite_target(target_row.get("target_rain_max_mmhr"), rmax)
    tarea10, missarea10 = finite_target(target_row.get("target_rain_area_10_km2"), np.nan)
    cap, _ = finite_target(target_row.get("target_cap_mmhr"), GLOBAL_RAIN_MAX_CAP_MMHR)
    cap = min(GLOBAL_RAIN_MAX_CAP_MMHR, cap)
    missing_targets = [name for name, miss in [("p95", miss95), ("p99", miss99), ("max", missmax), ("area10", missarea10)] if miss]
    if missing_targets:
        diag["calibration_issue"] = "missing_targets_fallback:" + ",".join(missing_targets)
        diag["calibration_ok"] = False
    t99 = max(t99, t95)
    tmax = max(tmax, t99)
    y95 = min(max(q95 * float(np.clip(t95 / (q95 + EPS), SCALE_MIN, SCALE_MAX)), q90), cap)
    y99 = min(max(q99 * float(np.clip(t99 / (q99 + EPS), SCALE_MIN, SCALE_MAX)), y95), cap)
    ymax = min(max(rmax * float(np.clip(tmax / (rmax + EPS), SCALE_MIN, SCALE_MAX_MAX)), y99), cap)
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
    calibrated = piecewise_tail_scale_map(r, q90, q95, q99, rmax, y95, y99, ymax)
    calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    calibrated[calibrated < 0.0] = 0.0
    calibrated, area_diag = calibrate_area_threshold(calibrated, tarea10, cell_area_km2)
    diag.update(area_diag)
    calibrated, cap_diag = apply_physical_caps(calibrated, cap)
    diag.update(cap_diag)
    if not np.isfinite(calibrated).all() or np.any(calibrated < 0.0):
        diag["calibration_ok"] = False
        diag["calibration_issue"] = (diag.get("calibration_issue", "") + "; nonfinite_or_negative").strip("; ")
        calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        calibrated[calibrated < 0.0] = 0.0
    return calibrated.astype(np.float32), diag


def build_truth_field(
    target_row: pd.Series,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    path = resolve_project_path(target_row.get("tif_path"))
    if not path.exists():
        raise FileNotFoundError(f"truth tif not found: {path}")
    center_lat = pd.to_numeric(pd.Series([target_row.get("lat")]), errors="coerce").iloc[0]
    center_lon = pd.to_numeric(pd.Series([target_row.get("lon_180")]), errors="coerce").iloc[0]
    move_dir = pd.to_numeric(pd.Series([target_row.get("move_dir_deg")]), errors="coerce").iloc[0]
    if pd.isna(center_lat) or pd.isna(center_lon) or pd.isna(move_dir):
        raise ValueError("truth center or move_dir_deg missing")
    rain, lat1d, lon1d, raster_meta = read_gpm_tif(path)
    if raster_meta["all_missing"]:
        raise ValueError("truth tif all missing")
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
    if sample_meta["nan_ratio"] > NAN_SKIP_THRESHOLD:
        raise ValueError(f"truth resampled nan ratio too high: {sample_meta['nan_ratio']:.3f}")
    sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
    sampled[sampled < 0.0] = 0.0
    if not np.isfinite(sampled).all():
        raise ValueError("truth field still contains nonfinite values")
    if np.all(sampled == 0.0):
        raise ValueError("truth field all zero after resampling")
    return sampled.astype("float32"), {**raster_meta, **sample_meta}


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
    return float(r_sorted[min(idx, len(r_sorted) - 1)])


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
    keys = [
        "rain_mean_mmhr",
        "rain_max_mmhr",
        "rain_p50_mmhr",
        "rain_p75_mmhr",
        "rain_p90_mmhr",
        "rain_p95_mmhr",
        "rain_p99_mmhr",
        "rain_area_1_km2",
        "rain_area_5_km2",
        "rain_area_10_km2",
        "rain_area_20_km2",
        "centroid_x_front_km",
        "centroid_y_left_km",
        "centroid_offset_km",
        "centroid_angle_deg",
        "asym_front_back_ratio",
        "asym_left_right_ratio",
        "anisotropy",
        "rain_radius_r50_km",
        "rain_radius_r80_km",
        "rain_radius_r90_km",
        "rain_band_width_km",
    ]
    out = {f"{prefix}_{key}": np.nan for key in keys}
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
        }
    )
    for threshold in [1, 5, 10, 20]:
        out[f"{prefix}_rain_area_{threshold}_km2"] = float(np.count_nonzero(valid & (rain >= threshold)) * cell_area)
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
    out[f"{prefix}_asym_front_back_ratio"] = safe_div(float(np.sum(weights[front])), float(np.sum(weights[back])))
    out[f"{prefix}_asym_left_right_ratio"] = safe_div(float(np.sum(weights[left])), float(np.sum(weights[right])))
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
            out[f"{prefix}_anisotropy"] = float((float(eigvals[1]) - float(eigvals[0])) / float(np.sum(eigvals)))
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    r50 = weighted_radius(radius, weights, 0.50)
    r80 = weighted_radius(radius, weights, 0.80)
    r90 = weighted_radius(radius, weights, 0.90)
    out[f"{prefix}_rain_radius_r50_km"] = r50
    out[f"{prefix}_rain_radius_r80_km"] = r80
    out[f"{prefix}_rain_radius_r90_km"] = r90
    out[f"{prefix}_rain_band_width_km"] = float(r90 - r50) if np.isfinite(r90) and np.isfinite(r50) else np.nan
    return out


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_grid_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    p = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(np.asarray(truth, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    diff = p - t
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "corr": corr_flat(p, t),
    }


def compute_threshold_metrics(pred: np.ndarray, truth: np.ndarray, threshold: float) -> Dict[str, float]:
    p = np.asarray(pred) >= threshold
    t = np.asarray(truth) >= threshold
    hit = int(np.count_nonzero(p & t))
    miss = int(np.count_nonzero((~p) & t))
    false_alarm = int(np.count_nonzero(p & (~t)))
    return {
        "hit": hit,
        "miss": miss,
        "false_alarm": false_alarm,
        "csi": float(hit / (hit + miss + false_alarm + EPS)),
        "pod": float(hit / (hit + miss + EPS)),
        "far": float(false_alarm / (hit + false_alarm + EPS)),
        "f1": float(2 * hit / (2 * hit + false_alarm + miss + EPS)),
    }


def compute_fss(pred: np.ndarray, truth: np.ndarray, threshold: float = 10.0, windows: Sequence[int] = (3, 5, 9)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    p_bin = (np.asarray(pred) >= threshold).astype(np.float32)
    t_bin = (np.asarray(truth) >= threshold).astype(np.float32)
    for w in windows:
        fp = uniform_filter(p_bin, size=w, mode="constant", cval=0.0)
        ft = uniform_filter(t_bin, size=w, mode="constant", cval=0.0)
        mse = float(np.mean((fp - ft) ** 2))
        ref = float(np.mean(fp ** 2 + ft ** 2))
        out[f"fss10_w{w}"] = float(1.0 - mse / (ref + EPS)) if ref > EPS else np.nan
    return out


def compute_timeslice_validation_metrics(
    target_row: pd.Series,
    model_version: str,
    pred: np.ndarray,
    truth: np.ndarray,
    topk_one: pd.DataFrame,
    generation_status: Mapping[str, object],
    calibration_targets: Mapping[str, float],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    validation_ok: bool = True,
    skip_reason: str = "",
) -> Dict[str, object]:
    truth_metrics = compute_rainfall_metrics_on_relative_grid(truth, x_front_km, y_left_km, x_grid, y_grid, "truth")
    pred_metrics = compute_rainfall_metrics_on_relative_grid(pred, x_front_km, y_left_km, x_grid, y_grid, "pred")
    grid_metrics = compute_grid_metrics(pred, truth)
    m10 = compute_threshold_metrics(pred, truth, 10.0)
    m20 = compute_threshold_metrics(pred, truth, 20.0)
    fss = compute_fss(pred, truth, 10.0, (3, 5, 9))
    distances = pd.to_numeric(topk_one.get("similarity_distance"), errors="coerce") if len(topk_one) else pd.Series(dtype=float)
    weights = pd.to_numeric(topk_one.get("similarity_weight"), errors="coerce") if len(topk_one) else pd.Series(dtype=float)
    row: Dict[str, object] = {
        "validation_event_uid": target_row.get("event_uid"),
        "typhoon_name": target_row.get("typhoon_name"),
        "validation_time": format_time(target_row.get("time")),
        "model_version": model_version,
        "field_index_within_event": int(target_row.get("field_index_within_event", -1)),
        "tif_path": target_row.get("tif_path"),
        "WND": target_row.get("WND"),
        "PRES": target_row.get("PRES"),
        "intensity": target_row.get("intensity"),
        "move_speed_kmh": target_row.get("move_speed_kmh"),
        "move_dir_deg": target_row.get("move_dir_deg"),
        "signed_coast_dist_km": target_row.get("signed_coast_dist_km"),
        "is_land": target_row.get("is_land"),
        "life_progress": target_row.get("life_progress"),
        "topk_count": int(len(topk_one)),
        "valid_template_count": int(generation_status.get("valid_template_count", 0)),
        "topk_unique_event_count": int(topk_one["history_event_uid"].nunique()) if "history_event_uid" in topk_one else 0,
        "topk_min_distance": float(distances.min(skipna=True)) if len(distances) and distances.notna().any() else np.nan,
        "topk_mean_distance": float(distances.mean(skipna=True)) if len(distances) and distances.notna().any() else np.nan,
        "topk_weight_sum": float(weights.sum(skipna=True)) if len(weights) else np.nan,
        "validation_ok": bool(validation_ok),
        "skip_reason": skip_reason,
    }
    row.update(calibration_targets)
    for key in [
        "rain_max_mmhr",
        "rain_p95_mmhr",
        "rain_p99_mmhr",
        "rain_area_10_km2",
        "rain_area_20_km2",
        "centroid_offset_km",
        "centroid_x_front_km",
        "centroid_y_left_km",
        "anisotropy",
        "rain_radius_r50_km",
        "rain_radius_r80_km",
        "rain_radius_r90_km",
        "rain_band_width_km",
        "asym_front_back_ratio",
        "asym_left_right_ratio",
    ]:
        row[f"truth_{key}"] = truth_metrics.get(f"truth_{key}", np.nan)
        row[f"pred_{key}"] = pred_metrics.get(f"pred_{key}", np.nan)
    row.update(grid_metrics)
    for key in ["rain_max_mmhr", "rain_p95_mmhr", "rain_p99_mmhr"]:
        short = key.replace("_mmhr", "")
        row[f"abs_error_{short}"] = abs(row[f"pred_{key}"] - row[f"truth_{key}"])
        row[f"rel_error_{short}"] = rel_abs_error(row[f"pred_{key}"], row[f"truth_{key}"])
    for area_key, suffix in [("rain_area_10_km2", "area_10"), ("rain_area_20_km2", "area_20")]:
        row[f"abs_error_{suffix}"] = abs(row[f"pred_{area_key}"] - row[f"truth_{area_key}"])
        row[f"rel_error_{suffix}"] = rel_abs_error(row[f"pred_{area_key}"], row[f"truth_{area_key}"])
    row["abs_error_centroid_offset"] = abs(row["pred_centroid_offset_km"] - row["truth_centroid_offset_km"])
    row["euclidean_error_centroid_xy"] = (
        float(
            math.hypot(
                row["pred_centroid_x_front_km"] - row["truth_centroid_x_front_km"],
                row["pred_centroid_y_left_km"] - row["truth_centroid_y_left_km"],
            )
        )
        if np.isfinite(row["pred_centroid_x_front_km"])
        and np.isfinite(row["truth_centroid_x_front_km"])
        and np.isfinite(row["pred_centroid_y_left_km"])
        and np.isfinite(row["truth_centroid_y_left_km"])
        else np.nan
    )
    row["abs_error_anisotropy"] = abs(row["pred_anisotropy"] - row["truth_anisotropy"])
    row["abs_error_r50"] = abs(row["pred_rain_radius_r50_km"] - row["truth_rain_radius_r50_km"])
    row["abs_error_r80"] = abs(row["pred_rain_radius_r80_km"] - row["truth_rain_radius_r80_km"])
    row["abs_error_r90"] = abs(row["pred_rain_radius_r90_km"] - row["truth_rain_radius_r90_km"])
    row["abs_error_band_width"] = abs(row["pred_rain_band_width_km"] - row["truth_rain_band_width_km"])
    row["abs_error_asym_front_back"] = abs(row["pred_asym_front_back_ratio"] - row["truth_asym_front_back_ratio"])
    row["abs_error_asym_left_right"] = abs(row["pred_asym_left_right_ratio"] - row["truth_asym_left_right_ratio"])
    row.update({"csi10": m10["csi"], "pod10": m10["pod"], "far10": m10["far"], "f1_10": m10["f1"]})
    row.update({"csi20": m20["csi"], "pod20": m20["pod"], "far20": m20["far"], "f1_20": m20["f1"]})
    row.update(fss)
    nan_metric_cols = ["rmse", "mae", "bias", "truth_rain_p95_mmhr", "pred_rain_p95_mmhr"]
    if any(pd.isna(row.get(c)) for c in nan_metric_cols):
        row["validation_ok"] = False
        row["skip_reason"] = (str(row.get("skip_reason", "")) + "; metric_nan").strip("; ")
    return row


def compute_event_duration_metrics(
    duration_state: Mapping[str, object],
    model_version: str,
    cell_area_km2: float,
) -> Dict[str, float]:
    truth10 = np.asarray(duration_state["truth10"], dtype=np.float32)
    truth20 = np.asarray(duration_state["truth20"], dtype=np.float32)
    pred10 = np.asarray(duration_state[f"{model_version}10"], dtype=np.float32)
    pred20 = np.asarray(duration_state[f"{model_version}20"], dtype=np.float32)
    out = {
        "duration10_total_area_time_truth": float(np.sum(truth10) * 0.5 * cell_area_km2),
        "duration10_total_area_time_pred": float(np.sum(pred10) * 0.5 * cell_area_km2),
        "duration20_total_area_time_truth": float(np.sum(truth20) * 0.5 * cell_area_km2),
        "duration20_total_area_time_pred": float(np.sum(pred20) * 0.5 * cell_area_km2),
        "duration10_max_grid_hours_truth": float(np.max(truth10) * 0.5) if truth10.size else np.nan,
        "duration10_max_grid_hours_pred": float(np.max(pred10) * 0.5) if pred10.size else np.nan,
        "duration20_max_grid_hours_truth": float(np.max(truth20) * 0.5) if truth20.size else np.nan,
        "duration20_max_grid_hours_pred": float(np.max(pred20) * 0.5) if pred20.size else np.nan,
    }
    out["duration10_area_time_abs_error"] = abs(out["duration10_total_area_time_pred"] - out["duration10_total_area_time_truth"])
    out["duration10_area_time_rel_error"] = rel_abs_error(out["duration10_total_area_time_pred"], out["duration10_total_area_time_truth"])
    out["duration20_area_time_abs_error"] = abs(out["duration20_total_area_time_pred"] - out["duration20_total_area_time_truth"])
    out["duration20_area_time_rel_error"] = rel_abs_error(out["duration20_total_area_time_pred"], out["duration20_total_area_time_truth"])
    out["duration10_max_grid_hours_abs_error"] = abs(out["duration10_max_grid_hours_pred"] - out["duration10_max_grid_hours_truth"])
    out["duration20_max_grid_hours_abs_error"] = abs(out["duration20_max_grid_hours_pred"] - out["duration20_max_grid_hours_truth"])
    return out


def mean_col(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df.get(col), errors="coerce").mean(skipna=True)) if col in df.columns else np.nan


def median_col(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df.get(col), errors="coerce").median(skipna=True)) if col in df.columns else np.nan


def build_event_summary(
    timeslice_metrics: pd.DataFrame,
    duration_metrics: Mapping[Tuple[str, str], Mapping[str, float]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    valid = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    for (event_uid, model_version), sub in valid.groupby(["validation_event_uid", "model_version"], sort=False):
        row: Dict[str, object] = {
            "validation_event_uid": event_uid,
            "typhoon_name": sub["typhoon_name"].iloc[0],
            "model_version": model_version,
            "n_valid_times": int(len(sub)),
            "rmse_mean": mean_col(sub, "rmse"),
            "mae_mean": mean_col(sub, "mae"),
            "bias_mean": mean_col(sub, "bias"),
            "corr_mean": mean_col(sub, "corr"),
        }
        for col in ["abs_error_rain_max", "abs_error_rain_p95", "abs_error_rain_p99"]:
            row[f"{col}_mean"] = mean_col(sub, col)
        for col in ["rel_error_rain_max", "rel_error_rain_p95", "rel_error_rain_p99"]:
            row[f"{col}_median"] = median_col(sub, col)
        for col in ["abs_error_area_10", "abs_error_area_20"]:
            row[f"{col}_mean"] = mean_col(sub, col)
        for col in ["rel_error_area_10", "rel_error_area_20"]:
            row[f"{col}_median"] = median_col(sub, col)
        for col in [
            "abs_error_centroid_offset",
            "euclidean_error_centroid_xy",
            "abs_error_anisotropy",
            "abs_error_r50",
            "abs_error_r80",
            "abs_error_r90",
            "abs_error_band_width",
        ]:
            row[f"{col}_mean"] = mean_col(sub, col)
        for col in ["csi10", "pod10", "far10", "f1_10", "csi20", "pod20", "far20", "f1_20"]:
            row[f"{col}_mean"] = mean_col(sub, col)
        row.update(duration_metrics.get((str(event_uid), str(model_version)), {}))
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_summary(event_summary: pd.DataFrame, timeslice_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    valid = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    for version in MODEL_VERSIONS:
        sub = valid.loc[valid["model_version"] == version]
        ev = event_summary.loc[event_summary["model_version"] == version]
        row: Dict[str, object] = {
            "model_version": version,
            "n_events": int(ev["validation_event_uid"].nunique()) if len(ev) else 0,
            "n_timeslices": int(len(sub)),
            "rmse_mean": mean_col(sub, "rmse"),
            "mae_mean": mean_col(sub, "mae"),
            "bias_mean": mean_col(sub, "bias"),
            "corr_mean": mean_col(sub, "corr"),
            "abs_error_rain_max_mean": mean_col(sub, "abs_error_rain_max"),
            "abs_error_rain_p95_mean": mean_col(sub, "abs_error_rain_p95"),
            "abs_error_rain_p99_mean": mean_col(sub, "abs_error_rain_p99"),
            "rel_error_rain_max_median": median_col(sub, "rel_error_rain_max"),
            "rel_error_rain_p95_median": median_col(sub, "rel_error_rain_p95"),
            "rel_error_rain_p99_median": median_col(sub, "rel_error_rain_p99"),
            "abs_error_area_10_mean": mean_col(sub, "abs_error_area_10"),
            "rel_error_area_10_median": median_col(sub, "rel_error_area_10"),
            "abs_error_area_20_mean": mean_col(sub, "abs_error_area_20"),
            "rel_error_area_20_median": median_col(sub, "rel_error_area_20"),
            "abs_error_centroid_offset_mean": mean_col(sub, "abs_error_centroid_offset"),
            "abs_error_anisotropy_mean": mean_col(sub, "abs_error_anisotropy"),
            "csi10_mean": mean_col(sub, "csi10"),
            "pod10_mean": mean_col(sub, "pod10"),
            "far10_mean": mean_col(sub, "far10"),
            "f1_10_mean": mean_col(sub, "f1_10"),
            "csi20_mean": mean_col(sub, "csi20"),
            "pod20_mean": mean_col(sub, "pod20"),
            "far20_mean": mean_col(sub, "far20"),
            "f1_20_mean": mean_col(sub, "f1_20"),
            "duration10_area_time_rel_error_median": median_col(ev, "duration10_area_time_rel_error"),
            "duration20_area_time_rel_error_median": median_col(ev, "duration20_area_time_rel_error"),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.loc[summary["model_version"] == "initial"].empty:
        return summary
    base = summary.loc[summary["model_version"] == "initial"].iloc[0]
    improvement_map = {
        "rmse_improvement_vs_initial": "rmse_mean",
        "p95_error_improvement_vs_initial": "abs_error_rain_p95_mean",
        "p99_error_improvement_vs_initial": "abs_error_rain_p99_mean",
        "area10_error_improvement_vs_initial": "abs_error_area_10_mean",
    }
    for out_col, metric_col in improvement_map.items():
        summary[out_col] = np.nan
        denom = float(base.get(metric_col, np.nan))
        if np.isfinite(denom) and abs(denom) > EPS:
            summary[out_col] = (denom - pd.to_numeric(summary[metric_col], errors="coerce")) / denom
    summary["csi10_improvement_vs_initial"] = pd.to_numeric(summary["csi10_mean"], errors="coerce") - float(base.get("csi10_mean", np.nan))
    return summary


# =========================
# Figures
# =========================


def plot_field(ax, field: np.ndarray, x_front_km: np.ndarray, y_left_km: np.ndarray, title: str, vmax: Optional[float] = None):
    im = ax.imshow(
        field,
        origin="lower",
        extent=[x_front_km[0], x_front_km[-1], y_left_km[0], y_left_km[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x_front_km")
    ax.set_ylabel("y_left_km")
    return im


def make_summary_figures(model_summary: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if model_summary.empty:
        return paths
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("rmse_mean", "RMSE"),
        ("mae_mean", "MAE"),
        ("corr_mean", "Corr"),
        ("abs_error_rain_p95_mean", "P95 abs err"),
        ("abs_error_rain_p99_mean", "P99 abs err"),
        ("abs_error_rain_max_mean", "Rmax abs err"),
        ("abs_error_area_10_mean", "Area10 abs err"),
        ("csi10_mean", "CSI10"),
    ]
    versions = model_summary["model_version"].astype(str).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), dpi=200)
    for ax, (col, label) in zip(axes.ravel(), metrics):
        vals = pd.to_numeric(model_summary[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(versions, vals, color=["#4c78a8", "#59a14f", "#f28e2b"])
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "validation_model_summary_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def make_event_timeseries_figures(timeslice_metrics: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if timeslice_metrics.empty:
        return paths
    fig_dir.mkdir(parents=True, exist_ok=True)
    for event_uid, sub in timeslice_metrics.groupby("validation_event_uid", sort=False):
        sub = sub.loc[sub["validation_ok"].astype(bool)].copy()
        if sub.empty:
            continue
        sub["validation_time"] = pd.to_datetime(sub["validation_time"], errors="coerce")
        name = sub["typhoon_name"].iloc[0]
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=200)
        truth = sub.drop_duplicates("validation_time").sort_values("validation_time")
        axes[0].plot(truth["validation_time"], truth["truth_rain_p95_mmhr"], color="black", linewidth=1.6, label="truth")
        axes[1].plot(truth["validation_time"], truth["truth_rain_area_10_km2"], color="black", linewidth=1.6, label="truth")
        styles = {"initial": "#4c78a8", "blend": "#59a14f", "calibrated": "#f28e2b"}
        for version, one in sub.groupby("model_version", sort=False):
            one = one.sort_values("validation_time")
            axes[0].plot(one["validation_time"], one["pred_rain_p95_mmhr"], color=styles.get(version), linewidth=1.0, label=version)
            axes[1].plot(one["validation_time"], one["pred_rain_area_10_km2"], color=styles.get(version), linewidth=1.0, label=version)
        axes[0].set_ylabel("P95 mm/hr")
        axes[1].set_ylabel("Area >=10 km2")
        axes[1].set_xlabel("validation_time")
        axes[0].set_title(f"{event_uid} {name} pseudo-missing validation")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(ncol=4, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = fig_dir / f"validation_event_{safe_name(event_uid)}_timeseries.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def make_spatial_compare_figures(
    representative_fields: Sequence[Mapping[str, object]],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    fig_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    fig_dir.mkdir(parents=True, exist_ok=True)
    for item in representative_fields:
        fields = item["fields"]
        truth = np.asarray(fields["truth"])
        vmax = float(np.nanpercentile(np.concatenate([truth.ravel(), fields["calibrated"].ravel()]), 99.5))
        vmax = max(vmax, 1.0)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), dpi=200)
        titles = ["truth", "initial", "blend", "calibrated", "calibrated - truth"]
        arrays = [
            fields["truth"],
            fields["initial"],
            fields["blend"],
            fields["calibrated"],
            fields["calibrated"] - fields["truth"],
        ]
        for ax, arr, title in zip(axes, arrays, titles):
            if title.endswith("truth") and title.startswith("calibrated"):
                im = ax.imshow(
                    arr,
                    origin="lower",
                    extent=[x_front_km[0], x_front_km[-1], y_left_km[0], y_left_km[-1]],
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.axvline(0, color="black", linewidth=0.4, alpha=0.8)
                ax.axhline(0, color="black", linewidth=0.4, alpha=0.8)
                ax.set_title(title, fontsize=9)
            else:
                im = plot_field(ax, arr, x_front_km, y_left_km, title, vmax=vmax)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.01)
        event_uid = item["event_uid"]
        time_label = pd.Timestamp(item["time"]).strftime("%Y%m%d_%H%M")
        fig.suptitle(f"{event_uid} {item['typhoon_name']} {time_label}", fontsize=10)
        path = fig_dir / f"validation_event_{safe_name(event_uid)}_spatial_compare_{time_label}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_truth_vs_pred_scatter(timeslice_metrics: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths: List[Path] = []
    sub = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    if sub.empty:
        return paths
    fig_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("truth_rain_p95_mmhr", "pred_rain_p95_mmhr", "P95"),
        ("truth_rain_p99_mmhr", "pred_rain_p99_mmhr", "P99"),
        ("truth_rain_area_10_km2", "pred_rain_area_10_km2", "Area >=10"),
        ("truth_rain_max_mmhr", "pred_rain_max_mmhr", "Rmax"),
    ]
    colors = {"initial": "#4c78a8", "blend": "#59a14f", "calibrated": "#f28e2b"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=200)
    for ax, (xcol, ycol, title) in zip(axes.ravel(), pairs):
        for version, one in sub.groupby("model_version", sort=False):
            x = pd.to_numeric(one[xcol], errors="coerce")
            y = pd.to_numeric(one[ycol], errors="coerce")
            ax.scatter(x, y, s=10, alpha=0.45, label=version, color=colors.get(version))
        lim = np.nanmax([pd.to_numeric(sub[xcol], errors="coerce").max(), pd.to_numeric(sub[ycol], errors="coerce").max()])
        if np.isfinite(lim) and lim > 0:
            ax.plot([0, lim], [0, lim], color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "validation_truth_vs_pred_extremes.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def make_duration_error_figure(event_summary: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths: List[Path] = []
    if event_summary.empty:
        return paths
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"initial": "#4c78a8", "blend": "#59a14f", "calibrated": "#f28e2b"}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for version, sub in event_summary.groupby("model_version", sort=False):
        x = pd.to_numeric(sub["duration10_total_area_time_truth"], errors="coerce")
        y = pd.to_numeric(sub["duration10_total_area_time_pred"], errors="coerce")
        ax.scatter(x, y, s=35, alpha=0.75, label=version, color=colors.get(version))
    lim = np.nanmax(
        [
            pd.to_numeric(event_summary["duration10_total_area_time_truth"], errors="coerce").max(),
            pd.to_numeric(event_summary["duration10_total_area_time_pred"], errors="coerce").max(),
        ]
    )
    if np.isfinite(lim) and lim > 0:
        ax.plot([0, lim], [0, lim], color="black", linewidth=0.8)
    ax.set_xlabel("truth duration10 area-time")
    ax.set_ylabel("pred duration10 area-time")
    ax.set_title("Duration >=10 mm/hr area-time validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "validation_duration_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def make_summary_figures_all(
    model_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    timeslice_metrics: pd.DataFrame,
    representative_fields: Sequence[Mapping[str, object]],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> List[Path]:
    if not MAKE_FIGURES:
        return []
    paths: List[Path] = []
    paths.extend(make_summary_figures(model_summary, FIGURE_DIR))
    paths.extend(make_event_timeseries_figures(timeslice_metrics, FIGURE_DIR))
    paths.extend(make_spatial_compare_figures(representative_fields, x_front_km, y_left_km, FIGURE_DIR))
    paths.extend(make_truth_vs_pred_scatter(timeslice_metrics, FIGURE_DIR))
    paths.extend(make_duration_error_figure(event_summary, FIGURE_DIR))
    return paths


# =========================
# QC report
# =========================


def format_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return "(none)"
    small = df[existing].head(max_rows).copy()
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda v: f"{v:.6g}" if pd.notna(v) and np.isfinite(v) else "NA")
        else:
            small[col] = small[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(existing) + " |"
    sep = "| " + " | ".join(["---"] * len(existing)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in existing) + " |" for _, row in small.iterrows()]
    return "\n".join([header, sep] + body)


def improvement_line(model_summary: pd.DataFrame, version: str, metric: str) -> str:
    sub = model_summary.loc[model_summary["model_version"] == version]
    if sub.empty or metric not in sub.columns:
        return "NA"
    value = pd.to_numeric(sub[metric], errors="coerce").iloc[0]
    return f"{value:.2%}" if np.isfinite(value) else "NA"


def write_qc_report(
    validation_events: pd.DataFrame,
    timeslice_metrics: pd.DataFrame,
    event_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    diagnostics: Mapping[str, object],
    figure_paths: Sequence[Path],
) -> None:
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    valid = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    skip = timeslice_metrics.loc[~timeslice_metrics["validation_ok"].astype(bool)].copy()
    selected_features = diagnostics.get("selected_features", [])
    leakage_used = [f for f in selected_features if is_leakage_feature(str(f))]
    self_match_count = int(diagnostics.get("self_match_count", 0))
    lines: List[str] = [
        "# Problem 2 Pseudo-Missing Validation QC Report",
        "",
        "## 1. Input and Output Files",
        f"- Historical library: `{HISTORICAL_LIBRARY_PATH}`",
        f"- EOF/PCA model: `{EOF_MODEL_PATH}`",
        f"- Validation events CSV: `{VALIDATION_EVENTS_OUTPUT_PATH}`",
        f"- Timeslice metrics CSV: `{TIMESLICE_METRICS_OUTPUT_PATH}`",
        f"- Event summary CSV: `{EVENT_SUMMARY_OUTPUT_PATH}`",
        f"- Model summary CSV: `{MODEL_SUMMARY_OUTPUT_PATH}`",
        f"- Representative fields NPZ: `{GENERATED_FIELDS_OUTPUT_PATH}`",
        f"- Figures directory: `{FIGURE_DIR}`",
        "",
        "## 2. Run Parameters",
        f"- N_VALIDATION_EVENTS: {N_VALIDATION_EVENTS}",
        f"- MAX_TIMES_PER_EVENT: {MAX_TIMES_PER_EVENT}",
        f"- TOPK: {TOPK}",
        f"- MAX_PER_HISTORY_EVENT: {MAX_PER_HISTORY_EVENT}",
        f"- MIN_VALID_TEMPLATES: {MIN_VALID_TEMPLATES}",
        f"- ENV_FEATURE_SET: {ENV_FEATURE_SET}",
        f"- D_env: mean squared standardized difference across selected environment features",
        f"- BETA_BLEND: {BETA_BLEND}",
        f"- USE_EOF_MODEL: {USE_EOF_MODEL}",
        f"- USE_EXTREME_CALIBRATION: {USE_EXTREME_CALIBRATION}",
        f"- RANDOM_SEED: {RANDOM_SEED}",
        "",
        "## 3. Validation Event Selection",
        format_table(
            validation_events,
            [
                "validation_event_uid",
                "typhoon_name",
                "start_time",
                "end_time",
                "n_total_times",
                "n_selected_times",
                "WND_max",
                "PRES_min",
                "rain_p95_mmhr_max",
                "selection_reason",
            ],
            30,
        ),
        "",
        "## 4. Filtering and Leakage Guard",
        f"- Training library excludes the current validation event for each run: yes",
        f"- Leakage fields used in retrieval: {'no' if not leakage_used else ', '.join(leakage_used)}",
        f"- Truth tif is read only in evaluation stage: yes",
        f"- KONG-REY / MAN-YI selected as validation events: no",
        f"- Top-K rows from the same validation event: {self_match_count}",
        f"- Safe retrieval features: {', '.join(map(str, selected_features))}",
        "",
        "## 5. Generation Process Statistics",
        f"- Total selected validation times: {int(validation_events['n_selected_times'].sum()) if 'n_selected_times' in validation_events else 'NA'}",
        f"- Timeslice metric rows: {len(timeslice_metrics)}",
        f"- Successful model-version rows: {int(valid.shape[0])}",
        f"- Skipped/flagged model-version rows: {int(skip.shape[0])}",
        f"- Skip reasons: {skip['skip_reason'].value_counts(dropna=False).to_dict() if len(skip) and 'skip_reason' in skip else {}}",
        f"- valid_template_count distribution: {stats_text(timeslice_metrics['valid_template_count']) if 'valid_template_count' in timeslice_metrics else 'NA'}",
        f"- topk_unique_event_count distribution: {stats_text(timeslice_metrics['topk_unique_event_count']) if 'topk_unique_event_count' in timeslice_metrics else 'NA'}",
        f"- Template cache counters: {diagnostics.get('template_counters', {})}",
        "",
        "## 6. Overall Performance by Version",
        format_table(
            model_summary,
            [
                "model_version",
                "n_events",
                "n_timeslices",
                "rmse_mean",
                "mae_mean",
                "bias_mean",
                "corr_mean",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "abs_error_rain_max_mean",
                "abs_error_area_10_mean",
                "abs_error_area_20_mean",
                "abs_error_centroid_offset_mean",
                "abs_error_anisotropy_mean",
                "csi10_mean",
                "pod10_mean",
                "far10_mean",
                "f1_10_mean",
                "csi20_mean",
                "f1_20_mean",
                "duration10_area_time_rel_error_median",
                "duration20_area_time_rel_error_median",
            ],
            10,
        ),
        "",
        "## 7. Module Gain Analysis",
        f"- Blend vs initial RMSE improvement: {improvement_line(model_summary, 'blend', 'rmse_improvement_vs_initial')}",
        f"- Calibrated vs initial RMSE improvement: {improvement_line(model_summary, 'calibrated', 'rmse_improvement_vs_initial')}",
        f"- Calibrated vs initial P95 error improvement: {improvement_line(model_summary, 'calibrated', 'p95_error_improvement_vs_initial')}",
        f"- Calibrated vs initial P99 error improvement: {improvement_line(model_summary, 'calibrated', 'p99_error_improvement_vs_initial')}",
        f"- Calibrated vs initial area10 error improvement: {improvement_line(model_summary, 'calibrated', 'area10_error_improvement_vs_initial')}",
        f"- Calibrated vs initial CSI10 change: {model_summary.loc[model_summary['model_version'].eq('calibrated'), 'csi10_improvement_vs_initial'].iloc[0]:.6g}"
        if "csi10_improvement_vs_initial" in model_summary.columns and model_summary["model_version"].eq("calibrated").any()
        else "- Calibrated vs initial CSI10 change: NA",
    ]

    if not model_summary.empty:
        initial = model_summary.loc[model_summary["model_version"] == "initial"].iloc[0]
        blend = model_summary.loc[model_summary["model_version"] == "blend"].iloc[0]
        calibrated = model_summary.loc[model_summary["model_version"] == "calibrated"].iloc[0]
        lines.extend(
            [
                f"- Blend spatial structure: centroid error {blend.get('abs_error_centroid_offset_mean', np.nan):.6g} vs initial {initial.get('abs_error_centroid_offset_mean', np.nan):.6g}; anisotropy error {blend.get('abs_error_anisotropy_mean', np.nan):.6g} vs initial {initial.get('abs_error_anisotropy_mean', np.nan):.6g}.",
                f"- Calibration extremes: P95/P99/area10 errors are {calibrated.get('abs_error_rain_p95_mean', np.nan):.6g}, {calibrated.get('abs_error_rain_p99_mean', np.nan):.6g}, {calibrated.get('abs_error_area_10_mean', np.nan):.6g}.",
                f"- Calibration detection: CSI10/F1_10 are {calibrated.get('csi10_mean', np.nan):.6g}, {calibrated.get('f1_10_mean', np.nan):.6g}.",
                f"- RMSE/correlation tradeoff: calibrated RMSE/corr are {calibrated.get('rmse_mean', np.nan):.6g}, {calibrated.get('corr_mean', np.nan):.6g}; initial RMSE/corr are {initial.get('rmse_mean', np.nan):.6g}, {initial.get('corr_mean', np.nan):.6g}.",
            ]
        )

    worst_rmse = valid.sort_values("rmse", ascending=False).head(10)
    worst_p95 = valid.sort_values("abs_error_rain_p95", ascending=False).head(10)
    worst_area10 = valid.sort_values("abs_error_area_10", ascending=False).head(10)
    worst_corr = valid.sort_values("corr", ascending=True).head(10)
    fail_cols = [
        "validation_event_uid",
        "validation_time",
        "model_version",
        "rmse",
        "abs_error_rain_p95",
        "abs_error_area_10",
        "corr",
        "WND",
        "PRES",
        "signed_coast_dist_km",
    ]
    lines.extend(
        [
            "",
            "## 8. Failure and Abnormal Samples",
            "### Top 10 RMSE",
            format_table(worst_rmse, fail_cols, 10),
            "",
            "### Top 10 P95 Error",
            format_table(worst_p95, fail_cols, 10),
            "",
            "### Top 10 Area10 Error",
            format_table(worst_area10, fail_cols, 10),
            "",
            "### Lowest 10 Correlation",
            format_table(worst_corr, fail_cols, 10),
            "",
            "Likely causes include analog mismatch during rapid intensity change, coastal terrain discontinuity, compact convective cores that are hard to recover from analog means, and sparse heavy-rain truth coverage near the edge of the storm-relative tile.",
            "",
            "## 9. Paper-Ready Conclusions",
            "- The initial Top-K log1p analog field provides a no-target-rainfall baseline that recovers broad storm-relative rainfall placement.",
            "- EOF/PCA blending acts as a large-scale structural regularizer and can be discussed through centroid, anisotropy, RMSE, and correlation changes.",
            "- Extreme quantile calibration directly targets Top-K-derived P95/P99/Rmax and heavy-rain area constraints without reading validation-event rain metrics.",
            "- P95/P99 and heavy-rain area are more stable validation targets than single-grid Rmax, which remains sensitive to small convective cores.",
            "- Threshold metrics such as CSI10 and F1_10 quantify whether heavy-rain identification improves after calibration.",
            "- The self-exclusion check confirms that validation-event samples do not enter Top-K templates.",
            "- The pseudo-missing experiment shows whether the model can generate physically plausible typhoon rainfall structures when the target GPM field is unavailable.",
            f"- Generated figure count: {len(figure_paths)}",
        ]
    )
    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def save_representative_npz(
    representative_fields: Sequence[Mapping[str, object]],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
) -> None:
    if not representative_fields:
        return
    fields_truth = []
    fields_initial = []
    fields_blend = []
    fields_calibrated = []
    event_uids = []
    typhoon_names = []
    times = []
    for item in representative_fields:
        fields = item["fields"]
        fields_truth.append(np.asarray(fields["truth"], dtype=np.float32))
        fields_initial.append(np.asarray(fields["initial"], dtype=np.float32))
        fields_blend.append(np.asarray(fields["blend"], dtype=np.float32))
        fields_calibrated.append(np.asarray(fields["calibrated"], dtype=np.float32))
        event_uids.append(str(item["event_uid"]))
        typhoon_names.append(str(item["typhoon_name"]))
        times.append(format_time(item["time"]))
    GENERATED_FIELDS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GENERATED_FIELDS_OUTPUT_PATH,
        truth=np.stack(fields_truth).astype(np.float32),
        initial=np.stack(fields_initial).astype(np.float32),
        blend=np.stack(fields_blend).astype(np.float32),
        calibrated=np.stack(fields_calibrated).astype(np.float32),
        validation_event_uid=np.asarray(event_uids, dtype="U"),
        typhoon_name=np.asarray(typhoon_names, dtype="U"),
        validation_time=np.asarray(times, dtype="U"),
        x_front_km=x_front_km.astype(np.float32),
        y_left_km=y_left_km.astype(np.float32),
    )


def choose_representative_indices(selected_times: pd.DataFrame) -> set:
    picks: set = set()
    if selected_times.empty:
        return picks
    if "rain_p95_mmhr" in selected_times and pd.to_numeric(selected_times["rain_p95_mmhr"], errors="coerce").notna().any():
        picks.add(int(pd.to_numeric(selected_times["rain_p95_mmhr"], errors="coerce").idxmax()))
    if "WND" in selected_times and pd.to_numeric(selected_times["WND"], errors="coerce").notna().any():
        picks.add(int(pd.to_numeric(selected_times["WND"], errors="coerce").idxmax()))
    if len(picks) < 2:
        picks.add(int(selected_times.index[len(selected_times) // 2]))
    return set(list(picks)[:2])


def main() -> None:
    np.random.seed(RANDOM_SEED)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_EVENTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[22] Loading historical library and EOF/PCA model")
    history = load_historical_library()
    eof_model = load_eof_pca_model()
    x_front_km, y_left_km, x_grid, y_grid = build_relative_grid()
    cell_area = cell_area_from_grid(x_front_km, y_left_km)

    print("[22] Selecting validation events")
    validation_events = select_validation_events(history)

    all_metric_rows: List[Dict[str, object]] = []
    duration_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    representative_fields: List[Dict[str, object]] = []
    global_qc_events: List[Dict[str, object]] = []
    template_counters: Counter = Counter()
    retrieval_diagnostics_all: List[Dict[str, object]] = []
    selected_feature_union: List[str] = []
    self_match_total = 0
    skipped_times: Counter = Counter()

    cache = TemplateCache(CACHE_SIZE)
    selected_counts: Dict[str, int] = {}
    iterator = iter_progress(validation_events.iterrows(), total=len(validation_events), desc="Validation events")
    for _, event_meta in iterator:
        event_uid = str(event_meta["validation_event_uid"])
        print(f"[22] Event {event_uid} {event_meta['typhoon_name']}: retrieval and generation")
        event_full = history.loc[history["event_uid"].astype(str) == event_uid].copy().sort_values("time").reset_index(drop=True)
        selected_times = select_validation_times_for_event(event_full)
        selected_counts[event_uid] = int(len(selected_times))
        train_history = history.loc[history["event_uid"].astype(str) != event_uid].copy().reset_index(drop=True)
        if train_history.empty:
            raise RuntimeError(f"Training library is empty after excluding validation event {event_uid}")
        if train_history["event_uid"].astype(str).eq(event_uid).any():
            raise RuntimeError(f"Self event {event_uid} remains in training library.")

        selected_components, skipped_features = select_safe_retrieval_features(train_history, selected_times)
        features = [f for feats in selected_components.values() for f in feats]
        for f in features:
            if f not in selected_feature_union:
                selected_feature_union.append(f)
        train_imp, target_imp, _ = impute_retrieval_features(train_history, selected_times, selected_components)
        standardization_params = compute_standardization_params(train_imp, features)
        topk, retrieval_diag = retrieve_topk_for_validation_times(
            train_imp,
            target_imp,
            selected_components,
            standardization_params,
        )
        retrieval_diag["validation_event_uid"] = event_uid
        retrieval_diag["skipped_features"] = skipped_features
        retrieval_diagnostics_all.append(retrieval_diag)
        self_match_total += int(retrieval_diag.get("self_match_count", 0))
        if retrieval_diag.get("self_match_count", 0):
            raise RuntimeError(f"Top-K leakage: validation event {event_uid} appears in its own Top-K.")
        topk_by_target = {tid: sub.copy() for tid, sub in topk.groupby("target_id", sort=False)}
        representative_idx = choose_representative_indices(selected_times)
        duration_state: Dict[str, np.ndarray] = {
            "truth10": np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
            "truth20": np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        }
        for version in MODEL_VERSIONS:
            duration_state[f"{version}10"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            duration_state[f"{version}20"] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        time_iter = iter_progress(selected_times.iterrows(), total=len(selected_times), desc=f"{event_uid} times")
        for local_df_index, target_row in time_iter:
            target_id = str(target_row["sample_id"])
            topk_one = topk_by_target.get(target_id, pd.DataFrame(columns=topk.columns))
            try:
                truth, _ = build_truth_field(target_row, x_grid, y_grid)
            except Exception as exc:
                reason = f"truth_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                global_qc_events.append({"event_uid": event_uid, "target_id": target_id, "reason": reason})
                continue
            try:
                rain_initial, log_initial, generation_status = generate_initial_field(
                    target_row,
                    topk_one,
                    cache,
                    x_grid,
                    y_grid,
                    global_qc_events,
                    template_counters,
                )
            except Exception as exc:
                reason = f"initial_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                global_qc_events.append({"event_uid": event_uid, "target_id": target_id, "reason": reason})
                continue
            if generation_status.get("valid_template_count", 0) < MIN_VALID_TEMPLATES:
                skipped_times["low_valid_template_count"] += 1
            try:
                rain_blend, log_blend, _ = apply_eof_blend(rain_initial, log_initial, eof_model, BETA_BLEND)
            except Exception as exc:
                reason = f"blend_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                global_qc_events.append({"event_uid": event_uid, "target_id": target_id, "reason": reason})
                continue
            calibration_targets = compute_weighted_calibration_targets(topk_one)
            try:
                rain_calibrated, cal_diag = calibrate_one_field_tail_enhancement(rain_blend, calibration_targets, cell_area)
            except Exception as exc:
                reason = f"calibrated_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                global_qc_events.append({"event_uid": event_uid, "target_id": target_id, "reason": reason})
                continue

            fields = {"initial": rain_initial, "blend": rain_blend, "calibrated": rain_calibrated}
            duration_state["truth10"] += (truth >= 10.0).astype(np.float32)
            duration_state["truth20"] += (truth >= 20.0).astype(np.float32)
            validation_ok = bool(generation_status.get("valid_template_count", 0) >= MIN_VALID_TEMPLATES)
            skip_reason = "" if validation_ok else "low_valid_template_count"
            for version, pred in fields.items():
                duration_state[f"{version}10"] += (pred >= 10.0).astype(np.float32)
                duration_state[f"{version}20"] += (pred >= 20.0).astype(np.float32)
                metric_row = compute_timeslice_validation_metrics(
                    target_row,
                    version,
                    pred,
                    truth,
                    topk_one,
                    generation_status,
                    calibration_targets,
                    x_front_km,
                    y_left_km,
                    x_grid,
                    y_grid,
                    validation_ok=validation_ok,
                    skip_reason=skip_reason,
                )
                metric_row["calibration_ok"] = bool(cal_diag.get("calibration_ok", True)) if version == "calibrated" else np.nan
                metric_row["calibration_issue"] = cal_diag.get("calibration_issue", "") if version == "calibrated" else ""
                all_metric_rows.append(metric_row)

            if int(local_df_index) in representative_idx and len([r for r in representative_fields if r["event_uid"] == event_uid]) < 2:
                representative_fields.append(
                    {
                        "event_uid": event_uid,
                        "typhoon_name": target_row.get("typhoon_name"),
                        "time": target_row.get("time"),
                        "fields": {
                            "truth": truth.astype(np.float32),
                            "initial": rain_initial.astype(np.float32),
                            "blend": rain_blend.astype(np.float32),
                            "calibrated": rain_calibrated.astype(np.float32),
                        },
                    }
                )

        for version in MODEL_VERSIONS:
            duration_metrics[(event_uid, version)] = compute_event_duration_metrics(duration_state, version, cell_area)

    validation_events["n_selected_times"] = validation_events["validation_event_uid"].astype(str).map(selected_counts).fillna(0).astype(int)
    validation_events.to_csv(VALIDATION_EVENTS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    timeslice_metrics = pd.DataFrame(all_metric_rows)
    if timeslice_metrics.empty:
        raise RuntimeError("No timeslice metrics were generated.")
    timeslice_metrics.to_csv(TIMESLICE_METRICS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    event_summary = build_event_summary(timeslice_metrics, duration_metrics)
    event_summary.to_csv(EVENT_SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    model_summary = build_model_summary(event_summary, timeslice_metrics)
    model_summary.to_csv(MODEL_SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    save_representative_npz(representative_fields, x_front_km, y_left_km)
    figure_paths = make_summary_figures_all(model_summary, event_summary, timeslice_metrics, representative_fields, x_front_km, y_left_km)

    diagnostics = {
        "selected_features": selected_feature_union,
        "self_match_count": self_match_total,
        "template_counters": dict(template_counters),
        "qc_events": global_qc_events[:200],
        "retrieval_diagnostics": retrieval_diagnostics_all,
        "skipped_times": dict(skipped_times),
        "cache_final_size": len(cache),
    }
    write_qc_report(validation_events, timeslice_metrics, event_summary, model_summary, diagnostics, figure_paths)

    print("[22] Completed pseudo-missing validation")
    print(f"[22] Validation events: {VALIDATION_EVENTS_OUTPUT_PATH}")
    print(f"[22] Timeslice metrics: {TIMESLICE_METRICS_OUTPUT_PATH}")
    print(f"[22] Event summary: {EVENT_SUMMARY_OUTPUT_PATH}")
    print(f"[22] Model summary: {MODEL_SUMMARY_OUTPUT_PATH}")
    print(f"[22] QC report: {QC_REPORT_PATH}")
    print(f"[22] Figures: {FIGURE_DIR} ({len(figure_paths)} files)")


if __name__ == "__main__":
    main()
