from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. Paths and constants
# =========================

TARGET_TRACK_PATH = Path("output/target_typhoon_tracks_2024_with_coast.csv")
HIST_REF_PATH = Path("data/processed/gpm_track_interpolated_features_clean.csv")

OUT_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/tables/field_audit")

OUT_TRACK_SAFE = OUT_DIR / "target_typhoon_inputs_2024_track_points_leakage_safe.csv"
OUT_HALFHOUR_SAFE = OUT_DIR / "target_typhoon_inputs_2024_halfhour_leakage_safe.csv"
OUT_HALFHOUR_X = OUT_DIR / "target_typhoon_model_x_2024_halfhour.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TARGETS = {
    "KONGREY": {
        "display_name": "KONG-REY",
        "source_file": "CH2024BST.txt",
        "start": pd.Timestamp("2024-10-24 00:00:00"),
        "end": pd.Timestamp("2024-11-02 23:59:59"),
    },
    "MANYI": {
        "display_name": "MAN-YI",
        "source_file": "CH2024BST.txt",
        "start": pd.Timestamp("2024-11-08 00:00:00"),
        "end": pd.Timestamp("2024-11-20 23:59:59"),
    },
}

AUDIT_TABLES = {
    "target_track_with_coast": TARGET_TRACK_PATH,
    "historical_gpm_precip_features": Path("data/processed/gpm_precip_features.csv"),
    "historical_gpm_track_interpolated_clean": HIST_REF_PATH,
    "historical_gpm_track_model_features_interp": Path(
        "data/processed/gpm_track_model_features_interp.csv"
    ),
    "historical_gpm_track_model_features_interp_env": Path(
        "data/processed/env_added/gpm_track_model_features_interp_env.csv"
    ),
    "historical_gpm_track_model_features": Path("data/processed/gpm_track_model_features.csv"),
}

MODEL_X_COLS = [
    "track_wind",
    "track_pressure",
    "pressure_deficit",
    "intensity_index",
    "track_move_speed_kmh",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

TARGET_SAFE_COLS = [
    "target_name_norm",
    "target_display_name",
    "event_uid",
    "source_file",
    "typhoon_id",
    "storm_seq",
    "typhoon_code",
    "record_count",
    "time",
    "cadence_hours",
    "center_lon",
    "center_lat",
    "track_lat",
    "track_lon_180",
    "track_intensity",
    "track_pressure",
    "track_wind",
    "track_dt_h",
    "track_move_distance_km",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "year",
    "month",
    "day",
    "hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",
    "wind_z",
    "pressure_deficit",
    "pressure_deficit_z",
    "intensity_index",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",
]

MOTION_EDGE_FILL_COLS = [
    "cadence_hours",
    "track_dt_h",
    "track_move_distance_km",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_wind_change_rate",
    "track_pressure_change_rate",
]


# =========================
# 2. Leakage rules
# =========================

LEAKAGE_PATTERNS = [
    re.compile(r"^rain_", re.IGNORECASE),
    re.compile(r"^rainband_", re.IGNORECASE),
    re.compile(r"^rainband10_", re.IGNORECASE),
    re.compile(r"^centroid_", re.IGNORECASE),
    re.compile(r"^rain_centroid_", re.IGNORECASE),
    re.compile(r"^anisotropy$", re.IGNORECASE),
    re.compile(r"^asym_", re.IGNORECASE),
    re.compile(r"^quad_", re.IGNORECASE),
    re.compile(r"^r(50|80|90)(_.*)?$", re.IGNORECASE),
    re.compile(r"^major_axis_km$", re.IGNORECASE),
    re.compile(r"^minor_axis_km$", re.IGNORECASE),
    re.compile(r"^orientation_deg$", re.IGNORECASE),
]

HISTORICAL_OUTPUT_EXACT = {
    "valid_count",
    "rain_valid_ratio",
    "rain_gini",
    "rain_entropy_norm",
    "rain_halfhour_mean_mm",
    "rain_halfhour_sum_mm_grid",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "rain_area_20_km2",
    "rain_area_20_equiv_radius_km",
    "rainband_width_km",
    "rainband_length_km",
    "rainband_aspect_ratio",
    "rainband_width10_km",
    "rainband_length10_km",
    "rainband_aspect_ratio10",
    "rainband_angle_to_motion_deg",
    "centroid_relative_to_motion_deg",
    "centroid_relative_to_motion_sin",
    "centroid_relative_to_motion_cos",
    "centroid_in_front",
}

METADATA_EXACT = {
    "event_uid",
    "gpm_event_uid",
    "track_event_uid",
    "source_file",
    "track_source_file",
    "source_path",
    "source_name",
    "filename",
    "file_name",
    "tif_path",
    "source_tif",
    "typhoon_id",
    "track_typhoon_id",
    "storm_seq",
    "track_storm_seq",
    "typhoon_code",
    "track_typhoon_code",
    "record_count",
    "track_record_count",
    "typhoon_name",
    "track_typhoon_name",
    "name_norm",
    "target_name_norm",
    "target_display_name",
    "source_file",
    "time",
    "time_end",
    "track_time",
    "matched_track_index",
    "match_status",
    "interp_match_status",
    "outside_track_time_range",
    "track_time_diff_h",
    "gpm_track_center_dist_km",
    "interp_center_error_km",
    "interp_center_error_ok",
    "cadence_hours",
}

TARGET_INPUT_EXACT = {
    "lat",
    "lon",
    "lon_180",
    "center_lon",
    "center_lat",
    "bbox_lon_min",
    "bbox_lon_max",
    "bbox_lat_min",
    "bbox_lat_max",
    "intensity",
    "pressure",
    "wind",
    "dt_h",
    "move_distance_km",
    "move_speed_kmh",
    "move_dir_deg",
    "move_dir_sin",
    "move_dir_cos",
    "wind_change_rate",
    "pressure_change_rate",
    "is_land",
    "coast_dist_km",
    "signed_coast_dist_km",
    "track_lat",
    "track_lon",
    "track_lon_180",
    "track_intensity",
    "track_pressure",
    "track_wind",
    "track_dt_h",
    "track_move_distance_km",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "track_is_land",
    "track_is_land_interp",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "year",
    "month",
    "day",
    "hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",
    "wind_z",
    "pressure_deficit",
    "pressure_deficit_z",
    "intensity_index",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",
}


# =========================
# 3. Utility functions
# =========================

def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def is_leakage_col(col: str) -> bool:
    col_clean = col.lstrip("\ufeff")
    if col_clean in HISTORICAL_OUTPUT_EXACT:
        return True
    return any(pattern.search(col_clean) for pattern in LEAKAGE_PATTERNS)


def classify_col(col: str) -> str:
    col_clean = col.lstrip("\ufeff")

    if is_leakage_col(col_clean):
        return "historical_output_or_calibration"

    if col_clean in METADATA_EXACT or col_clean.endswith("_match_status"):
        return "metadata_or_index"

    if col_clean in TARGET_INPUT_EXACT:
        return "target_available_input"

    if col_clean.startswith("track_"):
        return "target_available_input_review"

    return "review_unknown"


def action_for_col(field_class: str) -> str:
    if field_class == "historical_output_or_calibration":
        return "drop_from_target_input"
    if field_class == "metadata_or_index":
        return "keep_as_metadata_only"
    if field_class.startswith("target_available_input"):
        return "keep_as_candidate_input"
    return "manual_review_before_model_use"


def read_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)


def haversine_km(lon1, lat1, lon2, lat2):
    radius = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )
    deg = np.degrees(np.arctan2(x, y))
    return (deg + 360) % 360


def interp_numeric(t_src, y_src, t_new):
    y_src = pd.to_numeric(pd.Series(y_src), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_src)
    if valid.sum() < 2:
        return np.full_like(t_new, np.nan, dtype=float)
    return np.interp(t_new, t_src[valid], y_src[valid])


def interp_lon_degree(t_src, lon_src, t_new):
    lon_src = pd.to_numeric(pd.Series(lon_src), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lon_src)
    if valid.sum() < 2:
        return np.full_like(t_new, np.nan, dtype=float)

    lon_rad = np.deg2rad(lon_src[valid])
    lon_unwrap = np.unwrap(lon_rad)
    interp_rad = np.interp(t_new, t_src[valid], lon_unwrap)
    interp_lon = np.rad2deg(interp_rad)
    return ((interp_lon + 180) % 360) - 180


def datetime_to_seconds(series: pd.Series) -> np.ndarray:
    return series.astype("int64").to_numpy() / 1e9


def nearest_rows_by_time(source: pd.DataFrame, target_times: pd.Series) -> pd.DataFrame:
    source = source.sort_values("time").reset_index(drop=True)
    src_times = source["time"].values.astype("datetime64[ns]")
    rows = []

    for target_time in target_times.values.astype("datetime64[ns]"):
        pos = np.searchsorted(src_times, target_time)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(source):
            candidates.append(pos)
        best = min(
            candidates,
            key=lambda i: abs(
                (pd.Timestamp(target_time) - source.loc[i, "time"]).total_seconds()
            ),
        )
        rows.append(source.loc[best])

    return pd.DataFrame(rows).reset_index(drop=True)


def assert_no_leakage(df: pd.DataFrame, label: str) -> None:
    leaked = [col for col in df.columns if is_leakage_col(col)]
    if leaked:
        raise ValueError(f"{label} contains leakage columns: {leaked}")


def fill_initial_motion_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fill_cols = [col for col in MOTION_EDGE_FILL_COLS if col in df.columns]

    if not fill_cols:
        return df

    for _, idx in df.groupby("event_uid").groups.items():
        idx = list(idx)
        df.loc[idx, fill_cols] = df.loc[idx, fill_cols].bfill()

    return df


# =========================
# 4. Audit reports
# =========================

def audit_headers() -> pd.DataFrame:
    records = []

    for table_name, path in AUDIT_TABLES.items():
        columns = read_header(path)
        if not columns:
            records.append(
                {
                    "table_name": table_name,
                    "path": str(path),
                    "column": "",
                    "exists": False,
                    "field_class": "missing_table",
                    "banned_in_target_input": False,
                    "action": "check_path",
                }
            )
            continue

        for col in columns:
            field_class = classify_col(col)
            records.append(
                {
                    "table_name": table_name,
                    "path": str(path),
                    "column": col,
                    "exists": True,
                    "field_class": field_class,
                    "banned_in_target_input": is_leakage_col(col),
                    "action": action_for_col(field_class),
                }
            )

    audit = pd.DataFrame(records)
    audit.to_csv(REPORT_DIR / "field_classification_audit.csv", index=False, encoding="utf-8-sig")

    summary = (
        audit.groupby(["table_name", "field_class", "banned_in_target_input"])
        .size()
        .reset_index(name="n_columns")
    )
    summary.to_csv(REPORT_DIR / "field_classification_summary.csv", index=False, encoding="utf-8-sig")

    return audit


# =========================
# 5. Target input generation
# =========================

def load_reference_stats() -> dict[str, float]:
    if not HIST_REF_PATH.exists():
        raise FileNotFoundError(f"Missing historical reference table: {HIST_REF_PATH}")

    ref = pd.read_csv(HIST_REF_PATH, usecols=["track_wind", "track_pressure"])
    wind = pd.to_numeric(ref["track_wind"], errors="coerce")
    pressure = pd.to_numeric(ref["track_pressure"], errors="coerce")
    pressure_deficit = pressure.mean() - pressure

    stats = {
        "wind_mean": float(wind.mean()),
        "wind_std": float(wind.std()),
        "pressure_mean": float(pressure.mean()),
        "pressure_deficit_mean": float(pressure_deficit.mean()),
        "pressure_deficit_std": float(pressure_deficit.std()),
        "reference_rows": int(len(ref)),
    }

    pd.DataFrame([stats]).to_csv(
        REPORT_DIR / "feature_reference_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return stats


def validate_target_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_name_norm"] = df["typhoon_name"].apply(normalize_name)

    unknown = sorted(set(df["target_name_norm"]) - set(EXPECTED_TARGETS))
    if unknown:
        raise ValueError(f"Unexpected target names in target track table: {unknown}")

    records = []
    for name_norm, cfg in EXPECTED_TARGETS.items():
        sub = df[df["target_name_norm"] == name_norm].copy()
        if sub.empty:
            raise ValueError(f"Missing target typhoon: {cfg['display_name']}")

        if set(sub["source_file"]) != {cfg["source_file"]}:
            raise ValueError(
                f"{cfg['display_name']} source_file mismatch: {sorted(set(sub['source_file']))}"
            )

        if sub["time"].min() < cfg["start"] or sub["time"].max() > cfg["end"]:
            raise ValueError(
                f"{cfg['display_name']} time range is outside expected problem window"
            )

        if sub[["lat", "lon_180", "pressure", "wind"]].isna().any().any():
            raise ValueError(f"{cfg['display_name']} has missing core track values")

        if not sub["lat"].between(-90, 90).all():
            raise ValueError(f"{cfg['display_name']} latitude outside [-90, 90]")

        if not sub["lon_180"].between(-180, 180).all():
            raise ValueError(f"{cfg['display_name']} lon_180 outside [-180, 180]")

        records.append(
            {
                "target_name_norm": name_norm,
                "target_display_name": cfg["display_name"],
                "rows": len(sub),
                "actual_start": sub["time"].min(),
                "actual_end": sub["time"].max(),
                "expected_window_start": cfg["start"],
                "expected_window_end": cfg["end"],
                "min_lat": sub["lat"].min(),
                "max_lat": sub["lat"].max(),
                "min_lon_180": sub["lon_180"].min(),
                "max_lon_180": sub["lon_180"].max(),
                "min_pressure": sub["pressure"].min(),
                "max_wind": sub["wind"].max(),
                "event_uid": ",".join(sorted(sub["event_uid"].astype(str).unique())),
            }
        )

    check = pd.DataFrame(records)
    check.to_csv(REPORT_DIR / "target_name_time_position_check.csv", index=False, encoding="utf-8-sig")

    return df


def add_safe_model_features(df: pd.DataFrame, stats: dict[str, float]) -> pd.DataFrame:
    df = df.copy()

    df["center_lon"] = df["track_lon_180"]
    df["center_lat"] = df["track_lat"]

    move_rad = np.deg2rad(pd.to_numeric(df["track_move_dir_deg"], errors="coerce"))
    df["track_move_dir_sin"] = np.sin(move_rad)
    df["track_move_dir_cos"] = np.cos(move_rad)

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    d = pd.to_numeric(df["track_signed_coast_dist_km"], errors="coerce")
    df["is_near_coast_100km"] = ((d >= -100) & (d <= 100)).astype(int)
    df["is_near_coast_200km"] = ((d >= -200) & (d <= 200)).astype(int)
    df["is_offshore_far_300km"] = (d > 300).astype(int)
    df["is_inland_100km"] = (d < -100).astype(int)
    df["coast_influence_exp"] = np.exp(-np.abs(d) / 200.0)

    wind = pd.to_numeric(df["track_wind"], errors="coerce")
    pressure = pd.to_numeric(df["track_pressure"], errors="coerce")

    wind_std = stats["wind_std"] if stats["wind_std"] > 0 else np.nan
    pressure_deficit_std = (
        stats["pressure_deficit_std"] if stats["pressure_deficit_std"] > 0 else np.nan
    )

    df["wind_z"] = (wind - stats["wind_mean"]) / wind_std
    df["pressure_deficit"] = stats["pressure_mean"] - pressure
    df["pressure_deficit_z"] = (
        df["pressure_deficit"] - stats["pressure_deficit_mean"]
    ) / pressure_deficit_std
    df["intensity_index"] = 0.5 * df["wind_z"] + 0.5 * df["pressure_deficit_z"]

    wind_rate = pd.to_numeric(df["track_wind_change_rate"], errors="coerce")
    pressure_rate = pd.to_numeric(df["track_pressure_change_rate"], errors="coerce")
    df["is_intensifying_wind"] = (wind_rate > 0).astype(int)
    df["is_weakening_wind"] = (wind_rate < 0).astype(int)
    df["is_intensifying_pressure"] = (pressure_rate < 0).astype(int)
    df["is_weakening_pressure"] = (pressure_rate > 0).astype(int)

    return df


def make_track_point_inputs(raw: pd.DataFrame, stats: dict[str, float]) -> pd.DataFrame:
    df = raw.copy()
    df = df.rename(
        columns={
            "lat": "track_lat",
            "lon_180": "track_lon_180",
            "intensity": "track_intensity",
            "pressure": "track_pressure",
            "wind": "track_wind",
            "dt_h": "track_dt_h",
            "move_distance_km": "track_move_distance_km",
            "move_speed_kmh": "track_move_speed_kmh",
            "move_dir_deg": "track_move_dir_deg",
            "wind_change_rate": "track_wind_change_rate",
            "pressure_change_rate": "track_pressure_change_rate",
            "is_land": "track_is_land",
            "coast_dist_km": "track_coast_dist_km",
            "signed_coast_dist_km": "track_signed_coast_dist_km",
        }
    )
    df["target_display_name"] = df["target_name_norm"].map(
        {k: v["display_name"] for k, v in EXPECTED_TARGETS.items()}
    )
    df["cadence_hours"] = df.groupby("event_uid")["time"].diff().dt.total_seconds() / 3600
    df = fill_initial_motion_gaps(df)
    df = add_safe_model_features(df, stats)
    safe_cols = [col for col in TARGET_SAFE_COLS if col in df.columns]
    df = df[safe_cols].copy()
    assert_no_leakage(df, "track-point target input")
    return df


def make_halfhour_inputs(raw: pd.DataFrame, stats: dict[str, float]) -> pd.DataFrame:
    parts = []

    for event_uid, sub in raw.groupby("event_uid"):
        sub = sub.sort_values("time").reset_index(drop=True)
        times = pd.date_range(sub["time"].min(), sub["time"].max(), freq="30min")
        out = pd.DataFrame({"time": times})

        t_src = datetime_to_seconds(sub["time"])
        t_new = datetime_to_seconds(out["time"])

        out["track_lat"] = interp_numeric(t_src, sub["lat"], t_new)
        out["track_lon_180"] = interp_lon_degree(t_src, sub["lon_180"], t_new)
        out["track_pressure"] = interp_numeric(t_src, sub["pressure"], t_new)
        out["track_wind"] = interp_numeric(t_src, sub["wind"], t_new)
        out["track_coast_dist_km"] = interp_numeric(t_src, sub["coast_dist_km"], t_new)
        out["track_signed_coast_dist_km"] = interp_numeric(
            t_src,
            sub["signed_coast_dist_km"],
            t_new,
        )

        nearest = nearest_rows_by_time(sub, out["time"])
        for col in [
            "event_uid",
            "source_file",
            "typhoon_id",
            "storm_seq",
            "typhoon_code",
            "record_count",
            "target_name_norm",
        ]:
            out[col] = nearest[col].to_numpy()

        out["target_display_name"] = out["target_name_norm"].map(
            {k: v["display_name"] for k, v in EXPECTED_TARGETS.items()}
        )
        out["track_intensity"] = nearest["intensity"].to_numpy()
        out["track_is_land"] = (out["track_signed_coast_dist_km"] < 0).astype(int)
        out["cadence_hours"] = 0.5

        out = out.sort_values("time").reset_index(drop=True)
        dt_h = out["time"].diff().dt.total_seconds() / 3600

        prev_lon = out["track_lon_180"].shift(1)
        prev_lat = out["track_lat"].shift(1)
        dist = haversine_km(prev_lon, prev_lat, out["track_lon_180"], out["track_lat"])
        direction = bearing_deg(prev_lon, prev_lat, out["track_lon_180"], out["track_lat"])

        out["track_dt_h"] = dt_h
        out["track_move_distance_km"] = dist
        out["track_move_speed_kmh"] = dist / dt_h
        out["track_move_dir_deg"] = direction
        out["track_wind_change_rate"] = out["track_wind"].diff() / dt_h
        out["track_pressure_change_rate"] = out["track_pressure"].diff() / dt_h
        out = fill_initial_motion_gaps(out)

        parts.append(out)

    halfhour = pd.concat(parts, ignore_index=True)
    halfhour = add_safe_model_features(halfhour, stats)
    safe_cols = [col for col in TARGET_SAFE_COLS if col in halfhour.columns]
    halfhour = halfhour[safe_cols].copy()
    assert_no_leakage(halfhour, "half-hour target input")
    return halfhour


def write_leakage_check(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for label, df in outputs.items():
        leaked = [col for col in df.columns if is_leakage_col(col)]
        records.append(
            {
                "output": label,
                "n_columns": len(df.columns),
                "n_banned_columns": len(leaked),
                "banned_columns": ",".join(leaked),
                "status": "pass" if not leaked else "fail",
            }
        )

    check = pd.DataFrame(records)
    check.to_csv(REPORT_DIR / "target_input_leakage_check.csv", index=False, encoding="utf-8-sig")
    return check


def main() -> None:
    print("Running field audit and leakage guard...")

    audit = audit_headers()
    print("Field audit rows:", len(audit))
    print("Audit report:", REPORT_DIR / "field_classification_audit.csv")

    if not TARGET_TRACK_PATH.exists():
        raise FileNotFoundError(f"Missing target track table: {TARGET_TRACK_PATH}")

    raw = pd.read_csv(TARGET_TRACK_PATH, parse_dates=["time"])
    raw = validate_target_tracks(raw)
    stats = load_reference_stats()

    track_safe = make_track_point_inputs(raw, stats)
    halfhour_safe = make_halfhour_inputs(raw, stats)

    model_x_cols = [col for col in MODEL_X_COLS if col in halfhour_safe.columns]
    model_x = halfhour_safe[
        ["target_name_norm", "target_display_name", "event_uid", "time", *model_x_cols]
    ].copy()
    assert_no_leakage(model_x, "half-hour model X input")

    track_safe.to_csv(OUT_TRACK_SAFE, index=False, encoding="utf-8-sig")
    halfhour_safe.to_csv(OUT_HALFHOUR_SAFE, index=False, encoding="utf-8-sig")
    model_x.to_csv(OUT_HALFHOUR_X, index=False, encoding="utf-8-sig")

    leakage_check = write_leakage_check(
        {
            str(OUT_TRACK_SAFE): track_safe,
            str(OUT_HALFHOUR_SAFE): halfhour_safe,
            str(OUT_HALFHOUR_X): model_x,
        }
    )

    print("\nSafe target inputs written:")
    print(OUT_TRACK_SAFE, track_safe.shape)
    print(OUT_HALFHOUR_SAFE, halfhour_safe.shape)
    print(OUT_HALFHOUR_X, model_x.shape)

    print("\nLeakage check:")
    print(leakage_check.to_string(index=False))

    print("\nTarget time/position check:")
    print(pd.read_csv(REPORT_DIR / "target_name_time_position_check.csv").to_string(index=False))


if __name__ == "__main__":
    main()
