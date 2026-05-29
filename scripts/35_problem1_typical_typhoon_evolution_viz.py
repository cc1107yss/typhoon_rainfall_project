#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a comprehensive evolution figure for one typical historical typhoon.

Primary data file used by the current Problem-1 final chain:
    data/processed/env_added/gpm_track_model_features_interp_env.csv

The script searches a small set of final feature-table candidates, excludes
KONG-REY and MAN-YI, selects a data-complete historical event, and writes:
    outputs/figures/problem1_env/problem1_typical_typhoon_evolution.png
    outputs/figures/problem1_env/problem1_typical_typhoon_evolution.pdf
    outputs/tables/problem1_env/problem1_typical_typhoon_evolution_summary.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.gridspec import GridSpecFromSubplotSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
GPM_ROOT = PROJECT_ROOT / "data/raw/GPM_3IMERGHHE.07"
FIG_DIR = PROJECT_ROOT / "outputs/figures/problem1_env"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/problem1_env"
PNG_OUT = FIG_DIR / "problem1_typical_typhoon_evolution.png"
PDF_OUT = FIG_DIR / "problem1_typical_typhoon_evolution.pdf"
SUMMARY_OUT = TABLE_DIR / "problem1_typical_typhoon_evolution_summary.csv"

FEATURE_CANDIDATES = [
    PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env.csv",
    PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env_rainband_width.csv",
    PROJECT_ROOT / "data/processed/gpm_track_model_features_motion.csv",
    PROJECT_ROOT / "data/processed/gpm_track_model_features_interp.csv",
    PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv",
]

TARGET_NAMES = {"KONG_REY", "KONGREY", "MAN_YI", "MANYI"}
REQUIRED_BASE_COLS = {
    "time",
    "track_event_uid",
    "track_typhoon_name",
    "track_typhoon_id",
    "source_file",
    "gpm_event_uid",
    "track_lat",
    "track_lon_180",
    "track_wind",
    "track_move_speed_kmh",
    "rain_p95",
    "rain_area_10_km2",
    "centroid_offset_km",
    "anisotropy",
}
OPTIONAL_COLS = {
    "track_pressure",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "track_move_dir_deg",
    "track_move_dir_deg_interp",
    "rainband_width_km",
    "rainband_width_tif_path_exists",
    "rain_centroid_lon",
    "rain_centroid_lat",
}
NUMERIC_COLS = [
    "track_lat",
    "track_lon_180",
    "track_wind",
    "track_pressure",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_move_dir_deg_interp",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "rain_p95",
    "rain_area_10_km2",
    "centroid_offset_km",
    "anisotropy",
    "rainband_width_km",
    "rain_centroid_lon",
    "rain_centroid_lat",
]

KM_PER_DEG = 111.32
EPS = 1e-12

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from rainband_width_utils import compute_dual_rainband_width_metrics
except Exception as exc:  # pragma: no cover - explicit runtime fallback below
    compute_dual_rainband_width_metrics = None
    IMPORT_WIDTH_ERROR = exc
else:
    IMPORT_WIDTH_ERROR = None


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).upper()
    keep = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(keep).split("_"))


def pick_feature_file() -> Path:
    scored: List[Tuple[int, Path, List[str]]] = []
    for path in FEATURE_CANDIDATES:
        if not path.exists():
            continue
        cols = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        present = set(cols)
        missing = sorted(REQUIRED_BASE_COLS - present)
        if missing:
            scored.append((-len(missing), path, missing))
            continue
        bonus = 0
        if "rainband_width_km" in present:
            bonus += 3
        if "landfrac_500km" in present and "terrain_std_300km" in present:
            bonus += 2
        if "env_added" in str(path):
            bonus += 2
        scored.append((len(REQUIRED_BASE_COLS) + bonus, path, []))

    if not scored:
        raise FileNotFoundError("No candidate Problem-1 feature table was found.")

    scored.sort(key=lambda item: item[0], reverse=True)
    score, chosen, missing = scored[0]
    if missing or score <= 0:
        raise ValueError(f"Best candidate still misses required columns: {chosen}, missing={missing}")
    return chosen


def load_features(path: Path) -> pd.DataFrame:
    usecols = sorted((REQUIRED_BASE_COLS | OPTIONAL_COLS) & set(pd.read_csv(path, nrows=0).columns))
    df = pd.read_csv(path, usecols=usecols, parse_dates=["time"], low_memory=False)
    if "time_end" in df.columns:
        df["time_end"] = pd.to_datetime(df["time_end"], errors="coerce")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_event_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    names = out["track_typhoon_name"].map(normalize_name)
    out = out[~names.isin(TARGET_NAMES)].copy()
    out = out.dropna(subset=["time", "track_event_uid", "track_lat", "track_lon_180"])
    out = out.sort_values(["track_event_uid", "time"])
    return out


def finite_range(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0
    return float(s.max() - s.min())


def event_name(group: pd.DataFrame, col: str) -> str:
    if col not in group.columns:
        return ""
    s = group[col].dropna().astype(str)
    return "" if s.empty else s.iloc[0]


def select_typical_event(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    core = [
        "track_wind",
        "track_move_speed_kmh",
        "rain_p95",
        "rain_area_10_km2",
        "centroid_offset_km",
        "anisotropy",
    ]
    if "track_pressure" in df.columns:
        core.append("track_pressure")
    if "rainband_width_km" in df.columns:
        core.append("rainband_width_km")

    rows = []
    for uid, group in df.groupby("track_event_uid", sort=False):
        g = group.sort_values("time").copy()
        n = len(g)
        if n < 80:
            continue
        completeness = float(g[core].notna().all(axis=1).mean())
        if completeness < 0.90:
            continue
        if "rainband_width_tif_path_exists" in g.columns:
            tif_ratio = float(pd.Series(g["rainband_width_tif_path_exists"]).fillna(False).astype(bool).mean())
        else:
            tif_ratio = 1.0
        if tif_ratio < 0.90:
            continue

        rain_variation = finite_range(g["rain_p95"])
        area_variation = finite_range(g["rain_area_10_km2"])
        wind_variation = finite_range(g["track_wind"])
        offset_variation = finite_range(g["centroid_offset_km"])
        width_variation = finite_range(g["rainband_width_km"]) if "rainband_width_km" in g.columns else 0.0
        path_lon_span = finite_range(g["track_lon_180"])
        path_lat_span = finite_range(g["track_lat"])
        score = (
            n
            + 3.0 * rain_variation
            + 0.001 * area_variation
            + 0.20 * wind_variation
            + 0.05 * offset_variation
            + 0.02 * width_variation
            + 2.0 * (path_lon_span + path_lat_span)
        )
        rows.append(
            {
                "track_event_uid": uid,
                "track_typhoon_name": event_name(g, "track_typhoon_name"),
                "track_typhoon_id": event_name(g, "track_typhoon_id"),
                "start_time": g["time"].min(),
                "end_time": g["time"].max(),
                "n_times": n,
                "completeness": completeness,
                "tif_ratio": tif_ratio,
                "max_wind": float(g["track_wind"].max(skipna=True)),
                "min_pressure": float(g["track_pressure"].min(skipna=True)) if "track_pressure" in g.columns else np.nan,
                "max_rain_p95": float(g["rain_p95"].max(skipna=True)),
                "max_rain_area_10_km2": float(g["rain_area_10_km2"].max(skipna=True)),
                "max_centroid_offset_km": float(g["centroid_offset_km"].max(skipna=True)),
                "max_anisotropy": float(g["anisotropy"].max(skipna=True)),
                "max_rainband_width_km": float(g["rainband_width_km"].max(skipna=True))
                if "rainband_width_km" in g.columns
                else np.nan,
                "score": float(score),
            }
        )

    candidates = pd.DataFrame(rows).sort_values("score", ascending=False)
    if candidates.empty:
        raise RuntimeError("No suitable historical typhoon event found after filtering.")

    selected_uid = str(candidates.iloc[0]["track_event_uid"])
    selected = df.loc[df["track_event_uid"].eq(selected_uid)].sort_values("time").reset_index(drop=True)
    return selected_uid, selected, candidates


def resolve_tif_path(row: pd.Series) -> Optional[Path]:
    source_file = str(row.get("source_file", "")).strip()
    if not source_file or source_file.lower() == "nan":
        return None
    if Path(source_file).exists():
        return Path(source_file)

    event_uid = str(row.get("gpm_event_uid", "")).strip()
    candidates = []
    if event_uid and event_uid.lower() != "nan":
        candidates.append(GPM_ROOT / event_uid / source_file)
    candidates.append(GPM_ROOT / source_file)
    candidates.append(PROJECT_ROOT / source_file)
    for path in candidates:
        if path.exists():
            return path
    return None


def coordinate_grids(transform: rasterio.Affine, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    lon_grid = transform.a * col_grid + transform.b * row_grid + transform.c
    lat_grid = transform.d * col_grid + transform.e * row_grid + transform.f
    return lon_grid, lat_grid


def read_rain_field(tif_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else -9999
        valid = np.isfinite(arr) & (arr != nodata)
        rain = np.where(valid, np.maximum(arr, 0.0), np.nan)
        lon_grid, lat_grid = coordinate_grids(src.transform, arr.shape)
    return rain, lon_grid, lat_grid


def local_xy_km(lon_grid: np.ndarray, lat_grid: np.ndarray, center_lon: float, center_lat: float) -> Tuple[np.ndarray, np.ndarray]:
    dx_lon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    x_km = dx_lon * KM_PER_DEG * np.cos(np.deg2rad(center_lat))
    y_km = (lat_grid - center_lat) * KM_PER_DEG
    return x_km, y_km


def fallback_rainband_width(rain: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> float:
    mask = np.isfinite(rain) & np.isfinite(x_km) & np.isfinite(y_km) & (rain >= 1.0)
    if int(mask.sum()) < 5:
        return np.nan
    weights = rain[mask].ravel()
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= EPS:
        return np.nan
    x = x_km[mask].ravel()
    y = y_km[mask].ravel()
    x0 = x - float(np.sum(weights * x) / weight_sum)
    y0 = y - float(np.sum(weights * y) / weight_sum)
    cov = np.array(
        [
            [float(np.sum(weights * x0 * x0) / weight_sum), float(np.sum(weights * x0 * y0) / weight_sum)],
            [float(np.sum(weights * x0 * y0) / weight_sum), float(np.sum(weights * y0 * y0) / weight_sum)],
        ]
    )
    eigvals = np.linalg.eigvalsh(cov)
    if not np.all(np.isfinite(eigvals)):
        return np.nan
    return 4.0 * float(np.sqrt(max(eigvals[0], 0.0)))


def add_missing_rainband_width(event: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "rainband_width_km" in event.columns and event["rainband_width_km"].notna().any():
        return event, "rainband_width_km read from selected feature table"

    note = "rainband_width_km computed from GeoTIFF weighted covariance"
    if compute_dual_rainband_width_metrics is None:
        note += f"; rainband_width_utils import failed: {type(IMPORT_WIDTH_ERROR).__name__}"
    widths = []
    for _, row in event.iterrows():
        tif_path = resolve_tif_path(row)
        if tif_path is None:
            widths.append(np.nan)
            continue
        rain, lon_grid, lat_grid = read_rain_field(tif_path)
        x_km, y_km = local_xy_km(lon_grid, lat_grid, float(row["track_lon_180"]), float(row["track_lat"]))
        if compute_dual_rainband_width_metrics is not None:
            metrics = compute_dual_rainband_width_metrics(rain, x_km, y_km)
            widths.append(metrics.get("rainband_width_km", np.nan))
        else:
            widths.append(fallback_rainband_width(rain, x_km, y_km))
    out = event.copy()
    out["rainband_width_km"] = widths
    return out, note


def nearest_unique_position(positions: Iterable[int], target: int, n: int) -> int:
    used = set(int(p) for p in positions)
    target = max(0, min(n - 1, int(target)))
    if target not in used:
        return target
    for radius in range(1, n):
        for candidate in (target - radius, target + radius):
            if 0 <= candidate < n and candidate not in used:
                return candidate
    return target


def choose_snapshot_rows(event: pd.DataFrame) -> List[Tuple[str, int]]:
    g = event.reset_index(drop=True)
    n = len(g)
    peak_metric = "rain_p95" if g["rain_p95"].notna().any() else "track_wind"
    peak_pos = int(g[peak_metric].idxmax()) if g[peak_metric].notna().any() else int(round(0.60 * (n - 1)))

    early_pos = int(round(0.20 * (n - 1)))
    before_peak = g.loc[: max(peak_pos, 0)].copy()
    if "track_wind" in before_peak.columns and before_peak["track_wind"].notna().any():
        intensify_pos = int(before_peak["track_wind"].idxmax())
    else:
        intensify_pos = int(round(0.40 * (n - 1)))
    if intensify_pos == peak_pos:
        intensify_pos = int(round(0.40 * (n - 1)))

    after_peak = g.loc[min(peak_pos + 1, n - 1) :].copy()
    if not after_peak.empty:
        target80 = int(round(0.80 * (n - 1)))
        weakening_pos = int((after_peak.index.to_series() - target80).abs().idxmin())
    else:
        weakening_pos = int(round(0.80 * (n - 1)))

    selected: List[Tuple[str, int]] = []
    for label, pos in [
        ("early", early_pos),
        ("intensifying", intensify_pos),
        ("rain peak", peak_pos),
        ("weakening", weakening_pos),
    ]:
        unique_pos = nearest_unique_position([p for _, p in selected], pos, n)
        selected.append((label, unique_pos))
    return selected


def style_time_axis(ax: plt.Axes, show_xlabel: bool = False) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    if show_xlabel:
        ax.set_xlabel("Date in selected event")
    else:
        ax.tick_params(labelbottom=False)


def mark_stage_lines(axes: Iterable[plt.Axes], event: pd.DataFrame, stages: List[Tuple[str, int]]) -> None:
    for ax in axes:
        for label, pos in stages:
            t = event.loc[pos, "time"]
            ax.axvline(t, color="0.25", linestyle="--", linewidth=0.8, alpha=0.65)


def plot_line(ax: plt.Axes, event: pd.DataFrame, col: str, color: str, label: str, ylabel: str) -> None:
    ax.plot(event["time"], event[col], color=color, linewidth=1.6, label=label)
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_evolution(event: pd.DataFrame, stages: List[Tuple[str, int]], feature_file: Path, width_note: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    name = event_name(event, "track_typhoon_name") or "selected event"
    uid = event_name(event, "track_event_uid")
    fig = plt.figure(figsize=(16, 16), dpi=220)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.05, 1.05, 1.25, 1.25],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.34,
    )

    ax_path = fig.add_subplot(gs[0:2, 0:2])
    b_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 2:4], hspace=0.08)
    c_grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 2:4], hspace=0.08)
    ax_b1 = fig.add_subplot(b_grid[0, 0])
    ax_b2 = fig.add_subplot(b_grid[1, 0], sharex=ax_b1)
    ax_b3 = fig.add_subplot(b_grid[2, 0], sharex=ax_b1)
    ax_c1 = fig.add_subplot(c_grid[0, 0])
    ax_c2 = fig.add_subplot(c_grid[1, 0], sharex=ax_c1)
    ax_c3 = fig.add_subplot(c_grid[2, 0], sharex=ax_c1)
    snap_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[3, 0:2]),
        fig.add_subplot(gs[3, 2:4]),
    ]

    time_num = mdates.date2num(event["time"])
    sc = ax_path.scatter(
        event["track_lon_180"],
        event["track_lat"],
        c=time_num,
        s=13,
        cmap="viridis",
        edgecolors="none",
        zorder=3,
    )
    ax_path.plot(event["track_lon_180"], event["track_lat"], color="0.25", linewidth=1.1, zorder=2)
    stage_markers = ["o", "s", "^", "D"]
    for (label, pos), marker in zip(stages, stage_markers):
        row = event.loc[pos]
        ax_path.scatter(
            row["track_lon_180"],
            row["track_lat"],
            marker=marker,
            s=80,
            facecolor="white",
            edgecolor="black",
            linewidth=1.3,
            zorder=5,
        )
        ax_path.annotate(
            label,
            (row["track_lon_180"], row["track_lat"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
    ax_path.set_title(f"(a) Track: {name} ({uid})", loc="left", fontsize=12, fontweight="bold")
    ax_path.set_xlabel("Longitude (deg)")
    ax_path.set_ylabel("Latitude (deg)")
    ax_path.grid(True, alpha=0.28)
    cbar = fig.colorbar(sc, ax=ax_path, fraction=0.045, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    cbar.set_label("Time")

    plot_line(ax_b1, event, "track_wind", "#1f77b4", "WND", "WND")
    if "track_pressure" in event.columns and event["track_pressure"].notna().any():
        plot_line(ax_b2, event, "track_pressure", "#d62728", "PRES", "PRES")
        ax_b2.invert_yaxis()
    else:
        ax_b2.text(0.5, 0.5, "PRES unavailable", transform=ax_b2.transAxes, ha="center", va="center")
        ax_b2.set_ylabel("PRES")
    plot_line(ax_b3, event, "track_move_speed_kmh", "#2ca02c", "Move speed", "km/h")
    ax_b1.set_title("(b) Typhoon intensity and motion", loc="left", fontsize=12, fontweight="bold")
    for ax in [ax_b1, ax_b2]:
        style_time_axis(ax, show_xlabel=False)
    style_time_axis(ax_b3, show_xlabel=True)

    ax_c1.plot(event["time"], event["rain_p95"], color="#1f77b4", linewidth=1.5, label="rain_p95 (mm/hr)")
    ax_c1.set_ylabel("mm/hr", color="#1f77b4")
    ax_c1.tick_params(axis="y", labelcolor="#1f77b4")
    ax_c1b = ax_c1.twinx()
    ax_c1b.plot(event["time"], event["rain_area_10_km2"] / 1000.0, color="#ff7f0e", linewidth=1.5, label="rain_area_10")
    ax_c1b.set_ylabel("$10^3$ km$^2$", color="#ff7f0e")
    ax_c1b.tick_params(axis="y", labelcolor="#ff7f0e")
    lines, labels = ax_c1.get_legend_handles_labels()
    lines2, labels2 = ax_c1b.get_legend_handles_labels()
    ax_c1.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False, fontsize=8)

    ax_c2.plot(event["time"], event["centroid_offset_km"], color="#9467bd", linewidth=1.5, label="centroid_offset")
    ax_c2.plot(event["time"], event["rainband_width_km"], color="#8c564b", linewidth=1.5, linestyle="--", label="rainband_width")
    ax_c2.set_ylabel("km")
    ax_c2.legend(loc="upper right", frameon=False, fontsize=8)

    ax_c3.plot(event["time"], event["anisotropy"], color="#2ca02c", linewidth=1.5, label="anisotropy")
    ax_c3.set_ylabel("index")
    ax_c3.legend(loc="upper right", frameon=False, fontsize=8)
    ax_c1.set_title("(c) Rainfall structure metrics", loc="left", fontsize=12, fontweight="bold")
    for ax in [ax_c1, ax_c2]:
        style_time_axis(ax, show_xlabel=False)
    style_time_axis(ax_c3, show_xlabel=True)

    mark_stage_lines([ax_b1, ax_b2, ax_b3, ax_c1, ax_c2, ax_c3], event, stages)

    stage_fields = []
    vmax_values = []
    for label, pos in stages:
        row = event.loc[pos]
        tif_path = resolve_tif_path(row)
        if tif_path is None:
            stage_fields.append((label, pos, row, None, None, None))
            continue
        rain, lon_grid, lat_grid = read_rain_field(tif_path)
        stage_fields.append((label, pos, row, rain, lon_grid, lat_grid))
        vals = rain[np.isfinite(rain)]
        if vals.size:
            vmax_values.append(float(np.nanpercentile(vals, 99.0)))
    vmax = max(vmax_values) if vmax_values else 1.0
    vmax = max(vmax, 1.0)
    image_for_colorbar = None

    for ax, (label, pos, row, rain, lon_grid, lat_grid) in zip(snap_axes, stage_fields):
        if rain is None:
            ax.text(0.5, 0.5, "GeoTIFF missing", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        extent = [float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid)), float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))]
        image_for_colorbar = ax.imshow(
            rain,
            extent=extent,
            origin="upper",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        levels = [1, 5, 10, 20, 40]
        ax.contour(lon_grid, lat_grid, rain, levels=levels, colors="white", linewidths=0.45, alpha=0.75)
        ax.scatter(row["track_lon_180"], row["track_lat"], marker="+", s=80, linewidth=1.5, color="cyan", label="center")
        if "rain_centroid_lon" in row.index and np.isfinite(row.get("rain_centroid_lon", np.nan)):
            ax.scatter(row["rain_centroid_lon"], row["rain_centroid_lat"], marker="x", s=45, linewidth=1.1, color="white", label="rain centroid")
        ax.set_title(
            f"(d) {label}: {pd.Timestamp(row['time']).strftime('%Y-%m-%d %H:%M')}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, alpha=0.18, color="white", linewidth=0.45)
        ax.legend(loc="upper right", frameon=True, fontsize=7)

    if image_for_colorbar is not None:
        cb = fig.colorbar(image_for_colorbar, ax=snap_axes, fraction=0.025, pad=0.02)
        cb.set_label("GPM rain rate (mm/hr)")

    fig.suptitle(
        f"Typical historical typhoon evolution and rainfall structure: {name}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.01,
        0.01,
        f"Feature table: {feature_file.relative_to(PROJECT_ROOT)}; {width_note}",
        fontsize=7,
        color="0.35",
    )
    fig.savefig(PNG_OUT, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    event: pd.DataFrame,
    stages: List[Tuple[str, int]],
    feature_file: Path,
    width_note: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    row = {
        "selected_track_event_uid": event_name(event, "track_event_uid"),
        "selected_typhoon_name": event_name(event, "track_typhoon_name"),
        "selected_typhoon_id": event_name(event, "track_typhoon_id"),
        "start_time": event["time"].min(),
        "end_time": event["time"].max(),
        "n_times": int(len(event)),
        "max_wind": float(event["track_wind"].max(skipna=True)),
        "min_pressure": float(event["track_pressure"].min(skipna=True)) if "track_pressure" in event.columns else np.nan,
        "max_rain_p95": float(event["rain_p95"].max(skipna=True)),
        "max_rain_area_10_km2": float(event["rain_area_10_km2"].max(skipna=True)),
        "max_centroid_offset_km": float(event["centroid_offset_km"].max(skipna=True)),
        "max_anisotropy": float(event["anisotropy"].max(skipna=True)),
        "max_rainband_width_km": float(event["rainband_width_km"].max(skipna=True)),
        "feature_file": str(feature_file.relative_to(PROJECT_ROOT)),
        "gpm_root": str(GPM_ROOT.relative_to(PROJECT_ROOT)),
        "rainband_width_note": width_note,
        "candidate_rank_score": float(candidates.iloc[0]["score"]),
    }
    for label, pos in stages:
        stage_key = label.replace(" ", "_")
        stage_row = event.loc[pos]
        row[f"{stage_key}_time"] = stage_row["time"]
        row[f"{stage_key}_source_file"] = stage_row["source_file"]
        row[f"{stage_key}_wind"] = stage_row.get("track_wind", np.nan)
        row[f"{stage_key}_pressure"] = stage_row.get("track_pressure", np.nan)
        row[f"{stage_key}_rain_p95"] = stage_row.get("rain_p95", np.nan)
        row[f"{stage_key}_rain_area_10_km2"] = stage_row.get("rain_area_10_km2", np.nan)
        row[f"{stage_key}_centroid_offset_km"] = stage_row.get("centroid_offset_km", np.nan)
        row[f"{stage_key}_anisotropy"] = stage_row.get("anisotropy", np.nan)
        row[f"{stage_key}_rainband_width_km"] = stage_row.get("rainband_width_km", np.nan)
    return pd.DataFrame([row])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    feature_file = pick_feature_file()
    df = load_features(feature_file)
    df = clean_event_df(df)
    selected_uid, event, candidates = select_typical_event(df)
    event, width_note = add_missing_rainband_width(event)
    stages = choose_snapshot_rows(event)

    plot_evolution(event, stages, feature_file, width_note)
    summary = build_summary(event, stages, feature_file, width_note, candidates)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print(f"[35] Feature file: {feature_file.relative_to(PROJECT_ROOT)}")
    print(f"[35] Selected event: {selected_uid} / {event_name(event, 'track_typhoon_name')}")
    print(f"[35] Stage times: {[pd.Timestamp(event.loc[pos, 'time']).strftime('%Y-%m-%d %H:%M') for _, pos in stages]}")
    print(f"[35] Rainband width handling: {width_note}")
    print(f"[35] Figure PNG: {PNG_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[35] Figure PDF: {PDF_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[35] Summary CSV: {SUMMARY_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
