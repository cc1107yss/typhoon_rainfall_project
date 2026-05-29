#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_add_motion_relative_rain_features.py

逐格读取 GPM GeoTIFF，将雨场从地理坐标旋转到台风运动相对坐标系，
并把前后、左右、四象限降水结构指标并回插值后的模型特征表。

输入：
    data/processed/gpm_track_model_features_interp.csv
    data/raw/GPM_3IMERGHHE.07/<gpm_event_uid>/<source_file>.tif

输出：
    data/processed/gpm_track_model_features_motion.csv
    outputs/figures/problem1_motion/*.png
    outputs/tables/problem1_motion/*.csv / *.json / *.md

坐标约定：
    x_motion > 0: 台风移动前方(front)
    x_motion < 0: 台风移动后方(back)
    y_motion > 0: 面向移动方向的左侧(left)
    y_motion < 0: 面向移动方向的右侧(right)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from rainband_width_utils import (
    RAINBAND_WIDTH_COLS,
    RAINBAND_WIDTH_DIAGNOSTIC_COLS,
    compute_dual_rainband_width_metrics,
)

DEFAULT_INPUT = Path("data/processed/gpm_track_model_features_interp.csv")
DEFAULT_OUTPUT = Path("data/processed/gpm_track_model_features_motion.csv")
DEFAULT_GPM_ROOT = Path("data/raw/GPM_3IMERGHHE.07")
DEFAULT_FIGURE_DIR = Path("outputs/figures/problem1_motion")
DEFAULT_TABLE_DIR = Path("outputs/tables/problem1_motion")

KM_PER_DEG = 111.32
AREA_THRESHOLD_RATE = 10.0
EPS = 1e-12

MOTION_RAIN_FEATURE_COLS = [
    "rain_front_ratio",
    "rain_back_ratio",
    "rain_left_ratio",
    "rain_right_ratio",
    "rain_front_back_asym",
    "rain_left_right_asym",
    "rain_front_left_ratio",
    "rain_front_right_ratio",
    "rain_back_left_ratio",
    "rain_back_right_ratio",
    "rain_front_area_10_grid",
    "rain_back_area_10_grid",
    "rain_left_area_10_grid",
    "rain_right_area_10_grid",
    "rain_front_area_10_km2",
    "rain_back_area_10_km2",
    "rain_left_area_10_km2",
    "rain_right_area_10_km2",
    "rainband_angle_to_motion_deg",
    *RAINBAND_WIDTH_COLS,
]

MOTION_DIAGNOSTIC_COLS = [
    "motion_feature_status",
    "motion_tif_path_exists",
    "motion_center_lon_used",
    "motion_center_lat_used",
    "motion_dir_deg_used",
    *RAINBAND_WIDTH_DIAGNOSTIC_COLS,
]

ALL_NEW_COLS = MOTION_RAIN_FEATURE_COLS + MOTION_DIAGNOSTIC_COLS


def safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def first_finite(row: pd.Series, candidates: Iterable[str]) -> float:
    for col in candidates:
        if col in row.index:
            value = finite_or_nan(row[col])
            if np.isfinite(value):
                return value
    return np.nan


def empty_feature_row(status: str, tif_exists: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {col: np.nan for col in MOTION_RAIN_FEATURE_COLS}
    out.update({
        "motion_feature_status": status,
        "motion_tif_path_exists": bool(tif_exists),
        "motion_center_lon_used": np.nan,
        "motion_center_lat_used": np.nan,
        "motion_dir_deg_used": np.nan,
        "rainband_valid_grid_count": 0,
        "rainband_weight_sum": 0.0,
        "rainband10_valid_grid_count": 0,
        "rainband10_weight_sum": 0.0,
    })
    return out


def resolve_tif_path(row: pd.Series, gpm_root: Path) -> Tuple[Optional[Path], bool]:
    if "tif_path" in row.index and pd.notna(row["tif_path"]):
        p = Path(str(row["tif_path"]))
        if p.exists():
            return p, True
        if not p.is_absolute():
            p2 = Path.cwd() / p
            if p2.exists():
                return p2, True

    source_file = str(row.get("source_file", "")).strip()
    event_uid = str(row.get("gpm_event_uid", row.get("event_uid", ""))).strip()

    if not source_file or source_file.lower() == "nan":
        return None, False

    candidates: List[Path] = []
    if event_uid and event_uid.lower() != "nan":
        candidates.append(gpm_root / event_uid / source_file)
    candidates.append(gpm_root / source_file)
    candidates.append(Path(source_file))

    for candidate in candidates:
        if candidate.exists():
            return candidate, True

    return candidates[0] if candidates else None, False


def coordinate_grids(transform: rasterio.Affine, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)

    lon_grid = transform.a * col_grid + transform.b * row_grid + transform.c
    lat_grid = transform.d * col_grid + transform.e * row_grid + transform.f
    return lon_grid, lat_grid


def read_rain_and_coords(tif_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else -9999
        transform = src.transform
        crs = src.crs

        if crs is not None and str(crs).upper() != "EPSG:4326":
            warnings.warn(f"{tif_path.name} CRS is {crs}, expected EPSG:4326.")

        valid = np.isfinite(arr) & (arr != nodata)
        rain = np.where(valid, np.maximum(arr, 0.0), np.nan)
        lon_grid, lat_grid = coordinate_grids(transform, arr.shape)

    return rain, lon_grid, lat_grid, transform


def weighted_rainband_angle_to_motion(x_motion: np.ndarray, y_motion: np.ndarray, rain: np.ndarray, valid: np.ndarray) -> float:
    positive = valid & np.isfinite(rain) & (rain > 0)
    if int(positive.sum()) < 2:
        return np.nan

    weights = rain[positive].ravel()
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= EPS:
        return np.nan

    x = x_motion[positive].ravel()
    y = y_motion[positive].ravel()
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    x0 = x - x_mean
    y0 = y - y_mean

    cov_xx = float(np.average(x0 * x0, weights=weights))
    cov_yy = float(np.average(y0 * y0, weights=weights))
    cov_xy = float(np.average(x0 * y0, weights=weights))
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return np.nan

    major_i = int(np.argmax(eigvals))
    vx, vy = eigvecs[:, major_i]

    # 轴向角：0 表示降水带平行移动方向，90 表示垂直移动方向。
    raw = math.degrees(math.atan2(float(vy), float(vx)))
    signed_axis_angle = ((raw + 90.0) % 180.0) - 90.0
    return float(abs(signed_axis_angle))


def extract_motion_features(row: pd.Series, gpm_root: Path) -> Dict[str, object]:
    center_lon = first_finite(row, ["track_lon_180", "center_lon"])
    center_lat = first_finite(row, ["track_lat", "center_lat"])
    move_dir_deg = first_finite(row, ["track_move_dir_deg", "track_move_dir_deg_interp"])

    tif_path, tif_exists = resolve_tif_path(row, gpm_root)
    if not np.isfinite(center_lon) or not np.isfinite(center_lat) or not np.isfinite(move_dir_deg):
        out = empty_feature_row("missing_center_or_motion", tif_exists=tif_exists)
        out["motion_center_lon_used"] = center_lon
        out["motion_center_lat_used"] = center_lat
        out["motion_dir_deg_used"] = move_dir_deg
        return out

    if tif_path is None or not tif_exists:
        out = empty_feature_row("missing_tif", tif_exists=False)
        out["motion_center_lon_used"] = center_lon
        out["motion_center_lat_used"] = center_lat
        out["motion_dir_deg_used"] = move_dir_deg
        return out

    try:
        rain, lon_grid, lat_grid, transform = read_rain_and_coords(tif_path)
    except Exception as exc:  # pragma: no cover - kept for long batch robustness
        out = empty_feature_row(f"read_error:{type(exc).__name__}", tif_exists=tif_exists)
        out["motion_center_lon_used"] = center_lon
        out["motion_center_lat_used"] = center_lat
        out["motion_dir_deg_used"] = move_dir_deg
        return out

    valid = np.isfinite(rain)
    weights = np.where(valid, rain, 0.0)
    total_rain = float(np.sum(weights))

    move_rad = math.radians(move_dir_deg % 360.0)
    dx_lon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    x_east_km = dx_lon * KM_PER_DEG * math.cos(math.radians(center_lat))
    y_north_km = (lat_grid - center_lat) * KM_PER_DEG

    x_motion = x_east_km * math.sin(move_rad) + y_north_km * math.cos(move_rad)
    y_motion = -x_east_km * math.cos(move_rad) + y_north_km * math.sin(move_rad)

    front_mask = x_motion >= 0.0
    back_mask = ~front_mask
    left_mask = y_motion >= 0.0
    right_mask = ~left_mask

    front_sum = float(np.sum(weights[front_mask]))
    back_sum = float(np.sum(weights[back_mask]))
    left_sum = float(np.sum(weights[left_mask]))
    right_sum = float(np.sum(weights[right_mask]))
    front_left_sum = float(np.sum(weights[front_mask & left_mask]))
    front_right_sum = float(np.sum(weights[front_mask & right_mask]))
    back_left_sum = float(np.sum(weights[back_mask & left_mask]))
    back_right_sum = float(np.sum(weights[back_mask & right_mask]))

    mask10 = valid & (rain >= AREA_THRESHOLD_RATE)
    front_area_grid = int(np.count_nonzero(mask10 & front_mask))
    back_area_grid = int(np.count_nonzero(mask10 & back_mask))
    left_area_grid = int(np.count_nonzero(mask10 & left_mask))
    right_area_grid = int(np.count_nonzero(mask10 & right_mask))

    pixel_area_deg2 = abs(transform.a * transform.e - transform.b * transform.d)
    grid_area_km2 = KM_PER_DEG * KM_PER_DEG * pixel_area_deg2 * math.cos(math.radians(center_lat))
    grid_area_km2 = max(float(grid_area_km2), EPS)

    out = empty_feature_row("ok" if total_rain > EPS else "no_positive_rain", tif_exists=True)
    rainband_metrics = compute_dual_rainband_width_metrics(rain, x_motion, y_motion)

    out.update({
        "motion_center_lon_used": center_lon,
        "motion_center_lat_used": center_lat,
        "motion_dir_deg_used": move_dir_deg,
        "rain_front_ratio": safe_div(front_sum, total_rain),
        "rain_back_ratio": safe_div(back_sum, total_rain),
        "rain_left_ratio": safe_div(left_sum, total_rain),
        "rain_right_ratio": safe_div(right_sum, total_rain),
        "rain_front_back_asym": safe_div(front_sum - back_sum, front_sum + back_sum),
        "rain_left_right_asym": safe_div(left_sum - right_sum, left_sum + right_sum),
        "rain_front_left_ratio": safe_div(front_left_sum, total_rain),
        "rain_front_right_ratio": safe_div(front_right_sum, total_rain),
        "rain_back_left_ratio": safe_div(back_left_sum, total_rain),
        "rain_back_right_ratio": safe_div(back_right_sum, total_rain),
        "rain_front_area_10_grid": front_area_grid,
        "rain_back_area_10_grid": back_area_grid,
        "rain_left_area_10_grid": left_area_grid,
        "rain_right_area_10_grid": right_area_grid,
        "rain_front_area_10_km2": front_area_grid * grid_area_km2,
        "rain_back_area_10_km2": back_area_grid * grid_area_km2,
        "rain_left_area_10_km2": left_area_grid * grid_area_km2,
        "rain_right_area_10_km2": right_area_grid * grid_area_km2,
        "rainband_angle_to_motion_deg": weighted_rainband_angle_to_motion(x_motion, y_motion, rain, valid),
        **rainband_metrics,
    })
    return out


def write_describe_tables(df: pd.DataFrame, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)

    numeric = df[MOTION_RAIN_FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    desc = numeric.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).T
    desc.insert(0, "metric", desc.index)
    desc.to_csv(table_dir / "motion_relative_metric_describe.csv", index=False, encoding="utf-8-sig")

    if "track_intensity" in df.columns:
        by_intensity = df.groupby("track_intensity", dropna=False)[MOTION_RAIN_FEATURE_COLS].median(numeric_only=True)
        by_intensity.to_csv(table_dir / "motion_relative_median_by_intensity.csv", encoding="utf-8-sig")

    if "is_near_coast_100km" in df.columns:
        by_coast = df.groupby("is_near_coast_100km", dropna=False)[MOTION_RAIN_FEATURE_COLS].median(numeric_only=True)
        by_coast.to_csv(table_dir / "motion_relative_median_by_near_coast_100km.csv", encoding="utf-8-sig")

    if "track_move_speed_kmh" in df.columns:
        speed = pd.to_numeric(df["track_move_speed_kmh"], errors="coerce")
        valid_speed = speed.notna()
        if int(valid_speed.sum()) >= 10 and speed[valid_speed].nunique() >= 3:
            bins = pd.qcut(speed[valid_speed], q=3, labels=["slow", "middle", "fast"], duplicates="drop")
            tmp = df.loc[valid_speed, MOTION_RAIN_FEATURE_COLS].copy()
            tmp.insert(0, "move_speed_group", bins.astype(str).to_numpy())
            by_speed = tmp.groupby("move_speed_group", dropna=False).median(numeric_only=True)
            by_speed.to_csv(table_dir / "motion_relative_median_by_move_speed_tercile.csv", encoding="utf-8-sig")


def plot_box(df: pd.DataFrame, cols: List[str], labels: List[str], title: str, ylabel: str, out_path: Path) -> None:
    data = []
    used_labels = []
    for col, label in zip(cols, labels):
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if values.size > 0:
            data.append(values)
            used_labels.append(label)

    if not data:
        return

    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    box = ax.boxplot(data, tick_labels=used_labels, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], colors * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_hist(df: pd.DataFrame, cols: List[str], labels: List[str], title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    plotted = False
    for col, label, color in zip(cols, labels, colors):
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        ax.hist(values, bins=35, density=True, histtype="step", linewidth=2.0, label=label, color=color)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_figures(df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_box(
        df,
        ["rain_front_ratio", "rain_back_ratio", "rain_left_ratio", "rain_right_ratio"],
        ["Front", "Back", "Left", "Right"],
        "Rainfall Ratio in Motion-Relative Halves",
        "Rainfall ratio",
        figure_dir / "01_motion_half_ratio_boxplot.png",
    )
    plot_box(
        df,
        ["rain_front_left_ratio", "rain_front_right_ratio", "rain_back_left_ratio", "rain_back_right_ratio"],
        ["Front-left", "Front-right", "Back-left", "Back-right"],
        "Rainfall Ratio in Motion-Relative Quadrants",
        "Rainfall ratio",
        figure_dir / "02_motion_quadrant_ratio_boxplot.png",
    )
    plot_hist(
        df,
        ["rain_front_back_asym", "rain_left_right_asym"],
        ["Front-back asymmetry", "Left-right asymmetry"],
        "Motion-Relative Rainfall Asymmetry",
        "Asymmetry index",
        figure_dir / "03_motion_asymmetry_hist.png",
    )
    plot_hist(
        df,
        ["rainband_angle_to_motion_deg"],
        ["Rainband angle"],
        "Rainband Angle to Motion Direction",
        "Angle (degrees)",
        figure_dir / "04_rainband_angle_to_motion_hist.png",
    )
    plot_box(
        df,
        ["rain_front_area_10_km2", "rain_back_area_10_km2", "rain_left_area_10_km2", "rain_right_area_10_km2"],
        ["Front", "Back", "Left", "Right"],
        "Area >= 10 mm/hr in Motion-Relative Halves",
        "Area (km2)",
        figure_dir / "05_motion_area10_boxplot.png",
    )


def mean_median(df: pd.DataFrame, col: str) -> Dict[str, float]:
    s = pd.to_numeric(df[col], errors="coerce")
    return {
        "mean": float(s.mean(skipna=True)),
        "median": float(s.median(skipna=True)),
        "n": int(s.notna().sum()),
    }


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.1%}"


def num(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_conclusions(df: pd.DataFrame, table_dir: Path, figure_dir: Path, output_file: Path) -> None:
    status_counts = df["motion_feature_status"].value_counts(dropna=False).to_dict()
    valid = df[df["motion_feature_status"].eq("ok")].copy()

    half_cols = {
        "前方": "rain_front_ratio",
        "后方": "rain_back_ratio",
        "左侧": "rain_left_ratio",
        "右侧": "rain_right_ratio",
    }
    quad_cols = {
        "前左象限": "rain_front_left_ratio",
        "前右象限": "rain_front_right_ratio",
        "后左象限": "rain_back_left_ratio",
        "后右象限": "rain_back_right_ratio",
    }

    half_stats = {name: mean_median(valid, col) for name, col in half_cols.items()}
    quad_stats = {name: mean_median(valid, col) for name, col in quad_cols.items()}
    dominant_half_mean = max(half_stats, key=lambda k: half_stats[k]["mean"]) if half_stats else "NA"
    dominant_half_median = max(half_stats, key=lambda k: half_stats[k]["median"]) if half_stats else "NA"
    dominant_quad_mean = max(quad_stats, key=lambda k: quad_stats[k]["mean"]) if quad_stats else "NA"
    dominant_quad_median = max(quad_stats, key=lambda k: quad_stats[k]["median"]) if quad_stats else "NA"

    fb_asym = mean_median(valid, "rain_front_back_asym")
    lr_asym = mean_median(valid, "rain_left_right_asym")
    angle = mean_median(valid, "rainband_angle_to_motion_deg")
    area_front = mean_median(valid, "rain_front_area_10_km2")
    area_back = mean_median(valid, "rain_back_area_10_km2")
    area_left = mean_median(valid, "rain_left_area_10_km2")
    area_right = mean_median(valid, "rain_right_area_10_km2")

    summary = {
        "rows_total": int(len(df)),
        "rows_ok": int(len(valid)),
        "ok_rate": float(len(valid) / len(df)) if len(df) else np.nan,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "half_ratio_stats": half_stats,
        "quadrant_ratio_stats": quad_stats,
        "front_back_asym": fb_asym,
        "left_right_asym": lr_asym,
        "rainband_angle_to_motion_deg": angle,
        "area10_km2": {
            "front": area_front,
            "back": area_back,
            "left": area_left,
            "right": area_right,
        },
        "outputs": {
            "feature_file": str(output_file),
            "figure_dir": str(figure_dir),
            "table_dir": str(table_dir),
        },
    }
    with (table_dir / "motion_relative_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 运动相对坐标系降水结构指标结论",
        "",
        f"- 本次共处理 {len(df)} 个半小时时次，其中 {len(valid)} 个成功得到运动相对雨场结构指标，成功率为 {pct(summary['ok_rate'])}。状态计数：{summary['status_counts']}。",
        f"- 从降水总量占比看，均值最大的半区为{dominant_half_mean}，中位数最大的半区为{dominant_half_median}。前方/后方均值分别为 {pct(half_stats['前方']['mean'])}/{pct(half_stats['后方']['mean'])}，左侧/右侧均值分别为 {pct(half_stats['左侧']['mean'])}/{pct(half_stats['右侧']['mean'])}。",
        f"- 前后非对称指数的均值/中位数为 {num(fb_asym['mean'])}/{num(fb_asym['median'])}；左右非对称指数的均值/中位数为 {num(lr_asym['mean'])}/{num(lr_asym['median'])}。正值分别表示降水偏向移动前方、移动左侧。",
        f"- 四象限降水占比中，均值最大的象限为{dominant_quad_mean}，中位数最大的象限为{dominant_quad_median}。前左、前右、后左、后右均值分别为 {pct(quad_stats['前左象限']['mean'])}、{pct(quad_stats['前右象限']['mean'])}、{pct(quad_stats['后左象限']['mean'])}、{pct(quad_stats['后右象限']['mean'])}。",
        f"- 10 mm/hr 强降水面积的前方/后方中位数为 {num(area_front['median'], 1)}/{num(area_back['median'], 1)} km2，左侧/右侧中位数为 {num(area_left['median'], 1)}/{num(area_right['median'], 1)} km2。",
        f"- 降水带主轴与移动方向夹角的均值/中位数为 {num(angle['mean'], 1)}/{num(angle['median'], 1)} 度；数值接近 0 表示雨带更平行于移动方向，接近 90 表示更近似横切移动方向。",
        "",
        "这些结论只描述逐格旋转后的运动相对结构分布，不把东西/南北偏向误写为物理上的前后/左右偏向。",
    ]
    (table_dir / "motion_relative_conclusions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add motion-relative rain-structure features from GPM GeoTIFF fields.")
    parser.add_argument("--input-file", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-file", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpm-root", type=str, default=str(DEFAULT_GPM_ROOT))
    parser.add_argument("--figure-dir", type=str, default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--table-dir", type=str, default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: process the first N rows.")
    parser.add_argument("--skip-plots", action="store_true", help="Only write feature CSV and summary tables.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    gpm_root = Path(args.gpm_root)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"[motion] 读取输入表：{input_file}")
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
    for col in ["time", "time_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
        print(f"[motion] 调试模式，仅处理前 {len(df)} 行。")

    print(f"[motion] GPM root：{gpm_root}")
    print(f"[motion] 待处理时次：{len(df)}")

    records: List[Dict[str, object]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="motion-relative rain features"):
        records.append(extract_motion_features(row, gpm_root))

    feature_df = pd.DataFrame(records)
    df = pd.concat(
        [
            df.drop(columns=[c for c in ALL_NEW_COLS if c in df.columns], errors="ignore").reset_index(drop=True),
            feature_df[ALL_NEW_COLS].reset_index(drop=True),
        ],
        axis=1,
    )

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    write_describe_tables(df, table_dir)
    if not args.skip_plots:
        make_figures(df, figure_dir)
    write_conclusions(df, table_dir, figure_dir, output_file)

    status_counts = df["motion_feature_status"].value_counts(dropna=False)
    print("\n========== motion-relative rain features complete ==========")
    print(f"输出特征表：{output_file}")
    print(f"输出维度：{df.shape[0]} × {df.shape[1]}")
    print("状态计数：")
    print(status_counts.to_string())
    print(f"图表目录：{figure_dir}")
    print(f"表格/结论目录：{table_dir}")
    print(f"新增雨场结构字段数：{len(MOTION_RAIN_FEATURE_COLS)}")


if __name__ == "__main__":
    main()
