#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add covariance-based rainband-width features to the final Problem-1 env table."""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_COPY = PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env_rainband_width.csv"
DEFAULT_BACKUP = PROJECT_ROOT / "data/processed/env_added/gpm_track_model_features_interp_env_before_rainband_width.csv"
DEFAULT_GPM_ROOT = PROJECT_ROOT / "data/raw/GPM_3IMERGHHE.07"
QC_PATH = PROJECT_ROOT / "outputs/tables/problem1_env/problem1_rainband_width_generation_qc.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from rainband_width_utils import (  # noqa: E402
    RAINBAND_WIDTH_COLS,
    RAINBAND_WIDTH_DIAGNOSTIC_COLS,
    compute_dual_rainband_width_metrics,
)

EPS = 1e-12
STATUS_COLS = ["rainband_width_status", "rainband_width_tif_path_exists"]
ALL_NEW_COLS = RAINBAND_WIDTH_COLS + RAINBAND_WIDTH_DIAGNOSTIC_COLS + STATUS_COLS


def load_motion_module():
    path = SCRIPT_DIR / "17_add_motion_relative_rain_features.py"
    spec = importlib.util.spec_from_file_location("motion_relative_features_for_width", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import motion-relative helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_row(status: str, tif_exists: bool = False) -> Dict[str, object]:
    return {
        **{col: np.nan for col in RAINBAND_WIDTH_COLS},
        "rainband_valid_grid_count": 0,
        "rainband_weight_sum": 0.0,
        "rainband10_valid_grid_count": 0,
        "rainband10_weight_sum": 0.0,
        "rainband_width_status": status,
        "rainband_width_tif_path_exists": bool(tif_exists),
    }


def motion_grids(
    motion,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    center_lon: float,
    center_lat: float,
    move_dir_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    move_rad = math.radians(move_dir_deg % 360.0)
    dx_lon = ((lon_grid - center_lon + 180.0) % 360.0) - 180.0
    x_east_km = dx_lon * motion.KM_PER_DEG * math.cos(math.radians(center_lat))
    y_north_km = (lat_grid - center_lat) * motion.KM_PER_DEG
    x_motion = x_east_km * math.sin(move_rad) + y_north_km * math.cos(move_rad)
    y_motion = -x_east_km * math.cos(move_rad) + y_north_km * math.sin(move_rad)
    return x_motion, y_motion


def compute_one(row: pd.Series, motion, gpm_root: Path) -> Dict[str, object]:
    center_lon = motion.first_finite(row, ["track_lon_180", "center_lon"])
    center_lat = motion.first_finite(row, ["track_lat", "center_lat"])
    move_dir_deg = motion.first_finite(row, ["track_move_dir_deg", "track_move_dir_deg_interp"])
    tif_path, tif_exists = motion.resolve_tif_path(row, gpm_root)

    if not np.isfinite(center_lon) or not np.isfinite(center_lat) or not np.isfinite(move_dir_deg):
        return empty_row("missing_center_or_motion", tif_exists=tif_exists)
    if tif_path is None or not tif_exists:
        return empty_row("missing_tif", tif_exists=False)

    try:
        rain, lon_grid, lat_grid, _ = motion.read_rain_and_coords(tif_path)
        x_motion, y_motion = motion_grids(motion, lon_grid, lat_grid, center_lon, center_lat, move_dir_deg)
        metrics = compute_dual_rainband_width_metrics(rain, x_motion, y_motion)
    except Exception as exc:  # pragma: no cover - batch robustness
        out = empty_row(f"compute_error:{type(exc).__name__}", tif_exists=True)
        return out

    status = "ok"
    if int(metrics["rainband_valid_grid_count"]) < 5:
        status = "insufficient_main_rainband_points"
    elif not np.isfinite(float(metrics["rainband_width_km"])):
        status = "nonfinite_main_rainband_width"
    out = empty_row(status, tif_exists=True)
    out.update(metrics)
    return out


def summarize_qc(df: pd.DataFrame, output_file: Path, copy_file: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    status_counts = df["rainband_width_status"].value_counts(dropna=False).to_dict()
    for key, value in status_counts.items():
        records.append({"check": "status_count", "item": str(key), "value": int(value)})

    for col in RAINBAND_WIDTH_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        records.append({"check": "valid_count", "item": col, "value": int(s.notna().sum())})
        records.append({"check": "nan_count", "item": col, "value": int(s.isna().sum())})
        records.append({"check": "inf_count", "item": col, "value": int(np.isinf(s.to_numpy(dtype=float)).sum())})

    records.extend(
        [
            {"check": "rows_total", "item": "all", "value": int(len(df))},
            {"check": "output_file", "item": str(output_file.relative_to(PROJECT_ROOT)), "value": ""},
            {"check": "copy_file", "item": str(copy_file.relative_to(PROJECT_ROOT)), "value": ""},
            {"check": "coordinate_basis", "item": "GeoTIFF lon/lat rotated to storm motion using 111.32*cos(lat) and 111.32 km/deg", "value": ""},
        ]
    )
    qc = pd.DataFrame(records)
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(QC_PATH, index=False, encoding="utf-8-sig")
    return qc


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add rainband-width features to final Problem-1 env table.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copy-file", default=str(DEFAULT_COPY))
    parser.add_argument("--backup-file", default=str(DEFAULT_BACKUP))
    parser.add_argument("--gpm-root", default=str(DEFAULT_GPM_ROOT))
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    copy_file = Path(args.copy_file)
    backup_file = Path(args.backup_file)
    gpm_root = Path(args.gpm_root)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input table: {input_file}")
    if not gpm_root.exists():
        raise FileNotFoundError(f"Missing GPM root: {gpm_root}")

    motion = load_motion_module()
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    print(f"[32] Input: {input_file} {df.shape}")
    print(f"[32] GPM root: {gpm_root}")
    if output_file.resolve() == input_file.resolve() and not backup_file.exists():
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_file, backup_file)
        print(f"[32] Backup written: {backup_file}")

    records = [
        compute_one(row, motion, gpm_root)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="rainband width")
    ]
    metrics = pd.DataFrame(records)
    out = pd.concat(
        [
            df.drop(columns=[c for c in ALL_NEW_COLS if c in df.columns], errors="ignore").reset_index(drop=True),
            metrics[ALL_NEW_COLS].reset_index(drop=True),
        ],
        axis=1,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False, encoding="utf-8-sig")
    if copy_file.resolve() != output_file.resolve():
        copy_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(copy_file, index=False, encoding="utf-8-sig")

    qc = summarize_qc(out, output_file, copy_file)
    print("[32] Status counts:")
    print(out["rainband_width_status"].value_counts(dropna=False).to_string())
    print(f"[32] Output: {output_file} {out.shape}")
    print(f"[32] Copy: {copy_file}")
    print(f"[32] QC: {QC_PATH}")
    print(qc.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
