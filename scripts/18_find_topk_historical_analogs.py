#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_find_topk_historical_analogs.py

Problem 2: find the top-K historical half-hour analogs for each target
half-hour sample.

This script intentionally reads only tabular indexes produced by script 17.
It does not open the 201 x 201 GPM GeoTIFF rain fields; those are left for the
field generator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_LOCATION_FEATURES = ["track_lat", "track_lon_180"]
RATE_TO_HALFHOUR_MM = 0.5

TARGET_META_CANDIDATES = [
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
    "track_wind",
    "track_pressure",
    "pressure_deficit",
    "intensity_index",
    "track_move_speed_kmh",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
]

HISTORY_META_CANDIDATES = [
    "time",
    "time_end",
    "source_file",
    "gpm_event_uid",
    "track_event_uid",
    "track_typhoon_name",
    "track_lat",
    "track_lon_180",
    "center_lon",
    "center_lat",
    "bbox_lon_min",
    "bbox_lon_max",
    "bbox_lat_min",
    "bbox_lat_max",
    "tif_path",
    "tif_path_exists",
]

KEY_Y_METRICS = [
    "rain_mean",
    "rain_std",
    "rain_max",
    "rain_p90",
    "rain_p95",
    "rain_p99",
    "rain_area_10_km2",
    "rain_area_20_km2",
    "centroid_offset_km",
    "centroid_relative_to_motion_deg",
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
    "anisotropy",
    "rain_gini",
    "rain_entropy_norm",
]


def read_csv(path: Path, parse_dates: Sequence[str] = ("time", "time_end")) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def load_feature_cols(path: Path) -> List[str]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "x_feature" not in df.columns:
        raise ValueError(f"{path} must contain an x_feature column.")
    return df["x_feature"].dropna().astype(str).tolist()


def enrich_target_metadata(target: pd.DataFrame, full_path: Optional[Path]) -> Tuple[pd.DataFrame, Dict]:
    report: Dict = {"target_full_file": str(full_path) if full_path else None, "used": False}
    if full_path is None or not full_path.exists():
        return target, report

    full = read_csv(full_path, parse_dates=("time",))
    report["used"] = True
    report["full_shape"] = [int(full.shape[0]), int(full.shape[1])]

    merge_keys = [c for c in ["event_uid", "time"] if c in target.columns and c in full.columns]
    if len(merge_keys) < 2:
        report["merge_keys"] = merge_keys
        report["merged"] = False
        return target, report

    enrich_cols = [c for c in full.columns if c not in target.columns]
    for required in ["event_uid", "time"]:
        if required not in enrich_cols:
            enrich_cols.insert(0, required)
    enrich_cols = list(dict.fromkeys(enrich_cols))

    full_small = full[enrich_cols].drop_duplicates(subset=merge_keys)
    out = target.merge(full_small, on=merge_keys, how="left", validate="one_to_one")

    report["merge_keys"] = merge_keys
    report["merged"] = True
    report["added_columns"] = [c for c in out.columns if c not in target.columns]
    report["unmatched_rows"] = int(out[[c for c in report["added_columns"] if c in out.columns]].isna().all(axis=1).sum()) if report["added_columns"] else 0
    return out, report


def find_y_metric_cols(history: pd.DataFrame) -> List[str]:
    out = [c for c in KEY_Y_METRICS if c in history.columns]
    extra = [
        c for c in history.columns
        if (
            c.startswith("rain_")
            or c.startswith("centroid_")
            or c.startswith("r50_")
            or c.startswith("r80_")
            or c.startswith("r90_")
            or c.startswith("asym_")
            or c.startswith("quad_")
            or c.startswith("rainband_")
            or c in {"major_axis_km", "minor_axis_km", "anisotropy", "orientation_deg"}
        )
        and c not in out
    ]
    return out + extra


def stats_from_history(history: pd.DataFrame, feature: str) -> Dict[str, float]:
    s = pd.to_numeric(history[feature], errors="coerce")
    median = float(s.median(skipna=True))
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    std = float(s.std(skipna=True))
    scale = iqr if np.isfinite(iqr) and iqr > 1e-9 else std
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return {"feature": feature, "median": median, "scale": float(scale)}


def build_scaler_lookup(history: pd.DataFrame, scaler_path: Path, x_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    lookup: Dict[str, Dict[str, float]] = {}
    if scaler_path.exists():
        scaler = pd.read_csv(scaler_path, encoding="utf-8-sig")
        for row in scaler.to_dict(orient="records"):
            feature = str(row.get("feature", ""))
            if feature:
                lookup[feature] = {
                    "feature": feature,
                    "median": float(row.get("median", 0.0)),
                    "scale": float(row.get("scale", 1.0)),
                }

    for feature in x_cols:
        if feature not in lookup:
            lookup[feature] = stats_from_history(history, feature)
    return lookup


def standardize_frame(df: pd.DataFrame, x_cols: Sequence[str], scaler: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    arrs = []
    missing = np.zeros(len(df), dtype=np.int16)
    for feature in x_cols:
        s = pd.to_numeric(df[feature], errors="coerce")
        miss = s.isna().to_numpy()
        missing += miss.astype(np.int16)
        median = scaler[feature]["median"]
        scale = scaler[feature]["scale"]
        values = s.fillna(median).to_numpy(dtype=np.float64)
        arrs.append((values - median) / scale)
    matrix = np.column_stack(arrs).astype(np.float32)
    return matrix, missing


def batched_topk(
    target_z: np.ndarray,
    history_z: np.ndarray,
    k: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_target = target_z.shape[0]
    n_history = history_z.shape[0]
    k = min(k, n_history)

    out_idx = np.empty((n_target, k), dtype=np.int32)
    out_dist = np.empty((n_target, k), dtype=np.float32)

    history_norm = np.sum(history_z * history_z, axis=1, dtype=np.float64)
    history_t = history_z.T.astype(np.float32, copy=False)

    for start in range(0, n_target, batch_size):
        end = min(start + batch_size, n_target)
        batch = target_z[start:end]
        batch_norm = np.sum(batch * batch, axis=1, dtype=np.float64)
        dist_sq = batch_norm[:, None] + history_norm[None, :] - 2.0 * (batch @ history_t)
        np.maximum(dist_sq, 0.0, out=dist_sq)

        part = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist_sq, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        dist = np.sqrt(np.take_along_axis(dist_sq, idx, axis=1))

        out_idx[start:end] = idx.astype(np.int32)
        out_dist[start:end] = dist.astype(np.float32)

    return out_idx, out_dist


def value_as_output(value: object) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def prefixed_row(prefix: str, row: pd.Series, cols: Iterable[str], rename: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    rename = rename or {}
    out: Dict[str, object] = {}
    for col in cols:
        if col in row.index:
            out[rename.get(col, f"{prefix}{col}")] = value_as_output(row[col])
    return out


def add_rate_conversions(record: Dict[str, object], y_cols: Sequence[str], hist_row: pd.Series) -> None:
    for col in y_cols:
        if re.match(r"^rain_(mean|std|max|p\d+)$", col) and col in hist_row.index:
            value = pd.to_numeric(pd.Series([hist_row[col]]), errors="coerce").iloc[0]
            if pd.notna(value):
                record[f"history_{col}_halfhour_mm"] = float(value) * RATE_TO_HALFHOUR_MM


def target_summary(target: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    name_col = "target_name_norm" if "target_name_norm" in target.columns else None
    if name_col is None:
        return {}

    out: Dict[str, Dict[str, object]] = {}
    for name, sub in target.groupby(name_col, dropna=False):
        item: Dict[str, object] = {"rows": int(len(sub))}
        if "time" in sub.columns:
            item["time_min"] = str(sub["time"].min())
            item["time_max"] = str(sub["time"].max())
        for col in ["track_lon_180", "track_lat", "center_lon", "center_lat"]:
            if col in sub.columns:
                item[f"{col}_min"] = float(pd.to_numeric(sub[col], errors="coerce").min())
                item[f"{col}_max"] = float(pd.to_numeric(sub[col], errors="coerce").max())
        out[str(name)] = item
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find top-K historical analogs for each target half-hour.")
    parser.add_argument("--library-file", default="data/processed/problem2_historical_library_index.csv")
    parser.add_argument("--target-file", default="data/processed/problem2_target_model_x_aligned.csv")
    parser.add_argument("--target-full-file", default="data/processed/target_typhoon_inputs_2024_halfhour_leakage_safe.csv")
    parser.add_argument("--x-cols-file", default="outputs/tables/problem2/problem2_x_feature_cols.csv")
    parser.add_argument("--scaler-file", default="outputs/tables/problem2/problem2_robust_scaler_stats.csv")
    parser.add_argument("--out-file", default="data/processed/problem2_target_topk_similar_history.csv")
    parser.add_argument("--summary-file", default="outputs/tables/problem2/problem2_target_topk_similarity_summary.csv")
    parser.add_argument("--report-file", default="outputs/tables/problem2/target_topk_similarity_report.json")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-add-location-features", action="store_true",
                        help="Do not add track_lat/track_lon_180 even if they are available as safe metadata.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    library_path = Path(args.library_file)
    target_path = Path(args.target_file)
    target_full_path = Path(args.target_full_file) if args.target_full_file else None
    x_cols_path = Path(args.x_cols_file)
    scaler_path = Path(args.scaler_file)
    out_path = Path(args.out_file)
    summary_path = Path(args.summary_file)
    report_path = Path(args.report_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[18] Reading historical index: {library_path}")
    history = read_csv(library_path)
    print(f"[18] Reading target aligned X: {target_path}")
    target = read_csv(target_path, parse_dates=("time",))
    target, target_enrich_report = enrich_target_metadata(target, target_full_path)

    x_cols = load_feature_cols(x_cols_path)
    extra_features: List[str] = []
    if not args.no_add_location_features:
        for feature in DEFAULT_LOCATION_FEATURES:
            if feature not in x_cols and feature in history.columns and feature in target.columns:
                x_cols.append(feature)
                extra_features.append(feature)

    missing_in_history = [c for c in x_cols if c not in history.columns]
    missing_in_target = [c for c in x_cols if c not in target.columns]
    if missing_in_history or missing_in_target:
        raise RuntimeError(
            "Selected X features are not aligned. "
            f"missing_in_history={missing_in_history}; missing_in_target={missing_in_target}"
        )

    name_col = "track_typhoon_name" if "track_typhoon_name" in history.columns else None
    if name_col:
        target_like = history[name_col].map(normalize_name).str.contains("KONGREY|MANYI", na=False)
        if bool(target_like.any()):
            sample = history.loc[target_like, [c for c in ["time", name_col, "track_event_uid", "gpm_event_uid"] if c in history.columns]].head(10)
            raise RuntimeError(f"Historical library contains target-like storms: {sample.to_dict(orient='records')}")

    scaler = build_scaler_lookup(history, scaler_path, x_cols)
    history_z, history_missing_n = standardize_frame(history, x_cols, scaler)
    target_z, target_missing_n = standardize_frame(target, x_cols, scaler)

    print(f"[18] Finding top-{args.k} analogs: target={len(target)}, history={len(history)}, features={len(x_cols)}")
    top_idx, top_dist = batched_topk(target_z, history_z, args.k, args.batch_size)

    y_cols = find_y_metric_cols(history)
    target_cols = [c for c in TARGET_META_CANDIDATES if c in target.columns]
    history_cols = [c for c in HISTORY_META_CANDIDATES if c in history.columns]
    target_rename = {
        "target_name_norm": "target_name_norm",
        "target_display_name": "target_display_name",
        "event_uid": "target_event_uid",
        "source_file": "target_source_file",
        "time": "target_time",
    }
    history_rename = {
        "time": "history_time",
        "time_end": "history_time_end",
        "source_file": "history_source_file",
        "tif_path": "history_tif_path",
        "tif_path_exists": "history_tif_path_exists",
    }

    records: List[Dict[str, object]] = []
    norm_denominator = math.sqrt(len(x_cols))
    for target_i in range(len(target)):
        target_row = target.iloc[target_i]
        target_bits = prefixed_row("target_", target_row, target_cols, target_rename)
        for rank_i, hist_i in enumerate(top_idx[target_i], start=1):
            hist_row = history.iloc[int(hist_i)]
            distance = float(top_dist[target_i, rank_i - 1])
            rmse_distance = distance / norm_denominator if norm_denominator > 0 else distance
            record: Dict[str, object] = {
                "target_row_id": int(target_i),
                "analog_rank": int(rank_i),
                "analog_distance": distance,
                "analog_rmse_distance": rmse_distance,
                "analog_similarity_score": 1.0 / (1.0 + rmse_distance),
                "x_features_n": int(len(x_cols)),
                "target_x_missing_n": int(target_missing_n[target_i]),
                "history_x_missing_n": int(history_missing_n[int(hist_i)]),
            }
            record.update(target_bits)
            record.update(prefixed_row("history_", hist_row, history_cols, history_rename))
            for col in y_cols:
                if col in hist_row.index:
                    record[f"history_{col}"] = value_as_output(hist_row[col])
            add_rate_conversions(record, y_cols, hist_row)
            records.append(record)

    topk = pd.DataFrame(records)
    topk.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    group_cols = [c for c in ["target_name_norm", "target_display_name", "target_event_uid"] if c in topk.columns]
    if group_cols:
        for keys, sub in topk.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row.update({
                "target_times_n": int(sub["target_row_id"].nunique()),
                "topk_rows_n": int(len(sub)),
                "unique_history_storms_n": int(sub["history_track_typhoon_name"].nunique()) if "history_track_typhoon_name" in sub.columns else np.nan,
                "unique_history_events_n": int(sub["history_track_event_uid"].nunique()) if "history_track_event_uid" in sub.columns else np.nan,
                "rank1_mean_distance": float(sub.loc[sub["analog_rank"].eq(1), "analog_distance"].mean()),
                "rank1_mean_similarity": float(sub.loc[sub["analog_rank"].eq(1), "analog_similarity_score"].mean()),
            })
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report = {
        "script": "18_find_topk_historical_analogs.py",
        "k": int(args.k),
        "inputs": {
            "historical_library_index": str(library_path),
            "target_aligned_x": str(target_path),
            "target_full_metadata": str(target_full_path) if target_full_path else None,
            "x_feature_cols": str(x_cols_path),
            "robust_scaler_stats": str(scaler_path),
        },
        "outputs": {
            "target_topk_similar_history": str(out_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "history_shape": [int(history.shape[0]), int(history.shape[1])],
        "target_shape_after_metadata_enrichment": [int(target.shape[0]), int(target.shape[1])],
        "target_metadata_enrichment": target_enrich_report,
        "x_features_n": int(len(x_cols)),
        "x_features": x_cols,
        "extra_safe_features_added": extra_features,
        "target_summary": target_summary(target),
        "topk_rows_n": int(len(topk)),
        "history_tif_path_exists_rate_in_topk": float(topk["history_tif_path_exists"].mean()) if "history_tif_path_exists" in topk.columns else None,
        "gpm_precip_rate_unit": "mm/hr",
        "halfhour_accumulation_factor_hours": RATE_TO_HALFHOUR_MM,
        "unit_note": "A 10 mm/hr GPM snapshot corresponds to 5 mm accumulated precipitation over 30 minutes.",
        "hard_rules": {
            "target_like_history_names_n": 0,
            "target_required_names_present": sorted(target["target_name_norm"].dropna().astype(str).unique().tolist()) if "target_name_norm" in target.columns else [],
            "passed": True,
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== 18 top-K historical analog search complete ==========")
    print(f"Top-K table: {out_path}")
    print(f"Top-K shape: {topk.shape[0]} x {topk.shape[1]}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"X features: {len(x_cols)}")
    if extra_features:
        print(f"Extra safe path features added: {extra_features}")
    if 'history_tif_path_exists' in topk.columns:
        print(f"Top-K tif_path exists rate: {topk['history_tif_path_exists'].mean():.2%}")
    print("GPM rate unit: mm/hr; half-hour accumulation multiplier: 0.5")


if __name__ == "__main__":
    main()
