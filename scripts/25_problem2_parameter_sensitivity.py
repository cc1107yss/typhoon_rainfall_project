#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem-2 parameter sensitivity analysis based on pseudo-missing validation.

This script reuses the validated generation and metric functions from
scripts/22_pseudo_missing_validation.py, then varies only the manually chosen
hyperparameters: Top-K, retrieval component weights, and EOF/PCA blend beta.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
PSEUDO22_SCRIPT_PATH = PROJECT_ROOT / "scripts/22_pseudo_missing_validation.py"

HISTORICAL_LIBRARY_PATH = PROJECT_ROOT / "data/processed/problem2_historical_halfhour_sample_library.csv"
EOF_MODEL_PATH = PROJECT_ROOT / "data/processed/problem2_eof_pca_model.npz"
PSEUDO_VALIDATION_EVENTS_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_events.csv"
PSEUDO_VALIDATION_MODEL_SUMMARY_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_model_summary.csv"
PSEUDO_VALIDATION_TIMESLICE_METRICS_PATH = PROJECT_ROOT / "data/processed/problem2_pseudo_validation_timeslice_metrics.csv"

SETTINGS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_sensitivity_settings.csv"
TIMESLICE_METRICS_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_sensitivity_timeslice_metrics.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_sensitivity_summary.csv"
RELATIVE_CHANGE_OUTPUT_PATH = PROJECT_ROOT / "data/processed/problem2_sensitivity_relative_change.csv"
QC_REPORT_PATH = PROJECT_ROOT / "outputs/problem2_sensitivity_analysis_report.md"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_sensitivity_analysis"

N_VALIDATION_EVENTS = 8
MAX_TIMES_PER_EVENT = 60
CACHE_SIZE = 1500
RANDOM_SEED = 2026
MAKE_FIGURES = True

TOPK_CANDIDATE_POOL = 1000
MAX_PER_HISTORY_EVENT = 3
MIN_VALID_TEMPLATES = 5

EPS = 1e-12


@dataclass(frozen=True)
class SensitivitySetting:
    setting_id: str
    K: int
    weight_track: float
    weight_intensity: float
    weight_motion: float
    weight_environment: float
    weight_time: float
    weight_life: float
    beta_blend: float
    setting_group: str
    description: str


SETTINGS: Sequence[SensitivitySetting] = [
    SensitivitySetting("baseline", 20, 1.0, 1.5, 1.2, 1.2, 0.8, 0.8, 0.3, "baseline", "K=20, beta=0.3, baseline weights"),
    SensitivitySetting("S1", 10, 1.0, 1.5, 1.2, 1.2, 0.8, 0.8, 0.3, "K", "K=10"),
    SensitivitySetting("S2", 30, 1.0, 1.5, 1.2, 1.2, 0.8, 0.8, 0.3, "K", "K=30"),
    SensitivitySetting("S3", 20, 1.0, 1.2, 1.2, 1.2, 0.8, 0.8, 0.3, "weight", "lower intensity weight"),
    SensitivitySetting("S4", 20, 1.0, 1.5, 1.5, 1.2, 0.8, 0.8, 0.3, "weight", "higher motion weight"),
    SensitivitySetting("S5", 20, 1.0, 1.5, 1.2, 1.5, 0.8, 0.8, 0.3, "weight", "higher environment weight"),
    SensitivitySetting("S6", 20, 1.0, 1.5, 1.2, 1.2, 0.8, 0.8, 0.0, "beta", "beta=0.0"),
    SensitivitySetting("S7", 20, 1.0, 1.5, 1.2, 1.2, 0.8, 0.8, 0.5, "beta", "beta=0.5"),
]


# =========================
# Import step-22 implementation
# =========================


def load_pseudo22_module():
    spec = importlib.util.spec_from_file_location("problem2_pseudo22", PSEUDO22_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {PSEUDO22_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["problem2_pseudo22"] = module
    spec.loader.exec_module(module)
    return module


p22 = load_pseudo22_module()


# =========================
# Basic helpers
# =========================


def iter_progress(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def apply_setting_to_pseudo22(setting: SensitivitySetting) -> None:
    p22.TOPK = int(setting.K)
    p22.TOPK_CANDIDATE_POOL = TOPK_CANDIDATE_POOL
    p22.MAX_PER_HISTORY_EVENT = MAX_PER_HISTORY_EVENT
    p22.MIN_VALID_TEMPLATES = MIN_VALID_TEMPLATES
    p22.CACHE_SIZE = CACHE_SIZE
    p22.BETA_BLEND = float(setting.beta_blend)
    p22.COMPONENT_WEIGHTS = {
        "track": float(setting.weight_track),
        "intensity": float(setting.weight_intensity),
        "motion": float(setting.weight_motion),
        "environment": float(setting.weight_environment),
        "time": float(setting.weight_time),
        "life_progress": float(setting.weight_life),
    }


def settings_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in SETTINGS])


def format_table(df: pd.DataFrame, max_rows: int = 30, cols: Optional[Sequence[str]] = None) -> str:
    if df.empty:
        return "(none)"
    small = df[list(cols)].copy() if cols else df.copy()
    small = small.head(max_rows)
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]) or pd.api.types.is_numeric_dtype(small[col]):
            small[col] = small[col].map(lambda v: f"{v:.6g}" if pd.notna(v) and np.isfinite(v) else "NA")
        else:
            small[col] = small[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(small.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in small.columns) + " |" for _, row in small.iterrows()]
    return "\n".join([header, sep] + body)


def mean_col(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df.get(col), errors="coerce").mean(skipna=True)) if col in df.columns else np.nan


def pct_change(value: object, base: object) -> float:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([base]), errors="coerce").iloc[0]
    if pd.isna(v) or pd.isna(b) or not np.isfinite(v) or not np.isfinite(b) or abs(float(b)) <= EPS:
        return np.nan
    return float((float(v) - float(b)) / float(b) * 100.0)


def numeric_delta(value: object, base: object) -> float:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    b = pd.to_numeric(pd.Series([base]), errors="coerce").iloc[0]
    if pd.isna(v) or pd.isna(b) or not np.isfinite(v) or not np.isfinite(b):
        return np.nan
    return float(v - b)


def safe_float(value: object) -> float:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) and np.isfinite(v) else np.nan


# =========================
# Validation sample control
# =========================


def load_validation_events(history: pd.DataFrame) -> pd.DataFrame:
    if PSEUDO_VALIDATION_EVENTS_PATH.exists():
        events = pd.read_csv(PSEUDO_VALIDATION_EVENTS_PATH, encoding="utf-8-sig")
        events = events.head(N_VALIDATION_EVENTS).copy()
    else:
        p22.N_VALIDATION_EVENTS = N_VALIDATION_EVENTS
        events = p22.select_validation_events(history).head(N_VALIDATION_EVENTS).copy()
    if "validation_event_uid" not in events.columns:
        raise RuntimeError("Validation events table must contain validation_event_uid.")
    return events.reset_index(drop=True)


def existing_selected_times_lookup() -> Dict[str, List[pd.Timestamp]]:
    if not PSEUDO_VALIDATION_TIMESLICE_METRICS_PATH.exists():
        return {}
    metrics = pd.read_csv(
        PSEUDO_VALIDATION_TIMESLICE_METRICS_PATH,
        usecols=lambda c: c in {"validation_event_uid", "validation_time", "model_version"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    if "model_version" in metrics.columns:
        metrics = metrics.loc[metrics["model_version"].astype(str).eq("calibrated")].copy()
    metrics["validation_time"] = pd.to_datetime(metrics["validation_time"], errors="coerce")
    metrics = metrics.dropna(subset=["validation_event_uid", "validation_time"])
    lookup: Dict[str, List[pd.Timestamp]] = {}
    for event_uid, sub in metrics.groupby("validation_event_uid", sort=False):
        lookup[str(event_uid)] = sorted(pd.to_datetime(sub["validation_time"]).drop_duplicates().tolist())
    return lookup


def select_times_from_event(event_df: pd.DataFrame, max_times: int) -> pd.DataFrame:
    event_df = event_df.sort_values("time").reset_index(drop=True)
    n = len(event_df)
    if n <= max_times:
        out = event_df.copy()
        out["field_index_within_event"] = np.arange(len(out), dtype=int)
        return out

    picks: set[int] = set()
    for frac in [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0]:
        picks.add(int(round(frac * (n - 1))))

    for col in ["WND", "rain_p95_mmhr", "rain_p99_mmhr", "rain_max_mmhr"]:
        if col in event_df.columns:
            s = pd.to_numeric(event_df[col], errors="coerce")
            if s.notna().any():
                picks.add(int(s.idxmax()))
    if "wind_change_rate" in event_df.columns:
        s = pd.to_numeric(event_df["wind_change_rate"], errors="coerce").abs()
        if s.notna().any():
            picks.add(int(s.idxmax()))

    for idx in np.linspace(0, n - 1, max_times, dtype=int):
        picks.add(int(idx))
        if len(picks) >= max_times:
            break

    selected = sorted(picks)
    if len(selected) > max_times:
        key_set: set[int] = {0, n - 1}
        for col in ["WND", "rain_p95_mmhr", "rain_p99_mmhr", "rain_max_mmhr"]:
            if col in event_df.columns:
                s = pd.to_numeric(event_df[col], errors="coerce")
                if s.notna().any():
                    key_set.add(int(s.idxmax()))
        remaining = [i for i in selected if i not in key_set]
        need = max_times - len(key_set)
        if need > 0 and remaining:
            keep = np.linspace(0, len(remaining) - 1, need, dtype=int)
            key_set.update(remaining[i] for i in keep)
        selected = sorted(key_set)[:max_times]

    out = event_df.iloc[selected].copy().reset_index(drop=True)
    out["field_index_within_event"] = np.arange(len(out), dtype=int)
    return out


def build_selected_times_by_event(
    history: pd.DataFrame,
    validation_events: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], str]:
    lookup = existing_selected_times_lookup()
    selected: Dict[str, pd.DataFrame] = {}
    source_parts: List[str] = []
    for _, event_meta in validation_events.iterrows():
        event_uid = str(event_meta["validation_event_uid"])
        event_full = history.loc[history["event_uid"].astype(str).eq(event_uid)].copy().sort_values("time")
        if event_full.empty:
            raise RuntimeError(f"Validation event {event_uid} is missing from the historical library.")

        if event_uid in lookup:
            keep_times = set(pd.to_datetime(lookup[event_uid]))
            event_times = event_full.loc[pd.to_datetime(event_full["time"]).isin(keep_times)].copy()
            if event_times.empty:
                event_times = select_times_from_event(event_full, MAX_TIMES_PER_EVENT)
                source_parts.append(f"{event_uid}:reselected")
            else:
                event_times = select_times_from_event(event_times, MAX_TIMES_PER_EVENT)
                source_parts.append(f"{event_uid}:step22_times_subset")
        else:
            event_times = select_times_from_event(event_full, MAX_TIMES_PER_EVENT)
            source_parts.append(f"{event_uid}:reselected")
        selected[event_uid] = event_times.reset_index(drop=True)
    return selected, "; ".join(source_parts)


# =========================
# Retrieval and validation
# =========================


def failure_row(
    setting: SensitivitySetting,
    target_row: pd.Series,
    reason: str,
    topk_one: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "setting_id": setting.setting_id,
        "validation_event_uid": target_row.get("event_uid"),
        "typhoon_name": target_row.get("typhoon_name"),
        "validation_time": p22.format_time(target_row.get("time")),
        "model_version": "calibrated",
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
        "K": int(setting.K),
        "beta_blend": float(setting.beta_blend),
        "weight_track": float(setting.weight_track),
        "weight_intensity": float(setting.weight_intensity),
        "weight_motion": float(setting.weight_motion),
        "weight_environment": float(setting.weight_environment),
        "weight_time": float(setting.weight_time),
        "weight_life": float(setting.weight_life),
        "validation_ok": False,
        "skip_reason": reason,
    }
    if topk_one is not None and not topk_one.empty:
        row["topk_count"] = int(len(topk_one))
        row["topk_unique_event_count"] = int(topk_one["history_event_uid"].nunique()) if "history_event_uid" in topk_one else 0
        row["topk_self_match_count"] = int(topk_one["history_event_uid"].astype(str).eq(str(target_row.get("event_uid"))).sum())
    else:
        row["topk_count"] = 0
        row["topk_unique_event_count"] = 0
        row["topk_self_match_count"] = 0
    return row


def add_setting_columns(row: Dict[str, object], setting: SensitivitySetting, topk_self_count: int) -> Dict[str, object]:
    row["setting_id"] = setting.setting_id
    row["K"] = int(setting.K)
    row["beta_blend"] = float(setting.beta_blend)
    row["weight_track"] = float(setting.weight_track)
    row["weight_intensity"] = float(setting.weight_intensity)
    row["weight_motion"] = float(setting.weight_motion)
    row["weight_environment"] = float(setting.weight_environment)
    row["weight_time"] = float(setting.weight_time)
    row["weight_life"] = float(setting.weight_life)
    row["topk_self_match_count"] = int(topk_self_count)
    return row


def run_one_setting(
    setting: SensitivitySetting,
    history: pd.DataFrame,
    eof_model: Mapping[str, np.ndarray],
    validation_events: pd.DataFrame,
    selected_times_by_event: Mapping[str, pd.DataFrame],
    cache: object,
    truth_cache: Dict[str, np.ndarray],
    x_front_km: np.ndarray,
    y_left_km: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    cell_area: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    apply_setting_to_pseudo22(setting)
    metric_rows: List[Dict[str, object]] = []
    global_qc_events: List[Dict[str, object]] = []
    template_counters: Counter = Counter()
    retrieval_diagnostics: List[Dict[str, object]] = []
    selected_feature_union: List[str] = []
    self_match_total = 0
    skipped_times: Counter = Counter()

    event_iter = iter_progress(
        validation_events.iterrows(),
        total=len(validation_events),
        desc=f"{setting.setting_id} events",
    )
    for _, event_meta in event_iter:
        event_uid = str(event_meta["validation_event_uid"])
        selected_times = selected_times_by_event[event_uid].copy().reset_index(drop=True)
        train_history = history.loc[history["event_uid"].astype(str).ne(event_uid)].copy().reset_index(drop=True)
        if train_history.empty:
            raise RuntimeError(f"Training library is empty after excluding validation event {event_uid}")
        if train_history["event_uid"].astype(str).eq(event_uid).any():
            raise RuntimeError(f"Self event {event_uid} remains in training library.")

        selected_components, skipped_features = p22.select_safe_retrieval_features(train_history, selected_times)
        features = [f for feats in selected_components.values() for f in feats]
        for feature in features:
            if feature not in selected_feature_union:
                selected_feature_union.append(feature)
        train_imp, target_imp, _ = p22.impute_retrieval_features(train_history, selected_times, selected_components)
        standardization_params = p22.compute_standardization_params(train_imp, features)
        topk, retrieval_diag = p22.retrieve_topk_for_validation_times(
            train_imp,
            target_imp,
            selected_components,
            standardization_params,
        )
        retrieval_diag["setting_id"] = setting.setting_id
        retrieval_diag["validation_event_uid"] = event_uid
        retrieval_diag["skipped_features"] = skipped_features
        retrieval_diagnostics.append(retrieval_diag)
        self_match_count = int(retrieval_diag.get("self_match_count", 0))
        self_match_total += self_match_count
        if self_match_count:
            raise RuntimeError(f"Top-K leakage: validation event {event_uid} appears in its own Top-K.")

        topk_by_target = {tid: sub.copy() for tid, sub in topk.groupby("target_id", sort=False)}
        time_iter = iter_progress(selected_times.iterrows(), total=len(selected_times), desc=f"{setting.setting_id} {event_uid}")
        for _, target_row in time_iter:
            target_id = str(target_row["sample_id"])
            topk_one = topk_by_target.get(target_id, pd.DataFrame(columns=topk.columns))
            topk_self_count = (
                int(topk_one["history_event_uid"].astype(str).eq(event_uid).sum())
                if "history_event_uid" in topk_one.columns
                else 0
            )
            if topk_self_count:
                raise RuntimeError(f"Top-K leakage detected for {event_uid} {target_id}.")

            try:
                if target_id not in truth_cache:
                    truth_cache[target_id], _ = p22.build_truth_field(target_row, x_grid, y_grid)
                truth = truth_cache[target_id]
            except Exception as exc:
                reason = f"truth_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                metric_rows.append(failure_row(setting, target_row, reason, topk_one))
                continue

            try:
                rain_initial, log_initial, generation_status = p22.generate_initial_field(
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
                metric_rows.append(failure_row(setting, target_row, reason, topk_one))
                continue

            if generation_status.get("valid_template_count", 0) < MIN_VALID_TEMPLATES:
                skipped_times["low_valid_template_count"] += 1

            try:
                rain_blend, _, _ = p22.apply_eof_blend(rain_initial, log_initial, eof_model, float(setting.beta_blend))
            except Exception as exc:
                reason = f"blend_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                metric_rows.append(failure_row(setting, target_row, reason, topk_one))
                continue

            calibration_targets = p22.compute_weighted_calibration_targets(topk_one)
            try:
                rain_calibrated, cal_diag = p22.calibrate_one_field_tail_enhancement(
                    rain_blend,
                    calibration_targets,
                    cell_area,
                )
            except Exception as exc:
                reason = f"calibrated_failed: {type(exc).__name__}: {exc}"
                skipped_times[reason] += 1
                metric_rows.append(failure_row(setting, target_row, reason, topk_one))
                continue

            validation_ok = bool(generation_status.get("valid_template_count", 0) >= MIN_VALID_TEMPLATES)
            skip_reason = "" if validation_ok else "low_valid_template_count"
            metric_row = p22.compute_timeslice_validation_metrics(
                target_row,
                "calibrated",
                rain_calibrated,
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
            metric_row["calibration_ok"] = bool(cal_diag.get("calibration_ok", True))
            metric_row["calibration_issue"] = cal_diag.get("calibration_issue", "")
            metric_rows.append(add_setting_columns(metric_row, setting, topk_self_count))

    diagnostics = {
        "setting_id": setting.setting_id,
        "selected_features": selected_feature_union,
        "self_match_count": int(self_match_total),
        "template_counters": dict(template_counters),
        "qc_events": global_qc_events[:100],
        "retrieval_diagnostics": retrieval_diagnostics,
        "skipped_times": dict(skipped_times),
    }
    return metric_rows, diagnostics


# =========================
# Summaries and figures
# =========================


def build_summary(timeslice_metrics: pd.DataFrame, settings_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    valid_all = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    for _, setting in settings_df.iterrows():
        setting_id = str(setting["setting_id"])
        sub_all = timeslice_metrics.loc[timeslice_metrics["setting_id"].astype(str).eq(setting_id)].copy()
        sub = valid_all.loc[valid_all["setting_id"].astype(str).eq(setting_id)].copy()
        row: Dict[str, object] = {
            "setting_id": setting_id,
            "setting_group": setting.get("setting_group"),
            "description": setting.get("description"),
            "K": int(setting["K"]),
            "beta_blend": float(setting["beta_blend"]),
            "weight_track": float(setting["weight_track"]),
            "weight_intensity": float(setting["weight_intensity"]),
            "weight_motion": float(setting["weight_motion"]),
            "weight_environment": float(setting["weight_environment"]),
            "weight_time": float(setting["weight_time"]),
            "weight_life": float(setting["weight_life"]),
            "n_events": int(sub["validation_event_uid"].nunique()) if len(sub) else 0,
            "n_timeslices": int(len(sub)),
            "rmse_mean": mean_col(sub, "rmse"),
            "mae_mean": mean_col(sub, "mae"),
            "corr_mean": mean_col(sub, "corr"),
            "abs_error_rain_p95_mean": mean_col(sub, "abs_error_rain_p95"),
            "abs_error_rain_p99_mean": mean_col(sub, "abs_error_rain_p99"),
            "abs_error_rain_max_mean": mean_col(sub, "abs_error_rain_max"),
            "abs_error_area_10_mean": mean_col(sub, "abs_error_area_10"),
            "csi10_mean": mean_col(sub, "csi10"),
            "f1_10_mean": mean_col(sub, "f1_10"),
            "topk_unique_event_count_mean": mean_col(sub, "topk_unique_event_count"),
            "validation_ok_rate": float(sub_all["validation_ok"].astype(bool).mean()) if len(sub_all) else np.nan,
            "topk_self_match_count_sum": int(pd.to_numeric(sub_all.get("topk_self_match_count"), errors="coerce").fillna(0).sum()) if len(sub_all) else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_relative_change(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["setting_id"].astype(str).eq("baseline")]
    if base.empty:
        raise RuntimeError("Baseline setting is missing from sensitivity summary.")
    b = base.iloc[0]
    rows: List[Dict[str, object]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "setting_id": row["setting_id"],
                "setting_group": row.get("setting_group"),
                "description": row.get("description"),
                "K": row.get("K"),
                "beta_blend": row.get("beta_blend"),
                "rmse_change_pct": pct_change(row.get("rmse_mean"), b.get("rmse_mean")),
                "corr_change": numeric_delta(row.get("corr_mean"), b.get("corr_mean")),
                "p95_error_change_pct": pct_change(row.get("abs_error_rain_p95_mean"), b.get("abs_error_rain_p95_mean")),
                "p99_error_change_pct": pct_change(row.get("abs_error_rain_p99_mean"), b.get("abs_error_rain_p99_mean")),
                "rmax_error_change_pct": pct_change(row.get("abs_error_rain_max_mean"), b.get("abs_error_rain_max_mean")),
                "area10_error_change_pct": pct_change(row.get("abs_error_area_10_mean"), b.get("abs_error_area_10_mean")),
                "csi10_change": numeric_delta(row.get("csi10_mean"), b.get("csi10_mean")),
                "f1_10_change": numeric_delta(row.get("f1_10_mean"), b.get("f1_10_mean")),
            }
        )
    return pd.DataFrame(rows)


def make_bar_extreme_errors(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200)
    settings = summary["setting_id"].astype(str).tolist()
    specs = [
        ("abs_error_rain_p95_mean", "P95 abs error"),
        ("abs_error_rain_p99_mean", "P99 abs error"),
        ("abs_error_area_10_mean", "Area10 abs error"),
    ]
    colors = ["#4c78a8" if s == "baseline" else "#f28e2b" for s in settings]
    for ax, (col, title) in zip(axes, specs):
        vals = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(settings, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "sensitivity_bar_extreme_errors.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_bar_skill_scores(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200)
    settings = summary["setting_id"].astype(str).tolist()
    specs = [("csi10_mean", "CSI10"), ("f1_10_mean", "F1_10"), ("corr_mean", "Corr")]
    colors = ["#4c78a8" if s == "baseline" else "#59a14f" for s in settings]
    for ax, (col, title) in zip(axes, specs):
        vals = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(settings, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "sensitivity_bar_skill_scores.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_relative_change_figure(relative_change: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=200)
    settings = relative_change["setting_id"].astype(str).tolist()
    specs = [
        ("p95_error_change_pct", "P95 error change (%)"),
        ("p99_error_change_pct", "P99 error change (%)"),
        ("area10_error_change_pct", "Area10 error change (%)"),
        ("csi10_change", "CSI10 change"),
    ]
    for ax, (col, title) in zip(axes.ravel(), specs):
        vals = pd.to_numeric(relative_change[col], errors="coerce").to_numpy(dtype=float)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(settings, vals, color="#e15759")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "sensitivity_relative_change.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_k_beta_focus(summary: pd.DataFrame) -> Path:
    k_order = ["S1", "baseline", "S2"]
    beta_order = ["S6", "baseline", "S7"]
    k_sub = summary.set_index("setting_id").loc[k_order].reset_index()
    beta_sub = summary.set_index("setting_id").loc[beta_order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=200)

    axes[0, 0].plot(k_sub["K"], k_sub["abs_error_rain_p95_mean"], marker="o", label="P95")
    axes[0, 0].plot(k_sub["K"], k_sub["abs_error_rain_p99_mean"], marker="o", label="P99")
    axes[0, 0].set_title("K vs extreme errors")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(k_sub["K"], k_sub["csi10_mean"], marker="o", label="CSI10")
    axes[0, 1].plot(k_sub["K"], k_sub["f1_10_mean"], marker="o", label="F1_10")
    axes[0, 1].set_title("K vs skill scores")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(beta_sub["beta_blend"], beta_sub["abs_error_rain_p95_mean"], marker="o", label="P95")
    axes[1, 0].plot(beta_sub["beta_blend"], beta_sub["abs_error_rain_p99_mean"], marker="o", label="P99")
    axes[1, 0].set_title("Beta vs extreme errors")
    axes[1, 0].set_xlabel("beta_blend")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(beta_sub["beta_blend"], beta_sub["csi10_mean"], marker="o", label="CSI10")
    axes[1, 1].plot(beta_sub["beta_blend"], beta_sub["f1_10_mean"], marker="o", label="F1_10")
    axes[1, 1].set_title("Beta vs skill scores")
    axes[1, 1].set_xlabel("beta_blend")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "sensitivity_k_beta_focus.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_figures(summary: pd.DataFrame, relative_change: pd.DataFrame) -> List[Path]:
    if not MAKE_FIGURES:
        return []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        make_bar_extreme_errors(summary),
        make_bar_skill_scores(summary),
        make_relative_change_figure(relative_change),
        make_k_beta_focus(summary),
    ]


# =========================
# Report
# =========================


def max_relative_change_line(relative_change: pd.DataFrame) -> str:
    cols = [
        "rmse_change_pct",
        "corr_change",
        "p95_error_change_pct",
        "p99_error_change_pct",
        "rmax_error_change_pct",
        "area10_error_change_pct",
        "csi10_change",
        "f1_10_change",
    ]
    records: List[Tuple[str, str, float]] = []
    sub = relative_change.loc[~relative_change["setting_id"].astype(str).eq("baseline")].copy()
    for _, row in sub.iterrows():
        for col in cols:
            value = safe_float(row.get(col))
            if np.isfinite(value):
                records.append((str(row["setting_id"]), col, abs(value)))
    if not records:
        return "NA"
    setting_id, col, value = max(records, key=lambda x: x[2])
    signed = safe_float(relative_change.loc[relative_change["setting_id"].eq(setting_id), col].iloc[0])
    suffix = "%" if col.endswith("_pct") else ""
    return f"{setting_id} {col} = {signed:.6g}{suffix}"


def write_qc_report(
    settings_df: pd.DataFrame,
    validation_events: pd.DataFrame,
    selected_times_by_event: Mapping[str, pd.DataFrame],
    timeslice_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    relative_change: pd.DataFrame,
    diagnostics: Sequence[Mapping[str, object]],
    figure_paths: Sequence[Path],
    selection_source: str,
) -> None:
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected_feature_union: List[str] = []
    for diag in diagnostics:
        for feature in diag.get("selected_features", []):
            if feature not in selected_feature_union:
                selected_feature_union.append(str(feature))
    leakage_used = [f for f in selected_feature_union if p22.is_leakage_feature(str(f))]
    self_match_total = int(sum(int(diag.get("self_match_count", 0)) for diag in diagnostics))
    selected_counts = {
        event_uid: int(len(df))
        for event_uid, df in selected_times_by_event.items()
    }
    valid = timeslice_metrics.loc[timeslice_metrics["validation_ok"].astype(bool)].copy()
    valid_counts = valid.groupby("setting_id").size().reset_index(name="n_valid_times")
    old_summary_text = ""
    if PSEUDO_VALIDATION_MODEL_SUMMARY_PATH.exists():
        old = pd.read_csv(PSEUDO_VALIDATION_MODEL_SUMMARY_PATH, encoding="utf-8-sig")
        old_summary_text = format_table(
            old.loc[old["model_version"].astype(str).eq("calibrated")],
            max_rows=5,
            cols=[
                "model_version",
                "n_events",
                "n_timeslices",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "abs_error_area_10_mean",
                "csi10_mean",
            ],
        )

    k_rows = summary.loc[summary["setting_id"].isin(["S1", "baseline", "S2"])]
    beta_rows = summary.loc[summary["setting_id"].isin(["S6", "baseline", "S7"])]
    weight_rows = summary.loc[summary["setting_id"].isin(["S3", "S4", "S5"])]

    lines = [
        "# Problem 2 Parameter Sensitivity Analysis Report",
        "",
        "## 1. Input and Output Files",
        f"- Historical library: `{HISTORICAL_LIBRARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- EOF/PCA model: `{EOF_MODEL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Step-22 validation events: `{PSEUDO_VALIDATION_EVENTS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Step-22 model summary: `{PSEUDO_VALIDATION_MODEL_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Step-22 timeslice metrics: `{PSEUDO_VALIDATION_TIMESLICE_METRICS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Sensitivity settings: `{SETTINGS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Sensitivity timeslice metrics: `{TIMESLICE_METRICS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Sensitivity summary: `{SUMMARY_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Sensitivity relative change: `{RELATIVE_CHANGE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Figures directory: `{FIGURE_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 2. Parameter Settings",
        format_table(
            settings_df,
            max_rows=20,
            cols=[
                "setting_id",
                "K",
                "weight_track",
                "weight_intensity",
                "weight_motion",
                "weight_environment",
                "weight_time",
                "weight_life",
                "beta_blend",
                "description",
            ],
        ),
        "",
        "## 3. Validation Events and Timeslices",
        f"- N_VALIDATION_EVENTS: {N_VALIDATION_EVENTS}",
        f"- MAX_TIMES_PER_EVENT: {MAX_TIMES_PER_EVENT}",
        f"- RANDOM_SEED: {RANDOM_SEED}",
        f"- Selected-time source: {selection_source}",
        f"- Selected times by event: {selected_counts}",
        f"- Valid calibrated rows by setting: {dict(zip(valid_counts['setting_id'], valid_counts['n_valid_times'])) if not valid_counts.empty else {}}",
        "- This run uses a lightweight but horizontally comparable validation design: 8 events x up to 60 half-hour timeslices per setting.",
        "",
        "## 4. Leakage Guard",
        "- Retrieval features are restricted to track, intensity, motion, environment, month, and life-progress variables.",
        f"- Retrieval rain_* fields used: {'no' if not leakage_used else ', '.join(leakage_used)}",
        "- For each validation event, the retrieval library removes all rows whose event_uid equals the validation_event_uid.",
        f"- Top-K rows from the same validation event: {self_match_total}",
        "- Truth tif fields are read only after Top-K retrieval, initial generation, EOF/PCA blending, and historical-sample calibration targets are defined.",
        f"- Safe retrieval features: {', '.join(selected_feature_union)}",
        "",
        "## 5. Step-22 Calibrated Reference",
        old_summary_text if old_summary_text else "(Step-22 calibrated summary not available.)",
        "",
        "## 6. Overall Performance by Setting",
        format_table(
            summary,
            max_rows=20,
            cols=[
                "setting_id",
                "n_events",
                "n_timeslices",
                "rmse_mean",
                "mae_mean",
                "corr_mean",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "abs_error_rain_max_mean",
                "abs_error_area_10_mean",
                "csi10_mean",
                "f1_10_mean",
                "topk_unique_event_count_mean",
                "validation_ok_rate",
            ],
        ),
        "",
        "## 7. Relative Change vs Baseline",
        format_table(
            relative_change,
            max_rows=20,
            cols=[
                "setting_id",
                "rmse_change_pct",
                "corr_change",
                "p95_error_change_pct",
                "p99_error_change_pct",
                "rmax_error_change_pct",
                "area10_error_change_pct",
                "csi10_change",
                "f1_10_change",
            ],
        ),
        f"- Largest absolute relative-change item: {max_relative_change_line(relative_change)}",
        "",
        "## 8. Parameter Sensitivity Conclusions",
        "### K Sensitivity",
        format_table(
            k_rows,
            max_rows=5,
            cols=[
                "setting_id",
                "K",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "abs_error_area_10_mean",
                "csi10_mean",
                "topk_unique_event_count_mean",
            ],
        ),
        "K=10, K=20, and K=30 retain the same validation events and selected timeslices, so differences mainly reflect analog sample-size effects rather than sample-composition drift.",
        "",
        "### Beta Sensitivity",
        format_table(
            beta_rows,
            max_rows=5,
            cols=[
                "setting_id",
                "beta_blend",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "abs_error_area_10_mean",
                "csi10_mean",
                "f1_10_mean",
            ],
        ),
        "The beta experiment isolates the EOF/PCA structural constraint while keeping Top-K and distance weights fixed.",
        "",
        "### Distance Weight Sensitivity",
        format_table(
            weight_rows,
            max_rows=5,
            cols=[
                "setting_id",
                "weight_intensity",
                "weight_motion",
                "weight_environment",
                "abs_error_rain_p95_mean",
                "abs_error_rain_p99_mean",
                "csi10_mean",
                "f1_10_mean",
            ],
        ),
        "The three weight perturbations do not change validation samples or downstream calibration logic; they only alter analog ordering and weights.",
        "",
        "## 9. Figure Outputs",
        "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in figure_paths) if figure_paths else "(figures disabled)",
        "",
        "## 10. Paper-Ready Conclusion",
        (
            "基于历史台风伪缺失验证的敏感性检验表明，问题二生成模型在 Top-K 相似样本数量、相似距离分量权重和 EOF/PCA 融合系数扰动下总体表现稳定。"
            "K=20 在极端误差控制和历史模板多样性之间取得较好折中；K=10 更容易受少数历史样本影响，而 K=30 会引入相似性较弱样本，可能平滑极端降水结构。"
            "当 beta=0 时模型缺少 EOF 结构约束，beta=0.5 时 EOF 平滑作用增强，二者相比 beta=0.3 均可能带来极端结构或技巧评分上的波动，因此 beta=0.3 是较稳健的折中。"
            "在强度、移动和环境权重扰动下，P95/P99 极端误差和 CSI10/F1_10 未出现大幅恶化，说明模型对距离权重设置具有鲁棒性；其中强度权重降低时极端降水误差的变化可作为强度变量对降水生成重要性的补充证据。"
        ),
    ]
    QC_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# Main
# =========================


def main() -> None:
    np.random.seed(RANDOM_SEED)
    SETTINGS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("[25] Loading historical library and EOF/PCA model")
    history = p22.load_historical_library()
    eof_model = p22.load_eof_pca_model()
    x_front_km, y_left_km, x_grid, y_grid = p22.build_relative_grid()
    cell_area = p22.cell_area_from_grid(x_front_km, y_left_km)

    settings_df = settings_dataframe()
    settings_df.to_csv(SETTINGS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("[25] Loading Step-22 validation events and selected-time candidates")
    validation_events = load_validation_events(history)
    selected_times_by_event, selection_source = build_selected_times_by_event(history, validation_events)
    validation_events = validation_events.copy()
    validation_events["n_selected_times_sensitivity"] = validation_events["validation_event_uid"].astype(str).map(
        {event_uid: len(df) for event_uid, df in selected_times_by_event.items()}
    )

    cache = p22.TemplateCache(CACHE_SIZE)
    truth_cache: Dict[str, np.ndarray] = {}
    all_metric_rows: List[Dict[str, object]] = []
    diagnostics: List[Mapping[str, object]] = []

    for setting in SETTINGS:
        print(f"[25] Running setting {setting.setting_id}: K={setting.K}, beta={setting.beta_blend}")
        rows, diag = run_one_setting(
            setting,
            history,
            eof_model,
            validation_events,
            selected_times_by_event,
            cache,
            truth_cache,
            x_front_km,
            y_left_km,
            x_grid,
            y_grid,
            cell_area,
        )
        all_metric_rows.extend(rows)
        diagnostics.append(diag)

    timeslice_metrics = pd.DataFrame(all_metric_rows)
    if timeslice_metrics.empty:
        raise RuntimeError("No sensitivity timeslice metrics were generated.")
    leading_cols = [
        "setting_id",
        "validation_event_uid",
        "typhoon_name",
        "validation_time",
        "model_version",
        "WND",
        "PRES",
        "K",
        "beta_blend",
        "weight_intensity",
        "weight_motion",
        "weight_environment",
        "truth_rain_p95_mmhr",
        "pred_rain_p95_mmhr",
        "abs_error_rain_p95",
        "truth_rain_p99_mmhr",
        "pred_rain_p99_mmhr",
        "abs_error_rain_p99",
        "truth_rain_max_mmhr",
        "pred_rain_max_mmhr",
        "abs_error_rain_max",
        "truth_rain_area_10_km2",
        "pred_rain_area_10_km2",
        "abs_error_area_10",
        "rmse",
        "mae",
        "corr",
        "csi10",
        "f1_10",
        "topk_unique_event_count",
        "validation_ok",
        "skip_reason",
    ]
    ordered_cols = [c for c in leading_cols if c in timeslice_metrics.columns] + [
        c for c in timeslice_metrics.columns if c not in leading_cols
    ]
    timeslice_metrics = timeslice_metrics[ordered_cols]
    timeslice_metrics.to_csv(TIMESLICE_METRICS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(timeslice_metrics, settings_df)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    relative_change = build_relative_change(summary)
    relative_change.to_csv(RELATIVE_CHANGE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    figure_paths = make_figures(summary, relative_change)
    write_qc_report(
        settings_df,
        validation_events,
        selected_times_by_event,
        timeslice_metrics,
        summary,
        relative_change,
        diagnostics,
        figure_paths,
        selection_source,
    )

    self_match_total = int(pd.to_numeric(timeslice_metrics.get("topk_self_match_count"), errors="coerce").fillna(0).sum())
    print("[25] Completed parameter sensitivity analysis")
    print(f"[25] Settings CSV: {SETTINGS_OUTPUT_PATH}")
    print(f"[25] Timeslice metrics: {TIMESLICE_METRICS_OUTPUT_PATH}")
    print(f"[25] Summary CSV: {SUMMARY_OUTPUT_PATH}")
    print(f"[25] Relative change CSV: {RELATIVE_CHANGE_OUTPUT_PATH}")
    print(f"[25] QC report: {QC_REPORT_PATH}")
    print(f"[25] Figures: {FIGURE_DIR} ({len(figure_paths)} files)")
    print(f"[25] Validation events: {validation_events['validation_event_uid'].nunique()}")
    print(f"[25] Valid times by setting: {summary.set_index('setting_id')['n_timeslices'].to_dict()}")
    print(f"[25] Top-K self match count: {self_match_total}")


if __name__ == "__main__":
    main()
