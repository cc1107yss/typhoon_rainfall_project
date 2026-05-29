#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_build_historical_library.py

问题二：建立“相似历史时次类比生成”的历史样本库索引。

输入：
    data/processed/gpm_track_model_features_interp.csv
    data/processed/target_typhoon_model_x_2024_halfhour.csv  （可选，但建议已有）
    GPM_3IMERGHHE.07 根目录（可选，用于解析 tif_path）

输出：
    data/processed/problem2_historical_library_index.csv
    data/processed/problem2_target_model_x_aligned.csv        （若目标表存在）
    outputs/tables/problem2/historical_library_report.json
    outputs/tables/problem2/problem2_x_feature_cols.csv
    outputs/tables/problem2/problem2_y_metric_cols.csv
    outputs/tables/problem2/problem2_robust_scaler_stats.csv

设计原则：
    1. KONG-REY / MAN-YI 目标输入只允许路径、强度、移动、海陆、近岸、时间周期等变量。
    2. rain_*、centroid_*、anisotropy、asym_*、quad_*、r50/r80/r90 等降水派生列一律不得进入 X。
    3. 历史降水派生列只作为 Y 指标，用于校准、检验和解释。
    4. source_file 仅为 tif 文件名；真实读取雨场时需与 GPM 根目录 / gpm_event_uid 拼接。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# 1. 字段规则
# =========================

BANNED_RAIN_PATTERNS = [
    r"^rain_",
    r"rain",
    r"^centroid_",
    r"rain_centroid",
    r"^asym_",
    r"^quad_",
    r"^r50_",
    r"^r80_",
    r"^r90_",
    r"anisotropy",
    r"orientation",
    r"axis",
    r"gini",
    r"entropy",
    r"halfhour",
    r"valid_count",
    r"valid_ratio",
]

META_COLS = {
    "time",
    "time_end",
    "source_file",
    "tif_path",
    "tif_path_exists",
    "gpm_event_uid",
    "event_uid",
    "track_event_uid",
    "target_event_uid",
    "target_storm_uid",
    "storm_uid",
    "track_source_file",
    "track_typhoon_id",
    "track_storm_seq",
    "track_typhoon_code",
    "track_record_count",
    "track_typhoon_name",
    "typhoon_name",
    "storm_name",
    "name",
    "interp_center_error_km",
    "outside_track_time_range",
    "interp_match_status",
    "interp_center_error_ok",
    "center_lon",
    "center_lat",
    "bbox_lon_min",
    "bbox_lon_max",
    "bbox_lat_min",
    "bbox_lat_max",
    # 不建议把绝对年份和日期作为相似度特征；季节性用 month_sin/cos，日变化用 hour_sin/cos。
    "year",
    "day",
}

# 强推荐输入特征。若目标表存在，最终取“历史表 ∩ 目标表 ∩ 此列表/安全扩展”的列。
RECOMMENDED_X_ORDER = [
    # 中心位置
    "track_lat",
    "track_lon_180",

    # 强度
    "track_wind",
    "track_pressure",
    "track_intensity",
    "pressure_deficit",
    "intensity_index",
    "wind_z",
    "pressure_deficit_z",

    # 强度变化
    "track_wind_change_rate_interp",
    "track_pressure_change_rate_interp",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",

    # 移动
    "track_move_speed_kmh_interp",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "track_move_speed_kmh",
    "track_move_dir_deg_interp",
    "track_move_dir_deg",
    "track_dt_h_interp",
    "track_move_distance_km_interp",

    # 海陆和近岸
    "track_is_land_interp",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",

    # 时间周期
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]

# 允许补充的安全派生变量名模式，例如路径曲率、转向角等。
SAFE_EXTRA_X_PATTERNS = [
    r"turn",
    r"curv",
    r"curvature",
    r"bearing",
    r"motion",
    r"move",
    r"speed",
    r"dir_sin",
    r"dir_cos",
    r"land",
    r"coast",
    r"shore",
    r"month_sin",
    r"month_cos",
    r"hour_sin",
    r"hour_cos",
    r"wind",
    r"pres",
    r"pressure",
    r"intensity",
    r"lat$",
    r"lon",
]

Y_METRIC_ORDER = [
    "rain_mean",
    "rain_std",
    "rain_max",
    "rain_p50",
    "rain_p90",
    "rain_p95",
    "rain_p99",
    "rain_area_0p1_grid",
    "rain_area_1_grid",
    "rain_area_5_grid",
    "rain_area_10_grid",
    "rain_area_20_grid",
    "rain_area_50_grid",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "rain_area_20_km2",
    "rain_area_20_equiv_radius_km",
    "rain_centroid_lon",
    "rain_centroid_lat",
    "centroid_offset_km",
    "centroid_offset_dir_deg",
    "centroid_relative_to_motion_deg",
    "centroid_relative_to_motion_sin",
    "centroid_relative_to_motion_cos",
    "centroid_in_front",
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
    "r50_km",
    "r80_km",
    "r90_km",
    "asym_EW",
    "asym_NS",
    "quad_NE_ratio",
    "quad_SE_ratio",
    "quad_SW_ratio",
    "quad_NW_ratio",
    "major_axis_km",
    "minor_axis_km",
    "anisotropy",
    "orientation_deg",
    "rain_gini",
    "rain_entropy_norm",
]


def is_banned_rain_col(col: str) -> bool:
    c = str(col).strip().lower()
    return any(re.search(p, c) for p in BANNED_RAIN_PATTERNS)


def normalize_name(s: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())


def find_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_safely(path: Path, parse_time: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if parse_time:
        for c in ["time", "time_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def is_safe_extra_x(col: str) -> bool:
    c = col.lower()
    if col in META_COLS:
        return False
    if is_banned_rain_col(col):
        return False
    if c in {"month", "hour"}:
        # 建议不用原始 month/hour，避免与周期形式重复；确需使用可在 18 号脚本的权重配置中加入。
        return False
    if c in {"year", "day"}:
        return False
    return any(re.search(p, c) for p in SAFE_EXTRA_X_PATTERNS)


def select_x_columns(history: pd.DataFrame, target: Optional[pd.DataFrame]) -> Tuple[List[str], Dict]:
    """选择问题二相似度输入列。"""
    report: Dict = {}

    hist_numeric = {c for c in history.columns if pd.api.types.is_numeric_dtype(history[c])}

    if target is not None:
        target_numeric = {c for c in target.columns if pd.api.types.is_numeric_dtype(target[c])}
        common_numeric = hist_numeric & target_numeric
    else:
        target_numeric = set()
        common_numeric = hist_numeric

    # 第一优先级：推荐列表
    x_cols: List[str] = []
    for c in RECOMMENDED_X_ORDER:
        if c in common_numeric and not is_banned_rain_col(c) and c not in META_COLS:
            x_cols.append(c)

    # 第二优先级：目标表中出现、历史表也有、且命名安全的额外列
    if target is not None:
        for c in target.columns:
            if c in common_numeric and c not in x_cols and is_safe_extra_x(c):
                x_cols.append(c)
    else:
        for c in history.columns:
            if c in common_numeric and c not in x_cols and is_safe_extra_x(c):
                # 没有目标表时，只作为兜底；不主动加入过多列。
                if c in RECOMMENDED_X_ORDER:
                    x_cols.append(c)

    banned_leak = [c for c in x_cols if is_banned_rain_col(c)]
    if banned_leak:
        raise ValueError(f"X 列中发现降水派生泄漏字段：{banned_leak}")

    report["history_numeric_cols_n"] = len(hist_numeric)
    report["target_numeric_cols_n"] = len(target_numeric) if target is not None else None
    report["x_cols_selected_n"] = len(x_cols)
    report["x_cols_selected"] = x_cols
    report["recommended_missing_in_history"] = [c for c in RECOMMENDED_X_ORDER if c not in history.columns]
    if target is not None:
        report["recommended_missing_in_target"] = [c for c in RECOMMENDED_X_ORDER if c not in target.columns]
        report["target_numeric_not_used"] = sorted([
            c for c in target_numeric
            if c not in x_cols and c not in META_COLS and not is_banned_rain_col(c)
        ])

    return x_cols, report


def resolve_tif_path(row: pd.Series, gpm_root: Optional[Path]) -> Tuple[str, bool]:
    """根据 gpm_event_uid 和 source_file 拼接预期 tif 路径。"""
    source = str(row.get("source_file", ""))
    event = str(row.get("gpm_event_uid", row.get("event_uid", "")))

    if not source or source == "nan":
        return "", False

    if gpm_root is None:
        # 仍给出相对预期路径，便于后续在本地替换根目录。
        return str(Path(event) / source) if event else source, False

    candidates = []
    if event:
        candidates.append(gpm_root / event / source)
    candidates.append(gpm_root / source)

    for p in candidates:
        if p.exists():
            return str(p), True

    # 文件可能尚未解压或根目录未正确传入；保存最可能路径。
    return str(candidates[0]), False


def robust_scaler_stats(df: pd.DataFrame, x_cols: List[str]) -> pd.DataFrame:
    records = []
    for c in x_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        med = float(s.median(skipna=True))
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        std = float(s.std(skipna=True))
        # 防止常数列或近似常数列导致除零。
        scale = iqr if np.isfinite(iqr) and iqr > 1e-9 else std
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        records.append({
            "feature": c,
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "std": std,
            "scale": float(scale),
            "missing_rate": float(s.isna().mean()),
            "min": float(s.min(skipna=True)),
            "max": float(s.max(skipna=True)),
        })
    return pd.DataFrame(records)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build problem-2 historical analog library index.")
    parser.add_argument("--model-file", type=str, default=None,
                        help="默认优先寻找 data/processed/gpm_track_model_features_motion.csv，其次 interp 表")
    parser.add_argument("--target-x-file", type=str, default=None,
                        help="默认自动寻找 data/processed/target_typhoon_model_x_2024_halfhour.csv")
    parser.add_argument("--gpm-root", type=str, default=None,
                        help="GPM_3IMERGHHE.07 根目录；若不传，只保存相对预期路径")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--table-out-dir", type=str, default="outputs/tables/problem2")
    parser.add_argument("--allow-target-in-history", action="store_true",
                        help="默认不允许历史样本库含 KONG-REY / MAN-YI；调试时才打开")
    parser.add_argument("--validate-tif-paths", action="store_true",
                        help="检查拼接 tif_path 是否真实存在；大量文件时会略慢")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    cwd = Path.cwd()
    processed_dir = Path(args.processed_dir)
    table_out_dir = Path(args.table_out_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_out_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(args.model_file) if args.model_file else find_existing_path([
        processed_dir / "gpm_track_model_features_motion.csv",
        processed_dir / "gpm_track_model_features_interp.csv",
        cwd / "gpm_track_model_features_motion.csv",
        cwd / "gpm_track_model_features_interp.csv",
        Path("/mnt/data/gpm_track_model_features_motion.csv"),
        Path("/mnt/data/gpm_track_model_features_interp.csv"),
    ])
    if model_file is None:
        raise FileNotFoundError("未找到 gpm_track_model_features_motion.csv 或 gpm_track_model_features_interp.csv。")

    target_x_file = Path(args.target_x_file) if args.target_x_file else find_existing_path([
        processed_dir / "target_typhoon_model_x_2024_halfhour.csv",
        cwd / "target_typhoon_model_x_2024_halfhour.csv",
        Path("/mnt/data/target_typhoon_model_x_2024_halfhour.csv"),
    ])

    gpm_root = Path(args.gpm_root) if args.gpm_root else find_existing_path([
        cwd / "GPM_3IMERGHHE.07",
        cwd / "data" / "raw" / "GPM_3IMERGHHE.07",
        cwd / "data" / "GPM_3IMERGHHE.07",
        Path("/mnt/data/GPM_3IMERGHHE.07"),
    ])

    print(f"[17] 读取历史增强表：{model_file}")
    hist = read_csv_safely(model_file)

    # 基础质量筛选
    if "interp_match_status" in hist.columns:
        before = len(hist)
        hist = hist[hist["interp_match_status"].astype(str).str.lower().eq("ok")].copy()
        print(f"[17] interp_match_status == ok: {before} -> {len(hist)}")

    if "interp_center_error_ok" in hist.columns:
        before = len(hist)
        hist = hist[hist["interp_center_error_ok"].fillna(1).astype(int).eq(1)].copy()
        print(f"[17] interp_center_error_ok == 1: {before} -> {len(hist)}")

    # 目标台风不得出现在历史降水库中。
    target_like = pd.DataFrame()
    if "track_typhoon_name" in hist.columns:
        name_norm = hist["track_typhoon_name"].map(normalize_name)
        target_like = hist.loc[
            name_norm.str.contains("KONGREY|MANYI", na=False),
            ["time", "track_typhoon_name", "track_event_uid", "gpm_event_uid", "source_file"],
        ].drop_duplicates()

    if len(target_like) > 0 and not args.allow_target_in_history:
        raise RuntimeError(
            "历史样本库中发现 KONG-REY / MAN-YI 名称，疑似目标降水泄漏。"
            f"样例：{target_like.head(10).to_dict(orient='records')}"
        )

    target = None
    target_report: Dict = {
        "target_x_file": str(target_x_file) if target_x_file else None,
        "exists": bool(target_x_file and target_x_file.exists()),
    }
    if target_x_file and target_x_file.exists():
        print(f"[17] 读取目标安全输入表：{target_x_file}")
        target = read_csv_safely(target_x_file)

        target_banned = [c for c in target.columns if is_banned_rain_col(c)]
        if target_banned:
            raise ValueError(f"目标输入表出现禁止降水派生列：{target_banned}")

        target_report.update({
            "shape": [int(target.shape[0]), int(target.shape[1])],
            "columns": list(target.columns),
            "banned_columns_n": len(target_banned),
        })

        # 目标名称/事件/时间摘要
        for name_col in ["track_typhoon_name", "typhoon_name", "storm_name", "name"]:
            if name_col in target.columns:
                target_report["name_col"] = name_col
                target_report["name_values"] = sorted(target[name_col].dropna().astype(str).unique().tolist())
                break
        for uid_col in ["target_event_uid", "track_event_uid", "storm_uid"]:
            if uid_col in target.columns:
                target_report["uid_col"] = uid_col
                target_report["uid_values"] = sorted(target[uid_col].dropna().astype(str).unique().tolist())
                break
        if "time" in target.columns:
            target_report["time_min"] = str(target["time"].min())
            target_report["time_max"] = str(target["time"].max())

    x_cols, x_select_report = select_x_columns(hist, target)
    if not x_cols:
        raise RuntimeError("未能选出任何安全输入列，请检查目标表和历史表列名是否一致。")

    y_cols = [c for c in Y_METRIC_ORDER if c in hist.columns]
    if not {"rain_p95", "rain_p99", "rain_area_10_km2", "centroid_offset_km", "anisotropy"}.issubset(set(y_cols)):
        missing = sorted({"rain_p95", "rain_p99", "rain_area_10_km2", "centroid_offset_km", "anisotropy"} - set(y_cols))
        raise RuntimeError(f"历史输出关键指标缺失：{missing}")

    meta_cols = [c for c in [
        "time", "time_end",
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
    ] if c in hist.columns]

    # tif_path：如未传 GPM 根目录，也保存相对路径。
    if args.validate_tif_paths or gpm_root is not None:
        print(f"[17] 拼接 tif_path，GPM root = {gpm_root}")
    resolved = hist.apply(lambda r: resolve_tif_path(r, gpm_root), axis=1)
    hist["tif_path"] = [x[0] for x in resolved]
    hist["tif_path_exists"] = [bool(x[1]) for x in resolved]

    if args.validate_tif_paths and gpm_root is not None:
        exists_n = int(hist["tif_path_exists"].sum())
        print(f"[17] tif_path exists: {exists_n}/{len(hist)}")

    # 缺失值报告和 robust scaler
    scaler_df = robust_scaler_stats(hist, x_cols)
    severe_missing = scaler_df.loc[scaler_df["missing_rate"] > 0.05, ["feature", "missing_rate"]]

    # 输出历史样本库索引：meta + X + Y + tif_path
    library_cols = []
    for c in meta_cols + ["tif_path", "tif_path_exists"] + x_cols + y_cols:
        if c in hist.columns and c not in library_cols:
            library_cols.append(c)

    library = hist[library_cols].copy()

    # 不在17号脚本中填补 X 缺失，避免掩盖数据问题；仅输出 scaler median 供 18 号脚本统一填补。
    library_path = processed_dir / "problem2_historical_library_index.csv"
    library.to_csv(library_path, index=False, encoding="utf-8-sig")

    x_cols_path = table_out_dir / "problem2_x_feature_cols.csv"
    y_cols_path = table_out_dir / "problem2_y_metric_cols.csv"
    scaler_path = table_out_dir / "problem2_robust_scaler_stats.csv"

    pd.DataFrame({"x_feature": x_cols}).to_csv(x_cols_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_metric": y_cols}).to_csv(y_cols_path, index=False, encoding="utf-8-sig")
    scaler_df.to_csv(scaler_path, index=False, encoding="utf-8-sig")

    target_aligned_path = None
    if target is not None:
        # 对齐目标表列：保留所有索引列 + X列；不做降水列。
        target_meta_cols = [c for c in target.columns if c in META_COLS or re.search(r"(uid|name|time|seq|storm)", c, re.I)]
        target_aligned_cols = []
        for c in target_meta_cols + x_cols:
            if c in target.columns and c not in target_aligned_cols:
                target_aligned_cols.append(c)

        missing_x_in_target = [c for c in x_cols if c not in target.columns]
        if missing_x_in_target:
            raise RuntimeError(f"选定 X 列在目标表中缺失：{missing_x_in_target}")

        target_aligned = target[target_aligned_cols].copy()
        target_aligned_path = processed_dir / "problem2_target_model_x_aligned.csv"
        target_aligned.to_csv(target_aligned_path, index=False, encoding="utf-8-sig")
        target_report["aligned_path"] = str(target_aligned_path)
        target_report["aligned_shape"] = [int(target_aligned.shape[0]), int(target_aligned.shape[1])]

    report = {
        "script": "17_build_historical_library.py",
        "model_file": str(model_file),
        "target_x_file": str(target_x_file) if target_x_file else None,
        "gpm_root": str(gpm_root) if gpm_root else None,
        "history_shape_after_filter": [int(hist.shape[0]), int(hist.shape[1])],
        "history_time_min": str(hist["time"].min()) if "time" in hist.columns else None,
        "history_time_max": str(hist["time"].max()) if "time" in hist.columns else None,
        "history_gpm_event_n": int(hist["gpm_event_uid"].nunique()) if "gpm_event_uid" in hist.columns else None,
        "history_track_event_n": int(hist["track_event_uid"].nunique()) if "track_event_uid" in hist.columns else None,
        "target_like_in_history_n": int(len(target_like)),
        "target_like_in_history_sample": target_like.head(20).astype(str).to_dict(orient="records"),
        "x_select_report": x_select_report,
        "y_metric_cols_n": len(y_cols),
        "y_metric_cols": y_cols,
        "x_missing_rate_over_5pct": severe_missing.to_dict(orient="records"),
        "tif_path_exists_n": int(hist["tif_path_exists"].sum()),
        "tif_path_exists_rate": float(hist["tif_path_exists"].mean()),
        "outputs": {
            "historical_library_index": str(library_path),
            "target_model_x_aligned": str(target_aligned_path) if target_aligned_path else None,
            "x_feature_cols": str(x_cols_path),
            "y_metric_cols": str(y_cols_path),
            "robust_scaler_stats": str(scaler_path),
        },
        "target_report": target_report,
        "hard_rules": {
            "banned_rain_cols_in_x": [c for c in x_cols if is_banned_rain_col(c)],
            "banned_rain_cols_in_target": [] if target is None else [c for c in target.columns if is_banned_rain_col(c)],
            "owd_like_cols_in_x": [c for c in x_cols if "owd" in c.lower()],
            "passed": (
                len([c for c in x_cols if is_banned_rain_col(c)]) == 0
                and (target is None or len([c for c in target.columns if is_banned_rain_col(c)]) == 0)
                and len([c for c in x_cols if "owd" in c.lower()]) == 0
            ),
        },
    }

    report_path = table_out_dir / "historical_library_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== 17 历史样本库构建完成 ==========")
    print(f"历史库索引：{library_path}")
    print(f"历史库维度：{library.shape[0]} × {library.shape[1]}")
    print(f"X 特征数：{len(x_cols)}")
    print(f"Y 指标数：{len(y_cols)}")
    print(f"目标对齐表：{target_aligned_path if target_aligned_path else '未生成：未找到目标 X 表'}")
    print(f"tif_path 存在率：{report['tif_path_exists_rate']:.2%}")
    print(f"硬规则 passed：{report['hard_rules']['passed']}")
    print(f"报告：{report_path}")

    if len(severe_missing) > 0:
        print("\n[注意] 以下 X 特征缺失率 > 5%，18号相似检索前应决定是否剔除或填补：")
        print(severe_missing.to_string(index=False))

    if target is None:
        print("\n[注意] 未找到 target_typhoon_model_x_2024_halfhour.csv。")
        print("你本地已有该文件时，请在项目根目录运行本脚本，或用 --target-x-file 显式指定。")


if __name__ == "__main__":
    main()
