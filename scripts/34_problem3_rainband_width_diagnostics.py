#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rainband-width diagnostics for Problem-3 S0-S5 calibrated scenario fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rainband_width_utils import compute_dual_rainband_width_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = PROJECT_ROOT / "data/processed/problem3/problem3_scenario_rainfields_calibrated.npz"
EOF_MODEL_PATH = PROJECT_ROOT / "data/processed/problem2_env/problem2_eof_pca_model_env.npz"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/problem3"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem3"
REPORT_DIR = PROJECT_ROOT / "outputs/reports/problem3"

TIMESERIES_PATH = TABLE_DIR / "problem3_rainband_width_timeseries.csv"
SUMMARY_PATH = TABLE_DIR / "problem3_rainband_width_summary.csv"
FINAL_COMPARISON_PATH = TABLE_DIR / "problem3_final_scenario_comparison_table.csv"
FINAL_COMPARISON_ENHANCED_PATH = TABLE_DIR / "problem3_final_scenario_comparison_table_with_rainband_width.csv"
FIGURE_PATH = FIGURE_DIR / "problem3_rainband_width_bars.png"
REPORT_PATH = REPORT_DIR / "problem3_rainband_width_diagnostics_qc.md"


def load_npz() -> Dict[str, np.ndarray]:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing Problem-3 calibrated NPZ: {NPZ_PATH}")
    if not EOF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing EOF model grid axes: {EOF_MODEL_PATH}")
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        required = ["rainfields_calibrated_mmhr", "scenario_id", "scenario_name", "time"]
        missing = [key for key in required if key not in z.files]
        if missing:
            raise RuntimeError(f"Problem-3 NPZ missing arrays: {missing}")
        data = {key: z[key] for key in required}
    with np.load(EOF_MODEL_PATH, allow_pickle=True) as z:
        data["x_front_km"] = z["x_front_km"]
        data["y_left_km"] = z["y_left_km"]
    return data


def build_timeseries(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    rain = np.asarray(data["rainfields_calibrated_mmhr"], dtype=np.float32)
    x_front = np.asarray(data["x_front_km"], dtype=np.float64)
    y_left = np.asarray(data["y_left_km"], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_front, y_left)

    rows: List[Dict[str, object]] = []
    for i in range(rain.shape[0]):
        metrics = compute_dual_rainband_width_metrics(rain[i], x_grid, y_grid)
        rows.append(
            {
                "field_index": i,
                "scenario_id": str(np.asarray(data["scenario_id"])[i]),
                "scenario_name": str(np.asarray(data["scenario_name"])[i]),
                "time": str(np.asarray(data["time"])[i]),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out


def time_of_max(sub: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(sub[col], errors="coerce")
    if s.notna().sum() == 0:
        return ""
    return pd.Timestamp(sub.loc[s.idxmax(), "time"]).strftime("%Y-%m-%d %H:%M:%S")


def build_summary(ts: pd.DataFrame) -> pd.DataFrame:
    order = ["S0", "S1", "S2", "S3", "S4", "S5"]
    rows: List[Dict[str, object]] = []
    for sid in order:
        sub = ts.loc[ts["scenario_id"].astype(str).eq(sid)].copy()
        if sub.empty:
            continue
        width = pd.to_numeric(sub["rainband_width_km"], errors="coerce")
        width10 = pd.to_numeric(sub["rainband_width10_km"], errors="coerce")
        rows.append(
            {
                "scenario_id": sid,
                "scenario_name": str(sub["scenario_name"].iloc[0]),
                "n_times": int(len(sub)),
                "mean_rainband_width_km": float(width.mean(skipna=True)),
                "median_rainband_width_km": float(width.median(skipna=True)),
                "max_rainband_width_km": float(width.max(skipna=True)),
                "mean_rainband_width10_km": float(width10.mean(skipna=True)),
                "max_rainband_width10_km": float(width10.max(skipna=True)),
                "time_of_max_width": time_of_max(sub, "rainband_width_km"),
                "time_of_max_width10": time_of_max(sub, "rainband_width10_km"),
            }
        )
    return pd.DataFrame(rows)


def update_final_comparison(summary: pd.DataFrame) -> None:
    if not FINAL_COMPARISON_PATH.exists():
        return
    table = pd.read_csv(FINAL_COMPARISON_PATH, encoding="utf-8-sig")
    if "scenario_id" not in table.columns:
        return
    add_cols = [
        "scenario_id",
        "mean_rainband_width_km",
        "max_rainband_width_km",
        "mean_rainband_width10_km",
        "max_rainband_width10_km",
    ]
    merged = table.drop(columns=[c for c in add_cols if c in table.columns and c != "scenario_id"], errors="ignore").merge(
        summary[add_cols],
        on="scenario_id",
        how="left",
    )
    merged.to_csv(FINAL_COMPARISON_PATH, index=False, encoding="utf-8-sig")
    merged.to_csv(FINAL_COMPARISON_ENHANCED_PATH, index=False, encoding="utf-8-sig")


def make_figure(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ids = summary["scenario_id"].astype(str).tolist()
    x = np.arange(len(ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.bar(x - width / 2, summary["mean_rainband_width_km"], width=width, label="Mean B", color="#4477AA")
    ax.bar(x + width / 2, summary["max_rainband_width_km"], width=width, label="Max B", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("Rainband width (km)")
    ax.set_title("Problem-3 scenario rainband width diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}" if isinstance(x, (float, np.floating)) else str(x))
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_report(ts: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    quality = {
        "rows": int(len(ts)),
        "main_width_nan": int(pd.to_numeric(ts["rainband_width_km"], errors="coerce").isna().sum()),
        "width10_nan": int(pd.to_numeric(ts["rainband_width10_km"], errors="coerce").isna().sum()),
    }
    lines = [
        "# Problem-3 Rainband Width Diagnostics",
        "",
        f"- Input NPZ: `{NPZ_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Timeseries CSV: `{TIMESERIES_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Summary CSV: `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Figure: `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        "- The metrics diagnose generated S0-S5 rainfall fields and are not safe-input variables.",
        f"- Field-quality summary: `{quality}`",
        "",
        simple_markdown_table(summary),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    data = load_npz()
    ts = build_timeseries(data)
    summary = build_summary(ts)
    ts.to_csv(TIMESERIES_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    update_final_comparison(summary)
    make_figure(summary)
    write_report(ts, summary)
    print(f"[34] Timeseries: {TIMESERIES_PATH} {ts.shape}")
    print(f"[34] Summary: {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    print(f"[34] Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
