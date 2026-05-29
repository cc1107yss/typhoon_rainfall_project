#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rainband-width diagnostics for final Problem-2 env generated fields."""

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
NPZ_PATH = PROJECT_ROOT / "data/processed/problem2_env/problem2_generated_calibrated_fields_env.npz"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/problem2_env"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures/problem2_env"
REPORT_DIR = PROJECT_ROOT / "outputs/reports/problem2_env"

TIMESERIES_PATH = TABLE_DIR / "problem2_rainband_width_timeseries.csv"
SUMMARY_PATH = TABLE_DIR / "problem2_rainband_width_summary.csv"
FIGURE_PATH = FIGURE_DIR / "problem2_rainband_width_timeseries.png"
REPORT_PATH = REPORT_DIR / "problem2_rainband_width_diagnostics_qc.md"


def load_inputs() -> Dict[str, np.ndarray]:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing Problem-2 env calibrated NPZ: {NPZ_PATH}")
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        required = ["rain_mmhr_calibrated", "typhoon_name", "time", "x_front_km", "y_left_km"]
        missing = [key for key in required if key not in z.files]
        if missing:
            raise RuntimeError(f"Problem-2 NPZ missing arrays: {missing}")
        return {key: z[key] for key in required}


def build_timeseries(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    rain = np.asarray(data["rain_mmhr_calibrated"], dtype=np.float32)
    x_front = np.asarray(data["x_front_km"], dtype=np.float64)
    y_left = np.asarray(data["y_left_km"], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_front, y_left)

    rows: List[Dict[str, object]] = []
    for i in range(rain.shape[0]):
        metrics = compute_dual_rainband_width_metrics(rain[i], x_grid, y_grid)
        rows.append(
            {
                "field_index": i,
                "typhoon_name": str(np.asarray(data["typhoon_name"])[i]),
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
    rows: List[Dict[str, object]] = []
    for name, sub in ts.groupby("typhoon_name", sort=False):
        width = pd.to_numeric(sub["rainband_width_km"], errors="coerce")
        width10 = pd.to_numeric(sub["rainband_width10_km"], errors="coerce")
        rows.append(
            {
                "typhoon_name": name,
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


def make_figure(ts: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    names = list(ts["typhoon_name"].dropna().astype(str).unique())
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.2 * max(len(names), 1)), dpi=180, sharex=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        sub = ts.loc[ts["typhoon_name"].astype(str).eq(name)].sort_values("time")
        ax.plot(sub["time"], sub["rainband_width_km"], label="B >=1 mm/hr", color="#2F6F9F", linewidth=1.5)
        ax.plot(sub["time"], sub["rainband_width10_km"], label="B10 >=10 mm/hr", color="#C75146", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{name} generated rainband width")
        ax.set_ylabel("Width (km)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncols=2)
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
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
    field_quality = {
        "rows": int(len(ts)),
        "main_width_nan": int(pd.to_numeric(ts["rainband_width_km"], errors="coerce").isna().sum()),
        "width10_nan": int(pd.to_numeric(ts["rainband_width10_km"], errors="coerce").isna().sum()),
        "main_valid_min": int(pd.to_numeric(ts["rainband_valid_grid_count"], errors="coerce").min(skipna=True)),
        "heavy_valid_min": int(pd.to_numeric(ts["rainband10_valid_grid_count"], errors="coerce").min(skipna=True)),
    }
    lines = [
        "# Problem-2 Rainband Width Diagnostics",
        "",
        f"- Input NPZ: `{NPZ_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Timeseries CSV: `{TIMESERIES_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Summary CSV: `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Figure: `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        "- The metrics are diagnostics computed after generation; they are not target-typhoon input variables.",
        f"- Field-quality summary: `{field_quality}`",
        "",
        simple_markdown_table(summary),
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    ts = build_timeseries(data)
    summary = build_summary(ts)
    ts.to_csv(TIMESERIES_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    make_figure(ts)
    write_report(ts, summary)
    print(f"[33] Timeseries: {TIMESERIES_PATH} {ts.shape}")
    print(f"[33] Summary: {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    print(f"[33] Figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
