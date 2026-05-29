#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot historical typhoon tracks and GPM half-hour sample density."""

from __future__ import annotations

import zipfile
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_PATH = PROJECT_ROOT / "output/typhoon_track_features_with_coast.csv"
GPM_PATH = PROJECT_ROOT / "data/processed/gpm_track_model_features_interp.csv"
LAND_ZIP_PATH = PROJECT_ROOT / "ne_50m_land.zip"

FIG_DIR = PROJECT_ROOT / "outputs/figures/data_coverage"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/data_coverage"
FIG_PATH = FIG_DIR / "historical_tracks_gpm_sample_density.png"
SUMMARY_PATH = TABLE_DIR / "historical_tracks_gpm_sample_density_summary.csv"

LON_MIN, LON_MAX = 95.0, 180.0
LAT_MIN, LAT_MAX = 0.0, 65.0


def setup_font() -> None:
    candidates = [
        "Songti SC",
        "STSong",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
    ]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            break
    mpl.rcParams["axes.unicode_minus"] = False


def read_land_shapes(zip_path: Path) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        shp = next(tmp.glob("*.shp"))
        reader = shapefile.Reader(str(shp))
        for shp_rec in reader.shapes():
            pts = np.asarray(shp_rec.points, dtype=float)
            if pts.size == 0:
                continue
            parts = list(shp_rec.parts) + [len(pts)]
            for start, end in zip(parts[:-1], parts[1:]):
                ring = pts[start:end]
                if len(ring) < 3:
                    continue
                minx, miny = ring.min(axis=0)
                maxx, maxy = ring.max(axis=0)
                if maxx < LON_MIN or minx > LON_MAX or maxy < LAT_MIN or miny > LAT_MAX:
                    continue
                shapes.append(ring)
    return shapes


def make_track_segments(tracks: pd.DataFrame) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for _, group in tracks.sort_values("time").groupby("event_uid", sort=False):
        xy = group[["lon_180", "lat"]].to_numpy(dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 2:
            continue
        current = [xy[0]]
        for prev, cur in zip(xy[:-1], xy[1:]):
            if abs(cur[0] - prev[0]) > 25:
                if len(current) >= 2:
                    segments.append(np.asarray(current))
                current = [cur]
            else:
                current.append(cur)
        if len(current) >= 2:
            segments.append(np.asarray(current))
    return segments


def main() -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(TRACK_PATH, parse_dates=["time"])
    gpm = pd.read_csv(GPM_PATH, parse_dates=["time"])

    matched_event_ids = set(gpm["track_event_uid"].dropna().astype(str))
    matched_tracks = tracks[tracks["event_uid"].astype(str).isin(matched_event_ids)].copy()
    matched_tracks = matched_tracks[
        matched_tracks["lon_180"].between(LON_MIN, LON_MAX)
        & matched_tracks["lat"].between(LAT_MIN, LAT_MAX)
    ]

    gpm_plot = gpm[
        gpm["center_lon"].between(LON_MIN, LON_MAX)
        & gpm["center_lat"].between(LAT_MIN, LAT_MAX)
    ].copy()

    summary = {
        "track_event_count_matched_gpm": int(gpm["track_event_uid"].nunique()),
        "gpm_event_count": int(gpm["gpm_event_uid"].nunique()),
        "gpm_halfhour_sample_count": int(len(gpm)),
        "track_record_count_for_matched_events": int(len(matched_tracks)),
        "start_year": int(gpm["time"].dt.year.min()),
        "end_year": int(gpm["time"].dt.year.max()),
        "lon_min": float(gpm["center_lon"].min()),
        "lon_max": float(gpm["center_lon"].max()),
        "lat_min": float(gpm["center_lat"].min()),
        "lat_max": float(gpm["center_lat"].max()),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.2, 6.3), constrained_layout=True)
    ax.set_facecolor("#eef5f7")

    for ring in read_land_shapes(LAND_ZIP_PATH):
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor="#f4f0e6",
                edgecolor="#b8b1a4",
                linewidth=0.25,
                zorder=0,
            )
        )

    segments = make_track_segments(matched_tracks)
    ax.add_collection(
        LineCollection(
            segments,
            colors="#5f6670",
            linewidths=0.55,
            alpha=0.38,
            zorder=1,
        )
    )

    hb = ax.hexbin(
        gpm_plot["center_lon"],
        gpm_plot["center_lat"],
        gridsize=48,
        extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        mincnt=1,
        linewidths=0.0,
        cmap="YlOrRd",
        norm=LogNorm(),
        alpha=0.86,
        zorder=3,
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("GPM 半小时样本数/空间网格", fontsize=10)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_xticks(np.arange(100, 181, 10))
    ax.set_yticks(np.arange(0, 66, 10))
    ax.grid(True, color="white", linewidth=0.7, alpha=0.9, zorder=0)

    ax.set_title("历史台风轨迹与 GPM 降水样本空间覆盖", fontsize=14, weight="bold", pad=10)
    ax.plot([], [], color="#5f6670", lw=1.1, alpha=0.7, label="与 GPM 匹配的历史台风轨迹")
    ax.scatter([], [], c="#d94801", s=35, label="GPM 半小时样本点密度")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=9)

    stat_text = (
        f"GPM 事件数：{summary['gpm_event_count']} 场\n"
        f"半小时样本：{summary['gpm_halfhour_sample_count']:,} 个\n"
        f"匹配路径事件：{summary['track_event_count_matched_gpm']} 场\n"
        f"年份：{summary['start_year']}--{summary['end_year']}"
    )
    ax.text(
        0.985,
        0.965,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999999", alpha=0.92),
        zorder=5,
    )

    fig.savefig(FIG_PATH, dpi=320)
    plt.close(fig)

    print(f"Figure written: {FIG_PATH}")
    print(f"Summary written: {SUMMARY_PATH}")
    print(summary)


if __name__ == "__main__":
    main()
