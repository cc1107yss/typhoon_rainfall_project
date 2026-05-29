#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27_add_landfrac_terrain_features.py

Add supplementary environmental variables:
- landfrac_200km
- landfrac_500km
- terrain_mean_300km
- terrain_max_300km
- terrain_std_300km

Dependencies:
    pandas, numpy, tqdm, pyshp(shapefile), rasterio

No geopandas/fiona/gdalinfo required.

Important:
    ETOPO file may report CRS as EPSG:9518, but if its bounds are
    [-180, -90, 180, 90], we sample it directly with lon/lat.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import shapefile  # pyshp
import rasterio
from rasterio import features
from rasterio.transform import from_origin


LAT_CANDIDATES = [
    "track_lat", "lat", "center_lat", "gpm_center_lat",
    "interp_lat", "lat_interp", "typhoon_lat",
    "LAT", "latitude"
]

LON_CANDIDATES = [
    "track_lon_180", "lon_180", "track_lon", "lon",
    "center_lon", "gpm_center_lon", "interp_lon", "lon_interp",
    "typhoon_lon", "LONG", "longitude"
]

ENV_COLS = [
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",
]


def lon_to_180(lon: float) -> float:
    if pd.isna(lon):
        return np.nan
    return ((float(lon) + 180.0) % 360.0) - 180.0


def detect_lat_lon_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lat_col = next((c for c in LAT_CANDIDATES if c in df.columns), None)
    lon_col = next((c for c in LON_CANDIDATES if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(
            "Cannot detect latitude/longitude columns. "
            f"Available columns include: {list(df.columns)[:60]}"
        )
    return lat_col, lon_col


@lru_cache(maxsize=64)
def circle_offsets(radius_km: int, step_km: int) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-radius_km, radius_km + 1e-9, step_km, dtype=float)
    dx, dy = np.meshgrid(vals, vals)
    mask = dx**2 + dy**2 <= radius_km**2
    return dx[mask].ravel(), dy[mask].ravel()


def offsets_to_lonlat(
    lon0: float,
    lat0: float,
    dx_km: np.ndarray,
    dy_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lat0_rad = math.radians(float(lat0))
    cos_lat = max(math.cos(lat0_rad), 1e-6)

    lats = lat0 + dy_km / 111.32
    lons = lon0 + dx_km / (111.32 * cos_lat)

    lons = ((lons + 180.0) % 360.0) - 180.0
    return lons.astype(float), lats.astype(float)


class LandMaskRaster:
    def __init__(self, land_arr: np.ndarray, res_deg: float):
        self.land = land_arr.astype(np.uint8)
        self.res = float(res_deg)
        self.height, self.width = self.land.shape

    def covers(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)

        out = np.zeros(lons.shape[0], dtype=bool)
        valid = np.isfinite(lons) & np.isfinite(lats)
        if not valid.any():
            return out

        lonv = ((lons[valid] + 180.0) % 360.0) - 180.0
        latv = np.clip(lats[valid], -90.0 + 1e-9, 90.0 - 1e-9)

        cols = np.floor((lonv + 180.0) / self.res).astype(int)
        rows = np.floor((90.0 - latv) / self.res).astype(int)

        ok = (
            (rows >= 0) & (rows < self.height) &
            (cols >= 0) & (cols < self.width)
        )

        temp = np.zeros(lonv.shape[0], dtype=bool)
        temp[ok] = self.land[rows[ok], cols[ok]] > 0
        out[valid] = temp
        return out


def build_or_load_landmask(
    shp_path: Path,
    cache_dir: Path,
    res_deg: float,
) -> LandMaskRaster:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ne_50m_landmask_{str(res_deg).replace('.', 'p')}deg.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        land = data["land"].astype(np.uint8)
        res = float(data["res"])
        print(f"[LANDMASK] loaded cache: {cache_path}, shape={land.shape}, res={res}")
        return LandMaskRaster(land, res)

    if not shp_path.exists():
        raise FileNotFoundError(f"Natural Earth land shapefile not found: {shp_path}")

    width = int(round(360.0 / res_deg))
    height = int(round(180.0 / res_deg))
    transform = from_origin(-180.0, 90.0, res_deg, res_deg)

    print(f"[LANDMASK] rasterizing Natural Earth land polygons...")
    print(f"[LANDMASK] output shape=({height}, {width}), res={res_deg} deg")

    reader = shapefile.Reader(str(shp_path))
    shapes = []
    bad = 0

    for shp in reader.shapes():
        try:
            geom = shp.__geo_interface__
            shapes.append((geom, 1))
        except Exception:
            bad += 1

    if not shapes:
        raise RuntimeError("No valid geometries found in Natural Earth shapefile.")

    if bad:
        print(f"[LANDMASK] skipped invalid geometries: {bad}")

    land = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    np.savez_compressed(cache_path, land=land, res=np.array(res_deg))
    print(f"[LANDMASK] wrote cache: {cache_path}")

    return LandMaskRaster(land, res_deg)


class TerrainSampler:
    def __init__(self, dem_path: Path):
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM GeoTIFF not found: {dem_path}")

        self.src = rasterio.open(dem_path)
        print("[DEM] opened:", dem_path)
        print("[DEM] shape:", (self.src.height, self.src.width))
        print("[DEM] bounds:", self.src.bounds)
        print("[DEM] crs:", self.src.crs)
        print("[DEM] nodata:", self.src.nodata)

        # Do not transform CRS here.
        # ETOPO file may report EPSG:9518, but coordinates are lon/lat by bounds.
        self.bounds = self.src.bounds
        self.nodata = self.src.nodata

    def sample_lonlat(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)

        out = np.full(lons.shape[0], np.nan, dtype=float)

        valid = np.isfinite(lons) & np.isfinite(lats)
        if not valid.any():
            return out

        lonv = ((lons[valid] + 180.0) % 360.0) - 180.0
        latv = lats[valid]

        in_bounds = (
            (lonv >= self.bounds.left) & (lonv <= self.bounds.right) &
            (latv >= self.bounds.bottom) & (latv <= self.bounds.top)
        )

        vals = np.full(lonv.shape[0], np.nan, dtype=float)

        if in_bounds.any():
            coords = list(zip(lonv[in_bounds], latv[in_bounds]))
            sampled = np.array([v[0] for v in self.src.sample(coords)], dtype=float)

            if self.nodata is not None:
                sampled[np.isclose(sampled, self.nodata)] = np.nan
            sampled[sampled <= -99990] = np.nan

            vals[in_bounds] = sampled

        out[valid] = vals
        return out

    def close(self):
        self.src.close()


def safe_stats(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(values)), float(np.max(values)), float(np.std(values))


def compute_env_one_center(
    lat: float,
    lon: float,
    landmask: LandMaskRaster,
    terrain: TerrainSampler,
    sample_step_km: int,
) -> Dict[str, float]:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return {c: np.nan for c in ENV_COLS}

    lon = lon_to_180(lon)

    # 200 km land fraction
    dx200, dy200 = circle_offsets(200, sample_step_km)
    lons200, lats200 = offsets_to_lonlat(lon, lat, dx200, dy200)
    land200 = landmask.covers(lons200, lats200)
    landfrac_200 = float(np.mean(land200)) if land200.size else np.nan

    # 500 km land fraction
    dx500, dy500 = circle_offsets(500, sample_step_km)
    lons500, lats500 = offsets_to_lonlat(lon, lat, dx500, dy500)
    land500 = landmask.covers(lons500, lats500)
    landfrac_500 = float(np.mean(land500)) if land500.size else np.nan

    # 300 km terrain statistics, land points only
    dx300, dy300 = circle_offsets(300, sample_step_km)
    lons300, lats300 = offsets_to_lonlat(lon, lat, dx300, dy300)
    land300 = landmask.covers(lons300, lats300)

    elev = terrain.sample_lonlat(lons300, lats300)

    elev_land = elev[land300 & np.isfinite(elev)]
    # Ocean/bathymetry is excluded by land300; clip below sea-level land to 0 for orographic effect.
    elev_land = np.where(elev_land < 0.0, 0.0, elev_land)

    terrain_mean, terrain_max, terrain_std = safe_stats(elev_land)

    return {
        "landfrac_200km": landfrac_200,
        "landfrac_500km": landfrac_500,
        "terrain_mean_300km": terrain_mean,
        "terrain_max_300km": terrain_max,
        "terrain_std_300km": terrain_std,
    }


def add_env_features(
    df: pd.DataFrame,
    landmask: LandMaskRaster,
    terrain: TerrainSampler,
    sample_step_km: int,
    cache_round_deg: float,
) -> pd.DataFrame:
    lat_col, lon_col = detect_lat_lon_columns(df)
    print(f"[COLUMNS] lat={lat_col}, lon={lon_col}")

    out = df.copy()

    for c in ENV_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])

    cache: Dict[Tuple[float, float], Dict[str, float]] = {}
    records: List[Dict[str, float]] = []

    n = len(out)
    for lat, lon in tqdm(
        out[[lat_col, lon_col]].itertuples(index=False, name=None),
        total=n,
        desc="env features",
    ):
        lat_f = float(lat)
        lon_f = lon_to_180(float(lon))

        key = (
            round(lat_f / cache_round_deg) * cache_round_deg,
            round(lon_f / cache_round_deg) * cache_round_deg,
        )
        key = (round(key[0], 5), round(lon_to_180(key[1]), 5))

        if key not in cache:
            cache[key] = compute_env_one_center(
                lat=key[0],
                lon=key[1],
                landmask=landmask,
                terrain=terrain,
                sample_step_km=sample_step_km,
            )
        records.append(cache[key])

    env_df = pd.DataFrame(records)
    print(f"[CACHE] unique rounded centers: {len(cache)} / rows: {n}")

    return pd.concat([out.reset_index(drop=True), env_df.reset_index(drop=True)], axis=1)


def summarize(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    rows = []
    for c in ENV_COLS:
        if c not in df.columns:
            continue

        s = df[c]
        rows.append({
            "file": file_name,
            "variable": c,
            "n": int(len(s)),
            "missing": int(s.isna().sum()),
            "min": float(s.min(skipna=True)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "mean": float(s.mean(skipna=True)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max(skipna=True)),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-earth", required=True, type=Path)
    parser.add_argument("--dem", required=True, type=Path)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("data/processed/env_cache"), type=Path)
    parser.add_argument("--sample-step-km", default=50, type=int)
    parser.add_argument("--landmask-res", default=0.05, type=float)
    parser.add_argument("--cache-round-deg", default=0.02, type=float)
    parser.add_argument("--max-rows", default=None, type=int)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    landmask = build_or_load_landmask(
        shp_path=args.natural_earth,
        cache_dir=args.cache_dir,
        res_deg=args.landmask_res,
    )
    terrain = TerrainSampler(args.dem)

    summaries = []
    skipped = []

    for path in args.inputs:
        print("\n" + "=" * 80)
        print("[INPUT]", path)

        if not path.exists():
            print("[SKIP] missing file")
            skipped.append({"file": str(path), "reason": "missing"})
            continue

        df = pd.read_csv(path)
        if args.max_rows is not None:
            df = df.head(args.max_rows).copy()
            print(f"[SMOKE] using first {len(df)} rows")

        try:
            df_env = add_env_features(
                df=df,
                landmask=landmask,
                terrain=terrain,
                sample_step_km=args.sample_step_km,
                cache_round_deg=args.cache_round_deg,
            )
        except Exception as e:
            print("[SKIP] failed:", repr(e))
            skipped.append({"file": str(path), "reason": repr(e)})
            continue

        suffix = "_env_smoke.csv" if args.max_rows is not None else "_env.csv"
        out_path = args.output_dir / f"{path.stem}{suffix}"
        df_env.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("[WRITE]", out_path, "shape=", df_env.shape)

        summaries.append(summarize(df_env, out_path.name))

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = args.output_dir / (
            "env_feature_summary_smoke.csv" if args.max_rows is not None else "env_feature_summary.csv"
        )
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n[SUMMARY]", summary_path)
        print(summary.to_string(index=False))

    if skipped:
        skipped_df = pd.DataFrame(skipped)
        skipped_path = args.output_dir / (
            "env_feature_skipped_smoke.csv" if args.max_rows is not None else "env_feature_skipped.csv"
        )
        skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")
        print("\n[SKIPPED]", skipped_path)
        print(skipped_df.to_string(index=False))

    terrain.close()


if __name__ == "__main__":
    main()
