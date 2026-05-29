#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared rainband-width diagnostics for storm-relative rainfall fields."""

from __future__ import annotations

from typing import Dict

import numpy as np


EPS = 1e-12
MIN_VALID_POINTS = 5

RAINBAND_WIDTH_COLS = [
    "rainband_width_km",
    "rainband_length_km",
    "rainband_aspect_ratio",
    "rainband_width10_km",
    "rainband_length10_km",
    "rainband_aspect_ratio10",
]

RAINBAND_WIDTH_DIAGNOSTIC_COLS = [
    "rainband_valid_grid_count",
    "rainband_weight_sum",
    "rainband10_valid_grid_count",
    "rainband10_weight_sum",
]


def _empty(prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}width_km": np.nan,
        f"{prefix}length_km": np.nan,
        f"{prefix}aspect_ratio": np.nan,
        f"{prefix}valid_grid_count": 0,
        f"{prefix}weight_sum": 0.0,
    }


def rainband_width_for_threshold(
    rain_mmhr: np.ndarray,
    x_prime_km: np.ndarray,
    y_prime_km: np.ndarray,
    threshold_mmhr: float,
) -> Dict[str, float]:
    """Compute 4-sigma equivalent major-axis length and minor-axis width.

    The rainfall intensity itself is used as the weight. Grid cells below
    ``threshold_mmhr`` are excluded before the weighted covariance is computed.
    """
    rain = np.asarray(rain_mmhr, dtype=np.float64)
    x = np.asarray(x_prime_km, dtype=np.float64)
    y = np.asarray(y_prime_km, dtype=np.float64)

    if rain.shape != x.shape or rain.shape != y.shape:
        raise ValueError(f"Rain/x/y shape mismatch: {rain.shape}, {x.shape}, {y.shape}")

    mask = np.isfinite(rain) & np.isfinite(x) & np.isfinite(y) & (rain >= threshold_mmhr)
    n_valid = int(np.count_nonzero(mask))
    out = _empty()
    out["valid_grid_count"] = n_valid
    if n_valid < MIN_VALID_POINTS:
        return out

    weights = rain[mask].ravel()
    weight_sum = float(np.sum(weights))
    out["weight_sum"] = weight_sum
    if not np.isfinite(weight_sum) or weight_sum <= EPS:
        return out

    xx = x[mask].ravel()
    yy = y[mask].ravel()
    x_mean = float(np.sum(weights * xx) / weight_sum)
    y_mean = float(np.sum(weights * yy) / weight_sum)
    x0 = xx - x_mean
    y0 = yy - y_mean

    cov = np.array(
        [
            [float(np.sum(weights * x0 * x0) / weight_sum), float(np.sum(weights * x0 * y0) / weight_sum)],
            [float(np.sum(weights * x0 * y0) / weight_sum), float(np.sum(weights * y0 * y0) / weight_sum)],
        ],
        dtype=np.float64,
    )
    eigvals = np.linalg.eigvalsh(cov)
    if not np.all(np.isfinite(eigvals)):
        return out

    lam2 = max(float(eigvals[0]), 0.0)
    lam1 = max(float(eigvals[1]), 0.0)
    length = 4.0 * float(np.sqrt(lam1))
    width = 4.0 * float(np.sqrt(lam2))
    out["length_km"] = length
    out["width_km"] = width
    out["aspect_ratio"] = float(length / (width + EPS)) if np.isfinite(width) else np.nan
    return out


def compute_dual_rainband_width_metrics(
    rain_mmhr: np.ndarray,
    x_prime_km: np.ndarray,
    y_prime_km: np.ndarray,
) -> Dict[str, float]:
    """Compute main >=1 mm/hr and heavy >=10 mm/hr rainband-width metrics."""
    main = rainband_width_for_threshold(rain_mmhr, x_prime_km, y_prime_km, threshold_mmhr=1.0)
    heavy = rainband_width_for_threshold(rain_mmhr, x_prime_km, y_prime_km, threshold_mmhr=10.0)
    return {
        "rainband_width_km": main["width_km"],
        "rainband_length_km": main["length_km"],
        "rainband_aspect_ratio": main["aspect_ratio"],
        "rainband_width10_km": heavy["width_km"],
        "rainband_length10_km": heavy["length_km"],
        "rainband_aspect_ratio10": heavy["aspect_ratio"],
        "rainband_valid_grid_count": int(main["valid_grid_count"]),
        "rainband_weight_sum": float(main["weight_sum"]),
        "rainband10_valid_grid_count": int(heavy["valid_grid_count"]),
        "rainband10_weight_sum": float(heavy["weight_sum"]),
    }
