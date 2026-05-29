from pathlib import Path
import re
import math
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from tqdm import tqdm


# =========================
# 1. 路径配置
# =========================

GPM_ROOT = Path("data/raw/GPM_3IMERGHHE.07")
OUT_CSV = Path("data/processed/gpm_precip_features.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 文件名解析
# =========================

FILENAME_PATTERN = re.compile(
    r"3IMERG\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6}).*?"
    r"_center_(?P<center_lon>[-+]?\d+(?:\.\d+)?)E_(?P<center_lat>[-+]?\d+(?:\.\d+)?)N"
    r"_bbox_(?P<lon_min>[-+]?\d+(?:\.\d+)?)E_(?P<lon_max>[-+]?\d+(?:\.\d+)?)E_"
    r"(?P<lat_min>[-+]?\d+(?:\.\d+)?)N_(?P<lat_max>[-+]?\d+(?:\.\d+)?)N"
)


def parse_gpm_filename(tif_path: Path) -> dict:
    """
    从 GPM tif 文件名解析：
    日期、起止时间、台风中心、bbox。
    """
    name = tif_path.name
    match = FILENAME_PATTERN.search(name)

    if match is None:
        raise ValueError(f"文件名无法解析: {name}")

    info = match.groupdict()

    date = info["date"]
    start = info["start"]
    end = info["end"]

    time = pd.to_datetime(date + start, format="%Y%m%d%H%M%S")
    time_end = pd.to_datetime(date + end, format="%Y%m%d%H%M%S")

    return {
        "time": time,
        "time_end": time_end,
        "center_lon": float(info["center_lon"]),
        "center_lat": float(info["center_lat"]),
        "bbox_lon_min": float(info["lon_min"]),
        "bbox_lon_max": float(info["lon_max"]),
        "bbox_lat_min": float(info["lat_min"]),
        "bbox_lat_max": float(info["lat_max"]),
    }


# =========================
# 3. 经纬度与距离工具函数
# =========================

def haversine_km(lon1, lat1, lon2, lat2):
    """
    计算球面距离，单位 km。
    支持 numpy 数组。
    """
    radius = 6371.0

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    """
    从点1指向点2的方位角。
    0°为正北，90°为正东。
    """
    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    angle = math.degrees(math.atan2(x, y))
    return (angle + 360) % 360


def safe_div(a, b):
    """
    安全除法，避免除以 0。
    """
    if b == 0:
        return np.nan
    return a / b


# =========================
# 4. 单个 tif 的特征提取
# =========================

def extract_one_tif_features(tif_path: Path) -> dict:
    meta = parse_gpm_filename(tif_path)

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

        if crs is not None and str(crs).upper() != "EPSG:4326":
            warnings.warn(f"{tif_path.name} CRS 不是 EPSG:4326，而是 {crs}")

        if nodata is None:
            nodata = -9999

        valid = np.isfinite(arr) & (arr != nodata)
        rain = np.where(valid, arr, np.nan)

        # GPM 降水值单位是 mm/hr，半小时累计量 = mm/hr * 0.5
        rain_halfhour_mm = rain * 0.5

        # 生成每个格点的经纬度网格
        rows, cols = np.indices(arr.shape)
        lon_grid, lat_grid = xy(transform, rows, cols, offset="center")

        # rasterio.transform.xy 有时返回一维数组，所以必须 reshape 回 201 × 201
        lon_grid = np.array(lon_grid).reshape(arr.shape)
        lat_grid = np.array(lat_grid).reshape(arr.shape)

    center_lon = meta["center_lon"]
    center_lat = meta["center_lat"]

    valid_values = rain[valid]

    if valid_values.size == 0:
        return {
            **meta,
            "source_file": tif_path.name,
            "valid_count": 0,
            "rain_valid_ratio": 0.0,
        }

    # 只保留非负降水。少数反演数据可能有极小负值，直接截断为 0。
    rain_nonneg = np.where(np.isnan(rain), np.nan, np.maximum(rain, 0))
    rain_halfhour_nonneg = rain_nonneg * 0.5

    valid2 = np.isfinite(rain_nonneg)
    values = rain_nonneg[valid2]
    values_halfhour = rain_halfhour_nonneg[valid2]

    total_rate = np.nansum(values)
    total_halfhour = np.nansum(values_halfhour)

    # 有降水格点阈值
    mask01 = valid2 & (rain_nonneg > 0.1)
    mask1 = valid2 & (rain_nonneg >= 1)
    mask5 = valid2 & (rain_nonneg >= 5)
    mask10 = valid2 & (rain_nonneg >= 10)
    mask20 = valid2 & (rain_nonneg >= 20)
    mask50 = valid2 & (rain_nonneg >= 50)

    # 加权降水质心
    if total_rate > 0:
        weights = np.where(valid2, rain_nonneg, 0.0)
        weight_sum = np.nansum(weights)

        rain_centroid_lon = float(np.nansum(lon_grid * weights) / weight_sum)
        rain_centroid_lat = float(np.nansum(lat_grid * weights) / weight_sum)

        centroid_offset_km = float(
            haversine_km(
                center_lon,
                center_lat,
                rain_centroid_lon,
                rain_centroid_lat,
            )
        )

        centroid_offset_dir_deg = float(
            bearing_deg(
                center_lon,
                center_lat,
                rain_centroid_lon,
                rain_centroid_lat,
            )
        )
    else:
        rain_centroid_lon = np.nan
        rain_centroid_lat = np.nan
        centroid_offset_km = np.nan
        centroid_offset_dir_deg = np.nan

    # 每个格点到台风中心的距离
    dist_km = haversine_km(center_lon, center_lat, lon_grid, lat_grid)

    def weighted_radius_quantile(q):
        """
        累计降水权重达到 q 时的半径。
        例如 r80 表示 80% 降水量集中在中心多少 km 内。
        """
        if total_rate <= 0:
            return np.nan

        d = dist_km[valid2].ravel()
        w = rain_nonneg[valid2].ravel()

        order = np.argsort(d)
        d_sorted = d[order]
        w_sorted = w[order]

        cumsum = np.cumsum(w_sorted)
        target = q * cumsum[-1]

        idx = np.searchsorted(cumsum, target)
        idx = min(idx, len(d_sorted) - 1)

        return float(d_sorted[idx])

    r50 = weighted_radius_quantile(0.50)
    r80 = weighted_radius_quantile(0.80)
    r90 = weighted_radius_quantile(0.90)

    # 东西、南北非对称性
    east_sum = np.nansum(rain_nonneg[(lon_grid >= center_lon) & valid2])
    west_sum = np.nansum(rain_nonneg[(lon_grid < center_lon) & valid2])
    north_sum = np.nansum(rain_nonneg[(lat_grid >= center_lat) & valid2])
    south_sum = np.nansum(rain_nonneg[(lat_grid < center_lat) & valid2])

    asym_EW = safe_div(east_sum - west_sum, east_sum + west_sum)
    asym_NS = safe_div(north_sum - south_sum, north_sum + south_sum)

    # 四象限降水占比
    q_NE = np.nansum(
        rain_nonneg[(lon_grid >= center_lon) & (lat_grid >= center_lat) & valid2]
    )
    q_SE = np.nansum(
        rain_nonneg[(lon_grid >= center_lon) & (lat_grid < center_lat) & valid2]
    )
    q_SW = np.nansum(
        rain_nonneg[(lon_grid < center_lon) & (lat_grid < center_lat) & valid2]
    )
    q_NW = np.nansum(
        rain_nonneg[(lon_grid < center_lon) & (lat_grid >= center_lat) & valid2]
    )

    q_sum = q_NE + q_SE + q_SW + q_NW

    # 将经纬度近似转为相对 km 坐标，用于降水主轴分析
    x_km = (lon_grid - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_km = (lat_grid - center_lat) * 111.0

    # 降水形态协方差与主轴
    if total_rate > 0:
        w = rain_nonneg[valid2].ravel()
        x = x_km[valid2].ravel()
        y = y_km[valid2].ravel()

        x_mean = np.average(x, weights=w)
        y_mean = np.average(y, weights=w)

        x0 = x - x_mean
        y0 = y - y_mean

        cov_xx = np.average(x0 * x0, weights=w)
        cov_yy = np.average(y0 * y0, weights=w)
        cov_xy = np.average(x0 * y0, weights=w)

        cov = np.array(
            [
                [cov_xx, cov_xy],
                [cov_xy, cov_yy],
            ]
        )

        eigvals, eigvecs = np.linalg.eigh(cov)

        eig_order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[eig_order]
        eigvecs = eigvecs[:, eig_order]

        major_axis_km = float(np.sqrt(max(eigvals[0], 0)))
        minor_axis_km = float(np.sqrt(max(eigvals[1], 0)))

        anisotropy = float(
            safe_div(
                major_axis_km - minor_axis_km,
                major_axis_km + minor_axis_km,
            )
        )

        # 主轴方向：以正北为 0°，正东为 90°，范围压到 0-180°
        vx, vy = eigvecs[:, 0]
        orientation_deg = float((math.degrees(math.atan2(vx, vy)) + 360) % 180)
    else:
        major_axis_km = np.nan
        minor_axis_km = np.nan
        anisotropy = np.nan
        orientation_deg = np.nan

    # Gini：衡量降水集中程度
    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    if n > 0 and np.sum(sorted_vals) > 0:
        gini = float(
            (
                2
                * np.sum((np.arange(1, n + 1)) * sorted_vals)
                / (n * np.sum(sorted_vals))
            )
            - (n + 1) / n
        )
    else:
        gini = np.nan

    # 熵：降水越集中，归一化熵越低
    if total_rate > 0:
        p = values / np.sum(values)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
        entropy_norm = float(entropy / np.log(len(values))) if len(values) > 1 else np.nan
    else:
        entropy_norm = np.nan

    return {
        **meta,
        "source_file": tif_path.name,
        "valid_count": int(valid2.sum()),
        "rain_valid_ratio": float(valid2.mean()),

        # 原单位 mm/hr
        "rain_mean": float(np.nanmean(values)),
        "rain_std": float(np.nanstd(values)),
        "rain_max": float(np.nanmax(values)),
        "rain_p50": float(np.nanpercentile(values, 50)),
        "rain_p90": float(np.nanpercentile(values, 90)),
        "rain_p95": float(np.nanpercentile(values, 95)),
        "rain_p99": float(np.nanpercentile(values, 99)),

        # 半小时累计量，单位 mm
        "rain_halfhour_mean_mm": float(np.nanmean(values_halfhour)),
        "rain_halfhour_sum_mm_grid": float(total_halfhour),

        # 面积先用格点数表示
        "rain_area_0p1_grid": int(mask01.sum()),
        "rain_area_1_grid": int(mask1.sum()),
        "rain_area_5_grid": int(mask5.sum()),
        "rain_area_10_grid": int(mask10.sum()),
        "rain_area_20_grid": int(mask20.sum()),
        "rain_area_50_grid": int(mask50.sum()),

        # 降水中心偏移
        "rain_centroid_lon": rain_centroid_lon,
        "rain_centroid_lat": rain_centroid_lat,
        "centroid_offset_km": centroid_offset_km,
        "centroid_offset_dir_deg": centroid_offset_dir_deg,

        # 降水半径
        "r50_km": r50,
        "r80_km": r80,
        "r90_km": r90,

        # 非对称性
        "asym_EW": float(asym_EW),
        "asym_NS": float(asym_NS),

        # 四象限比例
        "quad_NE_ratio": float(safe_div(q_NE, q_sum)),
        "quad_SE_ratio": float(safe_div(q_SE, q_sum)),
        "quad_SW_ratio": float(safe_div(q_SW, q_sum)),
        "quad_NW_ratio": float(safe_div(q_NW, q_sum)),

        # 降水带形态
        "major_axis_km": major_axis_km,
        "minor_axis_km": minor_axis_km,
        "anisotropy": anisotropy,
        "orientation_deg": orientation_deg,

        # 集中度
        "rain_gini": gini,
        "rain_entropy_norm": entropy_norm,
    }


# =========================
# 5. 批量处理
# =========================

def main():
    print("当前工作目录:", Path.cwd())
    print("GPM_ROOT 绝对路径:", GPM_ROOT.resolve())
    print("GPM_ROOT 是否存在:", GPM_ROOT.exists())

    tif_files = sorted(GPM_ROOT.glob("*/*.tif"))

    print(f"发现 tif 文件数量: {len(tif_files)}")

    if len(tif_files) == 0:
        print("没有找到 tif 文件。请检查：")
        print("1. GPM 数据是否已经下载")
        print("2. GPM 数据是否已经解压")
        print("3. 数据是否放在 data/raw/GPM_3IMERGHHE.07")
        print("4. tif 是否位于 GPM_3IMERGHHE.07/事件目录/*.tif")
        return

    rows = []
    errors = []

    for tif_path in tqdm(tif_files):
        event_uid = tif_path.parent.name

        try:
            feat = extract_one_tif_features(tif_path)
            feat["event_uid"] = event_uid
            rows.append(feat)
        except Exception as e:
            errors.append((str(tif_path), str(e)))

            # 只打印前 10 个错误，避免终端刷屏
            if len(errors) <= 10:
                print(f"[跳过] {tif_path}: {e}")

    df = pd.DataFrame(rows)

    print(f"成功提取数量: {len(df)}")
    print(f"失败数量: {len(errors)}")

    if len(errors) > 10:
        print("错误较多，仅显示前 10 个。")

    if len(df) == 0:
        raise RuntimeError(
            "没有成功提取任何 tif 特征。当前已找到 tif，但全部处理失败，请检查前面的跳过报错。"
        )

    df = df.sort_values(["event_uid", "time"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"完成：{OUT_CSV}")
    print(df.head())
    print("输出表维度:", df.shape)


if __name__ == "__main__":
    main()