from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 1. 路径设置
# =========================

GPM_FEATURE_PATH = Path("data/processed/gpm_precip_features.csv")
TRACK_FEATURE_PATH = Path("data/processed/typhoon_track_features.csv")

# 前面 05 脚本生成的事件映射表
MAPPING_PATH = Path("data/processed/gpm_track_event_mapping.csv")

OUT_ALL_PATH = Path("data/processed/gpm_track_interpolated_features_all.csv")
OUT_CLEAN_PATH = Path("data/processed/gpm_track_interpolated_features_clean.csv")
OUT_QUALITY_PATH = Path("data/processed/gpm_track_interpolation_quality.csv")

OUT_ALL_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 参数设置
# =========================

# 中心误差阈值
OK_ERROR_KM = 80
SUSPICIOUS_ERROR_KM = 150

# GPM 时刻允许略微超出路径时间范围的容忍值
# 超出太多就标记 outside_time_range
TIME_EDGE_TOLERANCE_HOURS = 3


# =========================
# 3. 工具函数
# =========================

def haversine_km(lon1, lat1, lon2, lat2):
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
    从点1到点2的方位角。
    0°为北，90°为东。
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    deg = np.degrees(np.arctan2(x, y))
    return (deg + 360) % 360


def to_seconds(s):
    """
    datetime Series 转为 Unix 秒。
    """
    return s.astype("int64") / 1e9


def interp_numeric(t_src, y_src, t_new):
    """
    普通数值线性插值。
    """
    valid = np.isfinite(y_src)

    if valid.sum() < 2:
        return np.full_like(t_new, np.nan, dtype=float)

    return np.interp(t_new, t_src[valid], y_src[valid])


def interp_lon_degree(t_src, lon_src, t_new):
    """
    经度线性插值。
    用 unwrap 避免跨 180° 时出现跳变。
    """
    valid = np.isfinite(lon_src)

    if valid.sum() < 2:
        return np.full_like(t_new, np.nan, dtype=float)

    lon_rad = np.deg2rad(lon_src[valid])
    lon_unwrap = np.unwrap(lon_rad)

    interp_rad = np.interp(t_new, t_src[valid], lon_unwrap)
    interp_lon = np.rad2deg(interp_rad)

    # 归一到 [-180, 180]
    interp_lon = ((interp_lon + 180) % 360) - 180

    return interp_lon


def nearest_by_time(track_one, gpm_times):
    """
    对类别字段用最近时间匹配。
    """
    track_times = track_one["track_time"].values.astype("datetime64[ns]")
    result_index = []

    for t in gpm_times.values.astype("datetime64[ns]"):
        pos = np.searchsorted(track_times, t)

        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(track_times):
            candidates.append(pos)

        best = min(
            candidates,
            key=lambda i: abs((pd.Timestamp(t) - track_one.iloc[i]["track_time"]).total_seconds()),
        )

        result_index.append(track_one.index[best])

    return result_index


def classify_error(error_km, outside_time):
    if outside_time:
        return "outside_time_range"

    if pd.isna(error_km):
        return "missing_interpolation"

    if error_km <= OK_ERROR_KM:
        return "ok"

    if error_km <= SUSPICIOUS_ERROR_KM:
        return "suspicious"

    return "abnormal"


# =========================
# 4. 读取数据
# =========================

print("读取 GPM 特征表:", GPM_FEATURE_PATH)
gpm = pd.read_csv(
    GPM_FEATURE_PATH,
    parse_dates=["time", "time_end"],
    dtype={"event_uid": str},
)

print("读取路径特征表:", TRACK_FEATURE_PATH)
track = pd.read_csv(
    TRACK_FEATURE_PATH,
    parse_dates=["time"],
    dtype={"event_uid": str},
)

print("读取事件映射表:", MAPPING_PATH)
mapping = pd.read_csv(
    MAPPING_PATH,
    parse_dates=["track_time"],
    dtype={
        "gpm_event_uid": str,
        "track_event_uid": str,
    },
)

print("\nGPM 表维度:", gpm.shape)
print("路径表维度:", track.shape)
print("映射表维度:", mapping.shape)


# =========================
# 5. 字段重命名
# =========================

gpm = gpm.rename(columns={"event_uid": "gpm_event_uid"})
track = track.rename(columns={"event_uid": "track_event_uid", "time": "track_time"})

gpm["gpm_event_uid"] = gpm["gpm_event_uid"].astype(str)
track["track_event_uid"] = track["track_event_uid"].astype(str)
mapping["gpm_event_uid"] = mapping["gpm_event_uid"].astype(str)
mapping["track_event_uid"] = mapping["track_event_uid"].astype(str)


# =========================
# 6. 从映射表得到 GPM事件 → 路径事件 的唯一映射
# =========================
# 同一 gpm_event_uid 可能有多行，因为每个半小时都映射一次。
# 这里取出现次数最多的 track_event_uid 作为该 GPM 文件夹对应的路径事件。

event_map = (
    mapping.groupby(["gpm_event_uid", "track_event_uid"])
    .size()
    .reset_index(name="count")
    .sort_values(["gpm_event_uid", "count"], ascending=[True, False])
    .drop_duplicates("gpm_event_uid")
    .reset_index(drop=True)
)

print("\nGPM事件到路径事件映射数量:", len(event_map))
print(event_map.head(10))

gpm_to_track = dict(zip(event_map["gpm_event_uid"], event_map["track_event_uid"]))


# =========================
# 7. 准备路径字段
# =========================

required_track_cols = [
    "track_event_uid",
    "track_time",
    "lat",
    "lon_180",
    "pressure",
    "wind",
]

for col in required_track_cols:
    if col not in track.columns:
        raise ValueError(f"路径表缺少必要字段: {col}")

# 去除路径时间重复点
track = (
    track.sort_values(["track_event_uid", "track_time"])
    .drop_duplicates(["track_event_uid", "track_time"], keep="first")
    .reset_index(drop=True)
)


# =========================
# 8. 对每个 GPM 事件插值路径到半小时时刻
# =========================

all_parts = []
quality_records = []

for gpm_event_uid, gpm_one in tqdm(
    gpm.groupby("gpm_event_uid"),
    total=gpm["gpm_event_uid"].nunique(),
    desc="按 GPM 事件插值路径",
):
    gpm_one = gpm_one.sort_values("time").reset_index(drop=True)

    if gpm_event_uid not in gpm_to_track:
        gpm_one["track_event_uid"] = np.nan
        gpm_one["interp_match_status"] = "no_event_mapping"
        all_parts.append(gpm_one)
        continue

    track_event_uid = gpm_to_track[gpm_event_uid]

    track_one = track[track["track_event_uid"] == track_event_uid].copy()
    track_one = track_one.sort_values("track_time").reset_index(drop=True)

    if len(track_one) < 2:
        gpm_one["track_event_uid"] = track_event_uid
        gpm_one["interp_match_status"] = "insufficient_track_points"
        all_parts.append(gpm_one)
        continue

    # 时间轴
    t_src = to_seconds(track_one["track_time"]).values
    t_new = to_seconds(gpm_one["time"]).values

    track_start = track_one["track_time"].min()
    track_end = track_one["track_time"].max()

    gpm_time = gpm_one["time"]
    outside_time = (
        (gpm_time < track_start - pd.Timedelta(hours=TIME_EDGE_TOLERANCE_HOURS))
        | (gpm_time > track_end + pd.Timedelta(hours=TIME_EDGE_TOLERANCE_HOURS))
    ).values

    # 插值核心字段
    gpm_one["track_event_uid"] = track_event_uid
    gpm_one["track_lat"] = interp_numeric(t_src, track_one["lat"].astype(float).values, t_new)
    gpm_one["track_lon_180"] = interp_lon_degree(t_src, track_one["lon_180"].astype(float).values, t_new)

    gpm_one["track_pressure"] = interp_numeric(t_src, track_one["pressure"].astype(float).values, t_new)
    gpm_one["track_wind"] = interp_numeric(t_src, track_one["wind"].astype(float).values, t_new)

    # 可插值的连续字段
    optional_numeric_cols = [
        "coast_dist_km",
        "signed_coast_dist_km",
        "wind_change_rate",
        "pressure_change_rate",
    ]

    for col in optional_numeric_cols:
        if col in track_one.columns:
            gpm_one[f"track_{col}"] = interp_numeric(
                t_src,
                pd.to_numeric(track_one[col], errors="coerce").values,
                t_new,
            )

    # 类别/文本字段采用最近路径点
    nearest_idx = nearest_by_time(track_one, gpm_one["time"])
    nearest_rows = track_one.loc[nearest_idx].reset_index(drop=True)

    categorical_cols = [
        "source_file",
        "typhoon_id",
        "storm_seq",
        "typhoon_code",
        "record_count",
        "typhoon_name",
        "intensity",
        "is_land",
    ]

    for col in categorical_cols:
        if col in nearest_rows.columns:
            gpm_one[f"track_{col}"] = nearest_rows[col].values

    # 用插值后的 signed_coast_dist_km 重新推导陆地状态，优先保留一个连续逻辑
    if "track_signed_coast_dist_km" in gpm_one.columns:
        gpm_one["track_is_land_interp"] = (gpm_one["track_signed_coast_dist_km"] < 0).astype(int)

    # 计算 GPM 文件名中心 vs 插值路径中心误差
    gpm_one["interp_center_error_km"] = haversine_km(
        gpm_one["center_lon"],
        gpm_one["center_lat"],
        gpm_one["track_lon_180"],
        gpm_one["track_lat"],
    )

    gpm_one["outside_track_time_range"] = outside_time

    gpm_one["interp_match_status"] = [
        classify_error(err, out)
        for err, out in zip(gpm_one["interp_center_error_km"], outside_time)
    ]

    # 先加入，后面统一重算半小时移动速度和方向
    all_parts.append(gpm_one)

    quality_records.append(
        {
            "gpm_event_uid": gpm_event_uid,
            "track_event_uid": track_event_uid,
            "gpm_start": gpm_one["time"].min(),
            "gpm_end": gpm_one["time"].max(),
            "track_start": track_start,
            "track_end": track_end,
            "n_gpm_frames": len(gpm_one),
            "center_error_mean_km": gpm_one["interp_center_error_km"].mean(),
            "center_error_median_km": gpm_one["interp_center_error_km"].median(),
            "center_error_max_km": gpm_one["interp_center_error_km"].max(),
            "ok_count": (gpm_one["interp_match_status"] == "ok").sum(),
            "suspicious_count": (gpm_one["interp_match_status"] == "suspicious").sum(),
            "abnormal_count": (gpm_one["interp_match_status"] == "abnormal").sum(),
            "outside_time_count": (gpm_one["interp_match_status"] == "outside_time_range").sum(),
        }
    )


merged = pd.concat(all_parts, ignore_index=True)


# =========================
# 9. 基于插值后的半小时中心，重新计算移动速度、方向、变化率
# =========================

merged = merged.sort_values(["track_event_uid", "time"]).reset_index(drop=True)

merged["track_dt_h_interp"] = np.nan
merged["track_move_distance_km_interp"] = np.nan
merged["track_move_speed_kmh_interp"] = np.nan
merged["track_move_dir_deg_interp"] = np.nan
merged["track_wind_change_rate_interp"] = np.nan
merged["track_pressure_change_rate_interp"] = np.nan

for track_event_uid, idx in merged.groupby("track_event_uid").groups.items():
    if pd.isna(track_event_uid):
        continue

    sub = merged.loc[idx].sort_values("time")
    sub_idx = sub.index

    dt_h = sub["time"].diff().dt.total_seconds() / 3600

    prev_lon = sub["track_lon_180"].shift(1)
    prev_lat = sub["track_lat"].shift(1)

    dist = haversine_km(
        prev_lon,
        prev_lat,
        sub["track_lon_180"],
        sub["track_lat"],
    )

    direction = bearing_deg(
        prev_lon,
        prev_lat,
        sub["track_lon_180"],
        sub["track_lat"],
    )

    speed = dist / dt_h

    wind_rate = sub["track_wind"].diff() / dt_h
    pressure_rate = sub["track_pressure"].diff() / dt_h

    merged.loc[sub_idx, "track_dt_h_interp"] = dt_h.values
    merged.loc[sub_idx, "track_move_distance_km_interp"] = dist.values
    merged.loc[sub_idx, "track_move_speed_kmh_interp"] = speed.values
    merged.loc[sub_idx, "track_move_dir_deg_interp"] = direction.values
    merged.loc[sub_idx, "track_wind_change_rate_interp"] = wind_rate.values
    merged.loc[sub_idx, "track_pressure_change_rate_interp"] = pressure_rate.values


# =========================
# 10. 输出质量统计
# =========================

quality = pd.DataFrame(quality_records)

print("\n插值匹配状态统计:")
print(merged["interp_match_status"].value_counts(dropna=False))

print("\n中心误差统计:")
print(merged["interp_center_error_km"].describe())

print("\n逐事件质量前 20:")
print(
    quality.sort_values("center_error_median_km", ascending=False)
    .head(20)
)

quality.to_csv(OUT_QUALITY_PATH, index=False, encoding="utf-8-sig")


# =========================
# 11. 输出全量表和清洗表
# =========================

merged.to_csv(OUT_ALL_PATH, index=False, encoding="utf-8-sig")

clean = merged[merged["interp_match_status"].isin(["ok", "suspicious"])].copy()
clean.to_csv(OUT_CLEAN_PATH, index=False, encoding="utf-8-sig")

print("\n已输出全量插值表:")
print(OUT_ALL_PATH)
print("维度:", merged.shape)

print("\n已输出清洗插值表:")
print(OUT_CLEAN_PATH)
print("维度:", clean.shape)

print("\n已输出插值质量表:")
print(OUT_QUALITY_PATH)