from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 路径设置
# =========================

IN_PATH = Path("data/processed/gpm_track_interpolated_features_clean.csv")
OUT_PATH = Path("data/processed/gpm_track_model_features_interp.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取数据
# =========================

df = pd.read_csv(
    IN_PATH,
    parse_dates=["time", "time_end"],
    dtype={
        "gpm_event_uid": str,
        "track_event_uid": str,
        "track_typhoon_id": str,
        "track_typhoon_code": str,
    },
)

print("插值清洗表维度:", df.shape)


# =========================
# 3. 统一字段名
# =========================
# 插值脚本生成的是 *_interp 字段。
# 为了后续分析脚本复用，统一成标准字段名。

rename_map = {
    "track_move_speed_kmh_interp": "track_move_speed_kmh",
    "track_move_dir_deg_interp": "track_move_dir_deg",
    "track_wind_change_rate_interp": "track_wind_change_rate",
    "track_pressure_change_rate_interp": "track_pressure_change_rate",
    "track_dt_h_interp": "track_dt_h",
    "track_move_distance_km_interp": "track_move_distance_km",
}

for old, new in rename_map.items():
    if old in df.columns:
        df[new] = df[old]

# 陆地状态优先使用插值 signed_coast_dist 推导的结果
if "track_is_land_interp" in df.columns:
    df["track_is_land"] = df["track_is_land_interp"]

print("已统一插值移动字段。")


# =========================
# 4. 移动方向角度特征
# =========================

if "track_move_dir_deg" in df.columns:
    move_rad = np.deg2rad(df["track_move_dir_deg"])
    df["track_move_dir_sin"] = np.sin(move_rad)
    df["track_move_dir_cos"] = np.cos(move_rad)

    print("已生成: track_move_dir_sin, track_move_dir_cos")
else:
    print("警告: 未找到 track_move_dir_deg。")


# =========================
# 5. 时间特征
# =========================

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

print("已生成时间周期特征。")


# =========================
# 6. 海陆/近岸状态特征
# =========================

if "track_signed_coast_dist_km" in df.columns:
    d = pd.to_numeric(df["track_signed_coast_dist_km"], errors="coerce")

    df["is_near_coast_100km"] = ((d >= -100) & (d <= 100)).astype(int)
    df["is_near_coast_200km"] = ((d >= -200) & (d <= 200)).astype(int)
    df["is_offshore_far_300km"] = (d > 300).astype(int)
    df["is_inland_100km"] = (d < -100).astype(int)

    df["coast_influence_exp"] = np.exp(-np.abs(d) / 200.0)

    print("已生成近岸派生特征。")
else:
    print("警告: 未找到 track_signed_coast_dist_km。")


# =========================
# 7. 强度综合特征
# =========================

if {"track_wind", "track_pressure"}.issubset(df.columns):
    wind = pd.to_numeric(df["track_wind"], errors="coerce")
    pressure = pd.to_numeric(df["track_pressure"], errors="coerce")

    wind_z = (wind - wind.mean()) / wind.std()

    pressure_deficit = pressure.mean() - pressure
    pressure_deficit_z = (pressure_deficit - pressure_deficit.mean()) / pressure_deficit.std()

    df["wind_z"] = wind_z
    df["pressure_deficit"] = pressure_deficit
    df["pressure_deficit_z"] = pressure_deficit_z
    df["intensity_index"] = 0.5 * wind_z + 0.5 * pressure_deficit_z

    print("已生成强度综合特征。")
else:
    print("警告: 未找到 track_wind 或 track_pressure。")


# =========================
# 8. 增强/减弱状态特征
# =========================

if "track_wind_change_rate" in df.columns:
    df["is_intensifying_wind"] = (df["track_wind_change_rate"] > 0).astype(int)
    df["is_weakening_wind"] = (df["track_wind_change_rate"] < 0).astype(int)

if "track_pressure_change_rate" in df.columns:
    # 气压下降通常表示增强
    df["is_intensifying_pressure"] = (df["track_pressure_change_rate"] < 0).astype(int)
    df["is_weakening_pressure"] = (df["track_pressure_change_rate"] > 0).astype(int)

print("已生成增强/减弱状态特征。")


# =========================
# 9. 强降水面积 km² 与等效半径
# =========================

if {"rain_area_10_grid", "center_lat"}.issubset(df.columns):
    grid_area_km2 = 11.1 * 11.1 * np.cos(np.deg2rad(df["center_lat"]))
    grid_area_km2 = grid_area_km2.clip(lower=1e-6)

    area10_km2 = df["rain_area_10_grid"] * grid_area_km2
    df["rain_area_10_km2"] = area10_km2
    df["rain_area_10_equiv_radius_km"] = np.sqrt(area10_km2 / np.pi)

if {"rain_area_20_grid", "center_lat"}.issubset(df.columns):
    grid_area_km2 = 11.1 * 11.1 * np.cos(np.deg2rad(df["center_lat"]))
    grid_area_km2 = grid_area_km2.clip(lower=1e-6)

    area20_km2 = df["rain_area_20_grid"] * grid_area_km2
    df["rain_area_20_km2"] = area20_km2
    df["rain_area_20_equiv_radius_km"] = np.sqrt(area20_km2 / np.pi)

print("已生成强降水面积与等效半径。")


# =========================
# 10. 降水质心相对运动方向
# =========================

if {"centroid_offset_dir_deg", "track_move_dir_deg"}.issubset(df.columns):
    raw_diff = df["centroid_offset_dir_deg"] - df["track_move_dir_deg"]

    df["centroid_relative_to_motion_deg"] = ((raw_diff + 180) % 360) - 180

    rel_rad = np.deg2rad(df["centroid_relative_to_motion_deg"])
    df["centroid_relative_to_motion_sin"] = np.sin(rel_rad)
    df["centroid_relative_to_motion_cos"] = np.cos(rel_rad)

    df["centroid_in_front"] = (
        np.abs(df["centroid_relative_to_motion_deg"]) <= 90
    ).astype(int)

    print("已生成降水质心相对运动方向特征。")


# =========================
# 11. 插值质量特征保留
# =========================

if "interp_center_error_km" in df.columns:
    df["interp_center_error_ok"] = (df["interp_center_error_km"] <= 80).astype(int)

print("已保留插值质量特征。")


# =========================
# 12. 输出
# =========================

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n插值版增强特征表已输出:", OUT_PATH)
print("输出维度:", df.shape)

preview_cols = [
    "gpm_event_uid",
    "track_event_uid",
    "time",
    "center_lon",
    "center_lat",
    "track_lon_180",
    "track_lat",
    "interp_center_error_km",
    "track_pressure",
    "track_wind",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_move_dir_sin",
    "track_move_dir_cos",
    "track_signed_coast_dist_km",
    "track_is_land",
    "intensity_index",
    "rain_max",
    "rain_p95",
    "rain_area_10_km2",
]

preview_cols = [c for c in preview_cols if c in df.columns]

print("\n关键字段预览:")
print(df[preview_cols].head(10))