from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 路径设置
# =========================

IN_PATH = Path("data/processed/gpm_track_merged_features.csv")
OUT_PATH = Path("data/processed/gpm_track_model_features.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取数据
# =========================

df = pd.read_csv(
    IN_PATH,
    parse_dates=["time", "time_end", "track_time"],
    dtype={
        "gpm_event_uid": str,
        "track_event_uid": str,
        "track_typhoon_id": str,
        "track_typhoon_code": str,
    },
)

print("原始主表维度:", df.shape)


# =========================
# 3. 移动方向角度特征
# =========================
# 原始 move_dir_deg 是 0-360°，直接作为数值变量不够严谨。
# 例如 359° 和 1° 实际方向接近，但数值差很大。
# 因此转成 sin / cos。

if "track_move_dir_deg" in df.columns:
    move_rad = np.deg2rad(df["track_move_dir_deg"])

    df["track_move_dir_sin"] = np.sin(move_rad)
    df["track_move_dir_cos"] = np.cos(move_rad)

    print("已生成: track_move_dir_sin, track_move_dir_cos")
else:
    print("警告: 未找到 track_move_dir_deg，跳过方向角特征。")


# =========================
# 4. 时间特征
# =========================

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

# 日周期特征
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 月份周期特征
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

print("已生成时间特征: year, month, day, hour, hour_sin, hour_cos, month_sin, month_cos")


# =========================
# 5. 海陆/近岸状态特征
# =========================
# signed_coast_dist_km:
# 通常正值表示海上距岸，负值表示陆地内部距岸。
# is_land 已经是 0/1。
# 这里再构造近岸区、远海区、内陆区。

if "track_signed_coast_dist_km" in df.columns:
    d = df["track_signed_coast_dist_km"]

    df["is_near_coast_100km"] = ((d >= -100) & (d <= 100)).astype(int)
    df["is_near_coast_200km"] = ((d >= -200) & (d <= 200)).astype(int)
    df["is_offshore_far_300km"] = (d > 300).astype(int)
    df["is_inland_100km"] = (d < -100).astype(int)

    # 连续型近岸影响核：越接近海岸越接近 1，远离海岸衰减
    df["coast_influence_exp"] = np.exp(-np.abs(d) / 200.0)

    print("已生成海岸距离派生特征。")
else:
    print("警告: 未找到 track_signed_coast_dist_km，跳过近岸状态特征。")


# =========================
# 6. 强度综合特征
# =========================
# 风速越大、气压越低通常代表台风越强。
# 这里构造一个标准化强度指数，便于综合表达。

if {"track_wind", "track_pressure"}.issubset(df.columns):
    wind = pd.to_numeric(df["track_wind"], errors="coerce")
    pressure = pd.to_numeric(df["track_pressure"], errors="coerce")

    wind_z = (wind - wind.mean()) / wind.std()
    pressure_deficit = pressure.mean() - pressure
    pressure_deficit_z = (pressure_deficit - pressure_deficit.mean()) / pressure_deficit.std()

    df["wind_z"] = wind_z
    df["pressure_deficit"] = pressure_deficit
    df["pressure_deficit_z"] = pressure_deficit_z

    # 综合强度指数：风速标准化 + 气压亏损标准化
    df["intensity_index"] = 0.5 * wind_z + 0.5 * pressure_deficit_z

    print("已生成强度综合特征: wind_z, pressure_deficit, pressure_deficit_z, intensity_index")
else:
    print("警告: 未找到 track_wind 或 track_pressure，跳过强度综合特征。")


# =========================
# 7. 强度变化特征
# =========================

if "track_wind_change_rate" in df.columns:
    df["is_intensifying_wind"] = (df["track_wind_change_rate"] > 0).astype(int)
    df["is_weakening_wind"] = (df["track_wind_change_rate"] < 0).astype(int)

if "track_pressure_change_rate" in df.columns:
    # pressure_change_rate < 0 表示气压下降，通常对应增强
    df["is_intensifying_pressure"] = (df["track_pressure_change_rate"] < 0).astype(int)
    df["is_weakening_pressure"] = (df["track_pressure_change_rate"] > 0).astype(int)

print("已生成增强/减弱状态特征。")


# =========================
# 8. 降水核心半径修正特征
# =========================
# 原 r80_km 使用全域降水，容易被弱背景降水拉大。
# 这里先保留原值，同时构造一个更稳健的范围指标：
# area10_equiv_radius_km：将 rain_area_10_grid 转换成等效圆半径。
# GPM 分辨率约 0.1° × 0.1°，纬向约 11.1 km，经向按中心纬度修正。

if {"rain_area_10_grid", "center_lat"}.issubset(df.columns):
    grid_area_km2 = 11.1 * 11.1 * np.cos(np.deg2rad(df["center_lat"]))
    grid_area_km2 = grid_area_km2.clip(lower=1e-6)

    area10_km2 = df["rain_area_10_grid"] * grid_area_km2
    df["rain_area_10_km2"] = area10_km2
    df["rain_area_10_equiv_radius_km"] = np.sqrt(area10_km2 / np.pi)

    print("已生成强降水面积 km2 与等效半径。")

if {"rain_area_20_grid", "center_lat"}.issubset(df.columns):
    grid_area_km2 = 11.1 * 11.1 * np.cos(np.deg2rad(df["center_lat"]))
    grid_area_km2 = grid_area_km2.clip(lower=1e-6)

    area20_km2 = df["rain_area_20_grid"] * grid_area_km2
    df["rain_area_20_km2"] = area20_km2
    df["rain_area_20_equiv_radius_km"] = np.sqrt(area20_km2 / np.pi)

    print("已生成极强降水面积 km2 与等效半径。")


# =========================
# 9. 降水偏移方向相对移动方向
# =========================
# centroid_offset_dir_deg 是降水质心相对于台风中心的方向。
# track_move_dir_deg 是台风移动方向。
# 两者差值可以反映降水质心位于台风运动前方、后方、左侧或右侧。

if {"centroid_offset_dir_deg", "track_move_dir_deg"}.issubset(df.columns):
    raw_diff = df["centroid_offset_dir_deg"] - df["track_move_dir_deg"]

    # 归一到 [-180, 180]
    df["centroid_relative_to_motion_deg"] = ((raw_diff + 180) % 360) - 180

    rel_rad = np.deg2rad(df["centroid_relative_to_motion_deg"])
    df["centroid_relative_to_motion_sin"] = np.sin(rel_rad)
    df["centroid_relative_to_motion_cos"] = np.cos(rel_rad)

    # 前方: |角度差| <= 90；后方: 其余
    df["centroid_in_front"] = (np.abs(df["centroid_relative_to_motion_deg"]) <= 90).astype(int)

    print("已生成降水质心相对运动方向特征。")


# =========================
# 10. 输出
# =========================

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n增强特征表已输出:", OUT_PATH)
print("增强后维度:", df.shape)

print("\n新增字段预览:")
new_cols = [
    "track_move_dir_sin",
    "track_move_dir_cos",
    "year",
    "month",
    "hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",
    "pressure_deficit",
    "intensity_index",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "rain_area_20_km2",
    "rain_area_20_equiv_radius_km",
    "centroid_relative_to_motion_deg",
    "centroid_in_front",
]

new_cols = [c for c in new_cols if c in df.columns]
print(df[new_cols].head())