from pathlib import Path

import pandas as pd


MERGED_PATH = Path("data/processed/gpm_track_merged_features.csv")
OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    MERGED_PATH,
    parse_dates=["time", "time_end", "track_time"],
    dtype={
        "gpm_event_uid": str,
        "track_event_uid": str,
        "track_typhoon_id": str,
        "track_typhoon_code": str,
    },
)

print("合并主表维度:")
print(df.shape)

print("\n事件数量:")
print("GPM 事件数:", df["gpm_event_uid"].nunique())
print("路径事件数:", df["track_event_uid"].nunique())

print("\n匹配状态:")
print(df["match_status"].value_counts(dropna=False))

print("\n时间差统计:")
print(df["track_time_diff_h"].describe())

print("\n中心距离统计:")
print(df["gpm_track_center_dist_km"].describe())

# 检查关键建模字段缺失率
key_cols = [
    # 降水目标变量
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",

    # 路径/强度/环境解释变量
    "track_pressure",
    "track_wind",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
]

key_cols = [c for c in key_cols if c in df.columns]

missing = df[key_cols].isna().mean().sort_values(ascending=False)
print("\n关键字段缺失率:")
print(missing)

missing.to_csv(OUT_DIR / "merged_key_columns_missing_rate.csv", encoding="utf-8-sig")

# 每场台风的样本量
event_counts = (
    df.groupby(["track_event_uid", "track_typhoon_name"], dropna=False)
    .size()
    .reset_index(name="gpm_frame_count")
    .sort_values("gpm_frame_count", ascending=False)
)

print("\n每场台风 GPM 时次数量前 20:")
print(event_counts.head(20))

event_counts.to_csv(
    OUT_DIR / "merged_event_frame_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

# 降水变量描述统计
rain_cols = [
    "rain_max",
    "rain_p95",
    "rain_p99",
    "rain_area_5_grid",
    "rain_area_10_grid",
    "rain_area_20_grid",
    "centroid_offset_km",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",
    "rain_gini",
    "rain_entropy_norm",
]

rain_cols = [c for c in rain_cols if c in df.columns]

rain_desc = df[rain_cols].describe().T
print("\n降水分布特征描述统计:")
print(rain_desc)

rain_desc.to_csv(
    OUT_DIR / "merged_rain_feature_describe.csv",
    encoding="utf-8-sig",
)

# 路径强度变量描述统计
track_cols = [
    "track_pressure",
    "track_wind",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
]

track_cols = [c for c in track_cols if c in df.columns]

track_desc = df[track_cols].describe().T
print("\n路径/强度/环境特征描述统计:")
print(track_desc)

track_desc.to_csv(
    OUT_DIR / "merged_track_feature_describe.csv",
    encoding="utf-8-sig",
)

print("\n已输出检查表到 outputs/tables/")