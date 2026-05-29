import pandas as pd

df = pd.read_csv("data/processed/gpm_precip_features.csv")

print("表格维度：")
print(df.shape)

print("\n字段列表：")
print(df.columns.tolist())

cols = [
    "event_uid",
    "time",
    "center_lon",
    "center_lat",
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",
]

print("\n前 20 行关键字段：")
print(df[cols].head(20))

stat_cols = [
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "r80_km",
    "anisotropy",
]

print("\n核心特征描述统计：")
print(df[stat_cols].describe())