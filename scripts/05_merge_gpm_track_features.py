from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 1. 路径设置
# =========================

GPM_FEATURE_PATH = Path("data/processed/gpm_precip_features.csv")

# 这里要求你已经把 typhoon_track_features_with_coast.csv
# 复制/重命名为 data/processed/typhoon_track_features.csv
TRACK_FEATURE_PATH = Path("data/processed/typhoon_track_features.csv")

OUT_PATH = Path("data/processed/gpm_track_merged_features.csv")
MAPPING_OUT_PATH = Path("data/processed/gpm_track_event_mapping.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 参数设置
# =========================
# GPM 是半小时一次，路径数据通常是 6 小时一次。
# 对每个 GPM 文件，在路径表中找 ±3 小时内的候选路径点。
TIME_TOLERANCE_HOURS = 3

# 若最近路径中心与 GPM 文件名中心距离超过该阈值，则认为匹配不可靠。
# 先设为 250 km，后面根据输出统计决定是否收紧到 150 km。
MAX_CENTER_DISTANCE_KM = 250


# =========================
# 3. 工具函数
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


def choose_lon_col(track_df):
    """
    自动识别路径表经度字段。
    优先使用 lon_180，因为它通常是 [-180, 180] 口径。
    """
    if "lon_180" in track_df.columns:
        return "lon_180"
    if "lon" in track_df.columns:
        return "lon"

    raise ValueError("路径表中没有 lon_180 或 lon 字段。")


def choose_lat_col(track_df):
    """
    自动识别路径表纬度字段。
    """
    if "lat" in track_df.columns:
        return "lat"

    raise ValueError("路径表中没有 lat 字段。")


def prefix_track_col(col):
    """
    给路径表字段加前缀，避免和 GPM 字段混淆。
    但 track_event_uid 保持原名，因为它是后续建模主编号。
    """
    if col == "track_event_uid":
        return col

    if col == "track_time":
        return col

    if col.startswith("track_"):
        return col

    return f"track_{col}"


# =========================
# 4. 读取数据
# =========================

print("读取 GPM 降水特征表:", GPM_FEATURE_PATH)
gpm = pd.read_csv(
    GPM_FEATURE_PATH,
    parse_dates=["time", "time_end"],
    dtype={"event_uid": str},
)

print("读取台风路径特征表:", TRACK_FEATURE_PATH)
track = pd.read_csv(
    TRACK_FEATURE_PATH,
    parse_dates=["time"],
    dtype={"event_uid": str},
)

print("\n原始 GPM 表维度:", gpm.shape)
print("原始路径表维度:", track.shape)

print("\n原始 GPM 字段:")
print(gpm.columns.tolist())

print("\n原始路径字段:")
print(track.columns.tolist())


# =========================
# 5. 统一编号字段命名
# =========================

if "event_uid" not in gpm.columns:
    raise ValueError("GPM 表中缺少 event_uid 字段。")

if "event_uid" not in track.columns:
    raise ValueError("路径表中缺少 event_uid 字段。")

# GPM 的 event_uid 是 GPM 文件夹编号，不是你构造的路径事件编号
gpm = gpm.rename(columns={"event_uid": "gpm_event_uid"})

# 路径表的 event_uid 是你构造的台风事件唯一编号
track = track.rename(columns={"event_uid": "track_event_uid"})

# 路径表时间改名，避免和 GPM time 混淆
track = track.rename(columns={"time": "track_time"})

print("\n编号字段已重命名：")
print("GPM: event_uid -> gpm_event_uid")
print("Track: event_uid -> track_event_uid")
print("Track: time -> track_time")


# =========================
# 6. 基本字段检查
# =========================

required_gpm_cols = ["gpm_event_uid", "time", "time_end", "center_lon", "center_lat"]
for col in required_gpm_cols:
    if col not in gpm.columns:
        raise ValueError(f"GPM 表缺少必要字段: {col}")

lon_col = choose_lon_col(track)
lat_col = choose_lat_col(track)

required_track_cols = ["track_event_uid", "track_time", lon_col, lat_col]
for col in required_track_cols:
    if col not in track.columns:
        raise ValueError(f"路径表缺少必要字段: {col}")

print("\n路径经度字段:", lon_col)
print("路径纬度字段:", lat_col)


# =========================
# 7. 清洗缺失值
# =========================

gpm = gpm.dropna(
    subset=["gpm_event_uid", "time", "center_lon", "center_lat"]
).copy()

track = track.dropna(
    subset=["track_event_uid", "track_time", lon_col, lat_col]
).copy()

# 保证编号是字符串
gpm["gpm_event_uid"] = gpm["gpm_event_uid"].astype(str)
track["track_event_uid"] = track["track_event_uid"].astype(str)

gpm = gpm.reset_index(drop=True)
track = track.reset_index(drop=True)

print("\n清洗后 GPM 行数:", len(gpm))
print("清洗后路径行数:", len(track))

print("\nGPM 文件夹编号示例 gpm_event_uid:")
print(gpm["gpm_event_uid"].drop_duplicates().head(10).tolist())

print("\n路径事件编号示例 track_event_uid:")
print(track["track_event_uid"].drop_duplicates().head(10).tolist())

common_uid = set(gpm["gpm_event_uid"]) & set(track["track_event_uid"])
print("\n两个编号字段直接相同的数量:", len(common_uid))
print("说明：这里为 0 也正常，因为 GPM 文件夹编号和路径事件编号不是同一套体系。")


# =========================
# 8. 按时间排序，准备匹配
# =========================

track = track.sort_values("track_time").reset_index(drop=True)
track_times = track["track_time"].values.astype("datetime64[ns]")

print("\n开始匹配：时间窗口 ±", TIME_TOLERANCE_HOURS, "小时")
print("最大允许中心距离:", MAX_CENTER_DISTANCE_KM, "km")


# =========================
# 9. 对每个 GPM 时次匹配路径记录
# =========================
# 匹配规则：
# 1. 候选路径点必须在 GPM 时刻 ±3 小时内；
# 2. 若候选路径点有多个，选择与 GPM 文件名中心距离最近者；
# 3. 若最近距离超过 MAX_CENTER_DISTANCE_KM，则判为不可靠匹配。

match_records = []

for _, gpm_row in tqdm(gpm.iterrows(), total=len(gpm), desc="匹配 GPM 与路径"):
    gpm_time = np.datetime64(gpm_row["time"])

    time_min = gpm_time - np.timedelta64(TIME_TOLERANCE_HOURS, "h")
    time_max = gpm_time + np.timedelta64(TIME_TOLERANCE_HOURS, "h")

    left = np.searchsorted(track_times, time_min, side="left")
    right = np.searchsorted(track_times, time_max, side="right")

    # 没有时间候选
    if left >= right:
        match_records.append(
            {
                "matched_track_index": np.nan,
                "track_time": pd.NaT,
                "track_time_diff_h": np.nan,
                "gpm_track_center_dist_km": np.nan,
                "match_status": "no_time_candidate",
            }
        )
        continue

    candidates = track.iloc[left:right].copy()

    dists = haversine_km(
        gpm_row["center_lon"],
        gpm_row["center_lat"],
        candidates[lon_col].values,
        candidates[lat_col].values,
    )

    best_pos = int(np.nanargmin(dists))
    best_dist = float(dists[best_pos])

    best_track_index = int(candidates.index[best_pos])
    best_track_time = candidates.iloc[best_pos]["track_time"]

    time_diff_h = abs((gpm_row["time"] - best_track_time).total_seconds()) / 3600

    if best_dist > MAX_CENTER_DISTANCE_KM:
        match_records.append(
            {
                "matched_track_index": np.nan,
                "track_time": best_track_time,
                "track_time_diff_h": time_diff_h,
                "gpm_track_center_dist_km": best_dist,
                "match_status": "too_far",
            }
        )
    else:
        match_records.append(
            {
                "matched_track_index": best_track_index,
                "track_time": best_track_time,
                "track_time_diff_h": time_diff_h,
                "gpm_track_center_dist_km": best_dist,
                "match_status": "matched",
            }
        )

match_df = pd.DataFrame(match_records)

print("\n匹配状态统计:")
print(match_df["match_status"].value_counts(dropna=False))

matched_count = int((match_df["match_status"] == "matched").sum())
total_count = len(match_df)

print("\n成功匹配数量:", matched_count)
print("GPM 总数量:", total_count)
print("匹配比例:", matched_count / total_count)

print("\n时间差 track_time_diff_h 描述统计:")
print(match_df["track_time_diff_h"].describe())

print("\n中心距离 gpm_track_center_dist_km 描述统计:")
print(match_df["gpm_track_center_dist_km"].describe())


# =========================
# 10. 拼接路径字段
# =========================

# 路径表字段加前缀，但 track_event_uid 保持不变
track_for_output = track.rename(columns={col: prefix_track_col(col) for col in track.columns})

# track_time 已经在 match_df 中保留，这里删除，避免重复列
if "track_time" in track_for_output.columns:
    track_for_output = track_for_output.drop(columns=["track_time"])

track_rows = []

for matched_index in match_df["matched_track_index"]:
    if pd.isna(matched_index):
        empty_row = pd.Series({col: np.nan for col in track_for_output.columns})
        track_rows.append(empty_row)
    else:
        track_rows.append(track_for_output.loc[int(matched_index)])

track_matched = pd.DataFrame(track_rows).reset_index(drop=True)

merged = pd.concat(
    [
        gpm.reset_index(drop=True),
        match_df.reset_index(drop=True),
        track_matched.reset_index(drop=True),
    ],
    axis=1,
)

# 只保留可靠匹配行作为后续建模主表
merged_valid = merged[merged["match_status"] == "matched"].copy()

print("\n合并前总行数:", len(merged))
print("可靠匹配后行数:", len(merged_valid))
print("可靠匹配比例:", len(merged_valid) / len(merged))


# =========================
# 11. 生成 GPM 文件夹编号与路径事件编号映射表
# =========================

mapping_cols = [
    "gpm_event_uid",
    "track_event_uid",
    "track_time",
    "track_time_diff_h",
    "gpm_track_center_dist_km",
]

for optional_col in [
    "track_typhoon_name",
    "track_typhoon_id",
    "track_typhoon_code",
    f"track_{lon_col}",
    f"track_{lat_col}",
]:
    if optional_col in merged_valid.columns:
        mapping_cols.append(optional_col)

mapping_cols = [col for col in mapping_cols if col in merged_valid.columns]

event_mapping = (
    merged_valid[mapping_cols]
    .sort_values(["gpm_event_uid", "track_event_uid", "track_time"])
    .drop_duplicates()
    .reset_index(drop=True)
)

event_mapping.to_csv(MAPPING_OUT_PATH, index=False, encoding="utf-8-sig")

print("\n事件映射表已输出:", MAPPING_OUT_PATH)
print("事件映射表示例:")
print(event_mapping.head(10))


# =========================
# 12. 输出主合并表
# =========================

# 后续建模优先按 track_event_uid 分组，因此按它排序
sort_cols = ["track_event_uid", "time"]
sort_cols = [col for col in sort_cols if col in merged_valid.columns]

merged_valid = merged_valid.sort_values(sort_cols).reset_index(drop=True)

merged_valid.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n完成输出主合并表:", OUT_PATH)
print("输出维度:", merged_valid.shape)

print("\n主表前 5 行关键字段:")

show_cols = [
    "gpm_event_uid",
    "track_event_uid",
    "time",
    "track_time",
    "track_time_diff_h",
    "center_lon",
    "center_lat",
    f"track_{lon_col}",
    f"track_{lat_col}",
    "gpm_track_center_dist_km",
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",
]

for optional_col in [
    "track_typhoon_name",
    "track_typhoon_id",
    "track_typhoon_code",
    "track_pressure",
    "track_wind",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
]:
    if optional_col in merged_valid.columns:
        show_cols.append(optional_col)

show_cols = [col for col in show_cols if col in merged_valid.columns]

print(merged_valid[show_cols].head())

print("\n主表字段数量:", len(merged_valid.columns))
print("\n主表字段列表:")
print(merged_valid.columns.tolist())