from pathlib import Path
import re
import math
import pandas as pd
import numpy as np


# =========================
# 1. 路径设置
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "CMABSTdata"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "typhoon_track_features_basic.csv"


# =========================
# 2. 工具函数
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Haversine 球面距离公式。
    输入：纬度、经度，单位为度。
    输出：两点距离，单位为 km。
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    计算移动方向角。
    0° 表示向北，90° 表示向东，180° 表示向南，270° 表示向西。
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    )

    theta = math.degrees(math.atan2(x, y))
    return (theta + 360) % 360


def parse_header_line(parts):
    """
    解析 66666 开头的台风头行。

    以 CH2024BST.txt 中一行为例：
    66666 2401   38 0001 2401 0 6 EWINIAR 20250301

    字段解释：
    parts[1] = typhoon_id，台风编号；无名台风可能为 0000
    parts[2] = record_count，该台风路径记录条数
    parts[3] = storm_seq，该年份内事件序号
    parts[4] = typhoon_code，通常与 typhoon_id 相同；无名台风可能为 0000
    parts[7] = typhoon_name，英文名
    """
    typhoon_id = parts[1] if len(parts) > 1 else None
    record_count = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    storm_seq = parts[3] if len(parts) > 3 else None
    typhoon_code = parts[4] if len(parts) > 4 else None

    if len(parts) > 7:
        typhoon_name = parts[7].strip().upper()
        typhoon_name = typhoon_name.replace("(", "").replace(")", "")
    else:
        typhoon_name = None

    if typhoon_name in ["", None]:
        typhoon_name = "UNKNOWN"

    return typhoon_id, record_count, storm_seq, typhoon_code, typhoon_name


def parse_one_file(file_path):
    """
    解析单个 CH20xxBST.txt 文件。
    常见 CMABST 数据行格式：
    YYYYMMDDHH  grade  lat  lon  pressure  wind  ...
    其中 lat/lon 常以 0.1 度为单位，所以需要除以 10。
    """
    rows = []
    current_id = None
    current_name = None
    current_record_count = None
    current_storm_seq = None
    current_typhoon_code = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)

            # 台风头行
            if parts[0] == "66666":
                (
                    current_id,
                    current_record_count,
                    current_storm_seq,
                    current_typhoon_code,
                    current_name
                ) = parse_header_line(parts)
                continue

            # 数据行：第一列是 10 位时间 YYYYMMDDHH
            if re.fullmatch(r"\d{10}", parts[0]):
                if len(parts) < 6:
                    print(f"[跳过] 字段不足：{file_path.name} 第 {line_no} 行：{line}")
                    continue

                try:
                    time = pd.to_datetime(parts[0], format="%Y%m%d%H")
                    intensity = int(parts[1])
                    lat = int(parts[2]) / 10.0
                    lon = int(parts[3]) / 10.0
                    pressure = int(parts[4])
                    wind = int(parts[5])

                    rows.append({
                        "source_file": file_path.name,
                        "typhoon_id": current_id,
                        "storm_seq": current_storm_seq,
                        "typhoon_code": current_typhoon_code,
                        "record_count": current_record_count,
                        "typhoon_name": current_name,
                        "time": time,
                        "intensity": intensity,
                        "lat": lat,
                        "lon": lon,
                        "pressure": pressure,
                        "wind": wind,
                    })

                except Exception as e:
                    print(f"[跳过] 解析失败：{file_path.name} 第 {line_no} 行：{line}")
                    print(f"原因：{e}")

    return pd.DataFrame(rows)


def add_derived_features(df):
    """
    增加派生变量：
    - 移动速度 move_speed_kmh
    - 移动方向 move_dir_deg
    - 风速变化率 wind_change_rate
    - 气压变化率 pressure_change_rate
    """
    df = df.copy()
    df["lon_180"] = df["lon"].where(df["lon"] <= 180, df["lon"] - 360)
    # 若 typhoon_id 解析不稳定，用 source_file + typhoon_name 共同分组更稳
        # 构造唯一台风事件编号，避免多个无名台风 typhoon_id 都是 0000 而被错误合并
    df["event_uid"] = (
        df["source_file"].astype(str).str.replace("BST.txt", "", regex=False)
        + "_"
        + df["storm_seq"].astype(str).str.zfill(4)
    )

    df["group_key"] = df["event_uid"]

    df = df.sort_values(["group_key", "time"]).reset_index(drop=True)

    all_groups = []

    for _, g in df.groupby("group_key"):
        g = g.sort_values("time").copy()

        g["prev_time"] = g["time"].shift(1)
        g["prev_lat"] = g["lat"].shift(1)
        g["prev_lon"] = g["lon"].shift(1)
        g["prev_wind"] = g["wind"].shift(1)
        g["prev_pressure"] = g["pressure"].shift(1)

        g["dt_h"] = (g["time"] - g["prev_time"]).dt.total_seconds() / 3600

        distances = []
        bearings = []

        for _, row in g.iterrows():
            if pd.isna(row["dt_h"]) or row["dt_h"] <= 0:
                distances.append(np.nan)
                bearings.append(np.nan)
            else:
                d = haversine_km(
                    row["prev_lat"], row["prev_lon"],
                    row["lat"], row["lon"]
                )
                b = bearing_deg(
                    row["prev_lat"], row["prev_lon"],
                    row["lat"], row["lon"]
                )
                distances.append(d)
                bearings.append(b)

        g["move_distance_km"] = distances
        g["move_speed_kmh"] = g["move_distance_km"] / g["dt_h"]
        g["move_dir_deg"] = bearings

        g["wind_change_rate"] = (g["wind"] - g["prev_wind"]) / g["dt_h"]
        g["pressure_change_rate"] = (g["pressure"] - g["prev_pressure"]) / g["dt_h"]

        all_groups.append(g)

    df2 = pd.concat(all_groups, ignore_index=True)

    # 删除辅助列，保留基本建模字段
    keep_cols = [
        "event_uid",
        "source_file",
        "typhoon_id",
        "storm_seq",
        "typhoon_code",
        "record_count",
        "typhoon_name",
        "time",
        "intensity",
        "lat",
        "lon",
        "lon_180",
        "pressure",
        "wind",
        "dt_h",
        "move_distance_km",
        "move_speed_kmh",
        "move_dir_deg",
        "wind_change_rate",
        "pressure_change_rate",
    ]

    return df2[keep_cols]


# =========================
# 3. 主程序
# =========================
def main():
    txt_files = sorted(DATA_DIR.glob("CH20*BST.txt"))

    print("数据目录：", DATA_DIR)
    print("找到 txt 文件数：", len(txt_files))

    if len(txt_files) == 0:
        raise FileNotFoundError(
            f"没有在 {DATA_DIR} 找到 CH20*BST.txt，请检查文件夹路径。"
        )

    dfs = []
    for fp in txt_files:
        print("正在读取：", fp.name)
        df_one = parse_one_file(fp)
        print(f"  解析记录数：{len(df_one)}")
        dfs.append(df_one)

    raw_df = pd.concat(dfs, ignore_index=True)

    print("\n原始路径记录总数：", len(raw_df))
    print("年份文件：", raw_df["source_file"].nunique())
    print("台风名称示例：")
    print(raw_df["typhoon_name"].dropna().unique()[:20])

    feature_df = add_derived_features(raw_df)

    feature_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n已输出：", OUTPUT_FILE)
    print("输出行数：", len(feature_df))
    print("输出列：")
    print(feature_df.columns.tolist())

    print("\n前 10 行预览：")
    print(feature_df.head(10))


if __name__ == "__main__":
    main()