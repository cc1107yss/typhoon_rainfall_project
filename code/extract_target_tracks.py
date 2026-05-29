from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "output" / "typhoon_track_features_basic.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_name(x):
    """
    统一英文名格式：
    KONG-REY / KONG_REY / KONGREY -> KONGREY
    MAN-YI / MAN_YI / MANYI -> MANYI
    """
    if pd.isna(x):
        return ""
    x = str(x).upper()
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x


def save_target(df, target_name, start_date, end_date, output_name):
    """
    按台风名称 + 时间范围筛选目标台风路径。
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    sub = df[
        (df["source_file"] == "CH2024BST.txt")
        & (df["name_norm"] == target_name)
        & (df["time"] >= start)
        & (df["time"] <= end)
    ].copy()

    sub = sub.sort_values("time")

    out_file = OUTPUT_DIR / output_name
    sub.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print(f"目标台风：{target_name}")
    print(f"理论时间范围：{start_date} 至 {end_date}")
    print(f"筛选行数：{len(sub)}")
    print(f"输出文件：{out_file}")

    if len(sub) > 0:
        print("实际起止时间：", sub["time"].min(), "至", sub["time"].max())
        print("事件编号 event_uid：", sub["event_uid"].drop_duplicates().tolist())
        print("台风编号 typhoon_id：", sub["typhoon_id"].drop_duplicates().tolist())
        print("英文名 typhoon_name：", sub["typhoon_name"].drop_duplicates().tolist())
        print("\n前 5 行：")
        print(sub[[
            "event_uid", "typhoon_id", "storm_seq", "typhoon_name",
            "time", "lat", "lon", "lon_180",
            "intensity", "pressure", "wind",
            "move_speed_kmh", "move_dir_deg",
            "wind_change_rate", "pressure_change_rate"
        ]].head().to_string(index=False))

        print("\n后 5 行：")
        print(sub[[
            "event_uid", "typhoon_id", "storm_seq", "typhoon_name",
            "time", "lat", "lon", "lon_180",
            "intensity", "pressure", "wind",
            "move_speed_kmh", "move_dir_deg",
            "wind_change_rate", "pressure_change_rate"
        ]].tail().to_string(index=False))
    else:
        print("警告：没有筛选到记录。请检查台风英文名是否写作 KONGREY/MANYI 或时间范围是否匹配。")

    return sub


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df["name_norm"] = df["typhoon_name"].apply(normalize_name)

    print("读取路径特征表：", INPUT_FILE)
    print("总行数：", len(df))
    print("2024 年文件记录数：", (df["source_file"] == "CH2024BST.txt").sum())

    print("\n2024 年台风英文名列表：")
    names_2024 = (
        df[df["source_file"] == "CH2024BST.txt"]
        [["event_uid", "typhoon_id", "storm_seq", "typhoon_name", "name_norm"]]
        .drop_duplicates()
        .sort_values("storm_seq")
    )
    print(names_2024.to_string(index=False))

    kong_rey = save_target(
        df=df,
        target_name="KONGREY",
        start_date="2024-10-24 00:00:00",
        end_date="2024-11-02 23:59:59",
        output_name="kong_rey_track.csv"
    )

    man_yi = save_target(
        df=df,
        target_name="MANYI",
        start_date="2024-11-08 00:00:00",
        end_date="2024-11-20 23:59:59",
        output_name="man_yi_track.csv"
    )

    combined = pd.concat([kong_rey, man_yi], ignore_index=True)
    combined_file = OUTPUT_DIR / "target_typhoon_tracks_2024.csv"
    combined.to_csv(combined_file, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("合并目标台风路径表已输出：", combined_file)
    print("合并总行数：", len(combined))

    if len(combined) > 0:
        print("\n按台风统计：")
        print(combined.groupby("name_norm").agg(
            rows=("time", "count"),
            start_time=("time", "min"),
            end_time=("time", "max"),
            min_lat=("lat", "min"),
            max_lat=("lat", "max"),
            min_lon=("lon", "min"),
            max_lon=("lon", "max"),
            min_pressure=("pressure", "min"),
            max_wind=("wind", "max")
        ))


if __name__ == "__main__":
    main()
