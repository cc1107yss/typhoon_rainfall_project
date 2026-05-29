from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TRACK_FILE = OUTPUT_DIR / "target_typhoon_tracks_2024.csv"


def load_tracks():
    if not TRACK_FILE.exists():
        raise FileNotFoundError(f"找不到文件：{TRACK_FILE}")

    df = pd.read_csv(TRACK_FILE)
    df["time"] = pd.to_datetime(df["time"])

    # 后续画图统一用 lon_180；如果没有 lon_180，则退回 lon
    if "lon_180" not in df.columns:
        df["lon_180"] = df["lon"]

    df = df.sort_values(["event_uid", "time"]).reset_index(drop=True)
    return df


def plot_single_track(df, name_norm, output_name):
    sub = df[df["name_norm"] == name_norm].copy()
    if len(sub) == 0:
        print(f"没有找到 {name_norm}")
        return

    sub = sub.sort_values("time")

    title_name = sub["typhoon_name"].iloc[0]
    start_time = sub["time"].min()
    end_time = sub["time"].max()

    plt.figure(figsize=(8, 7))

    # 路径线
    plt.plot(
        sub["lon_180"],
        sub["lat"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=f"{title_name} track"
    )

    # 起点和终点
    plt.scatter(
        sub["lon_180"].iloc[0],
        sub["lat"].iloc[0],
        s=80,
        marker="s",
        label="Start"
    )
    plt.scatter(
        sub["lon_180"].iloc[-1],
        sub["lat"].iloc[-1],
        s=80,
        marker="X",
        label="End"
    )

    # 每隔 4 个点标注一次时间，避免太密
    for i, row in sub.iloc[::4].iterrows():
        label = row["time"].strftime("%m-%d %H")
        plt.text(
            row["lon_180"] + 0.2,
            row["lat"] + 0.2,
            label,
            fontsize=8
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"{title_name} Track\n"
        f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.    # 检查图不强制等比例，避免经纬跨度差异导致图像空白过大
    # plt.axis("equal")    plt.xlim(lon_min - 2, lon_max + 2)
    plt.ylim(lat_min - 2, lat_max + 2)axis("equal")

    # 自动留边
    lon_min, lon_max = sub["lon_180"].min(), sub["lon_180"].max()
    lat_min, lat_max = sub["lat"].min(), sub["lat"].max()
    plt.xlim(lon_min - 3, lon_max + 3)
    plt.ylim(lat_min - 3, lat_max + 3)

    out_path = FIG_DIR / output_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"已输出：{out_path}")


def plot_combined_tracks(df):
    plt.figure(figsize=(10, 8))

    for name_norm, sub in df.groupby("name_norm"):
        sub = sub.sort_values("time")
        title_name = sub["typhoon_name"].iloc[0]

        plt.plot(
            sub["lon_180"],
            sub["lat"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=title_name
        )

        # 起点终点
        plt.scatter(sub["lon_180"].iloc[0], sub["lat"].iloc[0], s=70, marker="s")
        plt.scatter(sub["lon_180"].iloc[-1], sub["lat"].iloc[-1], s=70, marker="X")

        # 标注台风名
        plt.text(
            sub["lon_180"].iloc[0] + 0.3,
            sub["lat"].iloc[0] + 0.3,
            f"{title_name} start",
            fontsize=9
        )
        plt.text(
            sub["lon_180"].iloc[-1] + 0.3,
            sub["lat"].iloc[-1] + 0.3,
            f"{title_name} end",
            fontsize=9
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Tracks of KONG-REY and MAN-YI in 2024")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.axis("equal")

    lon_min, lon_max = df["lon_180"].min(), df["lon_180"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    plt.xlim(lon_min - 5, lon_max + 5)
    plt.ylim(lat_min - 5, lat_max + 5)

    out_path = FIG_DIR / "target_typhoon_paths_2024.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"已输出：{out_path}")


def plot_intensity_time_series(df):
    """
    额外生成一张强度演变图：
    上半部分：最大风速
    下半部分：中心气压
    """
    for name_norm, sub in df.groupby("name_norm"):
        sub = sub.sort_values("time")
        title_name = sub["typhoon_name"].iloc[0]

        plt.figure(figsize=(10, 6))

        plt.plot(
            sub["time"],
            sub["wind"],
            marker="o",
            linewidth=1.5,
            label="Max wind speed"
        )
        plt.xlabel("Time")
        plt.ylabel("Max wind speed (m/s)")
        plt.title(f"{title_name} Wind Speed Evolution")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_path = FIG_DIR / f"{name_norm.lower()}_wind_timeseries.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"已输出：{out_path}")

        plt.figure(figsize=(10, 6))

        plt.plot(
            sub["time"],
            sub["pressure"],
            marker="o",
            linewidth=1.5,
            label="Central pressure"
        )
        plt.xlabel("Time")
        plt.ylabel("Central pressure (hPa)")
        plt.title(f"{title_name} Central Pressure Evolution")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_path = FIG_DIR / f"{name_norm.lower()}_pressure_timeseries.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"已输出：{out_path}")


def main():
    df = load_tracks()

    print("读取目标台风路径表：", TRACK_FILE)
    print("总行数：", len(df))
    print("包含台风：", df["typhoon_name"].drop_duplicates().tolist())

    plot_single_track(df, "KONGREY", "kong_rey_path.png")
    plot_single_track(df, "MANYI", "man_yi_path.png")
    plot_combined_tracks(df)
    plot_intensity_time_series(df)

    print("\n全部图片输出目录：", FIG_DIR)


if __name__ == "__main__":
    main()

