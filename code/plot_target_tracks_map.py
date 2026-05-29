from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TRACK_FILE = OUTPUT_DIR / "target_typhoon_tracks_2024.csv"
OUT_FILE = FIG_DIR / "target_typhoon_paths_map_2024.png"


def load_data():
    if not TRACK_FILE.exists():
        raise FileNotFoundError(f"找不到路径文件：{TRACK_FILE}")

    df = pd.read_csv(TRACK_FILE)
    df["time"] = pd.to_datetime(df["time"])

    if "lon_180" not in df.columns:
        df["lon_180"] = df["lon"]

    df = df.sort_values(["event_uid", "time"]).reset_index(drop=True)
    return df


def plot_map(df):
    # 地图显示范围：覆盖西北太平洋、菲律宾、南海、台湾、中国东南沿海、日本南部
    lon_min = min(df["lon_180"].min() - 5, 105)
    lon_max = max(df["lon_180"].max() + 5, 170)
    lat_min = min(df["lat"].min() - 5, 0)
    lat_max = max(df["lat"].max() + 5, 40)

    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=proj)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # 背景要素
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", zorder=1)

    # 经纬度网格
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.5,
        linestyle="--",
        alpha=0.6
    )
    gl.top_labels = False
    gl.right_labels = False

    # 两个台风分别画
    for name_norm, sub in df.groupby("name_norm"):
        sub = sub.sort_values("time")
        ty_name = sub["typhoon_name"].iloc[0]

        ax.plot(
            sub["lon_180"],
            sub["lat"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            transform=proj,
            label=ty_name,
            zorder=3
        )

        # 起点
        ax.scatter(
            sub["lon_180"].iloc[0],
            sub["lat"].iloc[0],
            marker="s",
            s=90,
            transform=proj,
            zorder=4
        )

        # 终点
        ax.scatter(
            sub["lon_180"].iloc[-1],
            sub["lat"].iloc[-1],
            marker="X",
            s=100,
            transform=proj,
            zorder=4
        )

        # 标注起点与终点
        ax.text(
            sub["lon_180"].iloc[0] + 0.5,
            sub["lat"].iloc[0] + 0.5,
            f"{ty_name} start\n{sub['time'].iloc[0].strftime('%m-%d %H')}",
            fontsize=9,
            transform=proj,
            zorder=5
        )

        ax.text(
            sub["lon_180"].iloc[-1] + 0.5,
            sub["lat"].iloc[-1] + 0.5,
            f"{ty_name} end\n{sub['time'].iloc[-1].strftime('%m-%d %H')}",
            fontsize=9,
            transform=proj,
            zorder=5
        )

        # 每隔 8 个点标注一次时间，避免文字过密
        for _, row in sub.iloc[::8].iterrows():
            ax.text(
                row["lon_180"] + 0.3,
                row["lat"] + 0.3,
                row["time"].strftime("%m-%d"),
                fontsize=8,
                transform=proj,
                zorder=5
            )

    ax.set_title(
        "Tracks of KONG-REY and MAN-YI in 2024",
        fontsize=16,
        pad=15
    )

    ax.legend(loc="lower left", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    plt.close()

    print(f"已输出论文版路径图：{OUT_FILE}")


def main():
    df = load_data()
    print("读取目标台风路径表：", TRACK_FILE)
    print("总行数：", len(df))
    print("包含台风：", df["typhoon_name"].drop_duplicates().tolist())
    plot_map(df)


if __name__ == "__main__":
    main()
