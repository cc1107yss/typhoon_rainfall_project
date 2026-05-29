from pathlib import Path
import pandas as pd

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union, transform
from pyproj import Transformer


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"

INPUT_ALL = OUTPUT_DIR / "typhoon_track_features_basic.csv"
INPUT_TARGET = OUTPUT_DIR / "target_typhoon_tracks_2024.csv"

OUTPUT_ALL = OUTPUT_DIR / "typhoon_track_features_with_coast.csv"
OUTPUT_TARGET = OUTPUT_DIR / "target_typhoon_tracks_2024_with_coast.csv"


def load_natural_earth_geometries():
    """
    使用 Cartopy 自带的 Natural Earth 数据：
    - land：陆地多边形，用于判断点是否在陆地上；
    - coastline：海岸线折线，用于计算距海岸线距离。
    """
    print("正在读取 Natural Earth 陆地边界和海岸线数据...")

    land_path = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land"
    )
    coast_path = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="coastline"
    )

    land_records = list(shpreader.Reader(land_path).geometries())
    coast_records = list(shpreader.Reader(coast_path).geometries())

    land_union = unary_union(land_records)
    coast_union = unary_union(coast_records)

    print("陆地多边形数量：", len(land_records))
    print("海岸线对象数量：", len(coast_records))

    return land_union, coast_union


def build_projected_geometry(coast_union):
    """
    为计算距离，将经纬度坐标投影到米制坐标 EPSG:3857。
    对本题西北太平洋—中国沿海区域，作为距岸距离近似足够使用。
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    coast_union_m = transform(transformer.transform, coast_union)
    return transformer, coast_union_m


def add_coast_features(df, land_union, coast_union_m, transformer):
    """
    为路径表逐点添加：
    - is_land
    - coast_dist_km
    - signed_coast_dist_km
    """
    df = df.copy()

    if "lon_180" not in df.columns:
        df["lon_180"] = df["lon"].where(df["lon"] <= 180, df["lon"] - 360)

    is_land_list = []
    dist_list = []

    total = len(df)

    for idx, row in df.iterrows():
        lon = float(row["lon_180"])
        lat = float(row["lat"])

        point_ll = Point(lon, lat)

        # 判断是否在陆地上。covers 比 contains 更稳，因为边界点也算陆地。
        is_land = land_union.covers(point_ll)

        x, y = transformer.transform(lon, lat)
        point_m = Point(x, y)

        dist_km = point_m.distance(coast_union_m) / 1000.0

        is_land_list.append(1 if is_land else 0)
        dist_list.append(dist_km)

        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1}/{total} 行")

    df["is_land"] = is_land_list
    df["coast_dist_km"] = dist_list
    df["signed_coast_dist_km"] = df["coast_dist_km"]
    df.loc[df["is_land"] == 1, "signed_coast_dist_km"] *= -1

    return df


def summarize(df, name):
    print("\n==============================")
    print(name)
    print("行数：", len(df))
    print("陆地点数量 is_land=1：", int(df["is_land"].sum()))
    print("海上点数量 is_land=0：", int((df["is_land"] == 0).sum()))
    print("距岸距离统计 km：")
    print(df[["coast_dist_km", "signed_coast_dist_km"]].describe())

    if "typhoon_name" in df.columns:
        print("\n按台风统计：")
        print(df.groupby("typhoon_name").agg(
            rows=("time", "count"),
            land_points=("is_land", "sum"),
            min_coast_dist=("coast_dist_km", "min"),
            mean_coast_dist=("coast_dist_km", "mean"),
            max_coast_dist=("coast_dist_km", "max")
        ))


def main():
    if not INPUT_ALL.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_ALL}")

    land_union, coast_union = load_natural_earth_geometries()
    transformer, coast_union_m = build_projected_geometry(coast_union)

    print("\n正在处理全量路径表...")
    df_all = pd.read_csv(INPUT_ALL)
    df_all_with = add_coast_features(df_all, land_union, coast_union_m, transformer)
    df_all_with.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")
    print("已输出：", OUTPUT_ALL)
    summarize(df_all_with, "全量路径表 coast 特征检查")

    if INPUT_TARGET.exists():
        print("\n正在处理 KONG-REY 与 MAN-YI 目标路径表...")
        df_target = pd.read_csv(INPUT_TARGET)
        df_target_with = add_coast_features(df_target, land_union, coast_union_m, transformer)
        df_target_with.to_csv(OUTPUT_TARGET, index=False, encoding="utf-8-sig")
        print("已输出：", OUTPUT_TARGET)
        summarize(df_target_with, "目标台风 coast 特征检查")
    else:
        print("未找到目标台风路径表，跳过：", INPUT_TARGET)


if __name__ == "__main__":
    main()
