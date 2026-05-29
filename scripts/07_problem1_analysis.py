from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# =========================
# 1. 路径设置
# =========================

MERGED_PATH = Path("data/processed/gpm_track_merged_features.csv")
FIG_DIR = Path("outputs/figures/problem1")
TABLE_DIR = Path("outputs/tables/problem1")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取数据
# =========================

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

print("主表维度:", df.shape)
print("路径事件数:", df["track_event_uid"].nunique())


# =========================
# 3. 变量定义
# =========================

x_cols = [
    "track_pressure",
    "track_wind",
    "track_move_speed_kmh",
    "track_move_dir_deg",
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "gpm_track_center_dist_km",
    "track_time_diff_h",
]

y_cols = [
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

x_cols = [c for c in x_cols if c in df.columns]
y_cols = [c for c in y_cols if c in df.columns]

model_df = df[x_cols + y_cols + ["track_event_uid", "track_typhoon_name"]].copy()

# 对建模所需变量转数值
for c in x_cols + y_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

print("解释变量:", x_cols)
print("响应变量:", y_cols)


# =========================
# 4. 相关系数矩阵：路径/强度/环境 vs 降水特征
# =========================

corr = model_df[x_cols + y_cols].corr(method="spearman")
corr_xy = corr.loc[x_cols, y_cols]

corr_xy.to_csv(TABLE_DIR / "spearman_correlation_x_to_rain_features.csv", encoding="utf-8-sig")

plt.figure(figsize=(14, 7))
plt.imshow(corr_xy.values, aspect="auto")
plt.colorbar(label="Spearman correlation")
plt.xticks(range(len(y_cols)), y_cols, rotation=60, ha="right")
plt.yticks(range(len(x_cols)), x_cols)
plt.title("Spearman correlation: typhoon factors vs rainfall distribution features")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_spearman_correlation_heatmap.png", dpi=300)
plt.show()


# =========================
# 5. 风速、气压与极端降水关系
# =========================

plot_df = model_df.dropna(subset=["track_wind", "track_pressure", "rain_max", "rain_p95"])

plt.figure(figsize=(7, 5))
plt.scatter(plot_df["track_wind"], plot_df["rain_max"], s=8, alpha=0.35)
plt.xlabel("Maximum wind speed")
plt.ylabel("Maximum rainfall (mm/hr)")
plt.title("Wind speed vs maximum rainfall")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_wind_vs_rain_max.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(plot_df["track_pressure"], plot_df["rain_max"], s=8, alpha=0.35)
plt.xlabel("Central pressure")
plt.ylabel("Maximum rainfall (mm/hr)")
plt.title("Central pressure vs maximum rainfall")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_pressure_vs_rain_max.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(plot_df["track_wind"], plot_df["rain_p95"], s=8, alpha=0.35)
plt.xlabel("Maximum wind speed")
plt.ylabel("P95 rainfall (mm/hr)")
plt.title("Wind speed vs P95 rainfall")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_wind_vs_rain_p95.png", dpi=300)
plt.show()


# =========================
# 6. 移动速度与降水中心偏移、非对称性关系
# =========================

move_df = model_df.dropna(
    subset=[
        "track_move_speed_kmh",
        "centroid_offset_km",
        "asym_EW",
        "asym_NS",
        "anisotropy",
    ]
)

plt.figure(figsize=(7, 5))
plt.scatter(move_df["track_move_speed_kmh"], move_df["centroid_offset_km"], s=8, alpha=0.35)
plt.xlabel("Moving speed (km/h)")
plt.ylabel("Rainfall centroid offset (km)")
plt.title("Moving speed vs rainfall centroid offset")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_move_speed_vs_centroid_offset.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(move_df["track_move_speed_kmh"], move_df["anisotropy"], s=8, alpha=0.35)
plt.xlabel("Moving speed (km/h)")
plt.ylabel("Rainband anisotropy")
plt.title("Moving speed vs rainband anisotropy")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_move_speed_vs_anisotropy.png", dpi=300)
plt.show()


# =========================
# 7. 登陆状态与降水特征箱线图
# =========================

land_df = model_df.dropna(subset=["track_is_land", "rain_max", "rain_area_10_grid", "centroid_offset_km"])
land_df["track_is_land"] = land_df["track_is_land"].astype(int)

def boxplot_by_land(y_col, filename, ylabel):
    sea = land_df.loc[land_df["track_is_land"] == 0, y_col].dropna()
    land = land_df.loc[land_df["track_is_land"] == 1, y_col].dropna()

    plt.figure(figsize=(6, 5))
    plt.boxplot([sea, land], labels=["Sea", "Land"], showfliers=False)
    plt.ylabel(ylabel)
    plt.title(f"Landfall effect on {y_col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.show()

boxplot_by_land("rain_max", "07_landfall_box_rain_max.png", "Maximum rainfall (mm/hr)")
boxplot_by_land("rain_area_10_grid", "08_landfall_box_area10.png", "Area >= 10 mm/hr (grid cells)")
boxplot_by_land("centroid_offset_km", "09_landfall_box_centroid_offset.png", "Centroid offset (km)")


# =========================
# 8. 距岸距离与强降水范围
# =========================

coast_df = model_df.dropna(
    subset=[
        "track_signed_coast_dist_km",
        "rain_area_10_grid",
        "rain_max",
        "centroid_offset_km",
    ]
)

plt.figure(figsize=(7, 5))
plt.scatter(coast_df["track_signed_coast_dist_km"], coast_df["rain_area_10_grid"], s=8, alpha=0.35)
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Signed distance to coast (km)")
plt.ylabel("Area >= 10 mm/hr (grid cells)")
plt.title("Coast distance vs heavy rainfall area")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_coast_distance_vs_area10.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(coast_df["track_signed_coast_dist_km"], coast_df["rain_max"], s=8, alpha=0.35)
plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Signed distance to coast (km)")
plt.ylabel("Maximum rainfall (mm/hr)")
plt.title("Coast distance vs maximum rainfall")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_coast_distance_vs_rain_max.png", dpi=300)
plt.show()


# =========================
# 9. 分强度等级汇总
# =========================

if "track_intensity" in df.columns:
    intensity_summary = (
        df.groupby("track_intensity")[y_cols]
        .agg(["mean", "median", "std", "count"])
    )
    intensity_summary.to_csv(TABLE_DIR / "rain_features_by_intensity.csv", encoding="utf-8-sig")
    print("\n已输出强度等级分组统计: rain_features_by_intensity.csv")


# =========================
# 10. 随机森林特征重要性
# =========================

rf_targets = [
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "anisotropy",
]

rf_targets = [c for c in rf_targets if c in model_df.columns]

importance_records = []

for target in rf_targets:
    sub = model_df[x_cols + [target]].dropna().copy()

    if len(sub) < 100:
        print(f"跳过 {target}：有效样本太少")
        continue

    X = sub[x_cols]
    y = sub[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    score_train = rf.score(X_train, y_train)
    score_test = rf.score(X_test, y_test)

    result = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    imp = pd.DataFrame(
        {
            "target": target,
            "feature": x_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "train_r2": score_train,
            "test_r2": score_test,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_records.append(imp)

    imp.to_csv(TABLE_DIR / f"rf_importance_{target}.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    top = imp.head(10).iloc[::-1]
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.title(f"Random forest feature importance: {target}\nTest R2={score_test:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"12_rf_importance_{target}.png", dpi=300)
    plt.show()

if importance_records:
    all_importance = pd.concat(importance_records, ignore_index=True)
    all_importance.to_csv(TABLE_DIR / "rf_importance_all_targets.csv", index=False, encoding="utf-8-sig")
    print("\n已输出随机森林特征重要性表: rf_importance_all_targets.csv")


# =========================
# 11. 输出变量相关性排序表
# =========================

corr_records = []

for y in y_cols:
    for x in x_cols:
        corr_records.append(
            {
                "rain_feature": y,
                "typhoon_factor": x,
                "spearman_corr": corr_xy.loc[x, y],
                "abs_corr": abs(corr_xy.loc[x, y]),
            }
        )

corr_rank = pd.DataFrame(corr_records).sort_values(
    ["rain_feature", "abs_corr"],
    ascending=[True, False],
)

corr_rank.to_csv(TABLE_DIR / "spearman_correlation_rank.csv", index=False, encoding="utf-8-sig")

print("\n已输出所有问题1分析图到:", FIG_DIR)
print("已输出所有问题1分析表到:", TABLE_DIR)
print("\n完成。")