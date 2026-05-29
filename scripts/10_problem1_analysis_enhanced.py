from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/typhoon_rainfall_matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =========================
# 1. 路径设置
# =========================

FEATURE_PATH = Path("data/processed/env_added/gpm_track_model_features_interp_env.csv")

FIG_DIR = Path("outputs/figures/problem1_env")
TABLE_DIR = Path("outputs/tables/problem1_env")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    labels = {
        "rainband_width_km": "雨带横向等效带宽 $B$ / km",
        "rainband_length_km": "雨带主轴等效长度 $L$ / km",
        "rainband_width10_km": "强降水雨带带宽 $B_{10}$ / km",
    }
    paper = summary[summary["metric"].isin(labels)].copy()
    paper["metric"] = paper["metric"].map(labels)
    for col in ["mean", "median", "q25", "q75"]:
        paper[col] = pd.to_numeric(paper[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    paper["n_valid"] = paper["n_valid"].astype(int)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{降水带宽指标统计}",
        "\\label{tab:rainband_width_stats_problem1}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "指标 & 有效样本 & 均值 & 中位数 & Q25 & Q75 \\\\",
        "\\midrule",
    ]
    for _, row in paper.iterrows():
        lines.append(
            f"{row['metric']} & {row['n_valid']} & {row['mean']} & {row['median']} & {row['q25']} & {row['q75']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rainband_distribution(model_df: pd.DataFrame) -> None:
    if "rainband_width_km" not in model_df.columns:
        return
    s = pd.to_numeric(model_df["rainband_width_km"], errors="coerce").dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.hist(s, bins=45, color="#4C78A8", alpha=0.78)
    q25, median, q75 = s.quantile([0.25, 0.50, 0.75])
    ax.axvline(median, color="#B33A3A", linewidth=2, label=f"Median = {median:.1f} km")
    ax.axvline(q25, color="#555555", linewidth=1.3, linestyle="--", label=f"Q25/Q75 = {q25:.1f}/{q75:.1f} km")
    ax.axvline(q75, color="#555555", linewidth=1.3, linestyle="--")
    ax.set_xlabel("Rainband width B (km)")
    ax.set_ylabel("Frequency")
    ax.set_title("Historical rainband width distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rainband_width_distribution.png")
    plt.close(fig)


def plot_rainband_relationships(model_df: pd.DataFrame) -> None:
    if "rainband_width_km" not in model_df.columns:
        return
    candidates = [
        ("track_move_speed_kmh", "Moving speed (km/h)"),
        ("track_wind", "WND / maximum wind"),
        ("track_pressure", "PRES / central pressure"),
        ("intensity_index", "Intensity index"),
        ("landfrac_500km", "Land fraction within 500 km"),
        ("terrain_std_300km", "Terrain std within 300 km"),
    ]
    pairs = [(col, label) for col, label in candidates if col in model_df.columns]
    if not pairs:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=180)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, pairs):
        sub = model_df[[col, "rainband_width_km"]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub["rainband_width_km"] = pd.to_numeric(sub["rainband_width_km"], errors="coerce")
        sub = sub.dropna()
        ax.scatter(sub[col], sub["rainband_width_km"], s=7, alpha=0.28, color="#2F6F9F", edgecolors="none")
        ax.set_xlabel(label)
        ax.set_ylabel("B (km)")
        ax.grid(alpha=0.22)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Rainband width relationships", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rainband_width_relationships.png")
    plt.close(fig)


def group_split_metrics(data: pd.DataFrame, x_cols: list[str], target: str) -> dict:
    if target not in data.columns or "track_event_uid" not in data.columns:
        return {}
    sub = data[x_cols + [target, "track_event_uid"]].dropna().copy()
    if len(sub) < 100 or sub["track_event_uid"].nunique() < 4:
        return {}
    X = sub[x_cols]
    y = sub[target]
    groups = sub["track_event_uid"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "group_split_r2": r2_score(y_test, pred),
        "group_split_mae": mean_absolute_error(y_test, pred),
        "group_split_rmse": mean_squared_error(y_test, pred) ** 0.5,
        "group_n_test_events": int(groups.iloc[test_idx].nunique()),
    }


def write_rainband_width_outputs(model_df: pd.DataFrame, x_cols: list[str], metric_df: pd.DataFrame, all_imp: pd.DataFrame) -> None:
    width_metrics = [
        "rainband_width_km",
        "rainband_length_km",
        "rainband_aspect_ratio",
        "rainband_width10_km",
        "rainband_length10_km",
        "rainband_aspect_ratio10",
    ]
    rows = []
    for metric in width_metrics:
        if metric not in model_df.columns:
            continue
        s = pd.to_numeric(model_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "n_valid": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)),
                "median": float(s.median(skipna=True)),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "problem1_rainband_width_summary.csv", index=False, encoding="utf-8-sig")
    write_latex_table(summary, TABLE_DIR / "problem1_rainband_width_latex_table.tex")
    plot_rainband_distribution(model_df)
    plot_rainband_relationships(model_df)

    target = "rainband_width_km"
    rf_rows = []
    if not metric_df.empty and target in set(metric_df["target"]):
        random_row = metric_df.loc[metric_df["target"].eq(target)].iloc[0].to_dict()
        group_row = group_split_metrics(model_df, x_cols, target)
        rf_rows.append(
            {
                "target": target,
                "n_samples": int(random_row["n_train"] + random_row["n_test"]),
                "random_split_r2": float(random_row["test_r2"]),
                "random_split_mae": float(random_row["test_mae"]),
                "random_split_rmse": float(random_row["test_rmse"]),
                "group_split_r2": group_row.get("group_split_r2", np.nan),
                "group_split_mae": group_row.get("group_split_mae", np.nan),
                "group_split_rmse": group_row.get("group_split_rmse", np.nan),
                "group_n_test_events": group_row.get("group_n_test_events", np.nan),
            }
        )
    pd.DataFrame(rf_rows).to_csv(
        TABLE_DIR / "problem1_rainband_width_rf_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not all_imp.empty and target in set(all_imp["target"]):
        top = all_imp.loc[all_imp["target"].eq(target)].sort_values("importance_mean", ascending=False).head(10).copy()
        top.insert(1, "rank", np.arange(1, len(top) + 1))
        top_out = top.rename(columns={"importance_mean": "importance"})[["target", "rank", "feature", "importance"]]
        top_out.to_csv(
            TABLE_DIR / "problem1_rainband_width_top_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)
        plot = top.iloc[::-1]
        ax.barh(plot["feature"], plot["importance_mean"], color="#3A7D74")
        ax.set_xlabel("Permutation importance")
        ax.set_title("Rainband width RF Top 10 features")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rainband_width_feature_importance.png")
        plt.close(fig)


# =========================
# 2. 读取数据
# =========================

df = pd.read_csv(
    FEATURE_PATH,
    parse_dates=["time", "time_end"],
    dtype={
        "gpm_event_uid": str,
        "track_event_uid": str,
        "track_typhoon_id": str,
        "track_typhoon_code": str,
    },
)

print("增强主表维度:", df.shape)
print("路径事件数:", df["track_event_uid"].nunique())


# =========================
# 3. 解释变量与响应变量
# =========================
# 增强版说明：
# 1. 不再直接使用 track_move_dir_deg，而使用 sin/cos；
# 2. 加入 intensity_index 表示综合强度；
# 3. 加入 coast_influence_exp 和 near_coast 状态；
# 4. 加入 hour/month 周期特征，控制日变化和季节性；
# 5. 加入 land fraction 与 terrain 变量，描述外部环境背景。

env_cols = [
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",
]

x_cols = [
    # 强度因素
    "track_wind",
    "track_pressure",
    "pressure_deficit",
    "intensity_index",

    # 移动因素
    "track_move_speed_kmh",
    "track_move_dir_sin",
    "track_move_dir_cos",

    # 强度变化因素
    "track_wind_change_rate",
    "track_pressure_change_rate",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",

    # 海陆与距岸因素
    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",

    # 外部环境变量
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",

    # 时间背景因素
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # 匹配误差控制变量
    "interp_center_error_km",
]

y_cols = [
    "rain_max",
    "rain_p95",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "centroid_offset_km",
    "anisotropy",
    "rainband_width_km",
]

x_cols = [c for c in x_cols if c in df.columns]
y_cols = [c for c in y_cols if c in df.columns]

rainband_diagnostic_cols = [
    "rainband_length_km",
    "rainband_aspect_ratio",
    "rainband_width10_km",
    "rainband_length10_km",
    "rainband_aspect_ratio10",
]
rainband_diagnostic_cols = [c for c in rainband_diagnostic_cols if c in df.columns and c not in y_cols]

model_df = df[x_cols + y_cols + rainband_diagnostic_cols + ["track_event_uid", "track_typhoon_name"]].copy()

for c in x_cols + y_cols + rainband_diagnostic_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

print("\n解释变量数量:", len(x_cols))
print(x_cols)

print("\n响应变量数量:", len(y_cols))
print(y_cols)


# =========================
# 4. Spearman 相关性热力图
# =========================

corr = model_df[x_cols + y_cols].corr(method="spearman")
corr_xy = corr.loc[x_cols, y_cols]

corr_xy.to_csv(
    TABLE_DIR / "enhanced_spearman_correlation_x_to_rain_features.csv",
    encoding="utf-8-sig",
)

env_cols_existing = [c for c in env_cols if c in corr_xy.index]
main_rain_cols = [c for c in y_cols if c in corr_xy.columns]
if env_cols_existing and main_rain_cols:
    env_corr = corr_xy.loc[env_cols_existing, main_rain_cols]
    env_corr.to_csv(
        TABLE_DIR / "env_spearman_to_main_rain_features.csv",
        encoding="utf-8-sig",
    )

    env_corr_long = (
        env_corr.reset_index(names="env_feature")
        .melt(
            id_vars="env_feature",
            var_name="rain_feature",
            value_name="spearman_corr",
        )
    )
    env_corr_long["abs_corr"] = env_corr_long["spearman_corr"].abs()
    env_corr_long = env_corr_long.sort_values(
        ["rain_feature", "abs_corr"],
        ascending=[True, False],
    )
    env_corr_long.to_csv(
        TABLE_DIR / "env_spearman_to_main_rain_features_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

plt.figure(figsize=(16, 9))
plt.imshow(corr_xy.values, aspect="auto")
plt.colorbar(label="Spearman correlation")
plt.xticks(range(len(y_cols)), y_cols, rotation=60, ha="right")
plt.yticks(range(len(x_cols)), x_cols)
plt.title("Enhanced Spearman correlation: typhoon factors vs rainfall features")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_enhanced_spearman_heatmap.png", dpi=300)
plt.close()


# =========================
# 5. 相关性排序表
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

corr_rank.to_csv(
    TABLE_DIR / "enhanced_spearman_correlation_rank.csv",
    index=False,
    encoding="utf-8-sig",
)

corr_top5 = (
    corr_rank.sort_values(["rain_feature", "abs_corr"], ascending=[True, False])
    .groupby("rain_feature")
    .head(5)
    .reset_index(drop=True)
)

corr_top5.to_csv(
    TABLE_DIR / "enhanced_spearman_top5_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 6. 典型关系图
# =========================

# 综合强度 vs P95 降水
if {"intensity_index", "rain_p95"}.issubset(model_df.columns):
    sub = model_df.dropna(subset=["intensity_index", "rain_p95"])

    plt.figure(figsize=(7, 5))
    plt.scatter(sub["intensity_index"], sub["rain_p95"], s=8, alpha=0.35)
    plt.xlabel("Intensity index")
    plt.ylabel("P95 rainfall (mm/hr)")
    plt.title("Intensity index vs P95 rainfall")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_intensity_index_vs_rain_p95.png", dpi=300)
    plt.close()

# 综合强度 vs 强降水面积
if {"intensity_index", "rain_area_10_km2"}.issubset(model_df.columns):
    sub = model_df.dropna(subset=["intensity_index", "rain_area_10_km2"])

    plt.figure(figsize=(7, 5))
    plt.scatter(sub["intensity_index"], sub["rain_area_10_km2"], s=8, alpha=0.35)
    plt.xlabel("Intensity index")
    plt.ylabel("Area >= 10 mm/hr (km²)")
    plt.title("Intensity index vs heavy rainfall area")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_intensity_index_vs_area10_km2.png", dpi=300)
    plt.close()

# 近岸影响 vs 强降水面积
if {"coast_influence_exp", "rain_area_10_km2"}.issubset(model_df.columns):
    sub = model_df.dropna(subset=["coast_influence_exp", "rain_area_10_km2"])

    plt.figure(figsize=(7, 5))
    plt.scatter(sub["coast_influence_exp"], sub["rain_area_10_km2"], s=8, alpha=0.35)
    plt.xlabel("Coast influence index")
    plt.ylabel("Area >= 10 mm/hr (km²)")
    plt.title("Coast influence vs heavy rainfall area")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_coast_influence_vs_area10_km2.png", dpi=300)
    plt.close()

# 移动速度 vs 降水质心偏移
if {"track_move_speed_kmh", "centroid_offset_km"}.issubset(model_df.columns):
    sub = model_df.dropna(subset=["track_move_speed_kmh", "centroid_offset_km"])

    plt.figure(figsize=(7, 5))
    plt.scatter(sub["track_move_speed_kmh"], sub["centroid_offset_km"], s=8, alpha=0.35)
    plt.xlabel("Moving speed (km/h)")
    plt.ylabel("Rainfall centroid offset (km)")
    plt.title("Moving speed vs rainfall centroid offset")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_move_speed_vs_centroid_offset.png", dpi=300)
    plt.close()

# 相对移动方向的降水质心位置分布
if "centroid_relative_to_motion_deg" in model_df.columns:
    sub = model_df.dropna(subset=["centroid_relative_to_motion_deg"])

    plt.figure(figsize=(7, 5))
    plt.hist(sub["centroid_relative_to_motion_deg"], bins=36)
    plt.xlabel("Centroid direction relative to motion (degree)")
    plt.ylabel("Frequency")
    plt.title("Rainfall centroid direction relative to typhoon motion")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_centroid_relative_to_motion_hist.png", dpi=300)
    plt.close()


# =========================
# 7. 随机森林特征重要性
# =========================

rf_targets = [
    "rain_max",
    "rain_p95",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "centroid_offset_km",
    "anisotropy",
    "rainband_width_km",
]

rf_targets = [c for c in rf_targets if c in model_df.columns]

importance_records = []
metric_records = []

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
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    pred_train = rf.predict(X_train)
    pred_test = rf.predict(X_test)

    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)
    train_rmse = mean_squared_error(y_train, pred_train) ** 0.5
    test_rmse = mean_squared_error(y_test, pred_test) ** 0.5

    metric_records.append(
        {
            "target": target,
            "split_type": "random_row_split",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        }
    )

    result = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )

    imp = pd.DataFrame(
        {
            "target": target,
            "feature": x_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        }
    ).sort_values("importance_mean", ascending=False)

    importance_records.append(imp)

    imp.to_csv(
        TABLE_DIR / f"enhanced_rf_importance_{target}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 6))
    top = imp.head(12).iloc[::-1]
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.title(f"Enhanced RF feature importance: {target}\nTest R²={test_r2:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"07_enhanced_rf_importance_{target}.png", dpi=300)
    plt.close()

    print(f"\n目标变量: {target}")
    print(f"训练 R2: {train_r2:.4f}")
    print(f"测试 R2: {test_r2:.4f}")
    print(f"测试 MAE: {test_mae:.4f}")
    print(imp.head(8)[["feature", "importance_mean", "importance_std"]])

if importance_records:
    all_imp = pd.concat(importance_records, ignore_index=True)

    all_imp.to_csv(
        TABLE_DIR / "enhanced_rf_importance_all_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top5 = (
        all_imp.sort_values(["target", "importance_mean"], ascending=[True, False])
        .groupby("target")
        .head(5)
        .reset_index(drop=True)
    )

    top5.to_csv(
        TABLE_DIR / "enhanced_rf_importance_top5_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top10 = (
        all_imp.sort_values(["target", "importance_mean"], ascending=[True, False])
        .groupby("target")
        .head(10)
        .reset_index(drop=True)
    )

    top10.to_csv(
        TABLE_DIR / "enhanced_rf_importance_top10_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    env_top10 = top10[top10["feature"].isin(env_cols_existing)].copy()
    env_top10.to_csv(
        TABLE_DIR / "env_features_in_rf_top10.csv",
        index=False,
        encoding="utf-8-sig",
    )

if metric_records:
    metric_df = pd.DataFrame(metric_records)
    metric_df.to_csv(
        TABLE_DIR / "enhanced_rf_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
else:
    metric_df = pd.DataFrame()

if importance_records:
    write_rainband_width_outputs(model_df, x_cols, metric_df, all_imp)


# =========================
# 8. 分组统计：近岸状态
# =========================

if "is_near_coast_100km" in model_df.columns:
    group_summary = (
        model_df.groupby("is_near_coast_100km")[y_cols]
        .agg(["mean", "median", "std", "count"])
    )

    group_summary.to_csv(
        TABLE_DIR / "rain_features_by_near_coast_100km.csv",
        encoding="utf-8-sig",
    )

    print("\n已输出近岸状态分组统计。")


print("\n已输出增强版问题1分析图到:", FIG_DIR)
print("已输出增强版问题1分析表到:", TABLE_DIR)
print("完成。")
