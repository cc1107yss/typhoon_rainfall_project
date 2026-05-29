from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================
# 1. 路径设置
# =========================

FEATURE_PATH = Path("data/processed/env_added/gpm_track_model_features_interp_env.csv")
OUT_DIR = Path("outputs/tables/problem1_env")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "group_validation_results.csv"
COMPARISON_PATH = OUT_DIR / "group_validation_comparison.csv"
PAPER_TABLE_PATH = OUT_DIR / "group_validation_comparison_for_paper.md"


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
print("台风事件数:", df["track_event_uid"].nunique())


# =========================
# 3. 解释变量与目标变量
# =========================

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
    "gpm_track_center_dist_km",
    "track_time_diff_h",
]

targets = [
    "rain_max",
    "rain_p95",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "centroid_offset_km",
    "anisotropy",
    "rainband_width_km",
]

x_cols = [c for c in x_cols if c in df.columns]
targets = [c for c in targets if c in df.columns]

print("\n解释变量数量:", len(x_cols))
print(x_cols)

print("\n目标变量:")
print(targets)

target_labels = {
    "rain_max": "最大降水强度",
    "rain_p95": "95分位降水强度",
    "rain_area_10_km2": "强降水面积",
    "rain_area_10_equiv_radius_km": "强降水等效半径",
    "centroid_offset_km": "降水中心偏移距离",
    "anisotropy": "降水空间非均匀性",
    "rainband_width_km": "雨带带宽",
}


# =========================
# 4. 模型函数
# =========================

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def fit_eval_random_split(data, target):
    """
    普通随机划分：会让同一台风的不同时次同时进入训练和测试。
    这个结果通常较乐观。
    """
    sub = data[x_cols + [target, "track_event_uid"]].dropna().copy()

    X = sub[x_cols]
    y = sub[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "target": target,
        "split_type": "random_row_split",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_events": sub.loc[X_train.index, "track_event_uid"].nunique(),
        "n_test_events": sub.loc[X_test.index, "track_event_uid"].nunique(),
        "train_r2": r2_score(y_train, pred_train),
        "test_r2": r2_score(y_test, pred_test),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": rmse(y_test, pred_test),
    }


def fit_eval_group_split(data, target):
    """
    按台风事件分组划分：
    同一 track_event_uid 不会同时出现在训练集和测试集。
    这个结果更能体现对未知台风事件的泛化能力。
    """
    sub = data[x_cols + [target, "track_event_uid"]].dropna().copy()

    X = sub[x_cols]
    y = sub[target]
    groups = sub["track_event_uid"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "target": target,
        "split_type": "group_event_split",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_events": train_groups.nunique(),
        "n_test_events": test_groups.nunique(),
        "train_r2": r2_score(y_train, pred_train),
        "test_r2": r2_score(y_test, pred_test),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": rmse(y_test, pred_test),
    }


def build_comparison_table(result):
    random_res = result[result["split_type"] == "random_row_split"].set_index("target")
    group_res = result[result["split_type"] == "group_event_split"].set_index("target")

    common_targets = [target for target in targets if target in random_res.index and target in group_res.index]
    records = []

    for target in common_targets:
        random_row = random_res.loc[target]
        group_row = group_res.loc[target]

        records.append(
            {
                "target": target,
                "target_label": target_labels.get(target, target),
                "random_test_r2": random_row["test_r2"],
                "group_test_r2": group_row["test_r2"],
                "test_r2_drop": random_row["test_r2"] - group_row["test_r2"],
                "random_test_mae": random_row["test_mae"],
                "group_test_mae": group_row["test_mae"],
                "mae_increase_pct": (group_row["test_mae"] / random_row["test_mae"] - 1) * 100,
                "random_test_rmse": random_row["test_rmse"],
                "group_test_rmse": group_row["test_rmse"],
                "rmse_increase_pct": (group_row["test_rmse"] / random_row["test_rmse"] - 1) * 100,
                "random_n_test_events": int(random_row["n_test_events"]),
                "group_n_test_events": int(group_row["n_test_events"]),
            }
        )

    return pd.DataFrame(records)


def dataframe_to_markdown(data):
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in data.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines) + "\n"


def build_paper_table(comparison):
    paper = comparison[
        [
            "target_label",
            "random_test_r2",
            "group_test_r2",
            "test_r2_drop",
            "random_test_mae",
            "group_test_mae",
            "mae_increase_pct",
            "random_n_test_events",
            "group_n_test_events",
        ]
    ].copy()

    paper["random_test_r2"] = paper["random_test_r2"].round(3)
    paper["group_test_r2"] = paper["group_test_r2"].round(3)
    paper["test_r2_drop"] = paper["test_r2_drop"].round(3)
    paper["random_test_mae"] = paper["random_test_mae"].round(3)
    paper["group_test_mae"] = paper["group_test_mae"].round(3)
    paper["mae_increase_pct"] = paper["mae_increase_pct"].round(1)

    paper = paper.rename(
        columns={
            "target_label": "指标",
            "random_test_r2": "随机划分R2",
            "group_test_r2": "按台风分组R2",
            "test_r2_drop": "R2下降",
            "random_test_mae": "随机划分MAE",
            "group_test_mae": "按台风分组MAE",
            "mae_increase_pct": "MAE增加(%)",
            "random_n_test_events": "随机测试事件数",
            "group_n_test_events": "分组测试事件数",
        }
    )

    return paper


# =========================
# 5. 执行验证
# =========================

records = []

for target in targets:
    print("\n==============================")
    print("目标变量:", target)

    random_res = fit_eval_random_split(df, target)
    group_res = fit_eval_group_split(df, target)

    records.append(random_res)
    records.append(group_res)

    print("普通随机划分:")
    print(
        "test_r2 =",
        round(random_res["test_r2"], 4),
        "test_mae =",
        round(random_res["test_mae"], 4),
        "test_rmse =",
        round(random_res["test_rmse"], 4),
    )

    print("按台风事件分组划分:")
    print(
        "test_r2 =",
        round(group_res["test_r2"], 4),
        "test_mae =",
        round(group_res["test_mae"], 4),
        "test_rmse =",
        round(group_res["test_rmse"], 4),
        "测试事件数 =",
        group_res["n_test_events"],
    )


# =========================
# 6. 输出结果
# =========================

result_df = pd.DataFrame(records)
result_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

comparison_df = build_comparison_table(result_df)
comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

paper_table = build_paper_table(comparison_df)
PAPER_TABLE_PATH.write_text(
    dataframe_to_markdown(paper_table),
    encoding="utf-8",
)

print("\n分组验证结果已输出:")
print(OUT_PATH)

print("\n随机划分与按台风事件分组对比表已输出:")
print(COMPARISON_PATH)
print(PAPER_TABLE_PATH)

print("\n汇总结果:")
print(result_df)
