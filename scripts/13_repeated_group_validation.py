from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


FEATURE_PATH = Path("data/processed/env_added/gpm_track_model_features_interp_env.csv")
OUT_DIR = Path("outputs/tables/problem1_env")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "repeated_group_validation_results.csv"


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


x_cols = [
    "track_wind",
    "track_pressure",
    "pressure_deficit",
    "intensity_index",

    "track_move_speed_kmh",
    "track_move_dir_sin",
    "track_move_dir_cos",

    "track_wind_change_rate",
    "track_pressure_change_rate",
    "is_intensifying_wind",
    "is_weakening_wind",
    "is_intensifying_pressure",
    "is_weakening_pressure",

    "track_is_land",
    "track_coast_dist_km",
    "track_signed_coast_dist_km",
    "is_near_coast_100km",
    "is_near_coast_200km",
    "is_offshore_far_300km",
    "is_inland_100km",
    "coast_influence_exp",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    "gpm_track_center_dist_km",
    "track_time_diff_h",
    "landfrac_200km",
    "landfrac_500km",
    "terrain_mean_300km",
    "terrain_max_300km",
    "terrain_std_300km",
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


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def baseline_metrics(y_train, y_test):
    """
    基准模型：永远预测训练集均值。
    如果 RF 的 R2 接近 0，说明大致和均值基准差不多。
    """
    pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    return {
        "baseline_r2": r2_score(y_test, pred),
        "baseline_mae": mean_absolute_error(y_test, pred),
        "baseline_rmse": rmse(y_test, pred),
    }


def evaluate_target(target, use_log=False, n_splits=20):
    sub = df[x_cols + [target, "track_event_uid"]].dropna().copy()

    X = sub[x_cols]
    y_raw = sub[target].astype(float)

    if use_log:
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()

    groups = sub["track_event_uid"]

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=0.25,
        random_state=42,
    )

    records = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        raw_train = y_raw.iloc[train_idx]
        raw_test = y_raw.iloc[test_idx]

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

        if use_log:
            # R2 既看 log 空间，也反变换回原尺度看 MAE/RMSE
            pred_test_raw = np.expm1(pred_test)
            pred_train_raw = np.expm1(pred_train)

            test_r2_raw = r2_score(raw_test, pred_test_raw)
            test_mae_raw = mean_absolute_error(raw_test, pred_test_raw)
            test_rmse_raw = rmse(raw_test, pred_test_raw)

            train_r2_raw = r2_score(raw_train, pred_train_raw)
            test_r2_model_space = r2_score(y_test, pred_test)
        else:
            pred_test_raw = pred_test
            pred_train_raw = pred_train

            test_r2_raw = r2_score(raw_test, pred_test_raw)
            test_mae_raw = mean_absolute_error(raw_test, pred_test_raw)
            test_rmse_raw = rmse(raw_test, pred_test_raw)

            train_r2_raw = r2_score(raw_train, pred_train_raw)
            test_r2_model_space = test_r2_raw

        base = baseline_metrics(raw_train.values, raw_test.values)

        records.append(
            {
                "target": target,
                "use_log1p_target": use_log,
                "split_id": split_id,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_train_events": train_groups.nunique(),
                "n_test_events": test_groups.nunique(),
                "train_r2_raw_scale": train_r2_raw,
                "test_r2_raw_scale": test_r2_raw,
                "test_r2_model_space": test_r2_model_space,
                "test_mae_raw_scale": test_mae_raw,
                "test_rmse_raw_scale": test_rmse_raw,
                **base,
            }
        )

    return pd.DataFrame(records)


all_records = []

for target in targets:
    print("\n==============================")
    print("目标变量:", target)

    normal_res = evaluate_target(target, use_log=False, n_splits=20)
    all_records.append(normal_res)

    print("原尺度分组验证 test_r2_raw_scale:")
    print(normal_res["test_r2_raw_scale"].describe())

    # 只对非负且偏态明显的变量做 log1p
    if target in [
        "rain_max",
        "rain_p95",
        "rain_area_10_km2",
        "rain_area_10_equiv_radius_km",
        "centroid_offset_km",
        "rainband_width_km",
    ]:
        log_res = evaluate_target(target, use_log=True, n_splits=20)
        all_records.append(log_res)

        print("log1p 分组验证 test_r2_raw_scale:")
        print(log_res["test_r2_raw_scale"].describe())


result = pd.concat(all_records, ignore_index=True)
result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n重复分组验证结果已输出:")
print(OUT_PATH)

summary = (
    result.groupby(["target", "use_log1p_target"])
    .agg(
        mean_test_r2=("test_r2_raw_scale", "mean"),
        median_test_r2=("test_r2_raw_scale", "median"),
        std_test_r2=("test_r2_raw_scale", "std"),
        mean_test_mae=("test_mae_raw_scale", "mean"),
        mean_test_rmse=("test_rmse_raw_scale", "mean"),
        mean_baseline_mae=("baseline_mae", "mean"),
        mean_baseline_rmse=("baseline_rmse", "mean"),
    )
    .reset_index()
)

summary_path = OUT_DIR / "repeated_group_validation_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n重复分组验证汇总:")
print(summary)

print("\n汇总表已输出:")
print(summary_path)
