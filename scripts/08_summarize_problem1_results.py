from pathlib import Path

import pandas as pd


TABLE_DIR = Path("outputs/tables/problem1")

corr_path = TABLE_DIR / "spearman_correlation_rank.csv"
rf_path = TABLE_DIR / "rf_importance_all_targets.csv"

corr = pd.read_csv(corr_path)
rf = pd.read_csv(rf_path)

print("========== Spearman 相关性：每个降水特征最相关的前 5 个台风因子 ==========")

targets = [
    "rain_max",
    "rain_p95",
    "rain_area_10_grid",
    "centroid_offset_km",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",
]

for target in targets:
    sub = corr[corr["rain_feature"] == target].copy()
    sub = sub.sort_values("abs_corr", ascending=False).head(5)

    print("\n---", target, "---")
    print(sub[["typhoon_factor", "spearman_corr", "abs_corr"]])


print("\n\n========== 随机森林置换重要性：每个目标变量前 8 个因子 ==========")

for target in rf["target"].unique():
    sub = rf[rf["target"] == target].copy()
    sub = sub.sort_values("importance_mean", ascending=False).head(8)

    print("\n---", target, "---")
    print(sub[["feature", "importance_mean", "importance_std", "test_r2"]])


# 输出一个简洁版汇总表
rf_top = (
    rf.sort_values(["target", "importance_mean"], ascending=[True, False])
    .groupby("target")
    .head(5)
    .reset_index(drop=True)
)

rf_top.to_csv(TABLE_DIR / "rf_importance_top5_summary.csv", index=False, encoding="utf-8-sig")

corr_top = (
    corr.sort_values(["rain_feature", "abs_corr"], ascending=[True, False])
    .groupby("rain_feature")
    .head(5)
    .reset_index(drop=True)
)

corr_top.to_csv(TABLE_DIR / "spearman_top5_summary.csv", index=False, encoding="utf-8-sig")

print("\n已输出：")
print(TABLE_DIR / "rf_importance_top5_summary.csv")
print(TABLE_DIR / "spearman_top5_summary.csv")