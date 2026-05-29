from pathlib import Path
import pandas as pd

TABLE_DIR = Path("outputs/tables/problem1_interp")

rf_path = TABLE_DIR / "enhanced_rf_importance_all_targets.csv"
corr_path = TABLE_DIR / "enhanced_spearman_correlation_rank.csv"

rf = pd.read_csv(rf_path)
corr = pd.read_csv(corr_path)

print("========== 增强版随机森林：各目标变量前 10 个重要因素 ==========")

for target in rf["target"].unique():
    sub = (
        rf[rf["target"] == target]
        .sort_values("importance_mean", ascending=False)
        .head(10)
    )

    print("\n---", target, "---")
    print("test_r2 =", sub["test_r2"].iloc[0])
    print(sub[["feature", "importance_mean", "importance_std"]])


print("\n\n========== 增强版 Spearman：各目标变量前 8 个相关因素 ==========")

targets = [
    "rain_max",
    "rain_p95",
    "rain_area_10_km2",
    "rain_area_10_equiv_radius_km",
    "centroid_offset_km",
    "centroid_relative_to_motion_deg",
    "asym_EW",
    "asym_NS",
    "r80_km",
    "anisotropy",
]

for target in targets:
    sub = corr[corr["rain_feature"] == target].copy()

    if len(sub) == 0:
        continue

    sub = sub.sort_values("abs_corr", ascending=False).head(8)

    print("\n---", target, "---")
    print(sub[["typhoon_factor", "spearman_corr", "abs_corr"]])


# 输出精简汇总表
rf_top5 = (
    rf.sort_values(["target", "importance_mean"], ascending=[True, False])
    .groupby("target")
    .head(5)
    .reset_index(drop=True)
)

corr_top5 = (
    corr.sort_values(["rain_feature", "abs_corr"], ascending=[True, False])
    .groupby("rain_feature")
    .head(5)
    .reset_index(drop=True)
)

rf_top5.to_csv(TABLE_DIR / "enhanced_rf_top5_for_paper.csv", index=False, encoding="utf-8-sig")
corr_top5.to_csv(TABLE_DIR / "enhanced_spearman_top5_for_paper.csv", index=False, encoding="utf-8-sig")

print("\n已输出论文用汇总表：")
print(TABLE_DIR / "enhanced_rf_top5_for_paper.csv")
print(TABLE_DIR / "enhanced_spearman_top5_for_paper.csv")