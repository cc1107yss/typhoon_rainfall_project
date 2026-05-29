from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 路径设置
# =========================

FEATURE_PATH = Path("data/processed/gpm_precip_features.csv")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取 GPM 特征表
# =========================

df = pd.read_csv(FEATURE_PATH, parse_dates=["time", "time_end"])

print("GPM 特征表维度:", df.shape)
print("事件数量:", df["event_uid"].nunique())


# =========================
# 3. 选择一场台风事件
# =========================
# 默认选择第一场事件。你后面可以改成别的事件编号，比如 2023060。

event_id = df["event_uid"].iloc[0]

one = df[df["event_uid"] == event_id].copy()
one = one.sort_values("time").reset_index(drop=True)

print("当前绘图事件:", event_id)
print("该事件时次数量:", len(one))
print("起止时间:", one["time"].min(), "至", one["time"].max())

print(
    one[
        [
            "time",
            "center_lon",
            "center_lat",
            "rain_max",
            "rain_p95",
            "rain_area_10_grid",
            "centroid_offset_km",
            "asym_EW",
            "asym_NS",
            "r80_km",
            "anisotropy",
        ]
    ].head()
)


# =========================
# 4. 图1：降水强度随时间演化
# =========================

plt.figure(figsize=(12, 5))
plt.plot(one["time"], one["rain_max"], label="Max rainfall")
plt.plot(one["time"], one["rain_p95"], label="P95 rainfall")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Rainfall intensity (mm/hr)")
plt.title(f"Event {event_id}: Rainfall intensity evolution")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"event_{event_id}_01_intensity.png"
plt.savefig(out_path, dpi=300)
print("已保存:", out_path)

plt.show()


# =========================
# 5. 图2：强降水范围随时间演化
# =========================

plt.figure(figsize=(12, 5))
plt.plot(one["time"], one["rain_area_5_grid"], label="Area >= 5 mm/hr")
plt.plot(one["time"], one["rain_area_10_grid"], label="Area >= 10 mm/hr")
plt.plot(one["time"], one["rain_area_20_grid"], label="Area >= 20 mm/hr")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Number of grid cells")
plt.title(f"Event {event_id}: Heavy rainfall area evolution")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"event_{event_id}_02_area.png"
plt.savefig(out_path, dpi=300)
print("已保存:", out_path)

plt.show()


# =========================
# 6. 图3：降水中心偏移与降水半径
# =========================

plt.figure(figsize=(12, 5))
plt.plot(one["time"], one["centroid_offset_km"], label="Centroid offset")
plt.plot(one["time"], one["r80_km"], label="R80 radius")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Distance (km)")
plt.title(f"Event {event_id}: Rainfall structure evolution")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"event_{event_id}_03_structure.png"
plt.savefig(out_path, dpi=300)
print("已保存:", out_path)

plt.show()


# =========================
# 7. 图4：东西、南北非对称性
# =========================

plt.figure(figsize=(12, 5))
plt.plot(one["time"], one["asym_EW"], label="East-West asymmetry")
plt.plot(one["time"], one["asym_NS"], label="North-South asymmetry")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Asymmetry index")
plt.title(f"Event {event_id}: Rainfall asymmetry evolution")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"event_{event_id}_04_asymmetry.png"
plt.savefig(out_path, dpi=300)
print("已保存:", out_path)

plt.show()


# =========================
# 8. 图5：降水带狭长程度
# =========================

plt.figure(figsize=(12, 5))
plt.plot(one["time"], one["anisotropy"], label="Anisotropy")
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.ylabel("Anisotropy")
plt.title(f"Event {event_id}: Rainband anisotropy evolution")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / f"event_{event_id}_05_anisotropy.png"
plt.savefig(out_path, dpi=300)
print("已保存:", out_path)

plt.show()