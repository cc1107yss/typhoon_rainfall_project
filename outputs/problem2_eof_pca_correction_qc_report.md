# Problem 2 EOF/PCA Structure Correction QC Report

## 1. 输入输出文件
- 19 号 initial NPZ: `data/processed/problem2_generated_initial_fields_topk_weighted.npz`
- 19 号 index: `data/processed/problem2_generated_initial_fields_index.csv`
- Top-K 表: `data/processed/problem2_target_topk_similar_history.csv`
- EOF/PCA model 输出: `data/processed/problem2_eof_pca_model.npz`
- coefficients CSV 输出: `data/processed/problem2_target_eof_coefficients.csv`
- blended NPZ 输出: `data/processed/problem2_generated_pca_blended_fields.npz`
- blended index CSV 输出: `data/processed/problem2_generated_pca_blended_fields_index.csv`
- figures 目录: `outputs/figures/problem2_eof_pca_correction`

## 2. 运行参数
- PCA_TRAINING_SOURCE: unique_topk_history_templates
- MAX_PCA_TEMPLATES: 5000
- 实际训练样本数: 2166
- N_COMPONENTS: 20
- BATCH_SIZE: 128
- BETA_BLEND: 0.3
- GRID_SIZE: 201
- GRID_EXTENT_KM: 1000.0
- RANDOM_SEED: 2026

## 3. PCA 训练数据统计
- Top-K 唯一历史模板数: 2166
- Top-K 目标同名历史模板排除数: 0
- 实际进入 PCA 候选样本数: 2166
- 成功读取和重采样数量: 2166
- 跳过数量: 0
- 跳过原因计数:
```
read_attempts    2166
templates_ok     2166
```
- 训练样本涉及历史台风事件数: 57
- 训练样本 typhoon_name Top 10:
```
history_typhoon_name
DUJUAN      184
KOPPU       183
KALMAEGI    127
MUIFA       120
KOINU       102
MALAKAS     100
YUTU         90
TALIM        90
HAIMA        83
MERANTI      78
```
- 训练样本 WND/PRES/rain_p95 摘要:
```
                       count    mean     std     min     50%     95%     max
history_WND             2166 29.2331 14.9205      10 23.1667 57.9792 68.3333
history_PRES            2166 977.845 26.4686 903.333     990    1004 1008.33
history_rain_p95_mmhr   2166 2.84966  1.3536    0.46    2.61  5.1375   13.51
```
- 是否存在目标台风样本: False (必须为 False)

## 4. PCA 模型统计
```
 component  explained_variance_ratio  cumulative_explained_variance_ratio
         1                  0.103664                             0.103664
         2                   0.08495                             0.188614
         3                 0.0516346                             0.240249
         4                 0.0447213                              0.28497
         5                 0.0303972                             0.315367
         6                 0.0240055                             0.339373
         7                 0.0207005                             0.360073
         8                 0.0176945                             0.377768
         9                 0.0162598                             0.394028
        10                 0.0149949                             0.409023
        11                 0.0132385                             0.422261
        12                 0.0118432                             0.434104
        13                 0.0107635                             0.444868
        14                 0.0100876                             0.454955
        15                0.00930471                              0.46426
        16                 0.0082945                             0.472555
        17                0.00798817                             0.480543
        18                0.00710673                              0.48765
        19                0.00687713                             0.494527
        20                0.00614157                             0.500668
```
- 前 5 个 EOF 的解释率: 0.103664, 0.08495, 0.0516346, 0.0447213, 0.0303972
- 前 10 个 EOF 的累计解释率: 0.409023
- 全部 N_COMPONENTS 累计解释率: 0.500668
- mean_log_field min/mean/max: 0.00339673 / 0.235589 / 1.76011

## 5. 目标投影与重构统计
- 目标样本数: 974
- eof_coefficients shape: 974 x 20
- rain_mmhr_eof shape: [974, 201, 201]
- rain_mmhr_blend shape: [974, 201, 201]
- EOF 重构 log RMSE 分布: min=0.0929754, mean=0.117479, p50=0.114671, p95=0.152312, max=0.188557
- EOF 重构 log Corr 分布: min=0.595867, mean=0.91251, p50=0.922135, p95=0.970275, max=0.975618
- initial NaN/Inf/负值/全零场: 0 / 0 / 0 / 0
- EOF NaN/Inf/负值/全零场: 0 / 0 / 0 / 0
- blend NaN/Inf/负值/全零场: 0 / 0 / 0 / 0

## 6. PCA-blended 场基本检查
- blend_rain_max_mmhr 分布: min=1.56712, mean=9.17361, p50=7.18613, p95=21.5543, max=26.553
- blend_rain_p95_mmhr 分布: min=0.418335, mean=1.41804, p50=1.3544, p95=2.39264, max=3.16419
- blend_rain_area_10_km2 分布: min=0, mean=6209.14, p50=0, p95=38535, max=46100
- blend_centroid_offset_km 分布: min=2.37683, mean=107.815, p50=89.6653, p95=281.896, max=416.868
- blend_anisotropy 分布: min=0.0154444, mean=0.212166, p50=0.211552, p95=0.363435, max=0.472633
- corr_initial_blend 分布: min=0.983166, mean=0.994515, p50=0.995279, p95=0.998423, max=0.998782
- rmse_initial_blend 分布: min=0.0421492, mean=0.0757636, p50=0.0696584, p95=0.116411, max=0.170888

## 7. 分台风统计
### KONG-REY
- 时刻数: 421
- blend_rain_max_mmhr 最大值: 23.1612 at 2024-10-29 22:30:00
- blend_rain_p95_mmhr 最大值: 2.49459 at 2024-10-30 09:30:00
- blend_rain_area_10_km2 最大值: 43800 at 2024-10-30 15:30:00
- blended 累计降水 proxy: 2.98563e+08
- 强降水持续时间 proxy: 80 hours
- 前 5 个 EOF 系数: EOF1: mean=-2.94085, std=19.8124, max_abs=41.3568; EOF2: mean=2.12772, std=15.5193, max_abs=35.0908; EOF3: mean=4.35595, std=9.73519, max_abs=21.342; EOF4: mean=2.87673, std=6.25256, max_abs=16.9845; EOF5: mean=-2.16297, std=7.77468, max_abs=22.4539
### MAN-YI
- 时刻数: 553
- blend_rain_max_mmhr 最大值: 26.553 at 2024-11-16 17:00:00
- blend_rain_p95_mmhr 最大值: 3.16419 at 2024-11-14 04:30:00
- blend_rain_area_10_km2 最大值: 46100 at 2024-11-16 12:30:00
- blended 累计降水 proxy: 4.36501e+08
- 强降水持续时间 proxy: 85 hours
- 前 5 个 EOF 系数: EOF1: mean=-0.182063, std=20.2375, max_abs=41.9195; EOF2: mean=1.77679, std=14.8426, max_abs=31.2236; EOF3: mean=-5.37652, std=7.13959, max_abs=22.3919; EOF4: mean=-1.51901, std=9.75488, max_abs=20.4795; EOF5: mean=3.2605, std=5.80843, max_abs=24.097

## 8. 与 19 号初始场的对比
- ratio_blend_to_initial_rain_max: P50=0.884414, P95=0.9623
- ratio_blend_to_initial_rain_p95: P50=0.971892, P95=1.01241
- ratio_blend_to_initial_area_10: P50=0.938034, P95=1.05936
- delta_blend_minus_initial_rain_max: mean=-1.14473, P95=-0.352525
- delta_blend_minus_initial_rain_p95: mean=-0.0372105, P95=0.019035

本步骤只做结构修正，不负责把极端值校准到历史分位数。若 PCA-blended 后极端峰值仍偏低，极端降水偏弱问题将在 21 号极端分位数校准中处理。

## 9. 时间连续性检查
### KONG-REY
- diff(blend_rain_max_mmhr): P95=2.34388, max=7.84819
- diff(blend_rain_p95_mmhr): P95=0.209899, max=0.689996
- diff(blend_rain_area_10_km2): P95=4100, max=27300
- diff(eof_coef_01): P95=3.91246, max=11.3996
- diff(eof_coef_02): P95=4.91621, max=21.9046
### MAN-YI
- diff(blend_rain_max_mmhr): P95=1.72406, max=6.97038
- diff(blend_rain_p95_mmhr): P95=0.157749, max=1.17214
- diff(blend_rain_area_10_km2): P95=3100, max=24200
- diff(eof_coef_01): P95=4.44303, max=21.6443
- diff(eof_coef_02): P95=3.78949, max=22.4763

## 10. 防泄漏声明
本步骤的 EOF/PCA 训练只使用历史 GPM 降水模板。目标台风 KONG-REY 和 MAN-YI 的真实 GPM 降水不存在且未被读取。目标台风 EOF 系数由 19 号初始生成场投影得到，不使用任何目标真实降水观测。rain_*、centroid_*、quad_*、anisotropy、rain_radius_*、rain_band_width_km 等历史降水指标未参与目标台风安全输入构造和 Top-K 检索，仅用于历史训练样本诊断与后续校准评价。

## 11. 图件
- 图件数量: 23
```
                  eof_explained_variance.png
                             eof_mode_01.png
                             eof_mode_02.png
                             eof_mode_03.png
                             eof_mode_04.png
                             eof_mode_05.png
                             eof_mode_06.png
                          mean_log_field.png
                         mean_rain_field.png
    KONG_REY_eof_coefficients_timeseries.png
      MAN_YI_eof_coefficients_timeseries.png
KONG_REY_pca_blend_compare_20241024_1800.png
KONG_REY_pca_blend_compare_20241030_0930.png
KONG_REY_pca_blend_compare_20241102_1200.png
  MAN_YI_pca_blend_compare_20241108_1200.png
  MAN_YI_pca_blend_compare_20241114_0500.png
  MAN_YI_pca_blend_compare_20241120_0000.png
KONG_REY_blend_cumulative_storm_relative.png
       KONG_REY_blend_max_storm_relative.png
  MAN_YI_blend_cumulative_storm_relative.png
         MAN_YI_blend_max_storm_relative.png
       KONG_REY_blend_metrics_timeseries.png
         MAN_YI_blend_metrics_timeseries.png
```

## 12. 异常与日志样例
```
(none)
```

## 13. 论文可写结论
- EOF1: 主要载荷偏向后侧/左侧，绝对载荷质心约 (-47.3, 30.3) km；可解释为主要大尺度雨带背景模态之一。
- EOF2: 主要载荷偏向前侧/左侧，绝对载荷质心约 (260.8, 145.5) km；可用于描述相对运动方向上的非对称变化。
- EOF3: 主要载荷偏向后侧/右侧，绝对载荷质心约 (-67.5, 54.1) km；可用于描述雨带横向偏移或局地增强/减弱结构。
- KONG-REY 的 EOF1/EOF2 系数标准差分别为 19.8/15.5，可在论文中结合路径阶段讨论结构演变。
- MAN-YI 的 EOF1/EOF2 系数标准差分别为 20.2/14.8，可在论文中结合路径阶段讨论结构演变。
- PCA-blended 场保持 19 号 Top-K 初始场主体，同时引入历史 EOF 低秩约束，可作为 21 号极端分位数校准的结构底图。
- 本步骤不负责极端峰值闭合；若 blended 后极端值偏弱，应在 21 号 R95/R99/Rmax 分位数校准中处理。
