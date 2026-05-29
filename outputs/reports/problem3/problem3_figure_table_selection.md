# 问题三正文图表筛选建议

## 推荐正文保留的图

正文建议保留 4 组图，围绕“基准-复合高风险”和“总体指标响应”展开：

1. S0 与 S5 累计降水对比图  
   - 使用 `outputs/figures/problem3/problem3_accum_rain_S0.png`
   - 使用 `outputs/figures/problem3/problem3_accum_rain_S5.png`
   - 用途：展示基准情景与复合高风险情景在固定地理落区上的累计降水差异。

2. S0 与 S5 duration10 对比图  
   - 使用 `outputs/figures/problem3/problem3_duration10_S0.png`
   - 使用 `outputs/figures/problem3/problem3_duration10_S5.png`
   - 用途：展示强降水持续时间在固定地理格点上的变化。

3. S0-S5 情景关键指标柱状图  
   - 使用 `outputs/figures/problem3/problem3_scenario_metric_bars.png`
   - 用途：同时比较最大累计降水、最大强降水面积和最大 D10。

4. 路径与累计降水质心偏移图  
   - 使用 `outputs/figures/problem3/problem3_path_and_centroid_shift.png`
   - 用途：解释 S2/S3 路径扰动主要改变风险落区，而不一定单调放大所有雨强指标。

可选正文补充图：

- `outputs/figures/problem3/problem3_relative_change_bars.png` 可作为相对变化图放在正文或附录。若正文篇幅有限，建议放附录。

## 建议正文保留的表

正文建议保留 2 张表：

1. 情景构造表  
   - 来源：`outputs/tables/problem3/problem3_scenario_design_table.csv`
   - 内容：S0-S5 的控制因子、数学操作、历史约束和物理含义。
   - 建议表号：表 7-1 或 `tab:problem3_scenarios`。

2. S0-S5 事件级指标与相对变化总表  
   - 来源：`outputs/tables/problem3/problem3_final_scenario_comparison_table.csv`
   - LaTeX 文件：`outputs/tables/problem3/problem3_paper_table_scenario_comparison_latex.tex`
   - 内容：最大累计降水、最大雨强、P99、area10、geo_D10、IoU 和相对变化率。
   - 建议表号：表 7-2 或 `tab:problem3_scenario_comparison`。

灵敏度解释表可根据篇幅放正文或附录：

- 来源：`outputs/tables/problem3/problem3_sensitivity_interpretation_table.csv`
- LaTeX 文件：`outputs/tables/problem3/problem3_paper_table_sensitivity_latex.tex`
- 若正文空间足够，可作为表 7-3；否则放附录并在正文概括其结论。

## 建议放附录或支撑材料的图表

- S1-S4 的单独累计降水图和 duration10 图。
- 完整的情景有效性审计表。
- 第二步生成 QC 表和 Top-K 检索距离表。
- 每时次基础指标表。

## 图件存在性检查

以下正文推荐图片均已存在：

| 图件 | 状态 |
|---|---|
| problem3_accum_rain_S0.png | exists |
| problem3_accum_rain_S5.png | exists |
| problem3_duration10_S0.png | exists |
| problem3_duration10_S5.png | exists |
| problem3_scenario_metric_bars.png | exists |
| problem3_relative_change_bars.png | exists |
| problem3_path_and_centroid_shift.png | exists |

## 正文解释提醒

- 累计降水图必须说明由 `0.5 x R_t` 累加得到，不能把 mm/hr 直接累加成 mm。
- duration10 图必须说明是固定地理格点持续时间。
- S0 是问题三统一链条内部基准，不是问题二 KONG-REY 的逐像元完全复刻。
- S2 的 area10 下降应解释为路径偏移导致强降水落区重新分布。
- S5 是复合高风险展示情景，不必要求所有单项指标都最大。
