# 问题三最终图表清单

## 正文必放图

| 图件 | 建议图号 | 文件 | 状态 | 正文作用 |
|---|---|---|---|---|
| S0 累计降水图 | 图 7-1(a) | `outputs/figures/problem3/problem3_accum_rain_S0.png` | exists | 展示问题三统一链条下的基准累计降水落区。 |
| S5 累计降水图 | 图 7-1(b) | `outputs/figures/problem3/problem3_accum_rain_S5.png` | exists | 与 S0 对比，展示复合高风险情景下累计降水增强和落区变化。 |
| S0 duration10 图 | 图 7-2(a) | `outputs/figures/problem3/problem3_duration10_S0.png` | exists | 展示基准情景固定地理格点强降水持续时间。 |
| S5 duration10 图 | 图 7-2(b) | `outputs/figures/problem3/problem3_duration10_S5.png` | exists | 与 S0 对比，展示复合情景下固定格点持续时间变化。 |
| S0-S5 关键指标柱状图 | 图 7-3 | `outputs/figures/problem3/problem3_scenario_metric_bars.png` | exists | 横向比较最大累计降水、最大强降水面积和最大 D10。 |
| 路径与累计降水质心偏移图 | 图 7-4 | `outputs/figures/problem3/problem3_path_and_centroid_shift.png` | exists | 解释路径扰动和登陆点扰动主要改变风险落区。 |

## 正文必放表

| 表格 | 建议表号 | 来源文件 | 正文作用 |
|---|---|---|---|
| 情景构造表 | 表 7-1 | `outputs/tables/problem3/problem3_scenario_design_table.csv` | 说明 S0-S5 的控制因子、数学操作、历史约束和物理意义。 |
| 事件级指标与相对变化表 | 表 7-2 | `outputs/tables/problem3/problem3_paper_table_scenario_comparison_latex.tex` | 集中呈现各情景的最大累计降水、最大雨强、P99、最大 area10、最大 D10 和相对 S0 的变化。 |

## 可放附录图

| 图件类型 | 文件建议 | 放附录理由 |
|---|---|---|
| S1-S4 累计降水单图 | `problem3_accum_rain_S1.png` 至 `problem3_accum_rain_S4.png` | 正文保留 S0/S5 代表图即可，单因子完整空间图可作支撑材料。 |
| S1-S4 duration10 单图 | `problem3_duration10_S1.png` 至 `problem3_duration10_S4.png` | 用于补充说明单因子情景下固定格点持续时间变化。 |
| 相对变化柱状图 | `problem3_relative_change_bars.png` | 若正文篇幅有限，可用事件级指标表替代，该图放附录辅助说明。 |

## 可放正文或附录的表

| 表格 | 文件 | 建议 |
|---|---|---|
| 灵敏度解释表 | `outputs/tables/problem3/problem3_paper_table_sensitivity_latex.tex` | 若论文篇幅足够可放正文；否则放附录，正文保留其文字总结。 |
| 情景有效性审计表 | `outputs/tables/problem3/problem3_scenario_validity_audit.csv` | 建议放附录或支撑材料，用于证明扰动受历史可比范围约束。 |
| 第二步生成 QC 表 | `outputs/tables/problem3/problem3_step2_generation_qc_summary.csv` | 建议放附录或支撑材料，用于说明雨场生成质量。 |

## 支撑材料应保留文件

- `data/processed/problem3/problem3_scenario_event_metrics.csv`
- `data/processed/problem3/problem3_scenario_relative_changes.csv`
- `data/processed/problem3/problem3_step3_metrics_qc.csv`
- `outputs/reports/problem3/problem3_step3_event_metrics_report.md`
- `outputs/tables/problem3/problem3_final_scenario_comparison_table.csv`
- `outputs/tables/problem3/problem3_sensitivity_interpretation_table.csv`
- `outputs/tables/problem3/problem3_scenario_design_table.csv`
- `outputs/tables/problem3/problem3_scenario_validity_audit.csv`

## 排版提醒

1. 事件级指标表列数较多，正式论文中建议使用横置表、缩放表或拆分为“绝对指标”和“相对变化”两张表。
2. 所有累计降水图的图注必须注明单位为 mm，且由半小时雨强按 $0.5R_t$ 折算得到。
3. 所有 duration10 图的图注必须注明单位为 h，并说明其为固定地理格点持续时间。
4. S0 与 S5 对比图应使用一致色标，避免视觉误判。
