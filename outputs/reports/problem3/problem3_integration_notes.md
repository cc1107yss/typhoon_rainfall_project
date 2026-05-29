# 问题三入总论文前排版整理说明

## 正式正文

- 正式问题三正文使用 `outputs/reports/problem3/problem3_latex_section_final.tex`。
- 该文件以 `problem3_latex_section_revised.tex` 为底稿，仅做总论文整合前的排版整理。
- 正文保留了 S0 baseline 的 `WARNING-but-usable` 说明：S0 最大累计降水 723.48 mm 与问题二 KONG-REY 的 722.05 mm 接近，但并非逐像元完全复刻，后续比较采用问题三统一链条下的 S0 作为内部对照。

## 正文主表

- 正文主表使用 `outputs/tables/problem3/problem3_compact_main_table.tex`。
- 主表保留 9 列：
  1. `scenario_id`
  2. `scenario_name`
  3. `max_accum_rain_mm`
  4. `max_area10_km2`
  5. `geo_D10_max_h`
  6. `delta_max_accum_rain_pct`
  7. `delta_max_area10_pct`
  8. `delta_geo_D10_max_pct`
  9. `brief_conclusion`
- 主表已使用 `\resizebox{\textwidth}{!}{...}` 降低过宽风险。若总论文模板未加载 `graphicx`，需在导言区加入 `\usepackage{graphicx}`。

## 完整宽表处理

- 原完整事件级指标宽表不建议直接放正文。
- 可作为附录或支撑材料保留：
  - `outputs/tables/problem3/problem3_final_scenario_comparison_table.csv`
  - `outputs/tables/problem3/problem3_paper_table_scenario_comparison_latex.tex`
- 若将完整 LaTeX 宽表放入附录，建议将表号标签改为 `tab:problem3_scenario_comparison_full`，避免与正文紧凑表 `tab:problem3_scenario_comparison` 重复。

## 关键说明保留情况

| 检查项 | 状态 | 位置 |
|---|---|---|
| mm/hr 到 mm 的 0.5 小时折算 | 已保留 | 建模思路、评价指标体系 |
| geo_D10 是固定地理格点持续时间 | 已保留 | 评价指标体系、质量检验 |
| S0 baseline WARNING-but-usable | 已保留 | 情景生成质量与基准一致性检验 |
| S2 area10 下降是落区重分布 | 已保留 | 情景结果分析 |
| S5 是复合风险情景，不是所有单项指标最大 | 已保留 | 灵敏度分析与物理解释 |

## 图表整合建议

正文建议保留：

1. S0 与 S5 累计降水对比图。
2. S0 与 S5 duration10 对比图。
3. S0-S5 关键指标柱状图。
4. 路径与累计降水质心偏移图。
5. 情景构造表。
6. 紧凑事件级指标主表。

附录或支撑材料建议保留：

1. S1-S4 单情景累计降水图和 duration10 图。
2. 完整事件级宽表。
3. 情景有效性审计表。
4. 生成质量 QC 表。
5. Top-K 检索和逐时次指标表。

## 整合结论

问题三正文已具备进入总论文整合的条件。当前仍需注意的唯一排版事项是：正文紧凑表使用 `\resizebox`，总论文模板需支持 `graphicx`；若不希望依赖该宏包，可将 `brief_conclusion` 列移入表注或改用横置表。
