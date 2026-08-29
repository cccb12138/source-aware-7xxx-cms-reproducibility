import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const OUT_DIR = "F:\\CC\\outputs\\paper_scope_clean_final";
const OUT_FILE = path.join(OUT_DIR, "论文写作前_结果与参考文献总表.xlsx");
const PREVIEW_DIR = path.join(OUT_DIR, "workbook_previews");

const paths = {
  decisions: "F:\\CC\\outputs\\paper_scope_clean_final\\model_decisions",
  shap: "F:\\CC\\outputs\\uts_scope_clean_final\\oof_shap",
  credibility: "F:\\CC\\outputs\\uts_scope_clean_final\\credibility",
  ysEl: "D:\\Jupyter\\Al7xxx_Traceable_Modeling\\results\\ys_el_scope_audit",
  mtl: "D:\\Jupyter\\Al7xxx_Traceable_Modeling\\results\\scope_clean_partial_label_mtl",
  matched: "D:\\Jupyter\\Al7xxx_Traceable_Modeling\\results\\matched_subset_final_robustness",
  figures: "F:\\CC\\outputs\\paper_results_v2",
};

const COLORS = {
  navy: "#17365D",
  blue: "#2F75B5",
  paleBlue: "#D9EAF7",
  teal: "#2F7D74",
  paleTeal: "#DDEFEA",
  orange: "#C55A11",
  paleOrange: "#FCE4D6",
  red: "#A61C1C",
  paleRed: "#F4CCCC",
  green: "#548235",
  paleGreen: "#E2F0D9",
  gray: "#666666",
  lightGray: "#F2F2F2",
  border: "#D9E2F3",
  white: "#FFFFFF",
};

async function csvObjects(file) {
  const text = await fs.readFile(file, "utf8");
  const temp = await Workbook.fromCSV(text, { sheetName: "Data" });
  const matrix = temp.worksheets.getItem("Data").getUsedRange().values;
  const headers = matrix[0].map((x) => String(x ?? "").replace(/^\uFEFF/, ""));
  return matrix.slice(1).map((row) =>
    Object.fromEntries(headers.map((header, i) => [header, row[i] ?? null]))
  );
}

function round(value, digits = 3) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  return Number(number.toFixed(digits));
}

function addTitle(sheet, title, subtitle, lastColumn) {
  const titleRange = sheet.getRange(`A1:${lastColumn}1`);
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 16 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  titleRange.format.rowHeight = 30;
  const subtitleRange = sheet.getRange(`A2:${lastColumn}2`);
  subtitleRange.merge();
  subtitleRange.values = [[subtitle]];
  subtitleRange.format = {
    fill: COLORS.paleBlue,
    font: { color: COLORS.navy, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  subtitleRange.format.rowHeight = 32;
  sheet.showGridLines = false;
}

function writeTable(sheet, startRow, headers, rows, options = {}) {
  const startCol = options.startCol ?? 1;
  const rowCount = rows.length + 1;
  const colCount = headers.length;
  const range = sheet.getRangeByIndexes(startRow - 1, startCol - 1, rowCount, colCount);
  range.values = [headers, ...rows];
  range.format = {
    font: { size: 10, color: "#222222" },
    verticalAlignment: "top",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
  const header = sheet.getRangeByIndexes(startRow - 1, startCol - 1, 1, colCount);
  header.format = {
    fill: options.headerFill ?? COLORS.blue,
    font: { bold: true, color: COLORS.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: COLORS.white },
  };
  header.format.rowHeight = 28;
  for (let i = 0; i < rows.length; i++) {
    if (i % 2 === 1) {
      sheet.getRangeByIndexes(startRow + i, startCol - 1, 1, colCount).format.fill = "#F8FBFE";
    }
  }
  range.format.autofitRows();
  return { startRow, endRow: startRow + rows.length, startCol, endCol: startCol + colCount - 1 };
}

function setWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function roleFill(role) {
  if (String(role).includes("主要") || String(role).includes("Primary")) return COLORS.paleGreen;
  if (String(role).includes("探索") || String(role).includes("补充")) return COLORS.paleOrange;
  if (String(role).includes("删除") || String(role).includes("不纳入")) return COLORS.paleRed;
  return COLORS.lightGray;
}

const finalMetrics = await csvObjects(path.join(paths.decisions, "final_metrics.csv"));
const models = await csvObjects(path.join(paths.decisions, "model_comparison.csv"));
const features = await csvObjects(path.join(paths.decisions, "feature_configuration_comparison.csv"));
const featureNested = await csvObjects(path.join(paths.decisions, "feature_nested_selection.csv"));
const augmentation = await csvObjects(path.join(paths.decisions, "augmentation_configuration_comparison.csv"));
const augmentationNested = await csvObjects(path.join(paths.decisions, "augmentation_nested_selection.csv"));
const shapImportance = await csvObjects(path.join(paths.shap, "global_importance_by_model.csv"));
const shapDirection = await csvObjects(path.join(paths.shap, "direction_stability.csv"));
const credBootstrap = await csvObjects(path.join(paths.credibility, "source_cluster_bootstrap_summary.csv"));
const adSummary = await csvObjects(path.join(paths.credibility, "applicability_summary.csv"));
const intervalSummary = await csvObjects(path.join(paths.credibility, "prediction_interval_summary.csv"));
const ysElOverall = await csvObjects(path.join(paths.ysEl, "ys_el_variant_overall_metrics.csv"));
const ysElBootstrap = await csvObjects(path.join(paths.ysEl, "scope_clean_source_bootstrap_summary.csv"));
const mtlMetrics = await csvObjects(path.join(paths.mtl, "metrics_oof_summary.csv"));
const matchedCorrelations = await csvObjects(path.join(paths.matched, "target_correlations_row_and_source_mean.csv"));
const matchedMulti = await csvObjects(path.join(paths.matched, "matched_multioutput_vs_independent_rf_metrics.csv"));
const figureIndex = await csvObjects(path.join(paths.figures, "figure_index_v2.csv"));

const uts = finalMetrics[0];
const ys = ysElOverall.find((x) => x.Task === "YS" && x.Variant === "Scope_Clean");
const el = ysElOverall.find((x) => x.Task === "EL" && x.Variant === "Scope_Clean");

const workbook = Workbook.create();

// 1. Overview
{
  const sheet = workbook.worksheets.add("总览");
  addTitle(
    sheet,
    "7xxx 铝合金可回溯建模：论文写作前总览",
    "锁定口径：YS=307、UTS=675、EL=537；正文以 UTS 为主要目标，YS/EL 为探索性目标；HV 不进入本轮主流程。",
    "H",
  );
  const decisions = [
    ["数据口径", "只使用 scope-clean：YS 307、UTS 675、EL 537", "已锁定", "旧 689/550 结果不得进入正文指标、图或结论"],
    ["主模型", "五特征 RF+XGBoost 等权集成", "已锁定", "Zn、Mg、Cu、Fe、Zr；同一 675 行、258 来源内重调参"],
    ["验证单位", "Source_Group 来源隔离五折 + 内层来源分组调参", "已锁定", "同一来源不跨训练/测试折"],
    ["增强", "最终模型不使用物理噪声增强", "已锁定", "五个外折内层一标准误差规则均选择不增强"],
    ["代理/派生特征", "不进入主模型和主 SHAP", "已锁定", "仅保留为敏感性/阴性消融"],
    ["MTL", "补充性阴性结果", "已锁定", "未稳定优于独立模型，避免夸大多任务收益"],
    ["HV", "本轮正文主流程不纳入", "已锁定", "可在展望中作为小样本探索方向"],
    ["解释边界", "SHAP 只解释预测关联，不作因果或微观机制证明", "已锁定", "物理解释必须由文献或独立实验支撑"],
  ];
  writeTable(sheet, 4, ["项目", "最终决定", "状态", "论文表述边界"], decisions);
  sheet.getRange("A14:H14").merge();
  sheet.getRange("A14").values = [["写作准备状态（自动汇总）"]];
  sheet.getRange("A14:H14").format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white, size: 12 },
  };
  writeTable(sheet, 15, ["检查项", "数量/状态", "说明"], [
    ["已确认可用的本地核心文献", null, "来自桌面三个文献文件夹，优先用于引言、领域背景和对比"],
    ["建议下载的领域文献", null, "用于补齐研究缺口、7xxx 冶金机理与近期数据驱动工作"],
    ["建议下载的方法学文献", null, "用于来源分组验证、嵌套 CV、SHAP、不确定性和报告规范"],
    ["图件自动核验", "17/17 通过", "最新图件已同步至 675 行重调参 UTS 模型"],
  ], { headerFill: COLORS.teal });
  sheet.getRange("B16:B18").values = [[21], [15], [20]];
  sheet.getRange("B16:B18").format = { fill: COLORS.paleTeal, font: { bold: true, color: COLORS.teal } };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 18, B: 39, C: 14, D: 54, E: 12, F: 12, G: 12, H: 12 });
}

// 2. Final results
{
  const sheet = workbook.worksheets.add("最终结果总表");
  addTitle(
    sheet,
    "最终结果总表",
    "所有正文指标均按来源隔离的 out-of-fold 预测汇总；UTS 为确认性主要目标，YS/EL 只作探索性描述。",
    "L",
  );
  const rows = [
    ["UTS", "主要", Number(uts.Rows), Number(uts.Sources), "Zn|Mg|Cu|Fe|Zr", "RF+XGBoost 等权集成", round(uts.R2, 3), round(uts.RMSE, 2), round(uts.MAE, 2), "MPa", "中等预测能力", "可作为正文主结果；必须同时报告来源自助法区间和适用域"],
    ["YS", "探索性", Number(ys.Rows), Number(ys.Sources), "直接成分特征", "独立模型", round(ys.R2, 3), round(ys.RMSE, 2), round(ys.MAE, 2), "MPa", "较弱", "不作高精度或可部署预测声明"],
    ["EL", "探索性", Number(el.Rows), Number(el.Sources), "直接成分特征", "独立模型", round(el.R2, 3), round(el.RMSE, 2), round(el.MAE, 2), "%", "较弱", "受工艺异质性和标签噪声影响，保留探索性定位"],
    ["HV", "不纳入主流程", null, null, null, null, null, null, null, "HV", "样本不足", "可在讨论/展望中说明，不与三目标并列"],
  ];
  writeTable(sheet, 4, ["目标", "论文角色", "行数", "来源数", "最终特征", "最终模型", "OOF R²", "RMSE", "MAE", "单位", "证据等级", "允许的结论"], rows);
  for (let i = 0; i < rows.length; i++) {
    sheet.getRangeByIndexes(4 + i, 0, 1, 12).format.fill = roleFill(rows[i][1]);
  }
  const bootRows = credBootstrap.map((x) => [
    x.Metric, round(x.Bootstrap_Median, 3), round(x.CI95_Lower, 3), round(x.CI95_Upper, 3),
    x.Metric === "R2" ? "95% CI 不跨 0，支持 UTS 具有稳定但非高精度的来源外预测信号" : "与点估计配套报告",
  ]);
  writeTable(sheet, 11, ["UTS 来源簇自助法指标", "中位数", "95%CI 下限", "95%CI 上限", "解释"], bootRows, { headerFill: COLORS.teal });
  const adRows = adSummary.map((x) => [
    x.AD_Status, Number(x.Rows), round(x.Row_Fraction, 3), Number(x.Sources),
    round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2),
  ]);
  writeTable(sheet, 19, ["适用域", "行数", "比例", "来源数", "R²", "RMSE", "MAE"], adRows, { headerFill: COLORS.orange });
  sheet.getRange("A24:L25").merge();
  sheet.getRange("A24").values = [[
    "核心判断：UTS 的 R²≈0.527、来源自助法中位数≈0.522（95% CI 0.416–0.619），足以支撑“来源感知条件下的中等预测能力”，但不足以支撑“高精度预测”或“直接工程部署”。适用域外 RMSE 明显升高，应作为实际使用边界。"
  ]];
  sheet.getRange("A24:L25").format = {
    fill: COLORS.paleOrange,
    font: { bold: true, color: COLORS.orange, size: 11 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 16, B: 14, C: 10, D: 10, E: 24, F: 24, G: 10, H: 11, I: 11, J: 9, K: 18, L: 48 });
}

// 3. Model and ablation decisions
{
  const sheet = workbook.worksheets.add("模型与消融");
  addTitle(
    sheet,
    "模型比较、稀疏特征与增强决策",
    "所有数值都来自同一 675 行、258 来源的 UTS scope-clean 数据；旧 689 行模型结果未被使用。",
    "J",
  );
  const modelRows = models
    .slice()
    .sort((a, b) => Number(b.R2) - Number(a.R2))
    .map((x) => [x.Model, Number(x.Rows), Number(x.Sources), round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2), round(x.Source_Macro_RMSE, 2), x.Model === "RF+XGB ensemble" ? "正文主模型" : "基线/对照"]);
  writeTable(sheet, 4, ["模型", "行数", "来源数", "R²", "RMSE", "MAE", "来源宏平均RMSE", "角色"], modelRows);
  const featureRows = features.map((x) => [
    x.Configuration, x.Features, round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2), round(x.Source_Macro_RMSE, 2),
    x.Configuration === "refined5" ? "选为正文主特征：总体 R² 最优，且可解释性与复杂度平衡较好" : "对照",
  ]);
  writeTable(sheet, 13, ["方案", "特征", "R²", "RMSE", "MAE", "来源宏平均RMSE", "结论"], featureRows, { headerFill: COLORS.teal });
  const nestedCounts = Object.entries(
    featureNested.reduce((acc, x) => {
      acc[x.Selected_Strategy] = (acc[x.Selected_Strategy] ?? 0) + 1;
      return acc;
    }, {})
  ).map(([strategy, count]) => [strategy, count, `${count}/5 外折`]);
  writeTable(sheet, 20, ["内层选择方案", "被选折数", "比例说明"], nestedCounts, { headerFill: COLORS.teal });
  const augRows = augmentation.map((x) => [
    x.Configuration, round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2), round(x.Source_Macro_RMSE, 2),
    x.Configuration === "no_augmentation" ? "最终采用" : "仅消融；不得表述为稳定增益",
  ]);
  writeTable(sheet, 26, ["增强方案", "R²", "RMSE", "MAE", "来源宏平均RMSE", "结论"], augRows, { headerFill: COLORS.orange });
  sheet.getRange("A32:J33").merge();
  sheet.getRange("A32").values = [[
    `嵌套选择核验：${augmentationNested.filter((x) => x.Selected_Strategy === "no_augmentation").length}/5 个外折选择“不增强”。半 σ 方案的汇总 R² 仅比不增强高约 0.002，但来源宏平均 RMSE 更差，故最终遵循嵌套选择而不采用增强。`
  ]];
  sheet.getRange("A32:J33").format = {
    fill: COLORS.paleOrange,
    font: { bold: true, color: COLORS.orange },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 25, B: 29, C: 10, D: 11, E: 11, F: 17, G: 23, H: 30, I: 12, J: 12 });
}

// 4. SHAP and credibility
{
  const sheet = workbook.worksheets.add("解释与可信度");
  addTitle(
    sheet,
    "UTS 模型解释与可信度证据",
    "SHAP 基于来源隔离 OOF 模型；其含义是预测关联，不是因果效应或微观组织机制证明。",
    "K",
  );
  const imp = shapImportance
    .filter((x) => x.Model === "Ensemble")
    .sort((a, b) => Number(a.Rank) - Number(b.Rank))
    .map((x) => [x.Feature, Number(x.Rank), round(x.Mean_Abs_SHAP, 2), round(x.Importance_Share, 3), round(x.Value_SHAP_Spearman, 3)]);
  writeTable(sheet, 4, ["特征", "重要性排序", "平均|SHAP|", "重要性占比", "特征值-SHAP Spearman"], imp);
  const dir = shapDirection
    .filter((x) => x.Model === "Ensemble")
    .map((x) => [
      x.Feature, round(x.Global_Value_SHAP_Spearman, 3), round(x.Fold_Spearman_Median, 3),
      Number(x.Positive_Folds_gt_0p1), Number(x.Negative_Folds_lt_minus0p1),
      String(x.Direction_Consistent_4of5), "仅可表述为跨折稳定预测方向",
    ]);
  writeTable(sheet, 12, ["特征", "全局相关", "折中位相关", "正向折数", "负向折数", "≥4/5方向一致", "解释边界"], dir, { headerFill: COLORS.teal });
  const intervalRows = intervalSummary
    .filter((x) => x.Scope === "All")
    .map((x) => [
      x.Method, round(x.Nominal_Coverage, 2), round(x.Row_Coverage, 3),
      round(x.Source_Simultaneous_Coverage, 3), round(x.Mean_Width_MPa, 1),
      "诊断性经验区间；不能宣称对未来外部数据具有无条件保证",
    ]);
  writeTable(sheet, 20, ["方法", "名义覆盖率", "逐行覆盖率", "来源同时覆盖率", "平均宽度(MPa)", "使用边界"], intervalRows, { headerFill: COLORS.orange });
  sheet.getRange("A27:K29").merge();
  sheet.getRange("A27").values = [[
    "解释性结论建议：Zn 的预测重要性最稳定；Fe 呈稳定负向关联；Cu、Mg、Zr 呈正向预测关联。但由于数据整合了不同合金状态、制造与热处理条件，不能把 SHAP 曲线直接写成元素的普适因果规律。正文应使用“associated with / predictive contribution”，并用冶金文献解释其合理性。"
  ]];
  sheet.getRange("A27:K29").format = {
    fill: COLORS.paleRed,
    font: { bold: true, color: COLORS.red },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 23, B: 15, C: 15, D: 15, E: 21, F: 18, G: 40, H: 14, I: 14, J: 14, K: 14 });
}

// 5. Multi-target evidence
{
  const sheet = workbook.worksheets.add("三目标与MTL");
  addTitle(
    sheet,
    "三目标定位、部分标签 MTL 与完整匹配子集",
    "该页用于支撑“UTS 为主要目标、YS/EL 为探索性、MTL 为补充阴性结果”的论文定位。",
    "K",
  );
  const independentVsMtl = mtlMetrics.map((x) => [
    x.Task, x.Feature_Set, x.Model, Number(x.Rows), Number(x.Sources),
    round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2), round(x.Source_Macro_MAE, 2),
  ]);
  writeTable(sheet, 4, ["目标", "特征集", "模型", "行数", "来源数", "R²", "RMSE", "MAE", "来源宏平均MAE"], independentVsMtl);
  const corrRows = matchedCorrelations.map((x) => [
    x.Level, x.Target_1, x.Target_2, Number(x.N), round(x.Spearman, 3), round(x.Pearson, 3),
  ]);
  writeTable(sheet, 19, ["层级", "目标1", "目标2", "N", "Spearman", "Pearson"], corrRows, { headerFill: COLORS.teal });
  const multiRows = matchedMulti.map((x) => [
    x.Task, x.Feature_Set, x.Model, Number(x.Rows), Number(x.Sources),
    round(x.R2, 3), round(x.RMSE, 2), round(x.MAE, 2),
  ]);
  writeTable(sheet, 29, ["目标", "特征集", "模型", "行数", "来源数", "R²", "RMSE", "MAE"], multiRows, { headerFill: COLORS.orange });
  sheet.getRange("A44:K46").merge();
  sheet.getRange("A44").values = [[
    "MTL 判断：部分标签 MTL 在 UTS、YS、EL 上均未稳定超过相同数据口径的独立模型，因此不应作为创新性能亮点。它的价值是证明“共享表示并不自动解决跨来源、跨工艺的目标异质性”；完整匹配 266 样本只用于同样本相关性和稳健性验证。"
  ]];
  sheet.getRange("A44:K46").format = {
    fill: COLORS.paleOrange,
    font: { bold: true, color: COLORS.orange },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 16, B: 16, C: 24, D: 10, E: 10, F: 10, G: 11, H: 11, I: 18, J: 12, K: 12 });
}

// 6. Claim-evidence map
{
  const sheet = workbook.worksheets.add("论点_证据");
  addTitle(
    sheet,
    "论文“论点—证据”对应表",
    "每条论点都给出允许措辞、直接证据、图表位置和禁止越界表述；可直接作为写作检查清单。",
    "J",
  );
  const rows = [
    ["C1", "数据异质且标签不完整，必须按目标分别定义样本口径", "确认性", "YS 307 / UTS 675 / EL 537；部分标签组合与 266 完整匹配子集", "Fig.1；最终结果总表", "使用“目标特异口径”", "把不同目标强行比较为同一训练集"],
    ["C2", "随机行切分会高估跨文献来源的泛化能力，因此采用来源隔离验证", "方法学核心", "258 个 UTS 来源固定在单一外折；来源重叠=0", "Fig.1c；方法章节；fold 输出", "使用“source-exclusive OOF”", "声称等同于真正外部前瞻验证"],
    ["C3", "五特征集成模型提供中等、可重复的 UTS 预测信号", "正文主论点", `R²=${round(uts.R2,3)}，RMSE=${round(uts.RMSE,2)} MPa，MAE=${round(uts.MAE,2)} MPa；来源自助法 R² 95%CI 0.416–0.619`, "Fig.2；Table 1；Fig.4a", "使用“moderate predictive performance”", "高精度、工程可部署、R²>0.6"],
    ["C4", "RF+XGBoost 集成优于单模型和线性/哑基线", "模型选择", "同一五特征、同一五外折：集成 R² 0.527，为比较模型最高", "Fig.7a；模型与消融", "强调相同切分与相同输入", "跨不同数据口径比较模型优劣"],
    ["C5", "五个直接成分特征是性能、稀疏性与解释性的折中", "特征决策", "refined5 总体 R² 0.527；full10 0.519；drop-Si 0.526；major3 0.462", "Fig.7b；模型与消融", "称为“预先限定候选中的最终方案”", "称为唯一最优物理变量集合"],
    ["C6", "物理噪声增强没有通过严格嵌套选择", "阴性消融", "5/5 外折选择不增强；半σ汇总微增不改善来源宏平均误差", "Fig.7c；补充表", "报告表面微增与嵌套选择结论", "只挑半σ结果写成稳定提升"],
    ["C7", "派生/代理描述符的表面收益未通过嵌套验证", "阴性消融", "强制比值/安全集高于直接特征，但嵌套选择回到直接特征", "Fig.7d；补充表", "用于说明防止选择偏倚", "把代理特征 SHAP 当作物理机制"],
    ["C8", "UTS 的 OOF SHAP 方向在来源外折间具有稳定性", "解释性", "Zn、Mg、Cu、Fe、Zr 均 ≥4/5 折方向一致；Zn 排名最稳定", "Fig.3；解释与可信度", "使用“预测关联/贡献”", "因果效应、析出机制已被模型证明"],
    ["C9", "适用域外样本误差更大，模型应带使用边界", "可信度", "域内 RMSE 67.05 MPa，域外 84.96 MPa；域外约10.4%行", "Fig.4b；解释与可信度", "将 AD 作为风险提示", "把 AD 阈值当作绝对安全边界"],
    ["C10", "预测区间可描述经验不确定性，但较宽", "可信度", "90% 行交叉共形覆盖 0.892，平均宽度约223 MPa", "Fig.4c；补充表", "诊断性经验区间", "宣称对未来任意数据保证90%覆盖"],
    ["C11", "YS 和 EL 只适合探索性讨论", "定位", "YS R² 0.161；EL R² 0.140；来源自助法 R² 区间跨0", "Fig.5；最终结果总表", "强调弱信号和不确定性", "与UTS并列为成功预测目标"],
    ["C12", "部分标签 MTL 未带来稳定收益", "补充阴性", "相同口径下 MTL 的三目标 R² 均低于对应独立模型", "Fig.5d；三目标与MTL", "讨论负迁移/异质性", "宣称MTL提升预测"],
    ["C13", "完整匹配子集支持目标关系分析而非主性能评估", "稳健性", "266 行、59 来源；YS–UTS Spearman≈0.943，强度–延伸率为负相关", "Fig.6；三目标与MTL", "作为同样本验证", "用266行替代各目标全部样本主结果"],
    ["C14", "模型方法学创新在于可追溯、来源感知和对阴性结果的诚实报告", "总创新点", "固定口径、嵌套选择、来源簇自助法、适用域、共形区间、阴性消融", "全文方法链；Fig.1–8", "定位为可信材料信息学工作流", "把算法本身宣称为全新"],
  ];
  writeTable(sheet, 4, ["编号", "核心论点", "论点等级", "直接数据证据", "图/表位置", "允许的措辞", "禁止/需避免"], rows);
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 8, B: 32, C: 14, D: 55, E: 24, F: 34, G: 36, H: 12, I: 12, J: 12 });
}

// 7. Figure index
{
  const sheet = workbook.worksheets.add("图表索引");
  addTitle(
    sheet,
    "正文图件与推荐位置",
    "共 8 张正文候选图；Fig.6 可视版面移至补充材料，其余图件保留无网格线版本。",
    "G",
  );
  const rows = figureIndex.map((x) => [
    Number(x.Figure), x.Title, x.Recommended_Location, x.Revision_or_Takeaway,
    `F:\\CC\\outputs\\paper_results_v2\\figures\\Fig${x.Figure}_*.png`,
    x.Figure === 6 ? "正文或补充" : "正文候选",
  ]);
  writeTable(sheet, 4, ["图号", "标题", "推荐位置", "核心作用", "文件位置", "优先级"], rows);
  sheet.getRange("A15:G16").merge();
  sheet.getRange("A15").values = [[
    "建议正文保留 7–8 张图：Fig.1 放方法/引言末尾，不必强行视作“结果图”；Fig.2–5 为结果主体；Fig.6 是完整匹配稳健性，可按版面移补充；Fig.7 为模型与消融决策；Fig.8 放讨论/局限。"
  ]];
  sheet.getRange("A15:G16").format = { fill: COLORS.paleBlue, font: { bold: true, color: COLORS.navy }, wrapText: true };
  sheet.freezePanes.freezeRows(3);
  setWidths(sheet, { A: 8, B: 38, C: 28, D: 52, E: 56, F: 16, G: 12 });
}

const existingReferences = [
  ["E01", "已有PDF", "Soofi et al.", 2022, "A feasibility study of machine learning-assisted alloy design using wrought aluminum alloys as an example", "Computational Materials Science", "10.1016/j.commatsci.2022.111783", "小数据、铝合金ML可行性与偏差-方差背景", "引言/讨论", "铝数据来源\\7...pdf"],
  ["E02", "已有PDF", "Jiang et al.", 2024, "Synchronously enhancing the strength, toughness, and stress corrosion resistance of high-end aluminum alloys via interpretable machine learning", "Acta Materialia", "10.1016/j.actamat.2024.119873", "解释型ML与高端铝合金多性能设计", "引言/相关工作", "铝数据来源\\48...pdf"],
  ["E03", "已有PDF", "作者见原文", 2025, "Accelerated discovery of Al-Zn-Mg-Cu alloys with high-strength and high-plasticity by machine learning", "Computational Materials Science", "10.1016/j.commatsci.2025.114121", "最新Al-Zn-Mg-Cu逆向设计；与本文验证口径对照", "引言/讨论", "参考论文\\Accelerated discovery...pdf"],
  ["E04", "已有PDF", "作者见原文", 2022, "Manipulation of mechanical properties of 7xxx aluminum alloy via a hybrid approach of machine learning and key experiments", "Journal of Materials Research and Technology", "10.1016/j.jmrt.2022.06.015", "7xxx三性能建模与实验验证；对比数据切分和结论边界", "引言/讨论", "铝数据来源\\32...pdf"],
  ["E05", "已有PDF", "作者见原文", 2024, "An Explainable Deep Learning Model Based on Microstructure–Property Relationship of Aluminum Alloys", "Integrating Materials and Manufacturing Innovation", "10.1007/s40192-024-00374-2", "铝合金可解释深度学习", "相关工作", "铝数据来源\\10...pdf"],
  ["E06", "已有PDF", "作者见原文", 2020, "Composition design of 7XXX aluminum alloys for stress corrosion cracking resistance using machine learning", "Materials Research Express", "10.1088/2053-1591/ab8492", "7xxx成分设计与耐蚀性能ML", "相关工作", "铝数据来源\\12...pdf"],
  ["E07", "已有PDF", "作者见原文", 2021, "Deep learning method for predicting the mechanical properties of aluminum alloys with small data sets", "Materials Today Communications", "10.1016/j.mtcomm.2021.102570", "小样本铝合金预测", "引言/讨论", "铝数据来源\\14...pdf"],
  ["E08", "已有PDF", "作者见原文", 2023, "Diffusion Model for Inverse Design of 7xxx Aluminum Alloys with Desired Property", "Metals and Materials International", "10.1007/s12540-023-01610-8", "7xxx逆向设计新方法", "相关工作", "铝数据来源\\17...pdf"],
  ["E09", "已有PDF", "作者见原文", 2025, "Discovery of ultra-high strength aluminum alloys by interpretable chain-based machine learning", "Materials & Design", "10.1016/j.matdes.2025.114289", "近期高强铝合金链式ML", "相关工作", "铝数据来源\\18...pdf"],
  ["E10", "已有PDF", "作者见原文", 2023, "Knowledge-aware design of high-strength aviation aluminum alloys via machine learning", "Journal of Materials Research and Technology", "10.1016/j.jmrt.2023.03.041", "知识感知与领域约束", "相关工作", "铝数据来源\\24...pdf"],
  ["E11", "已有PDF", "作者见原文", 2020, "Machine learning-aided design of aluminum alloys with high performance", "Materials Today Communications", "10.1016/j.mtcomm.2020.101897", "铝合金ML设计背景", "引言", "铝数据来源\\26...pdf"],
  ["E12", "已有PDF", "作者见原文", 2025, "Physical metallurgy-guided machine learning for strength-plasticity optimization in aluminum alloys", "Materials Today Communications", "10.1016/j.mtcomm.2025.113970", "物理冶金引导特征与强塑性优化", "相关工作/讨论", "铝数据来源\\40...pdf"],
  ["E13", "已有PDF", "作者见原文", 2025, "Predicting mechanical properties in aluminum alloys using large language models and physics-based feature engineering", "Materials Today Communications", "10.1016/j.mtcomm.2025.112843", "文献抽取与物理特征工程", "相关工作/展望", "铝数据来源\\41...pdf"],
  ["E14", "已有PDF", "Starke & Staley 等综述作者见原文", 2021, "Recent advances in the metallurgy of 7xxx aluminium alloys", "Metals", "10.3390/met11050718", "7xxx成分、析出、热处理和性能机理综述", "引言/SHAP物理讨论", "文献验证\\metals-11-00718-v2.pdf"],
  ["E15", "已有PDF", "作者见原文", 2020, "Materials informatics approach to understand aluminum alloys", "请核对原文", "待从PDF确认", "本地数据来源及材料信息学背景", "数据/相关工作", "曾提供PDF与补充CSV"],
  ["E16", "已有PDF", "作者见原文", 2021, "Metallurgical study linked to 7xxx alloy processing (Springer article)", "Metallurgical and Materials Transactions A", "10.1007/s11661-021-06279-5", "7xxx冶金/工艺补充证据，需核对题名", "讨论", "铝数据来源\\44...pdf"],
  ["E17", "已有PDF", "作者见原文", 2023, "Aluminum alloy study (JOM article)", "JOM", "10.1007/s11837-023-06025-9", "补充近期铝合金研究，需核对题名与用途", "备选", "铝数据来源\\45...pdf"],
  ["E18", "已有PDF", "作者见原文", 2019, "Aluminum alloy study", "International Journal of Minerals, Metallurgy and Materials", "10.1007/s12613-019-1894-6", "传统冶金对照，需核对题名", "备选", "铝数据来源\\46...pdf"],
  ["E19", "已有中文PDF", "刘萍 等", null, "7000系铝合金中主要合金元素对其性能及微观组织的影响", "中文期刊，见原文", "无/待核对", "Zn、Mg、Cu等元素机理中文综述", "SHAP物理讨论", "文献验证\\7000系...pdf"],
  ["E20", "已有中文PDF", "何宗政 等", null, "Zn/Mg比对7xxx系铝合金组织与性能的影响", "中文期刊，见原文", "无/待核对", "Zn/Mg比机理与敏感性分析依据", "SHAP物理讨论", "文献验证\\Zn_Mg比...pdf"],
  ["E21", "已有中文PDF", "霍望图 等", null, "高强7000(Al-Zn-Mg-Cu)系铝合金成形性研究进展", "中文期刊，见原文", "无/待核对", "高强7xxx成形性与强塑性权衡", "引言/讨论", "文献验证\\高强7000...pdf"],
];

const domainToDownload = [
  ["D01", "必下", "Li et al.", 2020, "Accelerated discovery of high-strength aluminum alloys by machine learning", "Communications Materials", "10.1038/s43246-020-00074-2", "7xxx/高强铝ML代表作；用于提出以往工作常见随机切分与高分指标的对比问题", "标题 + DOI"],
  ["D02", "必下", "Azarniya et al.", 2019, "Physico-metallurgical aspects of Al–Zn–Mg–Cu alloys: a review", "Journal of Alloys and Compounds", "10.1016/j.jallcom.2018.11.286", "7xxx合金析出、时效、成分与强塑性机理综述", "标题 + DOI"],
  ["D03", "高", "Butler et al.", 2018, "Machine learning for molecular and materials science", "Nature", "10.1038/s41586-018-0337-2", "材料ML总体背景", "标题 + DOI"],
  ["D04", "高", "Schmidt et al.", 2019, "Recent advances and applications of machine learning in solid-state materials science", "npj Computational Materials", "10.1038/s41524-019-0221-0", "材料机器学习综述", "标题 + DOI"],
  ["D05", "高", "Ramprasad et al.", 2017, "Machine learning in materials informatics: recent applications and prospects", "npj Computational Materials", "10.1038/npjcompumats.2017.22", "材料信息学经典综述", "标题 + DOI"],
  ["D06", "高", "Ward et al.", 2016, "A general-purpose machine learning framework for predicting properties of inorganic materials", "npj Computational Materials", "10.1038/npjcompumats.2016.28", "成分特征与通用材料预测", "标题 + DOI"],
  ["D07", "高", "Chang et al.", 2022, "Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework", "npj Computational Materials", "10.1038/s41524-022-00929-x", "材料小数据与跨域异质性", "标题 + DOI"],
  ["D08", "中", "Lookman et al.", 2019, "Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design", "npj Computational Materials", "10.1038/s41524-019-0153-8", "未来数据补充/主动学习展望", "标题 + DOI"],
  ["D09", "中", "Agrawal & Choudhary", 2016, "Perspective: Materials informatics and big data", "APL Materials", "10.1063/1.4946894", "材料信息学背景与数据质量", "标题 + DOI"],
  ["D10", "中", "Callister/Polmear 等", null, "Light Alloys: Metallurgy of the Light Metals / 7xxx aluminium metallurgy", "教材/专著", "ISBN或书名检索", "传统冶金基础，用于避免只靠ML论文解释元素效应", "7xxx aluminum alloy precipitation strengthening Zn Mg Cu Zr review"],
  ["D11", "高", "近期作者待筛", 2021, "7xxx aluminum alloy precipitation sequence η′/η, PFZ and aging review", "优先 Acta Materialia / Progress in Materials Science", "关键词检索", "Zn、Mg、Cu、Zr、Fe SHAP方向的物理文献依据", "\"7xxx aluminum alloy\" precipitation η' η PFZ aging review 2021 2022 2023 2024"],
  ["D12", "高", "近期作者待筛", 2021, "Effect of Fe-rich intermetallics on tensile properties of 7xxx aluminum alloys", "优先 Materials Science and Engineering A", "关键词检索", "解释Fe负向预测关联及其非因果边界", "\"7xxx\" Fe-rich intermetallic tensile strength elongation DOI"],
  ["D13", "高", "近期作者待筛", 2021, "Zr dispersoids and recrystallization resistance in Al-Zn-Mg-Cu alloys", "优先 Acta Materialia / MSEA", "关键词检索", "解释Zr的稳定正向预测关联", "\"Al-Zn-Mg-Cu\" Zr dispersoids recrystallization strength DOI 2021..2026"],
  ["D14", "中", "近期作者待筛", 2021, "Cu effect on precipitation and strength/corrosion trade-off in Al-Zn-Mg-Cu alloys", "优先 Corrosion Science / Acta Materialia", "关键词检索", "解释Cu正向贡献及强度-腐蚀权衡", "\"Al-Zn-Mg-Cu\" Cu precipitation strength corrosion DOI review"],
  ["D15", "中", "近期作者待筛", 2021, "Dataset shift and domain heterogeneity in experimental materials databases", "Digital Discovery / npj Computational Materials", "关键词检索", "支撑跨文献来源验证与适用域讨论", "\"experimental materials data\" dataset shift domain heterogeneity machine learning"],
];

const methodsToDownload = [
  ["M01", "必下", "Breiman", 2001, "Random Forests", "Machine Learning", "10.1023/A:1010933404324", "RF算法引用", "标题 + DOI"],
  ["M02", "必下", "Chen & Guestrin", 2016, "XGBoost: A Scalable Tree Boosting System", "KDD", "10.1145/2939672.2939785", "XGBoost算法引用", "标题 + DOI"],
  ["M03", "必下", "Lundberg & Lee", 2017, "A Unified Approach to Interpreting Model Predictions", "NeurIPS", "10.5555/3295222.3295230；arXiv:1705.07874", "SHAP方法及解释边界", "标题 + arXiv/DOI"],
  ["M04", "必下", "Akiba et al.", 2019, "Optuna: A Next-generation Hyperparameter Optimization Framework", "KDD", "10.1145/3292500.3330701；arXiv:1907.10902", "贝叶斯/TPE超参数优化软件引用", "标题 + DOI"],
  ["M05", "必下", "Varma & Simon", 2006, "Bias in error estimation when using cross-validation for model selection", "BMC Bioinformatics", "10.1186/1471-2105-7-91", "嵌套CV避免模型选择偏倚", "标题 + DOI"],
  ["M06", "必下", "Witman & Schindler", 2025, "MatFold: systematic insights into materials discovery models' performance through standardized cross-validation protocols", "Digital Discovery", "10.1039/D4DD00250D", "材料发现中的结构化/化学分组验证", "标题 + DOI"],
  ["M07", "必下", "Kapoor & Narayanan et al.", 2023, "Leakage and the reproducibility crisis in machine-learning-based science", "Patterns", "10.1016/j.patter.2023.100804", "数据泄漏、分组泄漏和可复现性", "标题 + DOI"],
  ["M08", "必下", "Kapoor et al.", 2024, "REFORMS: Consensus-based Recommendations for Machine-learning-based Science", "Science Advances", "10.1126/sciadv.adk3452", "研究目标、验证、透明报告和可复现性规范", "标题 + DOI"],
  ["M09", "高", "Vovk", 2015, "Cross-conformal predictors", "Annals of Mathematics and Artificial Intelligence", "10.1007/s10472-013-9368-4", "交叉共形预测方法", "标题 + DOI"],
  ["M10", "高", "Angelopoulos & Bates", 2023, "Conformal Prediction: A Gentle Introduction", "Foundations and Trends in Machine Learning", "10.1561/2200000101；arXiv:2107.07511", "共形预测概念、覆盖率与假设边界", "标题 + DOI"],
  ["M11", "高", "Field & Welsh", 2007, "Bootstrapping clustered data", "Journal of the Royal Statistical Society B", "关键词/题名检索", "来源簇自助法的统计依据", "\"cluster bootstrap\" Field Welsh 2007 DOI"],
  ["M12", "中", "Cheng et al.", 2013, "Bootstrapping for highly unbalanced clustered data", "Computational Statistics & Data Analysis", "10.1016/j.csda.2012.09.004", "不平衡来源簇重采样依据", "标题 + DOI"],
  ["M13", "高", "Kapoor et al.", 2024, "Setting standards for data driven materials science", "npj Computational Materials", "请按标题检索 DOI", "材料科学报告规范与开放数据", "标题精确检索"],
  ["M14", "高", "Chang et al.", 2021, "Materials representation and transfer learning for multi-property prediction", "Applied Physics Reviews", "10.1063/5.0047066", "多性能预测与迁移/共享表示背景", "标题 + DOI"],
  ["M15", "中", "Yates et al.", 2023, "Cross validation for model selection: a review with examples from ecology", "Ecological Monographs", "10.1002/ecm.1557", "模型选择与一标准误差思想", "标题 + DOI"],
  ["M16", "中", "作者见原文", 2021, "The One Standard Error Rule for Model Selection: Does It Work?", "Stats", "10.3390/stats4040051", "一标准误差规则的专门讨论", "标题 + DOI"],
  ["M17", "中", "Bates, Hastie & Tibshirani", 2024, "Cross-validation: what does it estimate and how well does it do it?", "Journal of the American Statistical Association", "按标题/arXiv:2104.00673检索", "CV不确定性与解释边界", "标题 + arXiv"],
  ["M18", "中", "Molnar", 2022, "Interpretable Machine Learning", "开放教材", "https://christophm.github.io/interpretable-ml-book/", "补充SHAP、关联与因果区别", "书名/网址"],
  ["M19", "中", "Pedregosa et al.", 2011, "Scikit-learn: Machine Learning in Python", "Journal of Machine Learning Research", "JMLR 12:2825-2830", "软件与实现引用", "标题精确检索"],
  ["M20", "中", "Paszke et al.", 2019, "PyTorch: An Imperative Style, High-Performance Deep Learning Library", "NeurIPS", "arXiv:1912.01703", "MTL实现的软件引用（若正文保留）", "标题 + arXiv"],
];

function addReferenceSheet(name, title, subtitle, data, headerFill) {
  const sheet = workbook.worksheets.add(name);
  addTitle(sheet, title, subtitle, "J");
  writeTable(sheet, 4, ["编号", "优先级/状态", "作者", "年份", "题名", "期刊/来源", "DOI/标识", "在本文中的用途", "下载检索词/章节", "本地位置"], data.map((row) => row.length === 9 ? [...row, ""] : row), { headerFill });
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 8, B: 14, C: 18, D: 9, E: 56, F: 32, G: 33, H: 48, I: 46, J: 44 });
}

addReferenceSheet(
  "已有文献",
  "本地已有的核心参考文献",
  "优先使用已下载论文；“需核对”的题名、作者或 DOI 必须在正式写作前从 PDF 首页确认，不可凭文件名直接生成最终参考文献。",
  existingReferences,
  COLORS.blue,
);
addReferenceSheet(
  "待下载领域",
  "建议下载的领域与冶金文献",
  "优先级“必下/高”先补齐；对只有关键词的条目，下载后再确定最终引用，不建议为了凑数量一次性下载大量弱相关论文。",
  domainToDownload,
  COLORS.teal,
);
addReferenceSheet(
  "待下载方法",
  "建议下载的方法学文献",
  "这些文献直接支撑来源隔离、嵌套模型选择、SHAP、不确定性和可复现报告，是本文方法学定位的关键参考。",
  methodsToDownload,
  COLORS.orange,
);

// 10. Remaining work
{
  const sheet = workbook.worksheets.add("写作前缺口");
  addTitle(
    sheet,
    "正式写作前仍需完成/确认的事项",
    "按优先级执行；这些不是要求重新建模，而是论文可审查性、参考文献准确性和最终输出一致性的收尾工作。",
    "H",
  );
  const rows = [
    ["P0", "确认数据版本冻结", "将最终三目标数据表、来源映射、排除清单和脚本打包并写版本号/日期", "用户+代码", "未完成", "写作前必须"],
    ["P0", "核对本地PDF元数据", "对“已有文献”中作者/年份/期刊/DOI待核对项逐篇检查首页或Crossref", "用户可下载/我可继续整理", "部分完成", "防止参考文献错误"],
    ["P0", "决定目标期刊", "确定期刊后统一参考文献样式、图宽、字数、补充材料结构", "用户", "待确认", "格式工作依赖此选择"],
    ["P0", "冻结最终图表", "确认更新后的 Fig.2–4、7 数值变化和全部图的最终版式", "用户", "待确认", "图已自动核验17/17"],
    ["P1", "补下载必需方法文献", "优先 M01–M10，至少覆盖 RF、XGB、SHAP、嵌套CV、材料分组CV、泄漏、REFORMS、共形", "用户", "待下载", "方法章节与讨论必需"],
    ["P1", "补下载必需领域文献", "优先 D01–D07、D11–D14，重点补7xxx元素/析出/工艺机理", "用户", "待下载", "支撑SHAP物理合理性但不把SHAP写成因果"],
    ["P1", "补充数据字典", "逐列定义单位、0与缺失的区别、检测/处理条件、派生字段公式", "我可继续整理", "待完成", "建议作为补充表S1"],
    ["P1", "生成最终表格", "Table 1数据结构；Table 2模型性能；Table 3可信度；补充消融与MTL表", "我可继续执行", "待图确认后", "本工作簿已提供数据基础"],
    ["P2", "决定Fig.6位置", "若正文图过多，将完整匹配子集图移至补充材料", "用户", "待确认", "不影响主结论"],
    ["P2", "外部验证展望", "若后续获得全新来源数据，保持模型冻结后做真正外部验证", "后续工作", "可选", "不是当前投稿前硬性条件"],
    ["P2", "论文写作", "按问题—缺口—方法—证据—边界链条撰写，不先写夸大的物理机制结论", "双方", "下一阶段", "建议单独任务进行"],
  ];
  writeTable(sheet, 4, ["优先级", "事项", "具体动作", "负责人", "当前状态", "为什么需要"], rows);
  sheet.getRange("A18:H20").merge();
  sheet.getRange("A18").values = [[
    "没有遗漏的建模硬性步骤：主模型、来源隔离验证、嵌套调参、模型/特征/增强消融、OOF SHAP、来源自助法、适用域、共形区间、部分标签 MTL 和完整匹配稳健性均已具备。当前最重要的不是继续追求更高 R²，而是冻结版本、补齐可核验文献元数据，并确保正文只使用本表锁定的指标。"
  ]];
  sheet.getRange("A18:H20").format = {
    fill: COLORS.paleGreen,
    font: { bold: true, color: COLORS.green, size: 11 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 10, B: 25, C: 58, D: 22, E: 18, F: 42, G: 12, H: 12 });
}

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

// Export, re-import, inspect, and render every sheet.
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(OUT_FILE);
const reloaded = await SpreadsheetFile.importXlsx(await (await import("@oai/artifact-tool")).FileBlob.load(OUT_FILE));

const sheetSummary = await reloaded.inspect({ kind: "sheet", include: "id,name", maxChars: 6000 });
await fs.writeFile(path.join(OUT_DIR, "workbook_sheet_inspection.ndjson"), sheetSummary.ndjson ?? String(sheetSummary), "utf8");

for (const sheetName of [
  "总览", "最终结果总表", "模型与消融", "解释与可信度", "三目标与MTL",
  "论点_证据", "图表索引", "已有文献", "待下载领域", "待下载方法", "写作前缺口",
]) {
  const preview = await reloaded.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

console.log(`Saved workbook: ${OUT_FILE}`);
console.log(`Rendered previews: ${PREVIEW_DIR}`);
