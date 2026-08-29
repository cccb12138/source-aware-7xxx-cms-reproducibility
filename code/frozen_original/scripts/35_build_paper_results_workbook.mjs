import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const inputPath = "F:/CC/outputs/paper_results_v1/paper_results_payload.json";
const outputDir = "F:/CC/outputs/paper_results_v1";
const previewDir = path.join(outputDir, "workbook_previews");
const outputPath = path.join(outputDir, "7xxx_论文结果数据表_v1.xlsx");

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
await fs.mkdir(previewDir, { recursive: true });

const sheetNotes = {
  README: "Approved positioning and workbook scope. This is an audit workbook, not the manuscript.",
  Workflow: "Completed modeling workflow and the rule applied at each stage.",
  Main_Metrics: "Target-level final positioning. UTS is confirmatory; YS/EL are exploratory; HV is not in the rebuilt main workflow.",
  UTS_Fold_Metrics: "Final UTS RF+XGBoost ensemble using source-exclusive outer folds.",
  UTS_Bootstrap: "Source-cluster bootstrap uncertainty; sources, rather than rows, are resampled.",
  UTS_LODO: "Pooled leave-one-dataset-out transfer validation.",
  UTS_SHAP: "Global SHAP importance by component model and ensemble.",
  SHAP_Direction: "Feature-value/SHAP direction stability across source folds.",
  Applicability_Domain: "Final UTS error inside and outside the defined applicability domain.",
  Conformal_Intervals: "Cross-conformal coverage and interval width; source-max is the conservative source-aware option.",
  YS_EL_Variants: "Scope and sensitivity variants for the exploratory targets.",
  YS_EL_Bootstrap: "Source-bootstrap intervals for YS and EL.",
  MTL_Comparison: "Partial-label shared MLP benchmark versus independent RF under identical folds and features.",
  Matched_Datasets: "Composition of the 266-row complete-label robustness subset.",
  Matched_Fair_Models: "All-label-trained versus matched-only-trained models, evaluated on the same 266 samples.",
  Matched_Joint_Models: "Native multi-output RF versus independent RF on complete labels.",
  Matched_Correlations: "Row-level and source-mean target correlations.",
  Excluded_Sources: "Only clearly out-of-scope material/test conditions; no residual-based deletion.",
  Figure_Index: "Candidate figure sequence and intended takeaway. Final manuscript placement remains editable.",
};

const sourceFiles = [
  ["UTS OOF + SHAP", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/uts_scope_clean_final/oof_shap"],
  ["UTS credibility", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/uts_scope_clean_final/credibility"],
  ["UTS scope audit", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/uts_systematic_scope_audit"],
  ["YS/EL scope audit", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/ys_el_scope_audit"],
  ["Partial-label MTL", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/scope_clean_partial_label_mtl"],
  ["Matched subset", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/matched_subset_final_robustness"],
  ["Candidate figures", "D:/Jupyter/Al7xxx_Traceable_Modeling/results/paper_results_v1/figures"],
];
payload.Source_Files = sourceFiles.map(([Output, Path]) => ({ Output, Path }));
sheetNotes.Source_Files = "Traceable locations of the model outputs used to populate this workbook.";

const orderedSheets = [
  "README", "Workflow", "Main_Metrics", "UTS_Fold_Metrics", "UTS_Bootstrap", "UTS_LODO",
  "UTS_SHAP", "SHAP_Direction", "Applicability_Domain", "Conformal_Intervals",
  "YS_EL_Variants", "YS_EL_Bootstrap", "MTL_Comparison", "Matched_Datasets",
  "Matched_Fair_Models", "Matched_Joint_Models", "Matched_Correlations", "Excluded_Sources",
  "Figure_Index", "Source_Files",
];

function columnName(index) {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function normalizedValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

function numberFormatFor(header) {
  const h = header.toLowerCase();
  if (h === "rows" || h === "sources" || h === "datasets" || h === "n" || h.includes("folds") || h === "step" || h === "rank") return "#,##0";
  if (h.includes("fraction") || h.includes("coverage") || h.includes("share")) return "0.0%";
  if (h === "r2" || h.includes("r2") || h.includes("spearman") || h.includes("pearson") || h.includes("correlation")) return "0.000";
  if (h.includes("rmse") || h.includes("mae") || h.includes("bias") || h.includes("width") || h.includes("shap") || h.includes("distance")) return "0.00";
  return null;
}

const previews = [];
for (const sheetName of orderedSheets) {
  const records = payload[sheetName] ?? [];
  const headers = records.length ? Object.keys(records[0]) : ["Note"];
  const rows = records.length ? records.map(record => headers.map(header => normalizedValue(record[header]))) : [["No records"]];
  const sheet = workbook.worksheets.add(sheetName.slice(0, 31));
  sheet.showGridLines = false;
  const lastCol = columnName(headers.length - 1);
  const titleRange = sheet.getRange(`A1:${lastCol}1`);
  titleRange.merge();
  titleRange.values = [[sheetName.replaceAll("_", " ")]];
  titleRange.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 15 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  titleRange.format.rowHeight = 28;

  const noteRange = sheet.getRange(`A2:${lastCol}2`);
  noteRange.merge();
  noteRange.values = [[sheetNotes[sheetName] ?? "Audit-ready model output."]];
  noteRange.format = {
    fill: "#DCE6F1",
    font: { color: "#334155", italic: true, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  noteRange.format.rowHeight = 30;

  sheet.getRange(`A4:${lastCol}4`).values = [headers];
  sheet.getRange(`A4:${lastCol}4`).format = {
    fill: "#5B9BD5",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#A7B6C2" },
  };
  sheet.getRange(`A4:${lastCol}4`).format.rowHeight = 30;
  sheet.getRange(`A5:${lastCol}${4 + rows.length}`).values = rows;
  const body = sheet.getRange(`A5:${lastCol}${4 + rows.length}`);
  body.format = {
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: "#E5E7EB" } },
  };
  body.format.rowHeight = 20;

  for (let col = 0; col < headers.length; col++) {
    const header = headers[col];
    const colLetter = columnName(col);
    const values = [header, ...rows.map(row => row[col] === null ? "" : String(row[col]))];
    const maxLen = Math.max(...values.slice(0, 60).map(v => v.length));
    const isLongText = /reason|decision|rule|output|path|file|takeaway|evidence|stage|item|title/i.test(header);
    const width = isLongText ? Math.min(58, Math.max(18, maxLen + 2)) : Math.min(22, Math.max(10, maxLen + 2));
    const range = sheet.getRange(`${colLetter}4:${colLetter}${4 + rows.length}`);
    range.format.columnWidth = width;
    if (isLongText) range.format.wrapText = true;
    const numberFormat = numberFormatFor(header);
    if (numberFormat) sheet.getRange(`${colLetter}5:${colLetter}${4 + rows.length}`).format.numberFormat = numberFormat;
  }

  if (rows.length > 1) {
    const table = sheet.tables.add(`A4:${lastCol}${4 + rows.length}`, true, `${sheetName.replace(/[^A-Za-z0-9]/g, "").slice(0, 20)}Table`);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }
  sheet.freezePanes.freezeRows(4);

  if (sheetName === "Main_Metrics") {
    const roleIndex = headers.indexOf("Role");
    if (roleIndex >= 0) {
      const roleCol = columnName(roleIndex);
      const roleRange = sheet.getRange(`${roleCol}5:${roleCol}${4 + rows.length}`);
      roleRange.conditionalFormats.add("containsText", { text: "Primary", format: { fill: "#DDEBF7", font: { bold: true, color: "#1F4E78" } } });
      roleRange.conditionalFormats.add("containsText", { text: "Exploratory", format: { fill: "#FFF2CC", font: { color: "#7F6000" } } });
    }
  }

  const renderLastRow = Math.min(4 + rows.length, 24);
  const preview = await workbook.render({ sheetName: sheet.name, range: `A1:${lastCol}${renderLastRow}`, scale: 1.15, format: "png" });
  const previewPath = path.join(previewDir, `${String(previews.length + 1).padStart(2, "0")}_${sheet.name}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  previews.push({ sheet: sheet.name, preview: previewPath, rows: rows.length, columns: headers.length });
}

const keyInspect = await workbook.inspect({
  kind: "table",
  range: "Main_Metrics!A1:J9",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 12,
  maxChars: 5000,
});
await fs.writeFile(path.join(outputDir, "workbook_key_inspect.ndjson"), keyInspect.ndjson, "utf8");

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "workbook_formula_error_scan.ndjson"), formulaErrors.ndjson, "utf8");

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
await fs.writeFile(path.join(outputDir, "workbook_preview_manifest.json"), JSON.stringify(previews, null, 2), "utf8");
console.log(JSON.stringify({ outputPath, sheets: previews.length, previews }, null, 2));
