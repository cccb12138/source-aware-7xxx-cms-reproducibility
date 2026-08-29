import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputDir = process.argv[2];
const outputXlsx = process.argv[3];
const previewDir = process.argv[4];

if (!inputDir || !outputXlsx || !previewDir) {
  throw new Error("Usage: node 46_build_supplementary_tables.mjs <csv_dir> <output.xlsx> <preview_dir>");
}

const sheetOrder = [
  ["cohort_summary.csv", "S1 Cohorts"],
  ["data_dictionary.csv", "S2 Data dictionary"],
  ["Table1_final_model_performance.csv", "S3 Final performance"],
  ["UTS_model_comparison.csv", "S4 Model comparison"],
  ["UTS_feature_ablation.csv", "S5 Feature ablation"],
  ["UTS_augmentation_ablation.csv", "S6 Augmentation"],
  ["UTS_SHAP_global_importance.csv", "S7 SHAP importance"],
  ["Table2_UTS_credibility.csv", "S8 UTS credibility"],
  ["UTS_applicability_domain.csv", "S9 Applicability domain"],
  ["UTS_prediction_intervals.csv", "S10 Prediction intervals"],
  ["UTS_source_bootstrap.csv", "S11 Source bootstrap"],
  ["UTS_leave_one_dataset_out.csv", "S12 Dataset transfer"],
  ["YS_EL_scope_metrics.csv", "S13 YS EL metrics"],
  ["Partial_label_MTL_metrics.csv", "S14 Partial label MTL"],
  ["Matched_target_correlations.csv", "S15 Target correlations"],
  ["Matched_fair_comparison.csv", "S16 Matched comparison"],
  ["Matched_joint_models.csv", "S17 Joint models"],
  ["Table3_matched_subset.csv", "S18 Matched summary"],
  ["figure_freeze_decision.csv", "S19 Figure decisions"],
];

function columnName(index) {
  let n = index + 1;
  let label = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    n = Math.floor((n - 1) / 26);
  }
  return label;
}

const workbook = Workbook.create();
const indexSheet = workbook.worksheets.add("Index");
const indexRows = [["Supplementary Tables", "Frozen source-aware 7xxx aluminium-alloy analysis"], ["Sheet", "Contents"]];

for (const [filename, sheetName] of sheetOrder) {
  const csvPath = path.join(inputDir, filename);
  try {
    await fs.access(csvPath);
  } catch {
    continue;
  }
  const csvText = await fs.readFile(csvPath, "utf8");
  const imported = await Workbook.fromCSV(csvText, { sheetName: "Imported" });
  const importedSheet = imported.worksheets.getItemAt(0);
  const used = importedSheet.getUsedRange(true);
  const values = used.values;
  if (!values || values.length === 0) continue;

  const sheet = workbook.worksheets.add(sheetName);
  const rowCount = values.length;
  const colCount = Math.max(...values.map((row) => row.length));
  const normalized = values.map((row) => [...row, ...Array(colCount - row.length).fill(null)]);
  const range = sheet.getRange(`A1:${columnName(colCount - 1)}${rowCount}`);
  range.values = normalized;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRange(`A1:${columnName(colCount - 1)}1`);
  header.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 30;
  range.format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
  range.format.autofitColumns();
  range.format.autofitRows();
  for (let col = 0; col < colCount; col += 1) {
    const target = sheet.getRange(`${columnName(col)}1:${columnName(col)}${rowCount}`);
    if (colCount > 8) target.format.columnWidth = Math.min(22, col < 3 ? 20 : 14);
  }
  if (rowCount > 1 && colCount > 0) {
    const safeName = `Table_${sheetName.replace(/[^A-Za-z0-9]/g, "_")}`.slice(0, 250);
    const table = sheet.tables.add(`A1:${columnName(colCount - 1)}${rowCount}`, true, safeName);
    table.style = "TableStyleMedium2";
    table.showFilterButton = true;
  }
  indexRows.push([sheetName, filename]);
}

indexSheet.getRange(`A1:B${indexRows.length}`).values = indexRows;
indexSheet.showGridLines = false;
indexSheet.freezePanes.freezeRows(2);
indexSheet.mergeCells("A1:B1");
indexSheet.getRange("A1:B1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 14 },
  verticalAlignment: "center",
};
indexSheet.getRange("A1:B1").format.rowHeight = 32;
indexSheet.getRange("A2:B2").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
};
indexSheet.getRange(`A2:B${indexRows.length}`).format.borders = { preset: "inside", style: "thin", color: "#B4C7E7" };
indexSheet.getRange("A:B").format.columnWidth = 30;

await fs.mkdir(path.dirname(outputXlsx), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputXlsx);

for (const sheet of workbook.worksheets.items) {
  const rendered = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  const bytes = new Uint8Array(await rendered.arrayBuffer());
  const safe = sheet.name.replace(/[^A-Za-z0-9_-]/g, "_");
  await fs.writeFile(path.join(previewDir, `${safe}.png`), bytes);
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 4,
  tableMaxCols: 8,
});
await fs.writeFile(`${outputXlsx}.inspect.ndjson`, inspection.ndjson, "utf8");
console.log(`Saved ${outputXlsx} with ${workbook.worksheets.items.length} sheets.`);
