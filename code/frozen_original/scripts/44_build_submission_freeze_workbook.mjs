import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  SpreadsheetFile,
  Workbook,
} from "@oai/artifact-tool";

const freezeDir = "F:/CC/outputs/submission_freeze_2026-07-25";
const outputPath = `${freezeDir}/提交冻结版_数据字典与最终表格.xlsx`;
const previewDir = `${freezeDir}/workbook_previews`;

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        inQuotes = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += c;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  if (rows.length && rows[0].length && rows[0][0].charCodeAt(0) === 0xfeff) {
    rows[0][0] = rows[0][0].slice(1);
  }
  return rows.filter((r) => r.some((v) => v !== ""));
}

async function readCsv(relativePath) {
  const text = await fs.readFile(`${freezeDir}/${relativePath}`, "utf8");
  return parseCsv(text);
}

function maybeNumber(value) {
  if (value === "") return "";
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

function typedRows(rows) {
  return rows.map((row, i) => (i === 0 ? row : row.map(maybeNumber)));
}

const COLORS = {
  navy: "#16324F",
  teal: "#1F6F78",
  pale: "#E8F1F3",
  paleBlue: "#EAF0F7",
  paleGold: "#F7F1DF",
  paleRed: "#F9E7E7",
  text: "#1F2933",
  muted: "#5B6770",
  border: "#C8D3D8",
  white: "#FFFFFF",
};

function columnName(index) {
  let result = "";
  let n = index + 1;
  while (n > 0) {
    const rem = (n - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function applyBaseStyle(sheet, rows, title, note = "") {
  const nRows = rows.length;
  const nCols = Math.max(...rows.map((r) => r.length));
  const lastCol = columnName(nCols - 1);
  sheet.showGridLines = false;
  sheet.mergeCells(`A1:${lastCol}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastCol}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 15, name: "Microsoft YaHei" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastCol}1`).format.rowHeight = 28;
  if (note) {
    sheet.mergeCells(`A2:${lastCol}2`);
    sheet.getRange("A2").values = [[note]];
    sheet.getRange(`A2:${lastCol}2`).format = {
      fill: COLORS.paleBlue,
      font: { color: COLORS.muted, italic: true, size: 9, name: "Microsoft YaHei" },
      wrapText: true,
      verticalAlignment: "center",
    };
    sheet.getRange(`A2:${lastCol}2`).format.rowHeight = 28;
  }
  sheet.getRange(`A3:${lastCol}${nRows + 2}`).values = typedRows(rows);
  sheet.getRange(`A3:${lastCol}3`).format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white, size: 10, name: "Microsoft YaHei" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: COLORS.navy } },
  };
  sheet.getRange(`A3:${lastCol}3`).format.rowHeight = 34;
  if (nRows > 1) {
    sheet.getRange(`A4:${lastCol}${nRows + 2}`).format = {
      font: { color: COLORS.text, size: 9, name: "Microsoft YaHei" },
      verticalAlignment: "center",
      borders: {
        insideHorizontal: { style: "thin", color: "#E2E8EB" },
        bottom: { style: "thin", color: COLORS.border },
      },
    };
  }
  sheet.freezePanes.freezeRows(3);
}

function setWidths(sheet, widths) {
  for (let i = 0; i < widths.length; i++) {
    sheet.getRange(`${columnName(i)}:${columnName(i)}`).format.columnWidth = widths[i];
  }
}

function formatNumericColumns(sheet, header, startRow, endRow) {
  header.forEach((name, i) => {
    const col = columnName(i);
    if (name.includes("pct") || name.includes("Coverage") || name.includes("Fraction")) {
      sheet.getRange(`${col}${startRow}:${col}${endRow}`).format.numberFormat = "0.0%";
    } else if (["R2", "Estimate", "Lower", "Upper", "R2_or_Spearman"].includes(name)) {
      sheet.getRange(`${col}${startRow}:${col}${endRow}`).format.numberFormat = "0.000";
    } else if (["RMSE", "MAE"].includes(name)) {
      sheet.getRange(`${col}${startRow}:${col}${endRow}`).format.numberFormat = "0.0";
    } else if (["Rows", "Sources", "Datasets", "Pages", "Bytes", "Columns"].includes(name)) {
      sheet.getRange(`${col}${startRow}:${col}${endRow}`).format.numberFormat = "#,##0";
    }
  });
}

function addTableSheet(workbook, name, title, note, rows, widths) {
  const sheet = workbook.worksheets.add(name);
  applyBaseStyle(sheet, rows, title, note);
  setWidths(sheet, widths);
  formatNumericColumns(sheet, rows[0], 4, rows.length + 2);
  return sheet;
}

const cohort = await readCsv("tables/cohort_summary.csv");
const dictionary = await readCsv("tables/data_dictionary.csv");
const finalResults = await readCsv("tables/Table1_final_model_performance.csv");
const credibility = await readCsv("tables/Table2_UTS_credibility.csv");
const matched = await readCsv("tables/Table3_matched_subset.csv");
const figures = await readCsv("tables/figure_freeze_decision.csv");
const references = await readCsv("manifests/reference_pdf_audit.csv");
const manifest = await readCsv("manifests/freeze_manifest.csv");

const workbook = Workbook.create();

const overview = workbook.worksheets.add("冻结说明");
overview.showGridLines = false;
overview.mergeCells("A1:H1");
overview.getRange("A1").values = [["7xxx铝合金论文投稿冻结版"]];
overview.getRange("A1:H1").format = {
  fill: COLORS.navy,
  font: { bold: true, color: COLORS.white, size: 17, name: "Microsoft YaHei" },
  verticalAlignment: "center",
};
overview.getRange("A1:H1").format.rowHeight = 32;
overview.mergeCells("A2:H2");
overview.getRange("A2").values = [[
  "冻结日期：2026-07-25。此后正文中的样本数、模型指标、图号和参考文献信息均应以本工作簿及同目录SHA-256清单为准。",
]];
overview.getRange("A2:H2").format = {
  fill: COLORS.paleBlue,
  font: { color: COLORS.muted, italic: true, size: 10, name: "Microsoft YaHei" },
  wrapText: true,
};
overview.getRange("A2:H2").format.rowHeight = 30;
overview.getRange("A4:B10").values = [
  ["冻结项目", "冻结结论"],
  ["主要目标", "UTS：675行，258个来源"],
  ["探索目标", "YS：307行；EL：537行"],
  ["部分标签数据", "689行"],
  ["完整匹配子集", "266行，59个来源"],
  ["正文图", "7张"],
  ["补充图", "Fig.S1：完整匹配子集稳健性"],
];
overview.getRange("A4:B4").format = {
  fill: COLORS.teal,
  font: { bold: true, color: COLORS.white, name: "Microsoft YaHei" },
};
overview.getRange("A5:A10").format = {
  fill: COLORS.pale,
  font: { bold: true, color: COLORS.text, name: "Microsoft YaHei" },
};
overview.getRange("A4:B10").format.borders = {
  outside: { style: "thin", color: COLORS.border },
  insideHorizontal: { style: "thin", color: COLORS.border },
};
overview.getRange("D4:F9").values = [
  ["最终UTS模型", "数值", "说明"],
  ["R²", 0.5268211291102216, "来源独占外层OOF"],
  ["RMSE", 69.12516659499569, "MPa"],
  ["MAE", 49.70170430226349, "MPa"],
  ["特征", "Zn, Mg, Cu, Fe, Zr", "直接成分"],
  ["定位", "中等预测能力", "不作因果或高精度声明"],
];
overview.getRange("D4:F4").format = {
  fill: COLORS.teal,
  font: { bold: true, color: COLORS.white, name: "Microsoft YaHei" },
};
overview.getRange("D5:D9").format = {
  fill: COLORS.pale,
  font: { bold: true, color: COLORS.text, name: "Microsoft YaHei" },
};
overview.getRange("D4:F9").format.borders = {
  outside: { style: "thin", color: COLORS.border },
  insideHorizontal: { style: "thin", color: COLORS.border },
};
overview.getRange("E5:E7").format.numberFormat = "0.000";
overview.mergeCells("A12:H12");
overview.getRange("A12").values = [["使用规则"]];
overview.getRange("A12:H12").format = {
  fill: COLORS.navy,
  font: { bold: true, color: COLORS.white, name: "Microsoft YaHei" },
};
overview.getRange("A13:H17").values = [
  ["1", "不要再引用旧的689条UTS模型或旧版四目标结论。", "", "", "", "", "", ""],
  ["2", "UTS是唯一主要确认性预测与SHAP解释目标；YS和EL为探索性目标。", "", "", "", "", "", ""],
  ["3", "派生成分描述符只用于敏感性分析，不进入最终UTS主模型。", "", "", "", "", "", ""],
  ["4", "Fig.6移至补充材料并重编号为Fig.S1；正文保留7张图。", "", "", "", "", "", ""],
  ["5", "参考文献库中Azarniya综述有3个完全相同副本，正式库只保留azarniya2019.pdf。", "", "", "", "", "", ""],
];
for (let r = 13; r <= 17; r++) {
  overview.mergeCells(`B${r}:H${r}`);
}
overview.getRange("A13:A17").format = {
  fill: COLORS.paleGold,
  font: { bold: true, color: COLORS.navy, name: "Microsoft YaHei" },
  horizontalAlignment: "center",
};
overview.getRange("B13:H17").format = {
  font: { color: COLORS.text, size: 10, name: "Microsoft YaHei" },
  wrapText: true,
};
overview.getRange("A13:H17").format.rowHeight = 25;
setWidths(overview, [10, 28, 8, 18, 18, 30, 16, 16]);
overview.freezePanes.freezeRows(2);

const dataVersion = addTableSheet(
  workbook,
  "数据版本",
  "冻结数据版本与用途",
  "样本数和来源数从冻结CSV重新统计；来源列只用于分组验证，不能作为预测特征。",
  cohort,
  [18, 12, 12, 12, 24, 34],
);

const dictSheet = addTableSheet(
  workbook,
  "数据字典",
  "统一数据字典",
  "缺失率按各冻结队列分别计算。最终UTS模型仅使用Zn、Mg、Cu、Fe和Zr。",
  dictionary,
  [23, 20, 15, 42, 34, 17, 17, 17, 17, 17],
);
dictSheet.freezePanes.freezeColumns(1);
const dictHeader = dictionary[0];
dictHeader.forEach((name, i) => {
  if (name.endsWith("_Missing_pct")) {
    const col = columnName(i);
    dictSheet.getRange(`${col}4:${col}${dictionary.length + 2}`).format.numberFormat = "0.0%";
  }
});

const resultsSheet = addTableSheet(
  workbook,
  "最终结果",
  "最终模型结果总表",
  "R²、RMSE和MAE均来自来源独占外层OOF预测；YS和EL不可与UTS同等强度表述。",
  finalResults,
  [12, 20, 24, 10, 10, 28, 12, 12, 12, 10, 34],
);
resultsSheet.getRange(`A4:K4`).format.fill = COLORS.paleGold;

const credSheet = addTableSheet(
  workbook,
  "UTS可信度",
  "UTS可信度、迁移与不确定性",
  "来源bootstrap、留一数据集验证、适用域和交叉保形区间共同限定模型可用范围。",
  credibility,
  [24, 18, 14, 14, 14, 42],
);

const matchedSheet = addTableSheet(
  workbook,
  "匹配子集",
  "266条完整匹配样本的稳健性验证",
  "该子集用于同一样本可比性检查，不替代各目标的全部可用标签数据。",
  matched,
  [20, 38, 16, 10, 18, 14, 14, 38],
);

const figureSheet = addTableSheet(
  workbook,
  "图表冻结",
  "正文与补充材料图表冻结决定",
  "正文共7张图；原Fig.6因信息密度高且与Fig.5部分重复，移至补充材料并改为Fig.S1。",
  figures,
  [16, 16, 38, 14, 14, 60],
);
figureSheet.getRange("D4:D11").conditionalFormats.add("containsText", {
  text: "补充材料",
  format: { fill: COLORS.paleGold, font: { bold: true, color: COLORS.navy } },
});

const refSheet = addTableSheet(
  workbook,
  "参考文献PDF核对",
  "桌面“论文参考文献”文件夹PDF核对",
  "共57份PDF：全部可打开；13份无文字层文件已视觉核对首页；Azarniya有2个冗余副本；Cu-Al备选文献不纳入正文。",
  references,
  [46, 12, 20, 18, 10, 10, 18, 45, 50, 32, 26, 48, 44],
);
refSheet.freezePanes.freezeColumns(1);
const statusIndex = references[0].indexOf("Audit_Status");
const statusCol = columnName(statusIndex);
refSheet.getRange(`${statusCol}4:${statusCol}${references.length + 2}`).conditionalFormats.add(
  "containsText",
  {
    text: "重复文件",
    format: { fill: COLORS.paleGold, font: { bold: true, color: COLORS.navy } },
  },
);
refSheet.getRange(`${statusCol}4:${statusCol}${references.length + 2}`).conditionalFormats.add(
  "containsText",
  {
    text: "研究对象为Cu-Al",
    format: { fill: COLORS.paleRed, font: { bold: true, color: "#8B1E1E" } },
  },
);

const manifestSheet = addTableSheet(
  workbook,
  "文件校验清单",
  "冻结文件SHA-256校验清单",
  "任何文件内容变化都会导致SHA-256变化；投稿前应再次比对此表。",
  manifest,
  [58, 18, 14, 10, 10, 70, 24],
);
manifestSheet.freezePanes.freezeColumns(1);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of [
  "冻结说明",
  "数据版本",
  "数据字典",
  "最终结果",
  "UTS可信度",
  "匹配子集",
  "图表冻结",
  "参考文献PDF核对",
  "文件校验清单",
]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(`${previewDir}/${sheetName}.png`, bytes);
}

const keyInspect = await workbook.inspect({
  kind: "table",
  range: "最终结果!A1:K7",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 12,
});
await fs.writeFile(
  `${freezeDir}/manifests/workbook_key_inspect.ndjson`,
  keyInspect.ndjson,
  "utf8",
);

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  `${freezeDir}/manifests/workbook_formula_error_scan.ndjson`,
  errorScan.ndjson,
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const workbookBytes = await fs.readFile(outputPath);
await fs.writeFile(
  `${freezeDir}/manifests/final_workbook_hash.json`,
  JSON.stringify(
    {
      file: path.basename(outputPath),
      bytes: workbookBytes.length,
      sha256: crypto.createHash("sha256").update(workbookBytes).digest("hex"),
    },
    null,
    2,
  ),
  "utf8",
);
console.log(outputPath);
