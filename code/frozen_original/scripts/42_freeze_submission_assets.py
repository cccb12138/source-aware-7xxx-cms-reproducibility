from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(r"F:\CC")
OUT = ROOT / "outputs" / "submission_freeze_2026-07-25"
REF_DIR = Path(r"C:\Users\dell\OneDrive\Desktop\论文参考文献")


CANONICAL_FILES = {
    "data/7000系铝合金_可回溯建模总表.xlsx": ROOT
    / "outputs"
    / "model_ready_rebuild"
    / "7000系铝合金_重新建模数据总表.xlsx",
    "data/YS_scope_clean_307.csv": ROOT
    / "outputs"
    / "ys_el_scope_audit"
    / "YS_scope_clean_with_outer_folds.csv",
    "data/UTS_scope_clean_675.csv": ROOT
    / "outputs"
    / "uts_systematic_scope_audit"
    / "UTS_scope_clean_with_outer_folds.csv",
    "data/EL_scope_clean_537.csv": ROOT
    / "outputs"
    / "ys_el_scope_audit"
    / "EL_scope_clean_with_outer_folds.csv",
    "data/Partial_label_MTL_689.csv": ROOT
    / "outputs"
    / "scope_clean_partial_label_mtl"
    / "MTL_scope_clean_with_outer_folds.csv",
    "data/Matched_complete_266.csv": ROOT
    / "outputs"
    / "matched_subset_final_robustness"
    / "matched_complete_266_with_outer_folds.csv",
    "results/UTS_final_metrics.csv": ROOT
    / "outputs"
    / "paper_scope_clean_final"
    / "model_decisions"
    / "final_metrics.csv",
    "results/UTS_final_oof_predictions.csv": ROOT
    / "outputs"
    / "paper_scope_clean_final"
    / "model_decisions"
    / "final_oof_predictions.csv",
    "results/YS_EL_scope_metrics.csv": ROOT
    / "outputs"
    / "ys_el_scope_audit"
    / "ys_el_variant_overall_metrics.csv",
    "results/Partial_label_MTL_metrics.csv": ROOT
    / "outputs"
    / "scope_clean_partial_label_mtl"
    / "metrics_oof_summary.csv",
    "results/UTS_model_comparison.csv": ROOT
    / "outputs"
    / "paper_scope_clean_final"
    / "model_decisions"
    / "model_comparison.csv",
    "results/UTS_feature_ablation.csv": ROOT
    / "outputs"
    / "paper_scope_clean_final"
    / "model_decisions"
    / "feature_configuration_comparison.csv",
    "results/UTS_augmentation_ablation.csv": ROOT
    / "outputs"
    / "paper_scope_clean_final"
    / "model_decisions"
    / "augmentation_configuration_comparison.csv",
    "results/UTS_source_bootstrap.csv": ROOT
    / "outputs"
    / "uts_scope_clean_final"
    / "credibility"
    / "source_cluster_bootstrap_summary.csv",
    "results/UTS_applicability_domain.csv": ROOT
    / "outputs"
    / "uts_scope_clean_final"
    / "credibility"
    / "applicability_summary.csv",
    "results/UTS_prediction_intervals.csv": ROOT
    / "outputs"
    / "uts_scope_clean_final"
    / "credibility"
    / "prediction_interval_summary.csv",
    "results/UTS_leave_one_dataset_out.csv": ROOT
    / "outputs"
    / "uts_systematic_scope_audit"
    / "scope_clean_leave_one_dataset_out_pooled_metrics.csv",
    "results/UTS_SHAP_global_importance.csv": ROOT
    / "outputs"
    / "uts_scope_clean_final"
    / "oof_shap"
    / "global_importance_by_model.csv",
    "results/Matched_fair_comparison.csv": ROOT
    / "outputs"
    / "matched_subset_final_robustness"
    / "fair_comparison_full_vs_matched_only_metrics.csv",
    "results/Matched_joint_models.csv": ROOT
    / "outputs"
    / "matched_subset_final_robustness"
    / "matched_multioutput_vs_independent_rf_metrics.csv",
    "results/Matched_target_correlations.csv": ROOT
    / "outputs"
    / "matched_subset_final_robustness"
    / "target_correlations_row_and_source_mean.csv",
}


MAIN_FIGURES = [1, 2, 3, 4, 5, 7, 8]
FIGURE_NAMES = {
    1: "data_structure_and_validation_design",
    2: "UTS_source_blocked_performance",
    3: "UTS_SHAP_importance_and_effects",
    4: "UTS_credibility_and_transfer",
    5: "target_positioning_and_MTL",
    6: "matched_subset_robustness",
    7: "nested_model_and_feature_decisions",
    8: "source_and_dataset_heterogeneity",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        return sum(1 for _ in reader), len(header)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_canonical_files() -> None:
    for rel, source in CANONICAL_FILES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination = OUT / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    figure_source = ROOT / "outputs" / "paper_results_v2" / "figures"
    for number, stem in FIGURE_NAMES.items():
        for ext in (".png", ".pdf"):
            source = figure_source / f"Fig{number}_{stem}{ext}"
            if not source.exists():
                raise FileNotFoundError(source)
            if number in MAIN_FIGURES:
                destination = OUT / "figures_main" / source.name
            else:
                destination = (
                    OUT
                    / "figures_supplement"
                    / f"FigS1_{stem}{ext}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    figure_data_source = ROOT / "outputs" / "paper_results_v2" / "figure_data"
    figure_data_destination = OUT / "figure_data"
    if figure_data_destination.exists():
        shutil.rmtree(figure_data_destination)
    shutil.copytree(figure_data_source, figure_data_destination)


def freeze_manifest() -> list[dict]:
    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        rel = path.relative_to(OUT).as_posix()
        if (
            path.name
            in {
                "freeze_manifest.csv",
                "freeze_manifest.json",
                "freeze_summary.json",
                "reference_pdf_audit_partial.csv",
                "workbook_key_inspect.ndjson",
                "workbook_formula_error_scan.ndjson",
                "提交冻结版_数据字典与最终表格.xlsx",
                "提交冻结版_数据字典与最终表格.xlsx.inspect.ndjson",
            }
            or rel.startswith("workbook_previews/")
            or path.name.startswith("submission_workbook_contact_")
        ):
            continue
        n_rows = ""
        n_cols = ""
        if path.suffix.lower() == ".csv":
            n_rows, n_cols = csv_shape(path)
        rows.append(
            {
                "Relative_Path": rel,
                "Category": rel.split("/", 1)[0],
                "Bytes": path.stat().st_size,
                "Rows": n_rows,
                "Columns": n_cols,
                "SHA256": sha256(path),
                "Frozen_At": datetime.now().isoformat(timespec="seconds"),
            }
        )
    write_csv(OUT / "manifests" / "freeze_manifest.csv", rows)
    (OUT / "manifests" / "freeze_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


def clean_doi(value: str) -> str:
    value = value.strip().rstrip(".,;:)]}>")
    value = value.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    return value.lower()


def reference_role(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("lundberg", "nips-2017", "barredoarrieta")):
        return "可解释性方法"
    if any(x in n for x in ("breiman", "chen2016", "akiba", "pedregosa", "1912.01703")):
        return "模型或软件方法"
    if any(
        x in n
        for x in (
            "cross-validation",
            "varma",
            "vovk",
            "yates",
            "samanta",
            "rj-2023",
            "3736575",
            "matfold",
            "piis266",
            "sciadv",
            "setting-standards",
            "stats-04",
        )
    ):
        return "验证、可信度或报告规范"
    if any(
        x in n
        for x in (
            "ramprasad",
            "ward2016",
            "butler2018",
            "lookman",
            "agrawal",
            "schmidt",
            "kong2021",
            "towards-overcoming",
            "coudert",
        )
    ):
        return "材料信息学与数据方法"
    return "7xxx铝合金领域或相关工作"


def reasonable_title(value: str | None) -> str:
    if not value:
        return ""
    value = " ".join(str(value).replace("\x00", " ").split())
    if value.lower() in {"untitled", "microsoft word", "title"}:
        return ""
    return value[:500]


def first_page_head(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [
        line
        for line in lines
        if 12 <= len(line) <= 220
        and not line.lower().startswith(("http", "doi:", "www."))
    ]
    return " | ".join(lines[:4])[:700]


def audit_references() -> list[dict]:
    preliminary = []
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(REF_DIR.glob("*.pdf")):
        digest = sha256(path)
        by_hash[digest].append(path)
        row = {
            "File_Name": path.name,
            "Bytes": path.stat().st_size,
            "SHA256": digest,
            "Duplicate_Group": digest[:12],
            "Pages": "",
            "Encrypted": "",
            "Text_Characters_Checked": 0,
            "PDF_Metadata_Title": "",
            "First_Page_Head": "",
            "Detected_DOI": "",
            "Reference_Role": reference_role(path.name),
            "Audit_Status": "",
            "Recommended_Action": "",
        }
        try:
            reader = PdfReader(str(path), strict=False)
            row["Pages"] = len(reader.pages)
            row["Encrypted"] = bool(reader.is_encrypted)
            metadata_title = ""
            if reader.metadata:
                metadata_title = reasonable_title(reader.metadata.get("/Title"))
            row["PDF_Metadata_Title"] = metadata_title
            # The first page is sufficient for integrity, title and DOI checks and
            # avoids very slow extraction from long scanned theses.
            page_indexes = [0] if len(reader.pages) else []
            texts = []
            for idx in sorted(set(page_indexes)):
                try:
                    texts.append(reader.pages[idx].extract_text() or "")
                except Exception:
                    texts.append("")
            combined = "\n".join(texts)
            row["Text_Characters_Checked"] = len(combined.strip())
            row["First_Page_Head"] = first_page_head(texts[0] if texts else "")
            dois = sorted({clean_doi(x) for x in DOI_RE.findall(combined)})
            row["Detected_DOI"] = "; ".join(dois[:8])
            if row["Text_Characters_Checked"] >= 200:
                row["Audit_Status"] = "通过-文件可打开且文字层可检索"
                row["Recommended_Action"] = "保留"
            else:
                row["Audit_Status"] = "需视觉核对-扫描件或文字层不足"
                row["Recommended_Action"] = "保留并进行人工题名/页码核对"
            if not row["Detected_DOI"]:
                row["Audit_Status"] += "；PDF内未识别DOI"
        except Exception as exc:
            row["Audit_Status"] = f"失败-无法读取PDF：{type(exc).__name__}"
            row["Recommended_Action"] = "重新下载或修复"
        preliminary.append(row)
        write_csv(OUT / "manifests" / "reference_pdf_audit_partial.csv", preliminary)

    preferred_duplicate = "azarniya2019.pdf"
    for row in preliminary:
        members = by_hash[row["SHA256"]]
        if len(members) > 1:
            if row["File_Name"] == preferred_duplicate:
                row["Audit_Status"] = "通过-与其他文件内容完全重复；指定为保留件"
                row["Recommended_Action"] = "仅保留此文件用于正式文献库"
            else:
                row["Audit_Status"] = "重复文件-SHA256完全一致"
                row["Recommended_Action"] = f"不引用此副本；保留件为{preferred_duplicate}"

    visually_checked_low_text = {
        "引言14.Deep learning method for predicting the...f aluminum alloys with small data sets.pdf",
        "引言26.Machine learning-aided design of aluminum alloys with high performance.pdf",
        "引言32.Manipulation of mechanical properties o...f machine learning and key experiments.pdf",
        "引言48.模仿Synchronously enhancing the strength, t...oys via interpretable machine learning.pdf",
        "引言7.A feasibility study of machine learning... wrought aluminum alloys as an example.pdf",
        "引言Accelerated discovery of Al-Zn-Mg-Cu al...nd high-plasticity by machine learning.pdf",
        "相关工作10.An Explainable Deep Learning Model Base...operty Relationship of Aluminum Alloys.pdf",
        "相关工作12.Composition design of 7XXX aluminum all...king resistance using machine learning.pdf",
        "相关工作17.Diffusion Model for Inverse Design of 7... Aluminum Alloys with Desired Property.pdf",
        "相关工作18.Discovery of ultra-high strength alumin...rpretable chain-based machine learning.pdf",
        "相关工作24.Knowledge-aware design of high-strength...n aluminum alloys via machine learning.pdf",
        "相关工作展望41.Predicting mechanical properties in alu... and physics-based feature engineering.pdf",
        "相关工作讨论40.Physical metallurgy-guided machine lear...ticity optimization in aluminum alloys.pdf",
    }
    for row in preliminary:
        if row["File_Name"] in visually_checked_low_text:
            row["Audit_Status"] = "通过-已视觉核对首页，题名与目标文献一致；无可检索文字层"
            row["Recommended_Action"] = "可保留；引用信息按首页或出版页面录入"
        if row["File_Name"] == "备选46.s12613-019-1894-6.pdf":
            row["Audit_Status"] = "通过-已视觉核对；研究对象为Cu-Al合金而非7xxx铝合金"
            row["Recommended_Action"] = "不纳入正文参考文献"
    write_csv(OUT / "manifests" / "reference_pdf_audit.csv", preliminary)
    return preliminary


DEFINITIONS = {
    "Model_Row_ID": ("记录标识", "", "建模表中的唯一行标识"),
    "Include_Main_Model": ("审计标记", "0/1", "是否进入原主建模候选范围"),
    "Sensitivity_Group": ("审计标记", "", "主分析或敏感性分析分层"),
    "Dataset": ("来源元数据", "", "数据集名称；用于迁移验证，不进入模型"),
    "Source_Group": ("来源元数据", "", "来源独占分组；同组仅进入一个外层折"),
    "DOI": ("来源元数据", "", "原始文献DOI；不得进入预测特征"),
    "Source_Reference": ("来源元数据", "", "可读的文献或数据来源说明"),
    "Source_File": ("来源元数据", "", "上游文件名"),
    "Source_Sheet": ("来源元数据", "", "上游工作表名"),
    "Original_Row_ID": ("记录标识", "", "上游数据中的原始行号"),
    "Original_Sample_ID": ("记录标识", "", "上游样本编号"),
    "Alloy_Grade": ("材料元数据", "", "合金名义牌号"),
    "Source_Type": ("来源元数据", "", "文献、公开数据集等来源类型"),
    "Value_Type": ("质量元数据", "", "实测值、估读值或整理值类型"),
    "Evidence_Level": ("质量元数据", "", "证据等级"),
    "Quality_Flag": ("质量元数据", "", "数据质量标记"),
    "Condition_Original": ("工艺元数据", "", "原始状态/处理条件文本"),
    "Temper": ("工艺元数据", "", "热处理状态"),
    "Process_Category": ("工艺元数据", "", "制造或加工类别"),
    "Quench_Medium": ("工艺元数据", "", "淬火介质"),
    "Al": ("化学成分", "wt.%", "铝含量；部分记录由余量计算"),
    "Zn": ("化学成分", "wt.%", "锌含量"),
    "Mg": ("化学成分", "wt.%", "镁含量"),
    "Cu": ("化学成分", "wt.%", "铜含量"),
    "Si": ("化学成分", "wt.%", "硅含量"),
    "Fe": ("化学成分", "wt.%", "铁含量"),
    "Mn": ("化学成分", "wt.%", "锰含量"),
    "Cr": ("化学成分", "wt.%", "铬含量"),
    "Ti": ("化学成分", "wt.%", "钛含量"),
    "Zr": ("化学成分", "wt.%", "锆含量"),
    "V": ("化学成分", "wt.%", "钒含量"),
    "Li": ("化学成分", "wt.%", "锂含量"),
    "Ni": ("化学成分", "wt.%", "镍含量"),
    "Sc": ("化学成分", "wt.%", "钪含量"),
    "Ag": ("化学成分", "wt.%", "银含量"),
    "Ce": ("化学成分", "wt.%", "铈含量"),
    "Sol_Temp_C": ("工艺参数", "°C", "固溶温度"),
    "Sol_Time_h": ("工艺参数", "h", "固溶时间"),
    "Cooling_Rate_K_s": ("工艺参数", "K/s", "冷却速率"),
    "Age1_Temp_C": ("工艺参数", "°C", "第一级时效温度"),
    "Age1_Time_h": ("工艺参数", "h", "第一级时效时间"),
    "Age2_Temp_C": ("工艺参数", "°C", "第二级时效温度"),
    "Age2_Time_h": ("工艺参数", "h", "第二级时效时间"),
    "Test_Temp_C": ("测试参数", "°C", "力学性能测试温度"),
    "Zn_Mg_Ratio": ("成分派生描述符", "", "Zn/Mg；分母接近零时按安全规则置缺失"),
    "Zn_Cu_Ratio": ("成分派生描述符", "", "Zn/Cu；仅用于敏感性分析"),
    "Mg_Cu_Ratio": ("成分派生描述符", "", "Mg/Cu；仅用于敏感性分析"),
    "Zn_Mg_Sum": ("成分派生描述符", "wt.%", "Zn+Mg"),
    "Zn_Mg_Cu_Sum": ("成分派生描述符", "wt.%", "Zn+Mg+Cu"),
    "Zn_Share_ZnMgCu": ("成分派生描述符", "", "Zn/(Zn+Mg+Cu)"),
    "Reported_Solute_Sum": ("成分派生描述符", "wt.%", "已报告合金元素含量之和"),
    "Process_Missing_Count": ("缺失性描述符", "count", "关键工艺字段缺失数量"),
    "YS_0.2pct_MPa": ("预测目标", "MPa", "0.2%屈服强度"),
    "UTS_MPa": ("预测目标", "MPa", "抗拉强度"),
    "EL_pct": ("预测目标", "%", "断后伸长率"),
    "Hardness_HV": ("探索目标", "HV", "维氏硬度；不进入本轮主模型"),
    "Mask_YS": ("标签掩码", "0/1", "YS标签是否存在"),
    "Mask_UTS": ("标签掩码", "0/1", "UTS标签是否存在"),
    "Mask_EL": ("标签掩码", "0/1", "EL标签是否存在"),
    "Observed_Target_Count": ("标签结构", "count", "YS、UTS和EL中已观测目标数量"),
    "Four_Target_Complete": ("标签结构", "0/1", "四目标是否完整"),
    "Outer_Fold": ("验证设计", "1-5", "来源独占外层折编号"),
    "Triple_Target_Complete": ("标签结构", "0/1", "YS、UTS和EL是否完整匹配"),
    "Sample_Key": ("记录标识", "", "Dataset与Original_Sample_ID构成的匹配键"),
    "Audit_Tier": ("审计标记", "", "研究范围审计层级"),
    "Audit_Decision": ("审计标记", "", "保留、敏感性或排除决定"),
    "Include_Scope_Clean_Model": ("审计标记", "0/1", "是否进入范围清洁模型"),
    "Decision_Reason_CN": ("审计说明", "", "范围决定的中文理由"),
}


def is_missing(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "na", "n/a", "none", "null"}


def build_data_dictionary() -> tuple[list[dict], list[dict]]:
    cohort_paths = {
        "YS_307": OUT / "data" / "YS_scope_clean_307.csv",
        "UTS_675": OUT / "data" / "UTS_scope_clean_675.csv",
        "EL_537": OUT / "data" / "EL_scope_clean_537.csv",
        "MTL_689": OUT / "data" / "Partial_label_MTL_689.csv",
        "Matched_266": OUT / "data" / "Matched_complete_266.csv",
    }
    cohort_rows = {name: read_csv_rows(path) for name, path in cohort_paths.items()}
    columns = []
    for rows in cohort_rows.values():
        for col in rows[0].keys():
            if col not in columns:
                columns.append(col)

    dictionary = []
    for col in columns:
        category, unit, definition = DEFINITIONS.get(
            col, ("待补充定义", "", "需在投稿前确认字段含义")
        )
        row = {
            "Variable": col,
            "Category": category,
            "Unit_or_Coding": unit,
            "Definition": definition,
            "UTS_Final_Model_Role": (
                "最终直接输入特征"
                if col in {"Zn", "Mg", "Cu", "Fe", "Zr"}
                else "目标"
                if col == "UTS_MPa"
                else "来源分组/审计，不进入模型"
                if category
                in {
                    "记录标识",
                    "来源元数据",
                    "质量元数据",
                    "审计标记",
                    "审计说明",
                    "验证设计",
                    "标签结构",
                    "标签掩码",
                }
                else "敏感性分析或未进入最终UTS模型"
            ),
        }
        for cohort, rows in cohort_rows.items():
            if col not in rows[0]:
                row[f"{cohort}_Missing_pct"] = ""
            else:
                missing = sum(is_missing(r.get(col)) for r in rows)
                row[f"{cohort}_Missing_pct"] = missing / len(rows)
        dictionary.append(row)

    summaries = []
    for cohort, rows in cohort_rows.items():
        target = {
            "YS_307": "YS_0.2pct_MPa",
            "UTS_675": "UTS_MPa",
            "EL_537": "EL_pct",
            "MTL_689": "YS/UTS/EL部分标签",
            "Matched_266": "YS+UTS+EL完整",
        }[cohort]
        summaries.append(
            {
                "Cohort": cohort,
                "Rows": len(rows),
                "Sources": len({r.get("Source_Group", "") for r in rows}),
                "Datasets": len({r.get("Dataset", "") for r in rows}),
                "Target_or_Use": target,
                "Paper_Role": {
                    "UTS_675": "主要确认性预测与SHAP解释",
                    "YS_307": "次要探索性预测",
                    "EL_537": "次要探索性预测",
                    "MTL_689": "部分标签MTL对照",
                    "Matched_266": "同一样本可比性与稳健性验证",
                }[cohort],
            }
        )
    write_csv(OUT / "tables" / "data_dictionary.csv", dictionary)
    write_csv(OUT / "tables" / "cohort_summary.csv", summaries)
    return dictionary, summaries


def build_final_tables() -> dict[str, list[dict]]:
    uts = read_csv_rows(OUT / "results" / "UTS_final_metrics.csv")[0]
    ys_el = read_csv_rows(OUT / "results" / "YS_EL_scope_metrics.csv")
    ys = next(r for r in ys_el if r["Task"] == "YS" and r["Variant"] == "Scope_Clean")
    el = next(r for r in ys_el if r["Task"] == "EL" and r["Variant"] == "Scope_Clean")
    main_results = [
        {
            "Task": "UTS",
            "Role": "主要确认性目标",
            "Model": "RF+XGBoost等权集成",
            "Rows": int(uts["Rows"]),
            "Sources": int(uts["Sources"]),
            "Features": "Zn, Mg, Cu, Fe, Zr",
            "R2": float(uts["R2"]),
            "RMSE": float(uts["RMSE"]),
            "MAE": float(uts["MAE"]),
            "Unit": "MPa",
            "Interpretation": "中等、来源感知的外推能力",
        },
        {
            "Task": "YS",
            "Role": "次要探索性目标",
            "Model": "Random Forest",
            "Rows": int(ys["Rows"]),
            "Sources": int(ys["Sources"]),
            "Features": "直接成分特征",
            "R2": float(ys["R2"]),
            "RMSE": float(ys["RMSE"]),
            "MAE": float(ys["MAE"]),
            "Unit": "MPa",
            "Interpretation": "较弱，不作高精度预测声明",
        },
        {
            "Task": "EL",
            "Role": "次要探索性目标",
            "Model": "Random Forest",
            "Rows": int(el["Rows"]),
            "Sources": int(el["Sources"]),
            "Features": "直接成分特征",
            "R2": float(el["R2"]),
            "RMSE": float(el["RMSE"]),
            "MAE": float(el["MAE"]),
            "Unit": "%",
            "Interpretation": "较弱，不作高精度预测声明",
        },
    ]

    bootstrap = read_csv_rows(OUT / "results" / "UTS_source_bootstrap.csv")
    r2_boot = next(r for r in bootstrap if r["Metric"] == "R2")
    ad = read_csv_rows(OUT / "results" / "UTS_applicability_domain.csv")
    ad_inside = next(r for r in ad if r["AD_Status"] == "Inside")
    ad_outside = next(r for r in ad if r["AD_Status"] == "Outside")
    lodo = read_csv_rows(OUT / "results" / "UTS_leave_one_dataset_out.csv")[0]
    intervals = read_csv_rows(OUT / "results" / "UTS_prediction_intervals.csv")
    row90 = next(
        r
        for r in intervals
        if r["Scope"] == "All"
        and r["Method"] == "RowCrossConformal"
        and r["Nominal_Coverage"] == "0.9"
    )
    row95 = next(
        r
        for r in intervals
        if r["Scope"] == "All"
        and r["Method"] == "RowCrossConformal"
        and r["Nominal_Coverage"] == "0.95"
    )
    credibility = [
        {
            "Analysis": "来源聚类bootstrap",
            "Metric": "R2",
            "Estimate": float(r2_boot["Bootstrap_Median"]),
            "Lower": float(r2_boot["CI95_Lower"]),
            "Upper": float(r2_boot["CI95_Upper"]),
            "Meaning": "跨来源重采样后的模型稳定性",
        },
        {
            "Analysis": "留一数据集迁移验证",
            "Metric": "R2",
            "Estimate": float(lodo["R2"]),
            "Lower": "",
            "Upper": "",
            "Meaning": "六个数据集之间的迁移能力",
        },
        {
            "Analysis": "适用域内",
            "Metric": "RMSE",
            "Estimate": float(ad_inside["RMSE"]),
            "Lower": "",
            "Upper": "",
            "Meaning": f"{float(ad_inside['Row_Fraction']):.1%}样本位于适用域内",
        },
        {
            "Analysis": "适用域外",
            "Metric": "RMSE",
            "Estimate": float(ad_outside["RMSE"]),
            "Lower": "",
            "Upper": "",
            "Meaning": "域外误差明显增加",
        },
        {
            "Analysis": "行级交叉保形区间",
            "Metric": "90%覆盖率",
            "Estimate": float(row90["Row_Coverage"]),
            "Lower": "",
            "Upper": "",
            "Meaning": f"平均宽度{float(row90['Mean_Width_MPa']):.1f} MPa",
        },
        {
            "Analysis": "行级交叉保形区间",
            "Metric": "95%覆盖率",
            "Estimate": float(row95["Row_Coverage"]),
            "Lower": "",
            "Upper": "",
            "Meaning": f"平均宽度{float(row95['Mean_Width_MPa']):.1f} MPa",
        },
    ]

    correlations = read_csv_rows(OUT / "results" / "Matched_target_correlations.csv")
    matched_metrics = read_csv_rows(OUT / "results" / "Matched_fair_comparison.csv")
    matched_table = []
    for row in correlations:
        matched_table.append(
            {
                "Section": "目标相关性",
                "Level_or_Model": row["Level"],
                "Task_or_Pair": f"{row['Target_A']}-{row['Target_B']}",
                "Rows": row["N"],
                "R2_or_Spearman": float(row["Spearman_r"]),
                "RMSE": "",
                "MAE": "",
                "Conclusion": "完整匹配子集上的秩相关",
            }
        )
    for row in matched_metrics:
        matched_table.append(
            {
                "Section": "同样本公平比较",
                "Level_or_Model": row["Model"],
                "Task_or_Pair": row["Task"],
                "Rows": row["Rows"],
                "R2_or_Spearman": float(row["R2"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "Conclusion": "匹配子集是更困难且分布不同的验证域",
            }
        )

    figure_table = []
    for number, stem in FIGURE_NAMES.items():
        main = number in MAIN_FIGURES
        figure_table.append(
            {
                "Frozen_Number": f"Fig.{number}" if main else "Fig.S1",
                "Original_Number": f"Fig.{number}",
                "Title_Key": stem,
                "Location": "正文" if main else "补充材料",
                "Decision": "保留" if main else "移出正文",
                "Reason": (
                    "承担主线证据"
                    if main
                    else "同样本可比性很重要，但四联图信息密度高且正文已有目标定位；正文报告关键数值即可"
                ),
            }
        )

    write_csv(OUT / "tables" / "Table1_final_model_performance.csv", main_results)
    write_csv(OUT / "tables" / "Table2_UTS_credibility.csv", credibility)
    write_csv(OUT / "tables" / "Table3_matched_subset.csv", matched_table)
    write_csv(OUT / "tables" / "figure_freeze_decision.csv", figure_table)
    return {
        "main_results": main_results,
        "credibility": credibility,
        "matched": matched_table,
        "figures": figure_table,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_canonical_files()
    reference_rows = audit_references()
    dictionary, cohorts = build_data_dictionary()
    tables = build_final_tables()
    manifest = freeze_manifest()
    summary = {
        "freeze_id": "submission_freeze_2026-07-25",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_data_rows": {
            row["Cohort"]: row["Rows"] for row in cohorts
        },
        "main_figures": len(MAIN_FIGURES),
        "supplement_figures": 1,
        "reference_pdf_count": len(reference_rows),
        "reference_open_failures": sum(
            str(r["Audit_Status"]).startswith("失败") for r in reference_rows
        ),
        "reference_low_text_files": sum(
            "需视觉核对" in str(r["Audit_Status"]) for r in reference_rows
        ),
        "reference_exact_duplicate_files": sum(
            str(r["Audit_Status"]).startswith("重复文件") for r in reference_rows
        ),
        "data_dictionary_variables": len(dictionary),
        "manifest_file_count": len(manifest),
        "final_uts_r2": tables["main_results"][0]["R2"],
    }
    (OUT / "manifests" / "freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
