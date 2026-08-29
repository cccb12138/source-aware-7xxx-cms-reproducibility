from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
SOURCE_ROOT = Path(r"C:\Users\dell\OneDrive\Desktop\论文优化数据")
OUTPUT_ROOT = Path(r"F:\CC\outputs\uts_systematic_scope_audit")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASK = "UTS"
TARGET = "UTS_MPa"
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260801

DEFINITE_EXCLUDE_DOIS = {
    "10.1016/j.jallcom.2015.10.108": "颗粒增强复合材料/摩擦搅拌加工，不属于普通单体7xxx铝合金",
    "10.1016/j.corsci.2008.09.031": "焊接接头在3.5% NaCl中的慢应变速率试验，不是常规室温母材UTS",
}

COMPOSITE_PATTERNS = [
    r"\bcomposite\b",
    r"matrix composite",
    r"reinforced",
    r"tib2 particle",
    r"sic particle",
    r"b4c",
    r"graphene",
    r"carbon nanotube",
    r"复合材料",
    r"颗粒增强",
]
WELD_PATTERNS = [
    r"friction stir weld",
    r"welded joint",
    r"weld joint",
    r"搅拌摩擦焊",
    r"焊接接头",
]
NONSTANDARD_TEST_PATTERNS = [
    r"slow strain rate",
    r"\bssrt\b",
    r"high[- ]temperature mechanical propert",
    r"elevated[- ]temperature tensile",
    r"hot tensile",
    r"高温力学性能",
]
SEVERE_PROCESS_PATTERNS = [
    r"\becap\b",
    r"equal[- ]channel angular",
    r"high pressure torsion",
    r"\bhpt\b",
    r"severe plastic deformation",
    r"multidirectional compression",
    r"friction stir processing",
    r"cryoroll",
    r"accumulative roll bonding",
    r"等通道转角",
    r"高压扭转",
    r"多向压缩",
    r"剧烈塑性变形",
]
POWDER_AM_PATTERNS = [
    r"powder metallurg",
    r"selective laser",
    r"additive manufactur",
    r"laser powder bed",
    r"cold spray",
    r"增材制造",
]


def norm_doi(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return text.rstrip("/.")


def compact_unique(values) -> str:
    seen = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.append(text)
    return " | ".join(seen)


def contains_any(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.I)]


def read_source_context() -> pd.DataFrame:
    frames = []

    acta = pd.read_excel(
        SOURCE_ROOT / "7xxx文献数据_UTS-KIC-ISSRT_清洗后835条.xlsx",
        sheet_name="UTS_7xxx_461",
    )
    acta["Context_Dataset"] = "Acta_UTS461"
    frames.append(acta)

    supp = pd.read_excel(
        SOURCE_ROOT / "7xxx文献补充数据_清洗后260条.xlsx",
        sheet_name="可入库数据_260",
    )
    supp["Context_Dataset"] = "补充文献260"
    frames.append(supp)

    aged = pd.read_excel(
        SOURCE_ROOT / "Mendeley四数据集_7000系清洗与建模缺口.xlsx",
        sheet_name="可追溯重建_Aged194",
    )
    aged["Context_Dataset"] = "Aged可追溯194"
    frames.append(aged)

    base = pd.read_excel(
        SOURCE_ROOT / "7xxx合并数据_文献63条+公开84条.xlsx",
        sheet_name="合并数据_147",
        header=1,
    )
    base["Context_Dataset"] = "基础合并147"
    base = base.rename(
        columns={
            "Caption_Original": "Original_Source",
            "Quality_Flags": "Quality_Flag",
            "Test_Temp(°C)": "Test_Temp_C",
            "UTS (MPa)": "UTS_MPa",
        }
    )
    frames.append(base)

    matinfo = pd.read_excel(
        SOURCE_ROOT / "7xxx文献数据_材料信息学_24条.xlsx",
        sheet_name="7000系数据_24",
    )
    matinfo["Context_Dataset"] = "材料信息学24"
    matinfo["Original_Source"] = (
        matinfo.get("Original_Database", "").astype(str)
        + " | "
        + matinfo.get("Aggregation_Note", "").astype(str)
    )
    frames.append(matinfo)

    four = pd.read_excel(
        SOURCE_ROOT / "四篇文献_7xxx可用数据与数据缺口统计.xlsx",
        sheet_name="高置信实验数据_12",
    )
    four["Context_Dataset"] = "四篇论文12"
    frames.append(four)

    keep = [
        "Context_Dataset",
        "Source_Group",
        "Original_Source",
        "DOI",
        "Source_File",
        "Source_Type",
        "Alloy_Grade",
        "Fabrication",
        "Process_Regime_Claimed",
        "Quality_Flag",
        "Test_Temp_C",
        "Temper_Original",
        "Alloy ID",
        "数据来源",
    ]
    normalized = []
    for frame in frames:
        part = frame.copy()
        for column in keep:
            if column not in part.columns:
                part[column] = np.nan
        normalized.append(part[keep])
    context = pd.concat(normalized, ignore_index=True)
    return context


def build_source_audit(data: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    source_rows = []
    for source, part in data.groupby("Source_Group", sort=False):
        external = context.loc[context["Source_Group"].astype(str).eq(str(source))]
        doi_values = [norm_doi(value) for value in pd.concat([part["DOI"], external["DOI"]])]
        doi_values = sorted({value for value in doi_values if value})
        text_values = []
        for column in [
            "Source_Reference",
            "Source_File",
            "Condition_Original",
            "Temper",
            "Process_Category",
            "Alloy_Grade",
        ]:
            text_values.extend(part[column].tolist())
        for column in [
            "Original_Source",
            "Source_File",
            "Alloy_Grade",
            "Fabrication",
            "Process_Regime_Claimed",
            "Quality_Flag",
            "Temper_Original",
            "Alloy ID",
            "数据来源",
        ]:
            text_values.extend(external[column].tolist())
        source_text = compact_unique(text_values)
        search_text = source_text.lower()

        exclusion_reasons = []
        doi_reasons = [DEFINITE_EXCLUDE_DOIS[doi] for doi in doi_values if doi in DEFINITE_EXCLUDE_DOIS]
        exclusion_reasons.extend(doi_reasons)
        composite_hits = contains_any(search_text, COMPOSITE_PATTERNS)
        weld_hits = contains_any(search_text, WELD_PATTERNS)
        nonstandard_hits = contains_any(search_text, NONSTANDARD_TEST_PATTERNS)
        if composite_hits:
            exclusion_reasons.append("题名/样本描述表明为增强复合材料，而非普通单体7xxx合金")
        if weld_hits:
            exclusion_reasons.append("题名/样本描述表明为焊缝或焊接接头，而非母材性能")
        if nonstandard_hits:
            exclusion_reasons.append("题名/样本描述表明为慢应变速率或高温等非标准目标条件")

        test_temps = pd.to_numeric(
            pd.concat([part["Test_Temp_C"], external["Test_Temp_C"]]),
            errors="coerce",
        ).dropna()
        nonambient_temp = bool(((test_temps < 10) | (test_temps > 40)).any())
        if nonambient_temp:
            exclusion_reasons.append("明确测试温度不在10–40 °C的近室温范围")

        severe_hits = contains_any(search_text, SEVERE_PROCESS_PATTERNS)
        powder_hits = contains_any(search_text, POWDER_AM_PATTERNS)
        secondary = bool(
            part["Source_Type"].fillna("").astype(str).str.contains("Secondary|Public_Database", case=False, regex=True).any()
            or part["Dataset"].isin(["Aged可追溯194", "材料信息学24"]).any()
        )

        if exclusion_reasons:
            decision = "Exclude_Definite_Scope_Mismatch"
            tier = "C_Exclude"
            use_main = False
        elif secondary:
            decision = "Retain_Secondary_Sensitivity"
            tier = "B2_Secondary_Database"
            use_main = True
        elif severe_hits or powder_hits:
            decision = "Retain_Special_Process_Sensitivity"
            tier = "B1_Special_Process"
            use_main = True
        else:
            decision = "Retain_Core_or_Process_Unresolved"
            tier = "A_Core_or_Unresolved"
            use_main = True

        source_rows.append(
            {
                "Source_Group": source,
                "Dataset": compact_unique(part["Dataset"]),
                "Rows": len(part),
                "Outer_Folds": compact_unique(part["Outer_Fold"]),
                "DOI_Normalized": " | ".join(doi_values),
                "Source_Type": compact_unique(part["Source_Type"]),
                "Evidence_Level": compact_unique(part["Evidence_Level"]),
                "Alloy_Grade": compact_unique(part["Alloy_Grade"]),
                "UTS_Min_MPa": pd.to_numeric(part[TARGET], errors="coerce").min(),
                "UTS_Max_MPa": pd.to_numeric(part[TARGET], errors="coerce").max(),
                "Test_Temp_Min_C": test_temps.min() if len(test_temps) else np.nan,
                "Test_Temp_Max_C": test_temps.max() if len(test_temps) else np.nan,
                "Composite_Keyword_Hits": " | ".join(composite_hits),
                "Weld_Keyword_Hits": " | ".join(weld_hits),
                "Nonstandard_Test_Hits": " | ".join(nonstandard_hits),
                "Special_Process_Hits": " | ".join(severe_hits + powder_hits),
                "Secondary_Provenance": secondary,
                "Audit_Tier": tier,
                "Audit_Decision": decision,
                "Include_Scope_Clean_Model": use_main,
                "Decision_Reason_CN": "；".join(dict.fromkeys(exclusion_reasons)) if exclusion_reasons else (
                    "二手数据库/二次汇编，保留在主范围但必须单列来源敏感性"
                    if secondary
                    else "特殊加工路径，真实但成分之外工艺影响较强，保留并单列敏感性"
                    if severe_hits or powder_hits
                    else "未发现明确研究对象或测试条件越界证据"
                ),
                "Source_Context": source_text,
            }
        )
    audit = pd.DataFrame(source_rows)
    order = {
        "C_Exclude": 0,
        "B1_Special_Process": 1,
        "B2_Secondary_Database": 2,
        "A_Core_or_Unresolved": 3,
    }
    audit["_order"] = audit["Audit_Tier"].map(order)
    return audit.sort_values(["_order", "Rows"], ascending=[True, False]).drop(columns="_order")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    return {
        "Rows": len(frame),
        "Sources": frame["Source_Group"].nunique(),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def load_parameters():
    root = config.OUTPUT_DIRS["single"]
    rf = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    return rf.loc[rf["Task"].eq(TASK)].copy(), xgb.loc[xgb["Task"].eq(TASK)].copy()


def evaluate_variant(name, data, model_module, rf_params, xgb_params):
    predictions = []
    fold_metrics = []
    for fold in sorted(data["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = data.loc[data["Outer_Fold"].eq(fold)].copy()
        if len(test) == 0 or len(train) == 0:
            continue
        rf_row = rf_params.loc[rf_params["Outer_Fold"].eq(fold)].iloc[0]
        xgb_row = xgb_params.loc[xgb_params["Outer_Fold"].eq(fold)].iloc[0]
        rf = model_module.rf_tuned(rf_row, SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, SEED + fold)
        rf.fit(train[FEATURES], train[TARGET])
        xgb.fit(train[FEATURES], train[TARGET])
        pred_rf = rf.predict(test[FEATURES])
        pred_xgb = xgb.predict(test[FEATURES])
        y_pred = (pred_rf + pred_xgb) / 2.0
        part = pd.DataFrame(
            {
                "Variant": name,
                "Outer_Fold": fold,
                "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                "Source_Group": test["Source_Group"].to_numpy(),
                "Dataset": test["Dataset"].to_numpy(),
                "y_true": test[TARGET].to_numpy(),
                "pred_rf": pred_rf,
                "pred_xgb": pred_xgb,
                "y_pred": y_pred,
            }
        )
        predictions.append(part)
        fold_metrics.append({"Variant": name, "Outer_Fold": fold, **metric_values(part)})
    oof = pd.concat(predictions, ignore_index=True)
    return oof, pd.DataFrame(fold_metrics), {"Variant": name, **metric_values(oof)}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "strict" / "UTS_with_outer_folds.csv")
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    context = read_source_context()
    audit = build_source_audit(data, context)
    decisions = audit[
        [
            "Source_Group",
            "Audit_Tier",
            "Audit_Decision",
            "Include_Scope_Clean_Model",
            "Decision_Reason_CN",
        ]
    ]
    rows = data.merge(decisions, on="Source_Group", how="left", validate="many_to_one")
    if rows["Audit_Decision"].isna().any():
        raise AssertionError("Some UTS rows were not assigned a scope decision")

    variants = {
        "Original_689": rows.copy(),
        "Scope_Clean": rows.loc[rows["Include_Scope_Clean_Model"]].copy(),
        "Direct_Literature_Sensitivity": rows.loc[
            rows["Include_Scope_Clean_Model"]
            & ~rows["Audit_Tier"].eq("B2_Secondary_Database")
        ].copy(),
        "Conventional_Core_Sensitivity": rows.loc[
            rows["Audit_Tier"].eq("A_Core_or_Unresolved")
        ].copy(),
    }

    model_module = load_module(
        "model_builders_scope_audit",
        PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py",
    )
    rf_params, xgb_params = load_parameters()
    all_oof = []
    all_fold_metrics = []
    overall_metrics = []
    for name, frame in variants.items():
        oof, fold_metrics, overall = evaluate_variant(
            name, frame, model_module, rf_params, xgb_params
        )
        all_oof.append(oof)
        all_fold_metrics.append(fold_metrics)
        overall_metrics.append(overall)

    audit.to_csv(OUTPUT_ROOT / "source_scope_audit_263.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(OUTPUT_ROOT / "row_scope_decisions_689.csv", index=False, encoding="utf-8-sig")
    variants["Scope_Clean"].to_csv(
        OUTPUT_ROOT / "UTS_scope_clean_with_outer_folds.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_oof, ignore_index=True).to_csv(
        OUTPUT_ROOT / "model_variant_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        OUTPUT_ROOT / "model_variant_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metrics = pd.DataFrame(overall_metrics)
    metrics.to_csv(OUTPUT_ROOT / "model_variant_overall_metrics.csv", index=False, encoding="utf-8-sig")

    hierarchy_module = load_module(
        "hierarchical_scope_clean",
        PROJECT_ROOT / "24_uts_hierarchical_source_validation.py",
    )
    hierarchy_rf_row, hierarchy_xgb_row = hierarchy_module.aggregate_parameters()
    hierarchy_metrics, hierarchy_predictions = hierarchy_module.evaluate_holdouts(
        variants["Scope_Clean"],
        model_module,
        hierarchy_rf_row,
        hierarchy_xgb_row,
    )
    hierarchy_pooled = hierarchy_module.pooled_dataset_metrics(hierarchy_predictions)
    hierarchy_metrics.to_csv(
        OUTPUT_ROOT / "scope_clean_hierarchical_holdout_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hierarchy_predictions.to_csv(
        OUTPUT_ROOT / "scope_clean_hierarchical_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hierarchy_pooled.to_csv(
        OUTPUT_ROOT / "scope_clean_leave_one_dataset_out_pooled_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        audit.groupby(["Audit_Tier", "Audit_Decision"], dropna=False)
        .agg(Sources=("Source_Group", "nunique"), Rows=("Rows", "sum"))
        .reset_index()
    )
    summary.to_csv(OUTPUT_ROOT / "scope_decision_summary.csv", index=False, encoding="utf-8-sig")
    excluded = audit.loc[audit["Audit_Tier"].eq("C_Exclude")].copy()
    config_out = {
        "input_rows": len(data),
        "input_sources": int(data["Source_Group"].nunique()),
        "scope_clean_rows": len(variants["Scope_Clean"]),
        "scope_clean_sources": int(variants["Scope_Clean"]["Source_Group"].nunique()),
        "excluded_rows": int(excluded["Rows"].sum()),
        "excluded_sources": len(excluded),
        "features": FEATURES,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "validation": "unchanged five source-exclusive outer folds; frozen prior hyperparameters",
        "selection_principle": "scope mismatch and test-condition rules only; prediction error was not used",
    }
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(config_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("SCOPE SUMMARY")
    print(summary.to_string(index=False))
    print("\nEXCLUDED SOURCES")
    print(
        excluded[
            ["Source_Group", "Dataset", "Rows", "DOI_Normalized", "Decision_Reason_CN", "Source_Context"]
        ].to_string(index=False)
    )
    print("\nMODEL VARIANT METRICS")
    print(metrics.to_string(index=False))
    print("\nSCOPE-CLEAN POOLED LEAVE-ONE-DATASET-OUT")
    print(hierarchy_pooled.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
