from pathlib import Path

import pandas as pd


ROOT = Path(r"F:\CC\outputs\IMMI_supplementary_package\tables\csv")
PUBLIC_SCHEMA = Path(r"F:\CC\outputs\github_release_candidate\data\schema")

CATEGORY = {
    "记录标识": "Record identifier",
    "审计标记": "Audit flag",
    "来源元数据": "Source metadata",
    "材料元数据": "Material metadata",
    "质量元数据": "Quality metadata",
    "工艺元数据": "Processing metadata",
    "化学成分": "Chemical composition",
    "工艺参数": "Processing parameter",
    "测试参数": "Test parameter",
    "成分派生描述符": "Derived composition descriptor",
    "缺失性描述符": "Missingness descriptor",
    "预测目标": "Prediction target",
    "验证设计": "Validation design",
    "审计说明": "Audit note",
    "探索目标": "Exploratory target",
    "标签掩码": "Label mask",
    "标签结构": "Label structure",
}

ROLE = {
    "来源分组/审计，不进入模型": "Source grouping or audit; excluded from predictors",
    "敏感性分析或未进入最终UTS模型": "Sensitivity analysis or excluded from the final UTS model",
    "最终直接输入特征": "Final direct input feature",
    "目标": "Prediction target",
}

DEFINITION = {
    "Model_Row_ID": "Unique row identifier in the modelling table",
    "Include_Main_Model": "Whether the record entered the original main-model candidate scope",
    "Sensitivity_Group": "Stratum for the primary or sensitivity analysis",
    "Dataset": "Dataset name used for transfer validation; excluded from predictors",
    "Source_Group": "Exclusive provenance group assigned to one outer fold only",
    "DOI": "DOI of the original publication; excluded from predictors",
    "Source_Reference": "Human-readable publication or dataset reference",
    "Source_File": "Upstream source filename",
    "Source_Sheet": "Upstream worksheet name",
    "Original_Row_ID": "Original row identifier in the upstream source",
    "Original_Sample_ID": "Original sample identifier in the upstream source",
    "Alloy_Grade": "Nominal alloy designation",
    "Source_Type": "Source type, such as literature or public dataset",
    "Value_Type": "Reported, graph-digitised, or curated value type",
    "Evidence_Level": "Evidence grade assigned during curation",
    "Quality_Flag": "Data-quality flag",
    "Condition_Original": "Original condition or treatment text",
    "Temper": "Temper designation",
    "Process_Category": "Manufacturing or processing category",
    "Quench_Medium": "Quenching medium",
    "Al": "Aluminium content; balance-derived for some records",
    "Zn": "Zinc content",
    "Mg": "Magnesium content",
    "Cu": "Copper content",
    "Si": "Silicon content",
    "Fe": "Iron content",
    "Mn": "Manganese content",
    "Cr": "Chromium content",
    "Ti": "Titanium content",
    "Zr": "Zirconium content",
    "V": "Vanadium content",
    "Li": "Lithium content",
    "Ni": "Nickel content",
    "Sc": "Scandium content",
    "Ag": "Silver content",
    "Ce": "Cerium content",
    "Sol_Temp_C": "Solution-treatment temperature",
    "Sol_Time_h": "Solution-treatment duration",
    "Cooling_Rate_K_s": "Cooling rate",
    "Age1_Temp_C": "First-stage ageing temperature",
    "Age1_Time_h": "First-stage ageing duration",
    "Age2_Temp_C": "Second-stage ageing temperature",
    "Age2_Time_h": "Second-stage ageing duration",
    "Test_Temp_C": "Mechanical-test temperature",
    "Zn_Mg_Ratio": "Zn/Mg; set to missing when the denominator approaches zero",
    "Zn_Cu_Ratio": "Zn/Cu; used only in sensitivity analysis",
    "Mg_Cu_Ratio": "Mg/Cu; used only in sensitivity analysis",
    "Zn_Mg_Sum": "Zn + Mg",
    "Zn_Mg_Cu_Sum": "Zn + Mg + Cu",
    "Zn_Share_ZnMgCu": "Zn / (Zn + Mg + Cu)",
    "Reported_Solute_Sum": "Sum of all reported alloying-element contents",
    "Process_Missing_Count": "Number of missing key processing fields",
    "YS_0.2pct_MPa": "0.2% proof strength",
    "Outer_Fold": "Source-exclusive outer-fold number",
    "Audit_Tier": "Research-scope audit tier",
    "Audit_Decision": "Retain, sensitivity, or exclusion decision",
    "Include_Scope_Clean_Model": "Whether the record entered the scope-clean model",
    "Decision_Reason_CN": "Original Chinese-language reason for the scope decision",
    "UTS_MPa": "Ultimate tensile strength",
    "EL_pct": "Elongation after fracture",
    "Hardness_HV": "Vickers hardness; excluded from the present main modelling",
    "Mask_YS": "Whether the YS label is observed",
    "Mask_UTS": "Whether the UTS label is observed",
    "Mask_EL": "Whether the EL label is observed",
    "Observed_Target_Count": "Number of observed targets among YS, UTS, and EL",
    "Four_Target_Complete": "Whether all four original targets are complete",
    "Triple_Target_Complete": "Whether YS, UTS, and EL are jointly observed",
    "Sample_Key": "Matching key formed from Dataset and Original_Sample_ID",
}


def save(frame: pd.DataFrame, filename: str, public: bool = False) -> None:
    frame.to_csv(ROOT / filename, index=False, encoding="utf-8-sig")
    if public:
        frame.to_csv(PUBLIC_SCHEMA / filename, index=False, encoding="utf-8-sig")


dictionary = pd.read_csv(ROOT / "data_dictionary.csv")
dictionary["Category"] = dictionary["Category"].replace(CATEGORY)
dictionary["Definition"] = dictionary["Variable"].map(DEFINITION).fillna(dictionary["Definition"])
dictionary["UTS_Final_Model_Role"] = dictionary["UTS_Final_Model_Role"].replace(ROLE)
save(dictionary, "data_dictionary.csv", public=True)

cohort = pd.read_csv(ROOT / "cohort_summary.csv")
cohort["Target_or_Use"] = cohort["Target_or_Use"].replace({
    "YS/UTS/EL部分标签": "Partially labelled YS/UTS/EL",
    "YS+UTS+EL完整": "Complete YS+UTS+EL",
})
cohort["Paper_Role"] = cohort["Paper_Role"].replace({
    "次要探索性预测": "Secondary exploratory prediction",
    "主要确认性预测与SHAP解释": "Primary confirmatory prediction and SHAP interpretation",
    "部分标签MTL对照": "Partial-label MTL comparison",
    "同一样本可比性与稳健性验证": "Same-sample comparability and robustness validation",
})
save(cohort, "cohort_summary.csv", public=True)

table1 = pd.read_csv(ROOT / "Table1_final_model_performance.csv")
for column in ["Role", "Model", "Features", "Interpretation"]:
    table1[column] = table1[column].replace({
        "主要确认性目标": "Primary confirmatory target",
        "次要探索性目标": "Secondary exploratory target",
        "RF+XGBoost等权集成": "Equal-weight RF+XGBoost ensemble",
        "直接成分特征": "Direct composition features",
        "中等、来源感知的外推能力": "Moderate source-aware generalisation",
        "较弱，不作高精度预测声明": "Weak; no high-accuracy predictive claim",
    })
save(table1, "Table1_final_model_performance.csv")

table2 = pd.read_csv(ROOT / "Table2_UTS_credibility.csv")
table2["Analysis"] = table2["Analysis"].replace({
    "来源聚类bootstrap": "Source-cluster bootstrap",
    "留一数据集迁移验证": "Leave-one-dataset-out transfer",
    "适用域内": "Inside applicability domain",
    "适用域外": "Outside applicability domain",
    "行级交叉保形区间": "Row-level cross-conformal interval",
})
table2["Metric"] = table2["Metric"].replace({"90%覆盖率": "90% coverage", "95%覆盖率": "95% coverage"})
table2["Meaning"] = table2["Meaning"].replace({
    "跨来源重采样后的模型稳定性": "Model stability after source-level resampling",
    "六个数据集之间的迁移能力": "Transfer across six contributing datasets",
    "89.6%样本位于适用域内": "89.6% of records are inside the applicability domain",
    "域外误差明显增加": "Error increases outside the applicability domain",
    "平均宽度223.2 MPa": "Mean interval width: 223.2 MPa",
    "平均宽度286.4 MPa": "Mean interval width: 286.4 MPa",
})
save(table2, "Table2_UTS_credibility.csv")

table3 = pd.read_csv(ROOT / "Table3_matched_subset.csv")
table3["Section"] = table3["Section"].replace({"目标相关性": "Target correlation", "同样本公平比较": "Same-sample fair comparison"})
table3["Conclusion"] = table3["Conclusion"].replace({
    "完整匹配子集上的秩相关": "Rank correlation in the complete matched subset",
    "匹配子集是更困难且分布不同的验证域": "The matched subset is a more difficult, distribution-shifted validation domain",
})
save(table3, "Table3_matched_subset.csv")

figures = pd.read_csv(ROOT / "figure_freeze_decision.csv")
figures["Location"] = figures["Location"].replace({"正文": "Main text", "补充材料": "Supplementary information"})
figures["Decision"] = figures["Decision"].replace({"保留": "Retain", "移出正文": "Move out of main text"})
figures["Reason"] = figures["Reason"].replace({
    "承担主线证据": "Supports the main evidence chain",
    "同样本可比性很重要，但四联图信息密度高且正文已有目标定位；正文报告关键数值即可": "Important same-sample comparison, but the four-panel figure is information-dense and the main text already positions the targets; report key values in the main text.",
})
save(figures, "figure_freeze_decision.csv")

print("Prepared English supplementary CSV files without changing numerical results.")
