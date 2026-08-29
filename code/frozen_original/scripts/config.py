from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKBOOK_PATH = Path(
    r"C:\Users\dell\OneDrive\Desktop\7000系铝合金_重新建模数据总表.xlsx"
)

RANDOM_SEED = 20260721
N_OUTER_FOLDS = 5
N_HV_FOLDS = 4

SHEETS = {
    "YS": "YS_模型数据",
    "UTS": "UTS_模型数据",
    "EL": "EL_模型数据",
    "HV": "HV_探索数据",
    "MTL": "MTL_三目标部分标签",
    "TRIPLE": "三目标完整匹配",
    "FOUR": "四目标稳健子集",
}

TARGET_COLUMNS = {
    "YS": "YS_0.2pct_MPa",
    "UTS": "UTS_MPa",
    "EL": "EL_pct",
    "HV": "Hardness_HV",
}

MASK_COLUMNS = {
    "YS": "Mask_YS",
    "UTS": "Mask_UTS",
    "EL": "Mask_EL",
}

COMPOSITION_FEATURES = [
    "Al", "Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr",
    "V", "Li", "Ni", "Sc", "Ag", "Ce",
]

# Primary model candidates. Al is omitted because it is the dependent balance
# component; very sparse V/Li/Ag/Ce are retained for audit but not main modeling.
MODEL_COMPOSITION_FEATURES = [
    "Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr", "Ni", "Sc",
]

MODEL_DERIVED_FEATURES = [
    "Zn_Mg_Ratio", "Zn_Mg_Sum", "Zn_Mg_Cu_Sum", "Zn_Share_ZnMgCu",
]

PRIMARY_FIXED_FEATURES = ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr", "Sc"]

DERIVED_FEATURES = [
    "Zn_Mg_Ratio", "Zn_Cu_Ratio", "Mg_Cu_Ratio", "Zn_Mg_Sum",
    "Zn_Mg_Cu_Sum", "Zn_Share_ZnMgCu", "Reported_Solute_Sum",
]

PROCESS_FEATURES = [
    "Sol_Temp_C", "Sol_Time_h", "Cooling_Rate_K_s", "Age1_Temp_C",
    "Age1_Time_h", "Age2_Temp_C", "Age2_Time_h", "Test_Temp_C",
]

CATEGORICAL_SENSITIVITY_FEATURES = [
    "Temper", "Process_Category", "Quench_Medium",
]

METADATA_COLUMNS = [
    "Model_Row_ID", "Include_Main_Model", "Sensitivity_Group", "Dataset",
    "Source_Group", "DOI", "Source_Reference", "Source_File", "Source_Sheet",
    "Original_Row_ID", "Original_Sample_ID", "Alloy_Grade", "Source_Type",
    "Value_Type", "Evidence_Level", "Quality_Flag", "Condition_Original",
]

AUDIT_ONLY_COLUMNS = ["Process_Missing_Count", "Observed_Target_Count", "Four_Target_Complete"]

ALL_TARGET_AND_MASK_COLUMNS = list(TARGET_COLUMNS.values()) + list(MASK_COLUMNS.values())

OUTPUT_DIRS = {
    "processed": PROJECT_ROOT / "data" / "processed",
    "folds": PROJECT_ROOT / "folds",
    "reports": PROJECT_ROOT / "reports" / "stage1",
    "single": PROJECT_ROOT / "results" / "single_target",
    "multi": PROJECT_ROOT / "results" / "multi_output",
    "mtl": PROJECT_ROOT / "results" / "mtl",
    "figures": PROJECT_ROOT / "figures",
}
