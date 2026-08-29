from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
ROOT = Path(r"F:\CC\outputs\paper_results_v1")
RESULTS = PROJECT / "results"
checks = []


def add(name, passed, detail):
    checks.append({"Check": name, "Passed": bool(passed), "Detail": str(detail)})


pred = pd.read_csv(RESULTS / "uts_scope_clean_final" / "oof_shap" / "scope_clean_oof_shap_values_wide.csv")
saved = pd.read_csv(RESULTS / "uts_scope_clean_final" / "oof_shap" / "prediction_metrics.csv")
overall = saved.loc[saved["Scope"].eq("All_OOF")].iloc[0]
calc = {
    "R2": r2_score(pred["y_true"], pred["y_pred"]),
    "RMSE": mean_squared_error(pred["y_true"], pred["y_pred"], squared=False),
    "MAE": mean_absolute_error(pred["y_true"], pred["y_pred"]),
}
for metric, value in calc.items():
    add(f"UTS_{metric}_recalculation", abs(value - overall[metric]) < 1e-10, f"saved={overall[metric]:.12g}; recalculated={value:.12g}")

importance = pd.read_csv(RESULTS / "uts_scope_clean_final" / "oof_shap" / "global_importance_by_model.csv")
ensemble = importance.loc[importance["Model"].eq("Ensemble")]
add("SHAP_importance_sums_to_one", abs(ensemble["Importance_Share"].sum() - 1) < 1e-10, ensemble["Importance_Share"].sum())
add("SHAP_expected_features", set(ensemble["Feature"]) == {"Zn", "Mg", "Cu", "Fe", "Zr"}, "|".join(ensemble.sort_values("Rank")["Feature"]))

matched = pd.read_csv(RESULTS / "matched_subset_final_robustness" / "matched_complete_266_with_outer_folds.csv")
corr = pd.read_csv(RESULTS / "matched_subset_final_robustness" / "target_correlations_row_and_source_mean.csv")
add("Matched_rows_266", len(matched) == 266, len(matched))
add("Matched_sources_59", matched["Source_Group"].nunique() == 59, matched["Source_Group"].nunique())
for left, right, left_col, right_col in [
    ("YS", "UTS", "YS_0.2pct_MPa", "UTS_MPa"),
    ("YS", "EL", "YS_0.2pct_MPa", "EL_pct"),
    ("UTS", "EL", "UTS_MPa", "EL_pct"),
]:
    calculated = matched[[left_col, right_col]].corr(method="spearman").iloc[0, 1]
    expected = corr.loc[(corr["Level"].eq("Row")) & corr["Target_A"].eq(left) & corr["Target_B"].eq(right), "Spearman_r"].iloc[0]
    add(f"Matched_correlation_{left}_{right}", abs(calculated - expected) < 1e-10, f"saved={expected:.12g}; recalculated={calculated:.12g}")

figures = sorted((ROOT / "figures").glob("Fig*.png"))
add("Six_candidate_figures", len(figures) == 6, len(figures))
for figure in figures:
    with Image.open(figure) as image:
        width, height = image.size
    add(f"Figure_nonempty_{figure.stem}", figure.stat().st_size > 100_000 and width > 2000 and height > 1200,
        f"bytes={figure.stat().st_size}; size={width}x{height}")

workbook = ROOT / "7xxx_论文结果数据表_v1.xlsx"
add("Workbook_created", workbook.exists() and workbook.stat().st_size > 30_000, f"bytes={workbook.stat().st_size if workbook.exists() else 0}")
error_scan = (ROOT / "workbook_formula_error_scan.ndjson").read_text(encoding="utf-8")
add("Workbook_no_formula_errors", "matched 0 entries" in error_scan, error_scan.strip())

result = pd.DataFrame(checks)
result.to_csv(ROOT / "paper_results_v1_verification.csv", index=False, encoding="utf-8-sig")
if not result["Passed"].all():
    raise AssertionError(result.loc[~result["Passed"]].to_string(index=False))
print(f"All {len(result)} paper-output checks passed.")
