from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(r"F:\CC\outputs\matched_subset_final_robustness")
TASKS = ("YS", "UTS", "EL")


def metric_check(summary_path, prediction_path, groups):
    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(prediction_path)
    rows = []
    for keys, part in predictions.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        selector = np.ones(len(summary), dtype=bool)
        for column, value in zip(groups, keys):
            selector &= summary[column].eq(value).to_numpy()
        saved = summary.loc[selector].iloc[0]
        calculated = {
            "R2": r2_score(part["y_true"], part["y_pred"]),
            "RMSE": mean_squared_error(part["y_true"], part["y_pred"], squared=False),
            "MAE": mean_absolute_error(part["y_true"], part["y_pred"]),
        }
        for metric, value in calculated.items():
            rows.append({
                "Check": "Metric_Recalculation",
                "Group": "|".join(map(str, keys)),
                "Metric": metric,
                "Saved": saved[metric],
                "Recalculated": value,
                "Absolute_Difference": abs(saved[metric] - value),
                "Passed": abs(saved[metric] - value) < 1e-10,
            })
    return rows


def main():
    matched = pd.read_csv(ROOT / "matched_complete_266_with_outer_folds.csv")
    matched_pred = pd.read_csv(ROOT / "matched_only_independent_ensemble_oof_predictions.csv")
    full_pred = pd.read_csv(ROOT / "full_models_restricted_to_matched_oof_predictions.csv")
    multi_pred = pd.read_csv(ROOT / "matched_multioutput_vs_independent_rf_oof_predictions.csv")
    checks = []

    basic = {
        "Matched_Rows_266": len(matched) == 266,
        "Matched_Sources_59": matched["Source_Group"].nunique() == 59,
        "Matched_Datasets_4": matched["Dataset"].nunique() == 4,
        "Unique_Sample_Keys": matched["Sample_Key"].nunique() == len(matched),
        "One_Fold_Per_Source": matched.groupby("Source_Group")["Outer_Fold"].nunique().max() == 1,
        "Five_Nonempty_Folds": set(matched["Outer_Fold"].unique()) == set(range(5)),
        "Independent_One_Prediction_Per_Task_Sample": not matched_pred.duplicated(["Task", "Sample_Key"]).any() and len(matched_pred) == 266 * 3,
        "Full_One_Prediction_Per_Task_Sample": not full_pred.duplicated(["Task", "Sample_Key"]).any() and len(full_pred) == 266 * 3,
        "Multioutput_Expected_Prediction_Count": len(multi_pred) == 266 * 3 * 2 * 2,
        "Multioutput_No_Duplicate_Model_Task_Feature_Sample": not multi_pred.duplicated(["Feature_Set", "Model", "Task", "Sample_Key"]).any(),
    }
    for name, passed in basic.items():
        checks.append({"Check": name, "Group": "All", "Metric": "Boolean", "Saved": np.nan, "Recalculated": np.nan, "Absolute_Difference": np.nan, "Passed": bool(passed)})

    for task in TASKS:
        a = matched_pred.loc[matched_pred["Task"].eq(task)].set_index("Sample_Key")["y_true"].sort_index()
        b = full_pred.loc[full_pred["Task"].eq(task)].set_index("Sample_Key")["y_true"].sort_index()
        passed = a.index.equals(b.index) and np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-10)
        checks.append({"Check": "Identical_Test_Truth", "Group": task, "Metric": "Boolean", "Saved": np.nan, "Recalculated": np.nan, "Absolute_Difference": np.nan, "Passed": bool(passed)})

    checks.extend(metric_check(
        ROOT / "fair_comparison_full_vs_matched_only_metrics.csv",
        ROOT / "matched_only_independent_ensemble_oof_predictions.csv",
        ["Model", "Task"],
    ))
    checks.extend(metric_check(
        ROOT / "fair_comparison_full_vs_matched_only_metrics.csv",
        ROOT / "full_models_restricted_to_matched_oof_predictions.csv",
        ["Model", "Task"],
    ))
    checks.extend(metric_check(
        ROOT / "matched_multioutput_vs_independent_rf_metrics.csv",
        ROOT / "matched_multioutput_vs_independent_rf_oof_predictions.csv",
        ["Feature_Set", "Model", "Task"],
    ))
    result = pd.DataFrame(checks)
    result.to_csv(ROOT / "verification_checks.csv", index=False, encoding="utf-8-sig")
    if not result["Passed"].all():
        raise AssertionError(result.loc[~result["Passed"]].to_string(index=False))
    print(f"All {len(result)} verification checks passed.")


if __name__ == "__main__":
    main()
