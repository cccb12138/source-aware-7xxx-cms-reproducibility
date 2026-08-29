from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
AUDIT_ROOT = PROJECT_ROOT / "results" / "uts_systematic_scope_audit"
PARAM_ROOT = Path(r"F:\CC\outputs\paper_scope_clean_final\model_decisions")
OUTPUT_ROOT = Path(r"F:\CC\outputs\uts_scope_clean_final") / "oof_shap"
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASK = "UTS"
TARGET = "UTS_MPa"
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260810
N_BOOTSTRAP = 3000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data() -> pd.DataFrame:
    data = pd.read_csv(AUDIT_ROOT / "UTS_scope_clean_with_outer_folds.csv")
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if len(data) != 675 or data["Source_Group"].nunique() != 258:
        raise AssertionError("Unexpected scope-clean UTS size")
    if data.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("A source occurs in more than one outer fold")
    return data


def rebuild_and_explain(data, base, model_module):
    rf_params = pd.read_csv(PARAM_ROOT / "rf_params_by_outer_fold.csv")
    xgb_params = pd.read_csv(PARAM_ROOT / "xgb_params_by_outer_fold.csv")
    saved = pd.read_csv(PARAM_ROOT / "final_oof_predictions.csv")
    saved = saved[["Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "saved_y_pred"})

    rows = []
    validation_rows = []
    for fold in sorted(data["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = data.loc[data["Outer_Fold"].eq(fold)].copy()
        if set(train["Source_Group"]) & set(test["Source_Group"]):
            raise AssertionError("Source leakage")

        rf_row = rf_params.loc[
            rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(fold)
        ].iloc[0]
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(fold)
        ].iloc[0]
        rf = model_module.rf_tuned(rf_row, SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, SEED + fold)
        rf.named_steps["model"].set_params(n_estimators=180)
        rf.fit(train[FEATURES], train[TARGET])
        xgb.fit(train[FEATURES], train[TARGET])

        x_rf = rf.named_steps["imputer"].transform(test[FEATURES])
        x_xgb = xgb.named_steps["imputer"].transform(test[FEATURES])
        if not np.allclose(x_rf, x_xgb, equal_nan=True):
            raise AssertionError("RF and XGBoost imputed matrices differ")
        pred_rf = rf.named_steps["model"].predict(x_rf)
        pred_xgb = xgb.named_steps["model"].predict(x_xgb)
        pred_ensemble = (pred_rf + pred_xgb) / 2.0

        explain_rf = shap.TreeExplainer(rf.named_steps["model"])
        explain_xgb = shap.TreeExplainer(xgb.named_steps["model"])
        sv_rf = base.shap_array(explain_rf, x_rf)
        sv_xgb = base.shap_array(explain_xgb, x_xgb)
        base_rf = base.scalar_expected_value(explain_rf)
        base_xgb = base.scalar_expected_value(explain_xgb)
        sv_ensemble = (sv_rf + sv_xgb) / 2.0
        base_ensemble = (base_rf + base_xgb) / 2.0

        validation_rows.append(
            {
                "Outer_Fold": fold,
                "Rows": len(test),
                "Train_Sources": train["Source_Group"].nunique(),
                "Test_Sources": test["Source_Group"].nunique(),
                "Source_Overlap": len(set(train["Source_Group"]) & set(test["Source_Group"])),
                "RF_Max_Additivity_Error": float(np.max(np.abs(base_rf + sv_rf.sum(axis=1) - pred_rf))),
                "XGB_Max_Additivity_Error": float(np.max(np.abs(base_xgb + sv_xgb.sum(axis=1) - pred_xgb))),
                "Ensemble_Max_Additivity_Error": float(
                    np.max(np.abs(base_ensemble + sv_ensemble.sum(axis=1) - pred_ensemble))
                ),
            }
        )

        for index, (_, row) in enumerate(test.iterrows()):
            record = {
                "Task": TASK,
                "Outer_Fold": fold,
                "Model_Row_ID": row["Model_Row_ID"],
                "Source_Group": row["Source_Group"],
                "Dataset": row["Dataset"],
                "Audit_Tier": row["Audit_Tier"],
                "Audit_Decision": row["Audit_Decision"],
                "y_true": row[TARGET],
                "pred_rf": pred_rf[index],
                "pred_xgb": pred_xgb[index],
                "y_pred": pred_ensemble[index],
                "Base_RF": base_rf,
                "Base_XGB": base_xgb,
                "Base_Ensemble": base_ensemble,
            }
            for feature_index, feature in enumerate(FEATURES):
                record[f"Value_{feature}"] = x_rf[index, feature_index]
                record[f"SHAP_RF_{feature}"] = sv_rf[index, feature_index]
                record[f"SHAP_XGB_{feature}"] = sv_xgb[index, feature_index]
                record[f"SHAP_Ensemble_{feature}"] = sv_ensemble[index, feature_index]
            rows.append(record)

    explained = pd.DataFrame(rows).merge(saved, on="Model_Row_ID", validate="one_to_one")
    explained["Saved_Prediction_Abs_Diff"] = (explained["y_pred"] - explained["saved_y_pred"]).abs()
    validation = pd.DataFrame(validation_rows)
    if explained["Saved_Prediction_Abs_Diff"].max() > 1e-5:
        raise AssertionError("OOF predictions do not reproduce the frozen scope-clean result")
    if validation[["RF_Max_Additivity_Error", "XGB_Max_Additivity_Error", "Ensemble_Max_Additivity_Error"]].max().max() > 1e-2:
        raise AssertionError("SHAP additivity error exceeds tolerance")
    return explained, validation


def prediction_metrics(explained: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("All_OOF", "All", explained)]
    scopes.extend(("Outer_Fold", int(fold), part) for fold, part in explained.groupby("Outer_Fold"))
    for scope, fold, part in scopes:
        rows.append(
            {
                "Scope": scope,
                "Fold": fold,
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                "R2": r2_score(part["y_true"], part["y_pred"]),
                "RMSE": mean_squared_error(part["y_true"], part["y_pred"], squared=False),
                "MAE": mean_absolute_error(part["y_true"], part["y_pred"]),
                "Bias": (part["y_pred"] - part["y_true"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def subgroup_importance(explained: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, part in explained.groupby(group_column, dropna=False):
        values = pd.Series(
            {feature: part[f"SHAP_Ensemble_{feature}"].abs().mean() for feature in FEATURES}
        )
        ranks = values.rank(method="min", ascending=False).astype(int)
        for feature in FEATURES:
            rows.append(
                {
                    "Grouping": group_column,
                    "Group": group,
                    "Rows": len(part),
                    "Sources": part["Source_Group"].nunique(),
                    "Feature": feature,
                    "Mean_Abs_SHAP": values[feature],
                    "Rank": int(ranks[feature]),
                    "Value_SHAP_Spearman": part[f"Value_{feature}"].corr(
                        part[f"SHAP_Ensemble_{feature}"], method="spearman"
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_old_clean(global_importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(
        PROJECT_ROOT / "results" / "uts_refined5_oof_shap" / "global_importance_by_model.csv"
    )
    old = old.loc[old["Model"].eq("Ensemble"), ["Feature", "Mean_Abs_SHAP", "Importance_Share", "Value_SHAP_Spearman", "Rank"]]
    old = old.rename(columns={column: f"Old_{column}" for column in old.columns if column != "Feature"})
    clean = global_importance.loc[
        global_importance["Model"].eq("Ensemble"),
        ["Feature", "Mean_Abs_SHAP", "Importance_Share", "Value_SHAP_Spearman", "Rank"],
    ].rename(columns={column: f"Clean_{column}" for column in ["Mean_Abs_SHAP", "Importance_Share", "Value_SHAP_Spearman", "Rank"]})
    comparison = old.merge(clean, on="Feature", validate="one_to_one")
    comparison["Rank_Change_Clean_Minus_Old"] = comparison["Clean_Rank"] - comparison["Old_Rank"]
    agreement = pd.DataFrame(
        [
            {
                "Comparison": "Old689_vs_ScopeClean675_Ensemble_Importance",
                "Features": len(comparison),
                "Rank_Spearman": spearmanr(comparison["Old_Rank"], comparison["Clean_Rank"]).statistic,
                "Importance_Share_Spearman": spearmanr(
                    comparison["Old_Importance_Share"], comparison["Clean_Importance_Share"]
                ).statistic,
                "Same_Top1": bool(
                    comparison.loc[comparison["Old_Rank"].idxmin(), "Feature"]
                    == comparison.loc[comparison["Clean_Rank"].idxmin(), "Feature"]
                ),
                "Max_Absolute_Rank_Change": int(comparison["Rank_Change_Clean_Minus_Old"].abs().max()),
            }
        ]
    )
    return comparison.sort_values("Clean_Rank"), agreement


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    base = load_module("scope_clean_shap_base", PROJECT_ROOT / "18_uts_oof_shap_stability.py")
    base.FEATURES = FEATURES
    base.N_BOOTSTRAP = N_BOOTSTRAP
    base.SEED = SEED
    model_module = base.load_model_module()
    explained, validation = rebuild_and_explain(data, base, model_module)
    metrics = prediction_metrics(explained)
    global_importance, fold_importance = base.model_importance(explained)
    agreement = base.rank_agreement(global_importance, fold_importance)
    directions = base.direction_stability(global_importance, fold_importance)
    boot_samples, boot_summary = base.source_cluster_bootstrap(explained)
    loo_values, loo_summary = base.leave_one_source_out(explained)
    subgroup = pd.concat(
        [subgroup_importance(explained, "Audit_Tier"), subgroup_importance(explained, "Dataset")],
        ignore_index=True,
    )
    explained.to_csv(OUTPUT_ROOT / "scope_clean_oof_shap_values_wide.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_ROOT / "shap_additivity_and_prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_ROOT / "prediction_metrics.csv", index=False, encoding="utf-8-sig")
    global_importance.to_csv(OUTPUT_ROOT / "global_importance_by_model.csv", index=False, encoding="utf-8-sig")
    fold_importance.to_csv(OUTPUT_ROOT / "fold_importance_and_direction.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(OUTPUT_ROOT / "model_and_fold_rank_agreement.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(OUTPUT_ROOT / "direction_stability.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(OUTPUT_ROOT / "source_bootstrap_importance_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUTPUT_ROOT / "source_bootstrap_importance_summary.csv", index=False, encoding="utf-8-sig")
    loo_values.to_csv(OUTPUT_ROOT / "leave_one_source_out_importance.csv", index=False, encoding="utf-8-sig")
    loo_summary.to_csv(OUTPUT_ROOT / "leave_one_source_out_summary.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(OUTPUT_ROOT / "subgroup_shap_importance.csv", index=False, encoding="utf-8-sig")
    base.save_figures(OUTPUT_ROOT, explained, boot_summary, fold_importance, global_importance)

    run_config = {
        "input": str(AUDIT_ROOT / "UTS_scope_clean_with_outer_folds.csv"),
        "rows": len(data),
        "sources": int(data["Source_Group"].nunique()),
        "features": FEATURES,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "outer_validation": "unchanged five source-exclusive folds",
        "hyperparameters": "retuned using only the 675-row scope-clean dataset",
        "feature_or_parameter_tuning_in_this_stage": False,
        "old_689_row_results_used": False,
        "source_bootstrap_iterations": N_BOOTSTRAP,
        "interpretation_scope": "predictive association; not a causal mechanism claim",
        "seed": SEED,
    }
    (OUTPUT_ROOT / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PREDICTION METRICS")
    print(metrics.to_string(index=False))
    print("\nENSEMBLE GLOBAL IMPORTANCE")
    print(global_importance.loc[global_importance["Model"].eq("Ensemble")].to_string(index=False))
    print("\nSOURCE BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print("\nDIRECTION STABILITY - ENSEMBLE")
    print(directions.loc[directions["Model"].eq("Ensemble")].to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
