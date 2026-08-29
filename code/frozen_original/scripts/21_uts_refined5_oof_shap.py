from __future__ import annotations

import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260728
N_BOOTSTRAP = 2000


def load_python_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rebuild_and_explain(base, model_module):
    data = pd.read_csv(
        config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv"
    )
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(
        pd.to_numeric, errors="coerce"
    )

    root = config.OUTPUT_DIRS["single"]
    rf_params = pd.read_csv(
        root / "nested_optuna_strict" / "metrics_by_outer_fold.csv"
    )
    xgb_params = pd.read_csv(
        root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv"
    )
    saved = pd.read_csv(
        config.PROJECT_ROOT
        / "results"
        / "uts_nested_sparse_feature_selection"
        / "outer_oof_predictions_all_strategies.csv"
    )
    saved = saved.loc[
        saved["Strategy"].eq("drop_si_sparse50"), ["Model_Row_ID", "y_pred"]
    ].rename(columns={"y_pred": "saved_y_pred"})

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
        rf = model_module.rf_tuned(rf_row, config.RANDOM_SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
        rf.fit(train[FEATURES], train[TARGET])
        xgb.fit(train[FEATURES], train[TARGET])

        x_rf = rf.named_steps["imputer"].transform(test[FEATURES])
        x_xgb = xgb.named_steps["imputer"].transform(test[FEATURES])
        if not np.allclose(x_rf, x_xgb, equal_nan=True):
            raise AssertionError("RF and XGBoost imputed test matrices differ")

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
                "RF_Max_Additivity_Error": float(
                    np.max(np.abs(base_rf + sv_rf.sum(axis=1) - pred_rf))
                ),
                "XGB_Max_Additivity_Error": float(
                    np.max(np.abs(base_xgb + sv_xgb.sum(axis=1) - pred_xgb))
                ),
                "Ensemble_Max_Additivity_Error": float(
                    np.max(
                        np.abs(
                            base_ensemble
                            + sv_ensemble.sum(axis=1)
                            - pred_ensemble
                        )
                    )
                ),
            }
        )

        for index, (_, row) in enumerate(test.iterrows()):
            record = {
                "Task": TASK,
                "Outer_Fold": fold,
                "Model_Row_ID": row["Model_Row_ID"],
                "Source_Group": row["Source_Group"],
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
    explained["Saved_Prediction_Abs_Diff"] = (
        explained["y_pred"] - explained["saved_y_pred"]
    ).abs()
    validation = pd.DataFrame(validation_rows)
    if explained["Saved_Prediction_Abs_Diff"].max() > 1e-4:
        raise AssertionError("Refined predictions do not reproduce saved OOF predictions")
    if (
        validation[
            [
                "RF_Max_Additivity_Error",
                "XGB_Max_Additivity_Error",
                "Ensemble_Max_Additivity_Error",
            ]
        ]
        .max()
        .max()
        > 1e-2
    ):
        raise AssertionError("SHAP additivity error exceeds tolerance")
    return explained, validation


def prediction_metrics(explained):
    y_true = explained["y_true"].to_numpy()
    y_pred = explained["y_pred"].to_numpy()
    rows = [
        {
            "Scope": "All_OOF",
            "Fold": "All",
            "Rows": len(explained),
            "Sources": explained["Source_Group"].nunique(),
            "R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "MAE": mean_absolute_error(y_true, y_pred),
        }
    ]
    for fold, part in explained.groupby("Outer_Fold"):
        rows.append(
            {
                "Scope": "Outer_Fold",
                "Fold": int(fold),
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                "R2": r2_score(part["y_true"], part["y_pred"]),
                "RMSE": mean_squared_error(
                    part["y_true"], part["y_pred"], squared=False
                ),
                "MAE": mean_absolute_error(part["y_true"], part["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    base = load_python_module(
        "uts_shap_base", config.PROJECT_ROOT / "18_uts_oof_shap_stability.py"
    )
    base.FEATURES = FEATURES
    base.N_BOOTSTRAP = N_BOOTSTRAP
    base.SEED = SEED
    model_module = base.load_model_module()
    explained, validation = rebuild_and_explain(base, model_module)
    metrics = prediction_metrics(explained)
    global_importance, fold_importance = base.model_importance(explained)
    agreement = base.rank_agreement(global_importance, fold_importance)
    directions = base.direction_stability(global_importance, fold_importance)
    boot_samples, boot_summary = base.source_cluster_bootstrap(explained)
    loo_values, loo_summary = base.leave_one_source_out(explained)

    out = config.PROJECT_ROOT / "results" / "uts_refined5_oof_shap"
    out.mkdir(parents=True, exist_ok=True)
    explained.to_csv(out / "uts_refined5_oof_shap_values_wide.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(out / "shap_additivity_and_prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "prediction_metrics.csv", index=False, encoding="utf-8-sig")
    global_importance.to_csv(out / "global_importance_by_model.csv", index=False, encoding="utf-8-sig")
    fold_importance.to_csv(out / "fold_importance_and_direction.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(out / "model_and_fold_rank_agreement.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(out / "direction_stability.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "source_bootstrap_importance_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "source_bootstrap_importance_summary.csv", index=False, encoding="utf-8-sig")
    loo_values.to_csv(out / "leave_one_source_out_importance.csv", index=False, encoding="utf-8-sig")
    loo_summary.to_csv(out / "leave_one_source_out_summary.csv", index=False, encoding="utf-8-sig")
    base.save_figures(out, explained, boot_summary, fold_importance, global_importance)

    run_config = {
        "task": TASK,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "features": FEATURES,
        "feature_rule": "stable intersection selected by nested source-grouped sparse-feature analysis",
        "explanation_rows": "original outer-fold test rows only",
        "training_augmentation": False,
        "source_bootstrap_iterations": N_BOOTSTRAP,
        "applicability_domain": "not reused from full10; must be recomputed for refined5",
        "interpretation_scope": "model associations, not causal material mechanisms",
        "seed": SEED,
    }
    (out / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("PREDICTION METRICS")
    print(metrics.to_string(index=False))
    print("\nENSEMBLE IMPORTANCE")
    print(
        global_importance.loc[global_importance["Model"].eq("Ensemble")].to_string(
            index=False
        )
    )
    print("\nBOOTSTRAP SUMMARY")
    print(boot_summary.to_string(index=False))
    print("\nRANK AGREEMENT")
    print(agreement.groupby("Comparison")["Spearman"].agg(["mean", "min", "max"]).to_string())
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
