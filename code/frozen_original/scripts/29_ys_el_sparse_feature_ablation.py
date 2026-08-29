from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
INPUT_ROOT = Path(r"F:\CC\outputs\ys_el_scope_audit")
OUTPUT_ROOT = Path(r"F:\CC\outputs\ys_el_sparse_feature_ablation")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASKS = ("YS", "EL")
TARGETS = {"YS": "YS_0.2pct_MPa", "EL": "EL_pct"}
FEATURE_SETS = {
    "major3": ["Zn", "Mg", "Cu"],
    "refined4": ["Zn", "Mg", "Cu", "Zr"],
    "refined5": ["Zn", "Mg", "Cu", "Fe", "Zr"],
    "dense7": ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Zr"],
    "drop_ni_sc": ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr"],
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def original_features(task: str, fold: int) -> list[str]:
    selected = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "baseline_strict" / "selected_features_by_fold.csv"
    )
    row = selected.loc[
        selected["Task"].eq(task)
        & selected["Feature_Set"].eq("composition_core")
        & selected["Model"].eq("RandomForest")
        & selected["Outer_Fold"].eq(fold)
    ].iloc[0]
    return str(row["Selected_Features"]).split("|")


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    source_values = []
    for _, part in frame.groupby("Source_Group"):
        error = part["y_pred"].to_numpy(dtype=float) - part["y_true"].to_numpy(dtype=float)
        source_values.append((np.abs(error).mean(), np.sqrt(np.square(error).mean())))
    source_values = np.asarray(source_values)
    return {
        "Rows": len(frame),
        "Sources": frame["Source_Group"].nunique(),
        "R2": float(r2_score(y, p)),
        "RMSE": float(mean_squared_error(y, p, squared=False)),
        "MAE": float(mean_absolute_error(y, p)),
        "Source_Macro_MAE": float(source_values[:, 0].mean()),
        "Source_Macro_RMSE": float(source_values[:, 1].mean()),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_module = load_module(
        "ys_el_ablation_models", PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    )
    xgb_params = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv"
    )
    prediction_parts = []
    fold_rows = []

    for task in TASKS:
        data = pd.read_csv(INPUT_ROOT / f"{task}_scope_clean_with_outer_folds.csv")
        target = TARGETS[task]
        data[target] = pd.to_numeric(data[target], errors="coerce")
        for fold in sorted(data["Outer_Fold"].unique()):
            fold = int(fold)
            train = data.loc[data["Outer_Fold"].ne(fold)].copy()
            test = data.loc[data["Outer_Fold"].eq(fold)].copy()
            xgb_row = xgb_params.loc[
                xgb_params["Task"].eq(task) & xgb_params["Outer_Fold"].eq(fold)
            ].iloc[0]
            candidates = {"original_fold_core": original_features(task, fold), **FEATURE_SETS}
            for feature_set, features in candidates.items():
                rf = model_module.rf_baseline(config.RANDOM_SEED + fold)
                xgb = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
                rf.fit(train[features], train[target])
                xgb.fit(train[features], train[target])
                pred_rf = rf.predict(test[features])
                pred_xgb = xgb.predict(test[features])
                pred = (pred_rf + pred_xgb) / 2.0
                part = pd.DataFrame(
                    {
                        "Task": task,
                        "Feature_Set": feature_set,
                        "Outer_Fold": fold,
                        "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                        "Source_Group": test["Source_Group"].to_numpy(),
                        "y_true": test[target].to_numpy(),
                        "pred_rf": pred_rf,
                        "pred_xgb": pred_xgb,
                        "y_pred": pred,
                    }
                )
                prediction_parts.append(part)
                fold_rows.append(
                    {
                        "Task": task,
                        "Feature_Set": feature_set,
                        "Outer_Fold": fold,
                        "N_Features": len(features),
                        "Features": "|".join(features),
                        **metrics(part),
                    }
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    overall_rows = []
    for (task, feature_set), part in predictions.groupby(["Task", "Feature_Set"]):
        overall_rows.append(
            {"Task": task, "Feature_Set": feature_set, **metrics(part)}
        )
    overall = pd.DataFrame(overall_rows).sort_values(["Task", "RMSE"])
    baseline = overall.loc[
        overall["Feature_Set"].eq("original_fold_core"),
        ["Task", "R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE"],
    ].rename(columns={column: f"Baseline_{column}" for column in ["R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE"]})
    comparison = overall.merge(baseline, on="Task", validate="many_to_one")
    comparison["Delta_R2"] = comparison["R2"] - comparison["Baseline_R2"]
    comparison["Delta_RMSE"] = comparison["RMSE"] - comparison["Baseline_RMSE"]
    comparison["Delta_MAE"] = comparison["MAE"] - comparison["Baseline_MAE"]
    comparison["Delta_Source_Macro_MAE"] = comparison["Source_Macro_MAE"] - comparison["Baseline_Source_Macro_MAE"]
    comparison["Exploratory_Improves_All_Key_Metrics"] = (
        comparison["Delta_R2"].gt(0)
        & comparison["Delta_RMSE"].lt(0)
        & comparison["Delta_MAE"].lt(0)
        & comparison["Delta_Source_Macro_MAE"].lt(0)
    )

    predictions.to_csv(OUTPUT_ROOT / "forced_feature_set_oof_predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_ROOT / "forced_feature_set_fold_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_ROOT / "forced_feature_set_overall_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_ROOT / "forced_feature_set_comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(
            {
                "tasks": TASKS,
                "input": "scope-clean YS and EL",
                "candidate_feature_sets": FEATURE_SETS,
                "model": "RandomForest baseline + frozen nested XGBoost; equal-weight ensemble",
                "outer_folds": "unchanged source-exclusive folds",
                "purpose": "exploratory forced-set ablation only; final choice requires nested selection",
                "parameter_tuning": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("FORCED FEATURE-SET OVERALL METRICS")
    print(overall.to_string(index=False))
    print("\nCOMPARISON TO ORIGINAL FOLD CORE")
    print(
        comparison[
            [
                "Task",
                "Feature_Set",
                "Delta_R2",
                "Delta_RMSE",
                "Delta_MAE",
                "Delta_Source_Macro_MAE",
                "Exploratory_Improves_All_Key_Metrics",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
