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
UTS_AUDIT_ROOT = PROJECT_ROOT / "results" / "uts_systematic_scope_audit"
STRICT_ROOT = PROJECT_ROOT / "data" / "processed" / "strict"
OUTPUT_ROOT = Path(r"F:\CC\outputs\ys_el_scope_audit")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASKS = ("YS", "EL")
SEED = config.RANDOM_SEED
N_BOOTSTRAP = 5000


EXTRA_SOURCE_DECISIONS = pd.DataFrame(
    [
        {
            "Source_Group": "10.1016/j.msea.2026.149868",
            "Audit_Tier": "B1_Special_Process",
            "Audit_Decision": "Retain_Special_Process_Sensitivity",
            "Include_Scope_Clean_Model": True,
            "Decision_Reason_CN": "7075高压扭转（HPT）材料；保留真实数据，但作为特殊加工敏感性层",
        },
        {
            "Source_Group": "10.3390/ma13225227",
            "Audit_Tier": "A_Core_or_Unresolved",
            "Audit_Decision": "Retain_Core_or_Process_Unresolved",
            "Include_Scope_Clean_Model": True,
            "Decision_Reason_CN": "普通7075锻造材料，未发现明确材料对象或测试条件越界证据",
        },
    ]
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    source_errors = []
    for _, part in frame.groupby("Source_Group"):
        error = part["y_pred"].to_numpy(dtype=float) - part["y_true"].to_numpy(dtype=float)
        source_errors.append((np.abs(error).mean(), np.sqrt(np.square(error).mean())))
    source_errors = np.asarray(source_errors)
    return {
        "Rows": len(frame),
        "Sources": frame["Source_Group"].nunique(),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "Source_Macro_MAE": float(source_errors[:, 0].mean()),
        "Source_Macro_RMSE": float(source_errors[:, 1].mean()),
    }


def load_decisions() -> pd.DataFrame:
    audit = pd.read_csv(UTS_AUDIT_ROOT / "source_scope_audit_263.csv")
    decisions = audit[
        [
            "Source_Group",
            "Audit_Tier",
            "Audit_Decision",
            "Include_Scope_Clean_Model",
            "Decision_Reason_CN",
        ]
    ].copy()
    decisions = pd.concat([decisions, EXTRA_SOURCE_DECISIONS], ignore_index=True)
    decisions = decisions.drop_duplicates("Source_Group", keep="last")
    decisions["Include_Scope_Clean_Model"] = (
        decisions["Include_Scope_Clean_Model"].astype(str).str.lower().map({"true": True, "false": False})
        .fillna(decisions["Include_Scope_Clean_Model"])
        .astype(bool)
    )
    return decisions


def load_task(task: str, decisions: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(STRICT_ROOT / f"{task}_with_outer_folds.csv")
    target = config.TARGET_COLUMNS[task]
    data[target] = pd.to_numeric(data[target], errors="coerce")
    rows = data.merge(decisions, on="Source_Group", how="left", validate="many_to_one")
    if rows["Audit_Decision"].isna().any():
        missing = sorted(rows.loc[rows["Audit_Decision"].isna(), "Source_Group"].astype(str).unique())
        raise AssertionError(f"{task}: unaudited sources: {missing}")
    if rows.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError(f"{task}: source appears in multiple outer folds")
    return rows


def selected_features(task: str, fold: int) -> list[str]:
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


def evaluate_variant(task, variant, data, model_module, xgb_params):
    target = config.TARGET_COLUMNS[task]
    parts = []
    fold_rows = []
    for fold in sorted(data["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = data.loc[data["Outer_Fold"].eq(fold)].copy()
        if set(train["Source_Group"]) & set(test["Source_Group"]):
            raise AssertionError(f"{task}/{variant}/fold{fold}: source leakage")
        features = selected_features(task, fold)
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(task) & xgb_params["Outer_Fold"].eq(fold)
        ].iloc[0]
        rf = model_module.rf_baseline(SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, SEED + fold)
        rf.fit(train[features], train[target])
        xgb.fit(train[features], train[target])
        pred_rf = rf.predict(test[features])
        pred_xgb = xgb.predict(test[features])
        y_pred = (pred_rf + pred_xgb) / 2.0
        part = pd.DataFrame(
            {
                "Task": task,
                "Variant": variant,
                "Outer_Fold": fold,
                "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                "Source_Group": test["Source_Group"].to_numpy(),
                "Dataset": test["Dataset"].to_numpy(),
                "Audit_Tier": test["Audit_Tier"].to_numpy(),
                "y_true": test[target].to_numpy(),
                "pred_rf": pred_rf,
                "pred_xgb": pred_xgb,
                "y_pred": y_pred,
                "Selected_Features": "|".join(features),
            }
        )
        parts.append(part)
        fold_rows.append(
            {
                "Task": task,
                "Variant": variant,
                "Outer_Fold": fold,
                "Train_Rows": len(train),
                "Train_Sources": train["Source_Group"].nunique(),
                "Test_Rows": len(test),
                "Test_Sources": test["Source_Group"].nunique(),
                "Selected_Features": "|".join(features),
                **metric_values(part),
            }
        )
    oof = pd.concat(parts, ignore_index=True)
    return oof, pd.DataFrame(fold_rows), {"Task": task, "Variant": variant, **metric_values(oof)}


def source_bootstrap(task: str, predictions: pd.DataFrame):
    rng = np.random.default_rng(20260803 + (0 if task == "YS" else 1000))
    grouped = {
        source: (part["y_true"].to_numpy(dtype=float), part["y_pred"].to_numpy(dtype=float))
        for source, part in predictions.groupby("Source_Group")
    }
    sources = np.asarray(list(grouped), dtype=object)
    rows = []
    for iteration in range(N_BOOTSTRAP):
        draw = sources[rng.integers(0, len(sources), size=len(sources))]
        y = np.concatenate([grouped[source][0] for source in draw])
        p = np.concatenate([grouped[source][1] for source in draw])
        rows.append(
            {
                "Task": task,
                "Iteration": iteration,
                "R2": r2_score(y, p),
                "RMSE": mean_squared_error(y, p, squared=False),
                "MAE": mean_absolute_error(y, p),
            }
        )
    samples = pd.DataFrame(rows)
    summary = []
    for metric in ["R2", "RMSE", "MAE"]:
        q = samples[metric].quantile([0.025, 0.5, 0.975])
        summary.append(
            {
                "Task": task,
                "Metric": metric,
                "Bootstrap_Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
            }
        )
    return samples, pd.DataFrame(summary)


def sparsity_table(task: str, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in config.MODEL_COMPOSITION_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        nonmissing = values.notna()
        nonzero = values.fillna(0).abs().gt(1e-12)
        rows.append(
            {
                "Task": task,
                "Feature": feature,
                "Rows": len(data),
                "Missing_Fraction": 1 - nonmissing.mean(),
                "Zero_or_Missing_Fraction": 1 - nonzero.mean(),
                "Nonzero_Rows": int(nonzero.sum()),
                "Unique_Numeric_Values": int(values.dropna().nunique()),
                "Candidate_Dense": bool(nonzero.mean() >= 0.20 and values.dropna().nunique() >= 5),
            }
        )
    return pd.DataFrame(rows)


def verify_original_reproduction(all_predictions: pd.DataFrame):
    saved = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "rf_xgb_ensemble_strict" / "oof_predictions.csv"
    )
    saved = saved.loc[saved["Task"].isin(TASKS), ["Task", "Model_Row_ID", "y_pred"]].rename(
        columns={"y_pred": "saved_y_pred"}
    )
    original = all_predictions.loc[
        all_predictions["Variant"].eq("Original"), ["Task", "Model_Row_ID", "y_pred"]
    ]
    check = original.merge(saved, on=["Task", "Model_Row_ID"], validate="one_to_one")
    check["Absolute_Difference"] = (check["y_pred"] - check["saved_y_pred"]).abs()
    if check["Absolute_Difference"].max() > 1.0:
        raise AssertionError(
            f"Original model reproduction failed: max diff={check['Absolute_Difference'].max()}"
        )
    return check


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions()
    model_module = load_module(
        "ys_el_scope_model_builders", PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    )
    xgb_params = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv"
    )

    task_rows = []
    audit_summaries = []
    all_predictions = []
    all_fold_metrics = []
    all_overall = []
    all_boot = []
    all_boot_summary = []
    all_sparsity = []

    for task in TASKS:
        rows = load_task(task, decisions)
        task_rows.append(rows.assign(Task_Audit=task))
        audit_summary = (
            rows.groupby(["Audit_Tier", "Audit_Decision"], dropna=False)
            .agg(Rows=("Model_Row_ID", "size"), Sources=("Source_Group", "nunique"))
            .reset_index()
        )
        audit_summary.insert(0, "Task", task)
        audit_summaries.append(audit_summary)

        variants = {
            "Original": rows.copy(),
            "Scope_Clean": rows.loc[rows["Include_Scope_Clean_Model"]].copy(),
            "Direct_Literature_Sensitivity": rows.loc[
                rows["Include_Scope_Clean_Model"] & ~rows["Audit_Tier"].eq("B2_Secondary_Database")
            ].copy(),
            "Conventional_Core_Sensitivity": rows.loc[rows["Audit_Tier"].eq("A_Core_or_Unresolved")].copy(),
        }
        variants["Scope_Clean"].to_csv(
            OUTPUT_ROOT / f"{task}_scope_clean_with_outer_folds.csv", index=False, encoding="utf-8-sig"
        )
        all_sparsity.append(sparsity_table(task, variants["Scope_Clean"]))
        for name, frame in variants.items():
            oof, fold_metrics, overall = evaluate_variant(
                task, name, frame, model_module, xgb_params
            )
            all_predictions.append(oof)
            all_fold_metrics.append(fold_metrics)
            all_overall.append(overall)
            if name == "Scope_Clean":
                boot, boot_summary = source_bootstrap(task, oof)
                all_boot.append(boot)
                all_boot_summary.append(boot_summary)

    predictions = pd.concat(all_predictions, ignore_index=True)
    reproduction = verify_original_reproduction(predictions)
    audit_rows = pd.concat(task_rows, ignore_index=True)
    audit_summary = pd.concat(audit_summaries, ignore_index=True)
    overall = pd.DataFrame(all_overall)
    folds = pd.concat(all_fold_metrics, ignore_index=True)
    boot = pd.concat(all_boot, ignore_index=True)
    boot_summary = pd.concat(all_boot_summary, ignore_index=True)
    sparsity = pd.concat(all_sparsity, ignore_index=True)

    audit_rows.to_csv(OUTPUT_ROOT / "ys_el_row_scope_decisions.csv", index=False, encoding="utf-8-sig")
    audit_summary.to_csv(OUTPUT_ROOT / "ys_el_scope_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_ROOT / "ys_el_variant_oof_predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_ROOT / "ys_el_variant_fold_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_ROOT / "ys_el_variant_overall_metrics.csv", index=False, encoding="utf-8-sig")
    reproduction.to_csv(OUTPUT_ROOT / "original_prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUTPUT_ROOT / "scope_clean_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUTPUT_ROOT / "scope_clean_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    sparsity.to_csv(OUTPUT_ROOT / "scope_clean_feature_sparsity.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "tasks": TASKS,
        "scope_rule": "same source-level material/test-condition decisions as UTS; two YS-only sources audited separately",
        "model": "RandomForest baseline + frozen nested XGBoost, equal-weight ensemble",
        "features": "original fold-specific composition_core features",
        "outer_validation": "unchanged five source-exclusive folds",
        "feature_or_parameter_tuning_in_this_stage": False,
        "source_bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("AUDIT SUMMARY")
    print(audit_summary.to_string(index=False))
    print("\nMODEL VARIANTS")
    print(overall.to_string(index=False))
    print("\nSCOPE-CLEAN BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print("\nFEATURE SPARSITY")
    print(sparsity.to_string(index=False))
    print(f"\nOriginal reproduction max difference: {reproduction['Absolute_Difference'].max():.6g}")
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
