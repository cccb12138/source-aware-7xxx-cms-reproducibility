from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
LOCAL_OUTPUTS = Path(r"F:\CC\outputs")
MTL_ROOT = LOCAL_OUTPUTS / "scope_clean_partial_label_mtl"
OUTPUT_ROOT = LOCAL_OUTPUTS / "matched_subset_final_robustness"
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASKS = ("YS", "UTS", "EL")
TARGETS = {
    "YS": config.TARGET_COLUMNS["YS"],
    "UTS": config.TARGET_COLUMNS["UTS"],
    "EL": config.TARGET_COLUMNS["EL"],
}
MATCHED_MODEL_FEATURES = {
    "YS": list(config.PRIMARY_FIXED_FEATURES),
    "UTS": ["Zn", "Mg", "Cu", "Fe", "Zr"],
    "EL": list(config.PRIMARY_FIXED_FEATURES),
}
COMMON_FEATURE_SETS = {
    "core10": list(config.PRIMARY_FIXED_FEATURES),
    "refined5": ["Zn", "Mg", "Cu", "Fe", "Zr"],
}
SEED = 20260812
N_BOOTSTRAP = 5000
N_CORR_BOOTSTRAP = 3000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_key(frame: pd.DataFrame) -> pd.Series:
    return frame[["Dataset", "Original_Sample_ID"]].fillna("").astype(str).agg("||".join, axis=1)


def score(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def source_macro(frame: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, part in frame.groupby("Source_Group"):
        err = part["y_pred"].to_numpy(dtype=float) - part["y_true"].to_numpy(dtype=float)
        rows.append((np.abs(err).mean(), np.sqrt(np.square(err).mean())))
    values = np.asarray(rows)
    return {
        "Source_Macro_MAE": float(values[:, 0].mean()),
        "Source_Macro_RMSE": float(values[:, 1].mean()),
    }


def summarize_predictions(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in predictions.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update({
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **score(part["y_true"], part["y_pred"]),
            **source_macro(part),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def load_matched() -> pd.DataFrame:
    data = pd.read_csv(MTL_ROOT / "MTL_scope_clean_with_outer_folds.csv")
    masks = [config.MASK_COLUMNS[task] for task in TASKS]
    data[masks] = data[masks].apply(pd.to_numeric, errors="coerce")
    matched = data.loc[data[masks].eq(1).all(axis=1)].copy()
    matched["Sample_Key"] = sample_key(matched)
    if matched["Sample_Key"].duplicated().any():
        dup = matched.loc[matched["Sample_Key"].duplicated(False), "Sample_Key"].tolist()
        raise AssertionError(f"Duplicate matched sample keys: {dup[:10]}")
    if matched.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("Source leakage in existing outer folds")
    for target in TARGETS.values():
        matched[target] = pd.to_numeric(matched[target], errors="coerce")
        if matched[target].isna().any():
            raise AssertionError(f"Matched target contains missing values: {target}")
    return matched


def correlation_tables(matched: pd.DataFrame):
    target_columns = list(TARGETS.values())
    source_means = matched.groupby("Source_Group", as_index=False)[target_columns].mean()
    rows = []
    boot_rows = []
    rng = np.random.default_rng(SEED + 1)
    grouped = {source: part[target_columns].to_numpy(dtype=float) for source, part in matched.groupby("Source_Group")}
    sources = np.asarray(list(grouped), dtype=object)

    for left_index, left_task in enumerate(TASKS):
        for right_task in TASKS[left_index + 1:]:
            left = TARGETS[left_task]
            right = TARGETS[right_task]
            for level, frame in (("Row", matched), ("Source_Mean", source_means)):
                rows.append({
                    "Level": level,
                    "Target_A": left_task,
                    "Target_B": right_task,
                    "N": len(frame),
                    "Spearman_r": float(spearmanr(frame[left], frame[right]).statistic),
                    "Pearson_r": float(pearsonr(frame[left], frame[right]).statistic),
                })
            for iteration in range(N_CORR_BOOTSTRAP):
                draw = sources[rng.integers(0, len(sources), size=len(sources))]
                values = np.concatenate([grouped[source] for source in draw], axis=0)
                boot_rows.append({
                    "Target_A": left_task,
                    "Target_B": right_task,
                    "Iteration": iteration,
                    "Spearman_r": spearmanr(values[:, TASKS.index(left_task)], values[:, TASKS.index(right_task)]).statistic,
                    "Pearson_r": pearsonr(values[:, TASKS.index(left_task)], values[:, TASKS.index(right_task)]).statistic,
                })
    boot = pd.DataFrame(boot_rows)
    summary = []
    for (left_task, right_task), part in boot.groupby(["Target_A", "Target_B"]):
        for metric in ("Spearman_r", "Pearson_r"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary.append({
                "Target_A": left_task,
                "Target_B": right_task,
                "Metric": metric,
                "Bootstrap_Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
            })
    return pd.DataFrame(rows), source_means, boot, pd.DataFrame(summary)


def parameter_tables():
    single_root = PROJECT_ROOT / "results" / "single_target"
    rf = pd.read_csv(single_root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(single_root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    return rf, xgb


def matched_independent_ensemble(matched: pd.DataFrame, model_module):
    rf_params, xgb_params = parameter_tables()
    parts = []
    fold_rows = []
    for task in TASKS:
        target = TARGETS[task]
        features = MATCHED_MODEL_FEATURES[task]
        for feature in features:
            matched[feature] = pd.to_numeric(matched[feature], errors="coerce")
        for fold in sorted(matched["Outer_Fold"].unique()):
            fold = int(fold)
            train = matched.loc[matched["Outer_Fold"].ne(fold)].copy()
            test = matched.loc[matched["Outer_Fold"].eq(fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError(f"Source leakage: {task}/fold{fold}")
            xgb_row = xgb_params.loc[xgb_params["Task"].eq(task) & xgb_params["Outer_Fold"].eq(fold)].iloc[0]
            if task == "UTS":
                rf_row = rf_params.loc[rf_params["Task"].eq(task) & rf_params["Outer_Fold"].eq(fold)].iloc[0]
                rf = model_module.rf_tuned(rf_row, SEED + fold)
            else:
                rf = model_module.rf_baseline(SEED + fold)
            xgb = model_module.xgb_tuned(xgb_row, SEED + fold)
            rf.fit(train[features], train[target])
            xgb.fit(train[features], train[target])
            pred_rf = rf.predict(test[features])
            pred_xgb = xgb.predict(test[features])
            pred = (pred_rf + pred_xgb) / 2.0
            part = pd.DataFrame({
                "Model": "Matched_Independent_RF_XGB_Mean",
                "Task": task,
                "Outer_Fold": fold,
                "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                "Sample_Key": test["Sample_Key"].to_numpy(),
                "Source_Group": test["Source_Group"].to_numpy(),
                "Dataset": test["Dataset"].to_numpy(),
                "y_true": test[target].to_numpy(),
                "pred_rf": pred_rf,
                "pred_xgb": pred_xgb,
                "y_pred": pred,
                "Selected_Features": "|".join(features),
            })
            parts.append(part)
            fold_rows.append({
                "Model": "Matched_Independent_RF_XGB_Mean",
                "Task": task,
                "Outer_Fold": fold,
                "Train_Rows": len(train),
                "Train_Sources": train["Source_Group"].nunique(),
                "Test_Rows": len(test),
                "Test_Sources": test["Source_Group"].nunique(),
                "Selected_Features": "|".join(features),
                **score(part["y_true"], part["y_pred"]),
            })
    predictions = pd.concat(parts, ignore_index=True)
    return predictions, pd.DataFrame(fold_rows), summarize_predictions(predictions, ["Model", "Task"])


def full_models_restricted_to_matched(matched: pd.DataFrame) -> pd.DataFrame:
    matched_keys = set(matched["Sample_Key"])
    rows = []
    target_clean_paths = {
        "YS": LOCAL_OUTPUTS / "ys_el_scope_audit" / "YS_scope_clean_with_outer_folds.csv",
        "UTS": LOCAL_OUTPUTS / "uts_systematic_scope_audit" / "UTS_scope_clean_with_outer_folds.csv",
        "EL": LOCAL_OUTPUTS / "ys_el_scope_audit" / "EL_scope_clean_with_outer_folds.csv",
    }
    uts_pred = pd.read_csv(LOCAL_OUTPUTS / "uts_scope_clean_final" / "oof_shap" / "scope_clean_oof_shap_values_wide.csv")
    ys_el_pred = pd.read_csv(LOCAL_OUTPUTS / "ys_el_scope_audit" / "ys_el_variant_oof_predictions.csv")
    ys_el_pred = ys_el_pred.loc[ys_el_pred["Variant"].eq("Scope_Clean")].copy()

    for task in TASKS:
        clean = pd.read_csv(target_clean_paths[task])
        clean["Sample_Key"] = sample_key(clean)
        if clean["Sample_Key"].duplicated().any():
            raise AssertionError(f"Duplicate full target keys: {task}")
        if task == "UTS":
            pred = uts_pred.loc[:, ["Model_Row_ID", "Outer_Fold", "Source_Group", "Dataset", "y_true", "y_pred"]].copy()
        else:
            pred = ys_el_pred.loc[ys_el_pred["Task"].eq(task), ["Model_Row_ID", "Outer_Fold", "Source_Group", "Dataset", "y_true", "y_pred"]].copy()
        pred = pred.merge(clean[["Model_Row_ID", "Sample_Key"]], on="Model_Row_ID", validate="one_to_one")
        pred = pred.loc[pred["Sample_Key"].isin(matched_keys)].copy()
        if len(pred) != len(matched):
            missing = matched_keys - set(pred["Sample_Key"])
            raise AssertionError(f"{task}: expected {len(matched)} matched predictions, got {len(pred)}; missing={list(missing)[:5]}")
        pred.insert(0, "Task", task)
        pred.insert(0, "Model", "Full_Label_Data_Model_restricted_to_matched")
        rows.append(pred)
    return pd.concat(rows, ignore_index=True)


def multioutput_rf_comparison(matched: pd.DataFrame):
    prediction_parts = []
    target_columns = [TARGETS[task] for task in TASKS]
    for feature_set, features in COMMON_FEATURE_SETS.items():
        work = matched.copy()
        work[features + target_columns] = work[features + target_columns].apply(pd.to_numeric, errors="coerce")
        for fold in sorted(work["Outer_Fold"].unique()):
            fold = int(fold)
            train = work.loc[work["Outer_Fold"].ne(fold)].copy()
            test = work.loc[work["Outer_Fold"].eq(fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError("Multi-output source leakage")
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(train[features])
            x_test = imputer.transform(test[features])

            joint = RandomForestRegressor(
                n_estimators=500,
                max_features=0.8,
                min_samples_leaf=2,
                random_state=SEED + fold,
                n_jobs=4,
            )
            joint.fit(x_train, train[target_columns])
            joint_pred = joint.predict(x_test)

            for task_index, task in enumerate(TASKS):
                independent = RandomForestRegressor(
                    n_estimators=500,
                    max_features=0.8,
                    min_samples_leaf=2,
                    random_state=SEED + 100 + task_index * 10 + fold,
                    n_jobs=4,
                )
                independent.fit(x_train, train[TARGETS[task]])
                independent_pred = independent.predict(x_test)
                for model, pred in (
                    ("Matched_MultiOutput_RF", joint_pred[:, task_index]),
                    ("Matched_Independent_RF", independent_pred),
                ):
                    prediction_parts.append(pd.DataFrame({
                        "Feature_Set": feature_set,
                        "Model": model,
                        "Task": task,
                        "Outer_Fold": fold,
                        "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                        "Sample_Key": test["Sample_Key"].to_numpy(),
                        "Source_Group": test["Source_Group"].to_numpy(),
                        "Dataset": test["Dataset"].to_numpy(),
                        "y_true": test[TARGETS[task]].to_numpy(),
                        "y_pred": pred,
                        "Selected_Features": "|".join(features),
                    }))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary = summarize_predictions(predictions, ["Feature_Set", "Model", "Task"])
    return predictions, summary


def source_bootstrap(predictions: pd.DataFrame, model_label: str):
    rng = np.random.default_rng(SEED + 500)
    rows = []
    for task in TASKS:
        part = predictions.loc[predictions["Task"].eq(task)].copy()
        grouped = {
            source: (group["y_true"].to_numpy(dtype=float), group["y_pred"].to_numpy(dtype=float))
            for source, group in part.groupby("Source_Group")
        }
        sources = np.asarray(list(grouped), dtype=object)
        for iteration in range(N_BOOTSTRAP):
            draw = sources[rng.integers(0, len(sources), size=len(sources))]
            y = np.concatenate([grouped[source][0] for source in draw])
            p = np.concatenate([grouped[source][1] for source in draw])
            rows.append({"Model": model_label, "Task": task, "Iteration": iteration, **score(y, p)})
    samples = pd.DataFrame(rows)
    summary = []
    for (model, task), part in samples.groupby(["Model", "Task"]):
        for metric in ("R2", "RMSE", "MAE"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary.append({
                "Model": model,
                "Task": task,
                "Metric": metric,
                "Bootstrap_Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
            })
    return samples, pd.DataFrame(summary)


def residual_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    wide = predictions.assign(Residual=predictions["y_pred"] - predictions["y_true"]).pivot(
        index="Sample_Key", columns="Task", values="Residual"
    )
    rows = []
    for i, left in enumerate(TASKS):
        for right in TASKS[i + 1:]:
            rows.append({
                "Task_A": left,
                "Task_B": right,
                "N": len(wide),
                "Spearman_Residual_r": float(spearmanr(wide[left], wide[right]).statistic),
                "Pearson_Residual_r": float(pearsonr(wide[left], wide[right]).statistic),
            })
    return pd.DataFrame(rows)


def dataset_fold_audit(matched: pd.DataFrame):
    by_dataset = matched.groupby("Dataset").agg(Rows=("Sample_Key", "size"), Sources=("Source_Group", "nunique")).reset_index()
    by_fold = matched.groupby("Outer_Fold").agg(Rows=("Sample_Key", "size"), Sources=("Source_Group", "nunique")).reset_index()
    return by_dataset, by_fold


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matched = load_matched()
    model_module = load_module("matched_model_builders", PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py")

    corr, source_means, corr_boot, corr_boot_summary = correlation_tables(matched)
    matched_pred, matched_folds, matched_summary = matched_independent_ensemble(matched, model_module)
    full_pred = full_models_restricted_to_matched(matched)
    full_summary = summarize_predictions(full_pred, ["Model", "Task"])
    fair_summary = pd.concat([full_summary, matched_summary], ignore_index=True)
    multi_pred, multi_summary = multioutput_rf_comparison(matched)
    boot_matched, boot_matched_summary = source_bootstrap(matched_pred, "Matched_Independent_RF_XGB_Mean")
    boot_full, boot_full_summary = source_bootstrap(full_pred, "Full_Label_Data_Model_restricted_to_matched")
    residual_corr = residual_correlations(matched_pred)
    by_dataset, by_fold = dataset_fold_audit(matched)

    matched.to_csv(OUTPUT_ROOT / "matched_complete_266_with_outer_folds.csv", index=False, encoding="utf-8-sig")
    by_dataset.to_csv(OUTPUT_ROOT / "matched_dataset_counts.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(OUTPUT_ROOT / "matched_fold_counts.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUTPUT_ROOT / "target_correlations_row_and_source_mean.csv", index=False, encoding="utf-8-sig")
    source_means.to_csv(OUTPUT_ROOT / "source_mean_targets.csv", index=False, encoding="utf-8-sig")
    corr_boot.to_csv(OUTPUT_ROOT / "target_correlation_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    corr_boot_summary.to_csv(OUTPUT_ROOT / "target_correlation_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    matched_pred.to_csv(OUTPUT_ROOT / "matched_only_independent_ensemble_oof_predictions.csv", index=False, encoding="utf-8-sig")
    matched_folds.to_csv(OUTPUT_ROOT / "matched_only_independent_ensemble_fold_metrics.csv", index=False, encoding="utf-8-sig")
    full_pred.to_csv(OUTPUT_ROOT / "full_models_restricted_to_matched_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fair_summary.to_csv(OUTPUT_ROOT / "fair_comparison_full_vs_matched_only_metrics.csv", index=False, encoding="utf-8-sig")
    multi_pred.to_csv(OUTPUT_ROOT / "matched_multioutput_vs_independent_rf_oof_predictions.csv", index=False, encoding="utf-8-sig")
    multi_summary.to_csv(OUTPUT_ROOT / "matched_multioutput_vs_independent_rf_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([boot_matched, boot_full], ignore_index=True).to_csv(OUTPUT_ROOT / "source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    pd.concat([boot_matched_summary, boot_full_summary], ignore_index=True).to_csv(OUTPUT_ROOT / "source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    residual_corr.to_csv(OUTPUT_ROOT / "matched_independent_residual_correlations.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "run_config.json").write_text(json.dumps({
        "rows": len(matched),
        "sources": int(matched["Source_Group"].nunique()),
        "datasets": int(matched["Dataset"].nunique()),
        "outer_validation": "existing five source-exclusive folds",
        "matched_independent_features": MATCHED_MODEL_FEATURES,
        "fair_comparison": "full-label-data source-blocked OOF predictions restricted to identical 266 samples versus models trained only on those 266 samples",
        "multioutput_comparison": "native multi-output RF versus three independent RF models under identical complete rows, features, and folds",
        "common_feature_sets": COMMON_FEATURE_SETS,
        "source_bootstrap_iterations": N_BOOTSTRAP,
        "correlation_source_bootstrap_iterations": N_CORR_BOOTSTRAP,
        "augmentation": False,
        "new_hyperparameter_tuning": False,
        "paper_figures_generated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MATCHED DATASET")
    print(pd.DataFrame([{"Rows": len(matched), "Sources": matched["Source_Group"].nunique(), "Datasets": matched["Dataset"].nunique()}]).to_string(index=False))
    print("\nTARGET CORRELATIONS")
    print(corr.to_string(index=False))
    print("\nFAIR COMPARISON")
    print(fair_summary.to_string(index=False))
    print("\nMULTIOUTPUT VS INDEPENDENT RF")
    print(multi_summary.to_string(index=False))
    print("\nSOURCE BOOTSTRAP")
    print(pd.concat([boot_matched_summary, boot_full_summary], ignore_index=True).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
