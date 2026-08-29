from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
AUDIT_ROOT = PROJECT_ROOT / "results" / "uts_systematic_scope_audit"
PARAM_ROOT = Path(r"F:\CC\outputs\paper_scope_clean_final\model_decisions")
SHAP_ROOT = Path(r"F:\CC\outputs\uts_scope_clean_final\oof_shap")
OUTPUT_ROOT = Path(r"F:\CC\outputs\uts_scope_clean_final\credibility")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASK = "UTS"
TARGET = "UTS_MPa"
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260802
N_BOOTSTRAP = 5000
INNER_SPLITS = 3
ALPHAS = (0.10, 0.05)


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
    return data


def load_predictions() -> pd.DataFrame:
    raw = pd.read_csv(SHAP_ROOT / "scope_clean_oof_shap_values_wide.csv")
    pred = raw[
        [
            "Task",
            "Outer_Fold",
            "Model_Row_ID",
            "Source_Group",
            "Dataset",
            "Audit_Tier",
            "y_true",
            "y_pred",
            "pred_rf",
            "pred_xgb",
        ]
    ].copy()
    pred["Selected_Model"] = "ScopeClean_Refined5_RF_XGBoost_OOF_Mean"
    pred["Residual"] = pred["y_pred"] - pred["y_true"]
    pred["Absolute_Error"] = pred["Residual"].abs()
    pred["Squared_Error"] = pred["Residual"].pow(2)
    pred["Model_Disagreement"] = (pred["pred_rf"] - pred["pred_xgb"]).abs()
    pred["Outer_Fold"] = pred["Outer_Fold"].astype(int)
    if len(pred) != 675 or pred["Source_Group"].nunique() != 258:
        raise AssertionError("Unexpected scope-clean prediction size")
    if pred.duplicated("Model_Row_ID").any():
        raise AssertionError("Duplicate UTS row identifiers")
    if pred.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("A source occurs in more than one outer fold")
    return pred


def applicability_domain(data: pd.DataFrame, pred: pd.DataFrame, credibility_base):
    merged = data.merge(
        pred[["Model_Row_ID", "y_true", "y_pred", "Absolute_Error", "Model_Disagreement"]],
        on="Model_Row_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_oof"),
    )
    if len(merged) != len(pred):
        raise AssertionError("Failed to merge all scope-clean rows for AD")

    diagnostics = []
    for fold in sorted(merged["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = merged.loc[merged["Outer_Fold"].eq(fold)].copy()
        x_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
        x_test = test[FEATURES].apply(pd.to_numeric, errors="coerce")

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
        test_scaled = scaler.transform(imputer.transform(x_test))
        k = min(5, max(1, len(train_scaled) - 1))
        train_nn = NearestNeighbors(n_neighbors=k + 1).fit(train_scaled)
        train_dist = train_nn.kneighbors(train_scaled, return_distance=True)[0][:, 1:].mean(axis=1)
        threshold = float(np.quantile(train_dist, 0.95))
        test_nn = NearestNeighbors(n_neighbors=k).fit(train_scaled)
        test_dist = test_nn.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
        threshold_safe = max(threshold, np.finfo(float).eps)

        for index, (_, row) in enumerate(test.iterrows()):
            diagnostics.append(
                {
                    "Task": TASK,
                    "Outer_Fold": fold,
                    "Model_Row_ID": row["Model_Row_ID"],
                    "Source_Group": row["Source_Group"],
                    "Dataset": row["Dataset"],
                    "Audit_Tier": row["Audit_Tier"],
                    "y_true": row["y_true"],
                    "y_pred": row["y_pred"],
                    "Absolute_Error": row["Absolute_Error"],
                    "Model_Disagreement": row["Model_Disagreement"],
                    "AD_Distance": test_dist[index],
                    "AD_Threshold": threshold,
                    "AD_Distance_Ratio": test_dist[index] / threshold_safe,
                    "AD_Status": "Inside" if test_dist[index] <= threshold else "Outside",
                    "AD_Features": "|".join(FEATURES),
                }
            )

    diag = pd.DataFrame(diagnostics)
    corr = spearmanr(diag["AD_Distance_Ratio"], diag["Absolute_Error"], nan_policy="omit").statistic
    rows = []
    for status in ("All", "Inside", "Outside"):
        part = diag if status == "All" else diag.loc[diag["AD_Status"].eq(status)]
        rows.append(
            {
                "Task": TASK,
                "AD_Status": status,
                "Rows": len(part),
                "Row_Fraction": len(part) / len(diag),
                "Sources": part["Source_Group"].nunique(),
                "Distance_Error_Spearman_All": corr if status == "All" else np.nan,
                **credibility_base.metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            }
        )
    return diag, pd.DataFrame(rows)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def nested_group_conformal(data: pd.DataFrame, pred: pd.DataFrame, model_module):
    rf_params = pd.read_csv(PARAM_ROOT / "rf_params_by_outer_fold.csv")
    xgb_params = pd.read_csv(PARAM_ROOT / "xgb_params_by_outer_fold.csv")
    calibration_rows = []
    quantile_rows = []
    interval_rows = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        outer_train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy()
        outer_test_pred = pred.loc[pred["Outer_Fold"].eq(outer_fold)].copy()
        if set(outer_train["Source_Group"]) & set(outer_test_pred["Source_Group"]):
            raise AssertionError("Source leakage in conformal outer split")

        rf_row = rf_params.loc[
            rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(outer_fold)
        ].iloc[0]
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(outer_fold)
        ].iloc[0]
        groups = outer_train["Source_Group"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=INNER_SPLITS)
        seen_ids = []
        fold_calibration = []
        for inner_fold, (fit_idx, calibration_idx) in enumerate(splitter.split(outer_train, groups=groups)):
            fit = outer_train.iloc[fit_idx]
            calibration = outer_train.iloc[calibration_idx]
            if set(fit["Source_Group"]) & set(calibration["Source_Group"]):
                raise AssertionError("Source leakage in conformal inner split")
            model_seed = SEED + outer_fold * 100 + inner_fold
            rf = model_module.rf_tuned(rf_row, model_seed)
            xgb = model_module.xgb_tuned(xgb_row, model_seed)
            rf.named_steps["model"].set_params(n_estimators=180)
            rf.fit(fit[FEATURES], fit[TARGET])
            xgb.fit(fit[FEATURES], fit[TARGET])
            y_pred = (rf.predict(calibration[FEATURES]) + xgb.predict(calibration[FEATURES])) / 2.0
            for position, (_, row) in enumerate(calibration.iterrows()):
                record = {
                    "Outer_Fold": outer_fold,
                    "Inner_Fold": inner_fold,
                    "Model_Row_ID": row["Model_Row_ID"],
                    "Source_Group": row["Source_Group"],
                    "y_true": row[TARGET],
                    "y_pred": y_pred[position],
                    "Absolute_Residual": abs(y_pred[position] - row[TARGET]),
                }
                calibration_rows.append(record)
                fold_calibration.append(record)
                seen_ids.append(row["Model_Row_ID"])
        if len(seen_ids) != len(outer_train) or len(set(seen_ids)) != len(outer_train):
            raise AssertionError("Inner cross-fitting did not cover outer training rows once")

        calibration_frame = pd.DataFrame(fold_calibration)
        row_scores = calibration_frame["Absolute_Residual"].to_numpy()
        source_scores = calibration_frame.groupby("Source_Group")["Absolute_Residual"].max().to_numpy()
        fold_intervals = outer_test_pred[
            ["Outer_Fold", "Model_Row_ID", "Source_Group", "Dataset", "Audit_Tier", "y_true", "y_pred"]
        ].copy()
        for alpha in ALPHAS:
            coverage = int(round((1.0 - alpha) * 100))
            for method, scores in (("RowCrossConformal", row_scores), ("SourceMaxCrossConformal", source_scores)):
                q = conformal_quantile(scores, alpha)
                quantile_rows.append(
                    {
                        "Outer_Fold": outer_fold,
                        "Method": method,
                        "Nominal_Coverage": 1.0 - alpha,
                        "Calibration_Rows": len(calibration_frame),
                        "Calibration_Sources": calibration_frame["Source_Group"].nunique(),
                        "Score_Count": len(scores),
                        "Quantile_MPa": q,
                    }
                )
                prefix = f"PI{coverage}_{method}"
                fold_intervals[f"{prefix}_Lower"] = fold_intervals["y_pred"] - q
                fold_intervals[f"{prefix}_Upper"] = fold_intervals["y_pred"] + q
                fold_intervals[f"{prefix}_Covered"] = (
                    fold_intervals["y_true"].between(
                        fold_intervals[f"{prefix}_Lower"], fold_intervals[f"{prefix}_Upper"]
                    ).astype(int)
                )
        interval_rows.append(fold_intervals)

    return pd.DataFrame(calibration_rows), pd.DataFrame(quantile_rows), pd.concat(interval_rows, ignore_index=True)


def compare_old_clean(boot_summary, ad_summary, pi_summary, disagreement_summary):
    rows = []
    old_boot = pd.read_csv(PROJECT_ROOT / "results" / "uts_refined5_credibility" / "source_cluster_bootstrap_summary.csv")
    for metric in ["R2", "RMSE", "MAE"]:
        old_value = old_boot.loc[old_boot["Metric"].eq(metric), "Bootstrap_Median"].iloc[0]
        clean_value = boot_summary.loc[boot_summary["Metric"].eq(metric), "Bootstrap_Median"].iloc[0]
        rows.append(
            {
                "Section": "SourceClusterBootstrap",
                "Metric": metric,
                "Old689": old_value,
                "Clean675": clean_value,
                "Clean_Minus_Old": clean_value - old_value,
            }
        )
    old_ad = pd.read_csv(PROJECT_ROOT / "results" / "uts_refined5_credibility" / "applicability_summary.csv")
    for status in ["Inside", "Outside"]:
        old_row = old_ad.loc[old_ad["AD_Status"].eq(status)].iloc[0]
        clean_row = ad_summary.loc[ad_summary["AD_Status"].eq(status)].iloc[0]
        for metric in ["Row_Fraction", "RMSE", "MAE"]:
            rows.append(
                {
                    "Section": f"Applicability_{status}",
                    "Metric": metric,
                    "Old689": old_row[metric],
                    "Clean675": clean_row[metric],
                    "Clean_Minus_Old": clean_row[metric] - old_row[metric],
                }
            )
    old_pi = pd.read_csv(PROJECT_ROOT / "results" / "uts_refined5_credibility" / "prediction_interval_summary.csv")
    for method in ["RowCrossConformal", "SourceMaxCrossConformal"]:
        for nominal in [0.90, 0.95]:
            old_row = old_pi.loc[
                old_pi["Scope"].eq("All")
                & old_pi["Method"].eq(method)
                & old_pi["Nominal_Coverage"].eq(nominal)
            ].iloc[0]
            clean_row = pi_summary.loc[
                pi_summary["Scope"].eq("All")
                & pi_summary["Method"].eq(method)
                & pi_summary["Nominal_Coverage"].eq(nominal)
            ].iloc[0]
            for metric in ["Row_Coverage", "Source_Simultaneous_Coverage", "Mean_Width_MPa"]:
                rows.append(
                    {
                        "Section": f"PI_{method}_{int(nominal*100)}",
                        "Metric": metric,
                        "Old689": old_row[metric],
                        "Clean675": clean_row[metric],
                        "Clean_Minus_Old": clean_row[metric] - old_row[metric],
                    }
                )
    old_dis = pd.read_csv(PROJECT_ROOT / "results" / "uts_refined5_credibility" / "model_disagreement_summary.csv").iloc[0]
    clean_dis = disagreement_summary.iloc[0]
    for metric in ["Disagreement_Error_Spearman", "Highest_vs_Lowest_MAE_Ratio"]:
        rows.append(
            {
                "Section": "ModelDisagreement",
                "Metric": metric,
                "Old689": old_dis[metric],
                "Clean675": clean_dis[metric],
                "Clean_Minus_Old": clean_dis[metric] - old_dis[metric],
            }
        )
    return pd.DataFrame(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    pred = load_predictions()
    credibility_base = load_module("scope_clean_credibility_base", PROJECT_ROOT / "13_model_credibility.py")
    credibility_base.N_BOOTSTRAP = N_BOOTSTRAP
    credibility_base.SEED = SEED
    credibility_ref = load_module("credibility_reference", PROJECT_ROOT / "22_uts_refined5_credibility.py")
    model_module = load_module("scope_clean_model_builders", PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py")

    pooled, folds = credibility_base.pooled_and_fold_metrics(pred)
    stability = credibility_base.fold_stability(folds)
    per_source = credibility_base.source_metrics(pred)
    boot, boot_summary = credibility_base.source_cluster_bootstrap(pred)
    ad, ad_summary = applicability_domain(data, pred, credibility_base)
    calibration, quantiles, intervals = nested_group_conformal(data, pred, model_module)
    pi_summary = credibility_ref.interval_summary(intervals)
    disagreement_rows, disagreement_bins, disagreement_summary = credibility_ref.disagreement_diagnostic(pred)
    pred.to_csv(OUTPUT_ROOT / "scope_clean_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUTPUT_ROOT / "pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_ROOT / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(OUTPUT_ROOT / "fold_stability_summary.csv", index=False, encoding="utf-8-sig")
    per_source.to_csv(OUTPUT_ROOT / "per_source_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUTPUT_ROOT / "source_cluster_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUTPUT_ROOT / "source_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    ad.to_csv(OUTPUT_ROOT / "applicability_row_diagnostics.csv", index=False, encoding="utf-8-sig")
    ad_summary.to_csv(OUTPUT_ROOT / "applicability_summary.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUTPUT_ROOT / "nested_calibration_residuals.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(OUTPUT_ROOT / "conformal_quantiles_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUTPUT_ROOT / "oof_prediction_intervals.csv", index=False, encoding="utf-8-sig")
    pi_summary.to_csv(OUTPUT_ROOT / "prediction_interval_summary.csv", index=False, encoding="utf-8-sig")
    disagreement_rows.to_csv(OUTPUT_ROOT / "model_disagreement_rows.csv", index=False, encoding="utf-8-sig")
    disagreement_bins.to_csv(OUTPUT_ROOT / "model_disagreement_risk_bins.csv", index=False, encoding="utf-8-sig")
    disagreement_summary.to_csv(OUTPUT_ROOT / "model_disagreement_summary.csv", index=False, encoding="utf-8-sig")
    credibility_ref.save_figures(OUTPUT_ROOT, ad, intervals, disagreement_bins)

    run_config = {
        "input_rows": len(data),
        "input_sources": int(data["Source_Group"].nunique()),
        "features": FEATURES,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "outer_validation": "unchanged five source-exclusive folds",
        "aggregate_uncertainty": f"{N_BOOTSTRAP} source-cluster bootstrap iterations",
        "applicability_domain": "fold-specific median imputation, standardization and mean 5-NN distance",
        "prediction_intervals": "3-fold inner source-group cross-conformal residuals",
        "feature_or_parameter_tuning_in_this_stage": False,
        "hyperparameters": "retuned using only the 675-row scope-clean dataset",
        "old_689_row_results_used": False,
        "interval_note": "diagnostic empirical interval, not a prospective external guarantee",
        "seed": SEED,
    }
    (OUTPUT_ROOT / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("POOLED OOF")
    print(pooled.to_string(index=False))
    print("\nSOURCE BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print("\nAPPLICABILITY DOMAIN")
    print(ad_summary.to_string(index=False))
    print("\nPREDICTION INTERVALS - ALL")
    print(pi_summary.loc[pi_summary["Scope"].eq("All")].to_string(index=False))
    print("\nMODEL DISAGREEMENT")
    print(disagreement_summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
