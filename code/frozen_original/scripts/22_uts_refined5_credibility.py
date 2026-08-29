from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260729
N_BOOTSTRAP = 3000
INNER_SPLITS = 3
ALPHAS = (0.10, 0.05)


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


def load_predictions() -> pd.DataFrame:
    path = (
        config.PROJECT_ROOT
        / "results"
        / "uts_refined5_oof_shap"
        / "uts_refined5_oof_shap_values_wide.csv"
    )
    raw = pd.read_csv(path)
    pred = raw[
        [
            "Task",
            "Outer_Fold",
            "Model_Row_ID",
            "Source_Group",
            "y_true",
            "y_pred",
            "pred_rf",
            "pred_xgb",
        ]
    ].copy()
    pred["Selected_Model"] = "Refined5_RF_XGBoost_OOF_Mean"
    pred["Residual"] = pred["y_pred"] - pred["y_true"]
    pred["Absolute_Error"] = pred["Residual"].abs()
    pred["Squared_Error"] = pred["Residual"].pow(2)
    pred["Model_Disagreement"] = (pred["pred_rf"] - pred["pred_xgb"]).abs()
    pred["Outer_Fold"] = pred["Outer_Fold"].astype(int)
    if len(pred) != 689 or pred["Source_Group"].nunique() != 263:
        raise AssertionError("Unexpected refined UTS prediction size")
    if pred.duplicated("Model_Row_ID").any():
        raise AssertionError("Duplicate UTS row identifiers")
    if pred.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("A source occurs in more than one outer fold")
    return pred


def applicability_domain(pred: pd.DataFrame, credibility_base):
    data = pd.read_csv(
        config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv"
    )
    merged = data.merge(
        pred[
            [
                "Model_Row_ID",
                "y_true",
                "y_pred",
                "Absolute_Error",
                "Model_Disagreement",
            ]
        ],
        on="Model_Row_ID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_oof"),
    )
    if len(merged) != len(pred):
        raise AssertionError("Failed to merge all refined OOF rows for AD")

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
        train_dist = train_nn.kneighbors(train_scaled, return_distance=True)[0][
            :, 1:
        ].mean(axis=1)
        threshold = float(np.quantile(train_dist, 0.95))
        test_nn = NearestNeighbors(n_neighbors=k).fit(train_scaled)
        test_dist = test_nn.kneighbors(test_scaled, return_distance=True)[0].mean(
            axis=1
        )
        threshold_safe = max(threshold, np.finfo(float).eps)

        for index, (_, row) in enumerate(test.iterrows()):
            diagnostics.append(
                {
                    "Task": TASK,
                    "Outer_Fold": fold,
                    "Model_Row_ID": row["Model_Row_ID"],
                    "Source_Group": row["Source_Group"],
                    "y_true": row["y_true"],
                    "y_pred": row["y_pred"],
                    "Absolute_Error": row["Absolute_Error"],
                    "Model_Disagreement": row["Model_Disagreement"],
                    "AD_Distance": test_dist[index],
                    "AD_Threshold": threshold,
                    "AD_Distance_Ratio": test_dist[index] / threshold_safe,
                    "AD_Status": "Inside"
                    if test_dist[index] <= threshold
                    else "Outside",
                    "AD_Features": "|".join(FEATURES),
                }
            )

    diag = pd.DataFrame(diagnostics)
    corr = spearmanr(
        diag["AD_Distance_Ratio"], diag["Absolute_Error"], nan_policy="omit"
    ).statistic
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
                **credibility_base.metrics(
                    part["y_true"].to_numpy(), part["y_pred"].to_numpy()
                ),
            }
        )
    return diag, pd.DataFrame(rows)


def nested_group_conformal(pred: pd.DataFrame, model_module):
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
            rf_params["Task"].eq(TASK)
            & rf_params["Outer_Fold"].eq(outer_fold)
        ].iloc[0]
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(TASK)
            & xgb_params["Outer_Fold"].eq(outer_fold)
        ].iloc[0]
        groups = outer_train["Source_Group"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=INNER_SPLITS)
        seen_ids = []
        fold_calibration = []
        for inner_fold, (fit_idx, calibration_idx) in enumerate(
            splitter.split(outer_train, groups=groups)
        ):
            fit = outer_train.iloc[fit_idx]
            calibration = outer_train.iloc[calibration_idx]
            if set(fit["Source_Group"]) & set(calibration["Source_Group"]):
                raise AssertionError("Source leakage in conformal inner split")

            model_seed = SEED + outer_fold * 100 + inner_fold
            rf = model_module.rf_tuned(rf_row, model_seed)
            xgb = model_module.xgb_tuned(xgb_row, model_seed)
            rf.fit(fit[FEATURES], fit[TARGET])
            xgb.fit(fit[FEATURES], fit[TARGET])
            pred_rf = rf.predict(calibration[FEATURES])
            pred_xgb = xgb.predict(calibration[FEATURES])
            y_pred = (pred_rf + pred_xgb) / 2.0
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
        source_scores = (
            calibration_frame.groupby("Source_Group")["Absolute_Residual"]
            .max()
            .to_numpy()
        )

        fold_intervals = outer_test_pred[
            [
                "Outer_Fold",
                "Model_Row_ID",
                "Source_Group",
                "y_true",
                "y_pred",
            ]
        ].copy()
        for alpha in ALPHAS:
            coverage = int(round((1.0 - alpha) * 100))
            for method, scores in (
                ("RowCrossConformal", row_scores),
                ("SourceMaxCrossConformal", source_scores),
            ):
                q = conformal_quantile(scores, alpha)
                quantile_rows.append(
                    {
                        "Outer_Fold": outer_fold,
                        "Method": method,
                        "Nominal_Coverage": 1.0 - alpha,
                        "Calibration_Rows": len(calibration_frame),
                        "Calibration_Sources": calibration_frame[
                            "Source_Group"
                        ].nunique(),
                        "Score_Count": len(scores),
                        "Quantile_MPa": q,
                    }
                )
                prefix = f"PI{coverage}_{method}"
                fold_intervals[f"{prefix}_Lower"] = fold_intervals["y_pred"] - q
                fold_intervals[f"{prefix}_Upper"] = fold_intervals["y_pred"] + q
                fold_intervals[f"{prefix}_Covered"] = (
                    fold_intervals["y_true"]
                    .between(
                        fold_intervals[f"{prefix}_Lower"],
                        fold_intervals[f"{prefix}_Upper"],
                    )
                    .astype(int)
                )
        interval_rows.append(fold_intervals)

    return (
        pd.DataFrame(calibration_rows),
        pd.DataFrame(quantile_rows),
        pd.concat(interval_rows, ignore_index=True),
    )


def interval_summary(intervals: pd.DataFrame):
    rows = []
    for method in ("RowCrossConformal", "SourceMaxCrossConformal"):
        for nominal in (0.90, 0.95):
            coverage = int(round(nominal * 100))
            prefix = f"PI{coverage}_{method}"
            for scope, fold, part in [("All", "All", intervals)]:
                source_covered = part.groupby("Source_Group")[
                    f"{prefix}_Covered"
                ].min()
                rows.append(
                    {
                        "Scope": scope,
                        "Outer_Fold": fold,
                        "Method": method,
                        "Nominal_Coverage": nominal,
                        "Rows": len(part),
                        "Sources": part["Source_Group"].nunique(),
                        "Row_Coverage": part[f"{prefix}_Covered"].mean(),
                        "Source_Simultaneous_Coverage": source_covered.mean(),
                        "Mean_Width_MPa": (
                            part[f"{prefix}_Upper"] - part[f"{prefix}_Lower"]
                        ).mean(),
                        "Median_Width_MPa": (
                            part[f"{prefix}_Upper"] - part[f"{prefix}_Lower"]
                        ).median(),
                    }
                )
            for outer_fold, part in intervals.groupby("Outer_Fold"):
                source_covered = part.groupby("Source_Group")[
                    f"{prefix}_Covered"
                ].min()
                rows.append(
                    {
                        "Scope": "Outer_Fold",
                        "Outer_Fold": int(outer_fold),
                        "Method": method,
                        "Nominal_Coverage": nominal,
                        "Rows": len(part),
                        "Sources": part["Source_Group"].nunique(),
                        "Row_Coverage": part[f"{prefix}_Covered"].mean(),
                        "Source_Simultaneous_Coverage": source_covered.mean(),
                        "Mean_Width_MPa": (
                            part[f"{prefix}_Upper"] - part[f"{prefix}_Lower"]
                        ).mean(),
                        "Median_Width_MPa": (
                            part[f"{prefix}_Upper"] - part[f"{prefix}_Lower"]
                        ).median(),
                    }
                )
    return pd.DataFrame(rows)


def disagreement_diagnostic(pred: pd.DataFrame):
    corr = spearmanr(
        pred["Model_Disagreement"], pred["Absolute_Error"], nan_policy="omit"
    ).statistic
    ranked = pred.copy()
    ranked["Disagreement_Quintile"] = pd.qcut(
        ranked["Model_Disagreement"], q=5, labels=False, duplicates="drop"
    )
    bins = (
        ranked.groupby("Disagreement_Quintile")
        .agg(
            Rows=("Model_Row_ID", "size"),
            Sources=("Source_Group", "nunique"),
            Disagreement_Mean=("Model_Disagreement", "mean"),
            Disagreement_Min=("Model_Disagreement", "min"),
            Disagreement_Max=("Model_Disagreement", "max"),
            Absolute_Error_Mean=("Absolute_Error", "mean"),
            Absolute_Error_Median=("Absolute_Error", "median"),
            Absolute_Error_P90=("Absolute_Error", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "Rows": len(pred),
                "Sources": pred["Source_Group"].nunique(),
                "Disagreement_Error_Spearman": corr,
                "Lowest_Quintile_MAE": bins.iloc[0]["Absolute_Error_Mean"],
                "Highest_Quintile_MAE": bins.iloc[-1]["Absolute_Error_Mean"],
                "Highest_vs_Lowest_MAE_Ratio": bins.iloc[-1][
                    "Absolute_Error_Mean"
                ]
                / bins.iloc[0]["Absolute_Error_Mean"],
            }
        ]
    )
    return ranked, bins, summary


def save_figures(out: Path, ad: pd.DataFrame, intervals: pd.DataFrame, bins: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    colors = ad["AD_Status"].map({"Inside": "#4C72B0", "Outside": "#C44E52"})
    ax.scatter(
        ad["AD_Distance_Ratio"],
        ad["Absolute_Error"],
        c=colors,
        s=22,
        alpha=0.65,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("5-feature AD distance / fold threshold")
    ax.set_ylabel("Absolute error (MPa)")
    ax.set_title("UTS refined model: applicability distance vs error")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "refined5_ad_distance_vs_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        bins["Disagreement_Quintile"] + 1,
        bins["Absolute_Error_Mean"],
        marker="o",
        color="#C44E52",
    )
    ax.set_xlabel("RF-XGBoost disagreement quintile")
    ax.set_ylabel("Mean absolute error (MPa)")
    ax.set_title("Model disagreement as an error-risk indicator")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "model_disagreement_risk.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    methods = ["RowCrossConformal", "SourceMaxCrossConformal"]
    labels = ["Row cross-conformal", "Source-max cross-conformal"]
    coverage_rows = []
    for method, label in zip(methods, labels):
        for nominal in (0.90, 0.95):
            prefix = f"PI{int(nominal * 100)}_{method}"
            coverage_rows.append(
                {
                    "Method": label,
                    "Nominal": nominal,
                    "Observed": intervals[f"{prefix}_Covered"].mean(),
                    "Width": (
                        intervals[f"{prefix}_Upper"]
                        - intervals[f"{prefix}_Lower"]
                    ).mean(),
                }
            )
    frame = pd.DataFrame(coverage_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(methods))
    width = 0.34
    for offset, nominal in zip((-width / 2, width / 2), (0.90, 0.95)):
        part = frame.loc[frame["Nominal"].eq(nominal)]
        axes[0].bar(x + offset, part["Observed"], width=width, label=f"{int(nominal*100)}%")
        axes[1].bar(x + offset, part["Width"], width=width, label=f"{int(nominal*100)}%")
    axes[0].axhline(0.90, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Observed row coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean interval width (MPa)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["Row", "Source-max"])
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle("Nested source-group conformal interval diagnostics")
    fig.tight_layout()
    fig.savefig(out / "conformal_coverage_and_width.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    credibility_base = load_python_module(
        "credibility_base", config.PROJECT_ROOT / "13_model_credibility.py"
    )
    credibility_base.N_BOOTSTRAP = N_BOOTSTRAP
    credibility_base.SEED = SEED
    model_module = load_python_module(
        "model_builders",
        config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py",
    )
    pred = load_predictions()

    pooled, folds = credibility_base.pooled_and_fold_metrics(pred)
    stability = credibility_base.fold_stability(folds)
    per_source = credibility_base.source_metrics(pred)
    boot, boot_summary = credibility_base.source_cluster_bootstrap(pred)
    ad, ad_summary = applicability_domain(pred, credibility_base)
    calibration, quantiles, intervals = nested_group_conformal(pred, model_module)
    pi_summary = interval_summary(intervals)
    disagreement_rows, disagreement_bins, disagreement_summary = (
        disagreement_diagnostic(pred)
    )

    out = config.PROJECT_ROOT / "results" / "uts_refined5_credibility"
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "refined5_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(out / "pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "fold_stability_summary.csv", index=False, encoding="utf-8-sig")
    per_source.to_csv(out / "per_source_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(out / "source_cluster_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "source_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    ad.to_csv(out / "applicability_row_diagnostics.csv", index=False, encoding="utf-8-sig")
    ad_summary.to_csv(out / "applicability_summary.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(out / "nested_calibration_residuals.csv", index=False, encoding="utf-8-sig")
    quantiles.to_csv(out / "conformal_quantiles_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(out / "oof_prediction_intervals.csv", index=False, encoding="utf-8-sig")
    pi_summary.to_csv(out / "prediction_interval_summary.csv", index=False, encoding="utf-8-sig")
    disagreement_rows.to_csv(out / "model_disagreement_rows.csv", index=False, encoding="utf-8-sig")
    disagreement_bins.to_csv(out / "model_disagreement_risk_bins.csv", index=False, encoding="utf-8-sig")
    disagreement_summary.to_csv(out / "model_disagreement_summary.csv", index=False, encoding="utf-8-sig")
    save_figures(out, ad, intervals, disagreement_bins)

    run_config = {
        "task": TASK,
        "features": FEATURES,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "outer_validation": "fixed 5-fold source-group split",
        "applicability_domain": "fold-specific median imputation + standardization + mean 5-NN distance; 95th percentile training threshold",
        "aggregate_uncertainty": f"{N_BOOTSTRAP} source-cluster bootstrap iterations",
        "prediction_intervals": "3-fold inner source-group cross-conformal absolute residuals; row and source-maximum scores",
        "interval_note": "cross-conformal diagnostic, not an external prospective calibration guarantee",
        "feature_or_parameter_tuning_in_this_stage": False,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("POOLED OOF")
    print(pooled.to_string(index=False))
    print("\nSOURCE BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print("\nAPPLICABILITY DOMAIN")
    print(ad_summary.to_string(index=False))
    print("\nPREDICTION INTERVALS - ALL OOF")
    print(pi_summary.loc[pi_summary["Scope"].eq("All")].to_string(index=False))
    print("\nMODEL DISAGREEMENT")
    print(disagreement_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
