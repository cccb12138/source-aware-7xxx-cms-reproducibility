from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260730
N_RANDOM_REMOVALS = 3000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    return {
        "Rows": len(frame),
        "Sources": frame["Source_Group"].nunique(),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def size_bin(rows: int) -> str:
    if rows == 1:
        return "1"
    if rows <= 3:
        return "2-3"
    if rows <= 6:
        return "4-6"
    return "7+"


def load_inputs():
    data = pd.read_csv(
        config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv"
    )
    pred = pd.read_csv(
        config.PROJECT_ROOT
        / "results"
        / "uts_refined5_credibility"
        / "refined5_oof_predictions.csv"
    )
    source_metrics = pd.read_csv(
        config.PROJECT_ROOT
        / "results"
        / "uts_refined5_credibility"
        / "per_source_metrics.csv"
    )
    ad = pd.read_csv(
        config.PROJECT_ROOT
        / "results"
        / "uts_refined5_credibility"
        / "applicability_row_diagnostics.csv"
    )
    outlier_sources = sorted(
        source_metrics.loc[source_metrics["MAE_Outlier"].eq(True), "Source_Group"]
        .astype(str)
        .tolist()
    )
    if len(outlier_sources) != 14:
        raise AssertionError(f"Expected 14 audit sources, found {len(outlier_sources)}")
    if len(pred) != len(data):
        raise AssertionError("Prediction and processed data row counts differ")
    return data, pred, source_metrics, ad, outlier_sources


def build_audit_tables(data, pred, source_metrics, ad, outlier_sources):
    rows = data.loc[data["Source_Group"].astype(str).isin(outlier_sources)].copy()
    rows = rows.merge(
        pred[
            [
                "Model_Row_ID",
                "y_pred",
                "Residual",
                "Absolute_Error",
                "Model_Disagreement",
            ]
        ],
        on="Model_Row_ID",
        how="left",
        validate="one_to_one",
    )
    rows = rows.merge(
        ad[
            [
                "Model_Row_ID",
                "AD_Status",
                "AD_Distance_Ratio",
            ]
        ],
        on="Model_Row_ID",
        how="left",
        validate="one_to_one",
    )

    summary_rows = []
    for source, part in rows.groupby("Source_Group", sort=False):
        comps = part[FEATURES].apply(pd.to_numeric, errors="coerce").round(6)
        uts = pd.to_numeric(part[TARGET], errors="coerce")
        doi_values = sorted(
            {
                str(value).strip()
                for value in part["DOI"].dropna()
                if str(value).strip()
            }
        )
        evidence = sorted(
            {str(value).strip() for value in part["Evidence_Level"].dropna() if str(value).strip()}
        )
        quality = sorted(
            {str(value).strip() for value in part["Quality_Flag"].dropna() if str(value).strip()}
        )
        temper_nonblank = part["Temper"].fillna("").astype(str).str.strip().ne("")
        process_missing = pd.to_numeric(part["Process_Missing_Count"], errors="coerce")
        flags = []
        if not doi_values:
            flags.append("DOI_missing")
        if any("Medium" in value or "secondary" in value.lower() for value in evidence):
            flags.append("secondary_or_medium_traceability")
        if process_missing.median() >= 7:
            flags.append("process_metadata_mostly_missing")
        if temper_nonblank.mean() < 0.5:
            flags.append("temper_mostly_missing")
        if len(part) >= 2 and comps.drop_duplicates().shape[0] == 1 and uts.max() - uts.min() >= 100:
            flags.append("same_refined5_composition_wide_UTS")
        if part["AD_Status"].eq("Outside").mean() >= 0.5:
            flags.append("mostly_outside_AD")
        critical_bad = (
            pd.to_numeric(part["Zn"], errors="coerce").le(0)
            | pd.to_numeric(part["Al"], errors="coerce").lt(75)
            | pd.to_numeric(part["Al"], errors="coerce").gt(99.9)
            | pd.to_numeric(part["Cu"], errors="coerce").gt(6)
            | pd.to_numeric(part["Ti"], errors="coerce").gt(1)
            | pd.to_numeric(part["Zr"], errors="coerce").gt(1)
        ).any()
        if critical_bad:
            flags.append("critical_composition_rule_failure")

        recommendation = (
            "manual_source_verification_before_decision"
            if "DOI_missing" in flags or "secondary_or_medium_traceability" in flags
            else "retain_as_valid_domain_shift_unless_source_check_finds_error"
        )
        summary_rows.append(
            {
                "Source_Group": source,
                "Outer_Fold": int(part["Outer_Fold"].iloc[0]),
                "Rows": len(part),
                "Dataset": "|".join(sorted(part["Dataset"].dropna().astype(str).unique())),
                "DOI": "|".join(doi_values),
                "Source_Type": "|".join(sorted(part["Source_Type"].dropna().astype(str).unique())),
                "Evidence_Level": "|".join(evidence),
                "Quality_Flag": "|".join(quality),
                "Alloy_Grades": "|".join(sorted(part["Alloy_Grade"].dropna().astype(str).unique())),
                "Unique_Refined5_Compositions": comps.drop_duplicates().shape[0],
                "UTS_Min_MPa": uts.min(),
                "UTS_Max_MPa": uts.max(),
                "UTS_Range_MPa": uts.max() - uts.min(),
                "UTS_Mean_MPa": uts.mean(),
                "Source_MAE_MPa": source_metrics.loc[
                    source_metrics["Source_Group"].astype(str).eq(str(source)), "MAE"
                ].iloc[0],
                "Process_Missing_Median": process_missing.median(),
                "Temper_Reported_Fraction": temper_nonblank.mean(),
                "Outside_AD_Fraction": part["AD_Status"].eq("Outside").mean(),
                "Critical_Composition_Failure": bool(critical_bad),
                "Audit_Flags": "|".join(flags) if flags else "none_detected",
                "Objective_Error_Confirmed": False,
                "Automatic_Exclusion_Justified": False,
                "Recommended_Action": recommendation,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("Source_MAE_MPa", ascending=False)
    return rows, summary


def retrain_without_audit_sources(data, outlier_sources, model_module):
    kept = data.loc[~data["Source_Group"].astype(str).isin(outlier_sources)].copy()
    root = config.OUTPUT_DIRS["single"]
    rf_params = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb_params = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    rows = []
    fold_rows = []
    for fold in sorted(kept["Outer_Fold"].unique()):
        fold = int(fold)
        train = kept.loc[kept["Outer_Fold"].ne(fold)].copy()
        test = kept.loc[kept["Outer_Fold"].eq(fold)].copy()
        rf_row = rf_params.loc[
            rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(fold)
        ].iloc[0]
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(fold)
        ].iloc[0]
        rf = model_module.rf_tuned(rf_row, SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, SEED + fold)
        rf.fit(train[FEATURES], train[TARGET])
        xgb.fit(train[FEATURES], train[TARGET])
        y_pred = (rf.predict(test[FEATURES]) + xgb.predict(test[FEATURES])) / 2.0
        part = pd.DataFrame(
            {
                "Outer_Fold": fold,
                "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                "Source_Group": test["Source_Group"].to_numpy(),
                "y_true": test[TARGET].to_numpy(),
                "y_pred": y_pred,
            }
        )
        fold_rows.append({"Outer_Fold": fold, **metrics(part)})
        rows.append(part)
    oof = pd.concat(rows, ignore_index=True)
    return oof, pd.DataFrame(fold_rows), metrics(oof)


def matched_random_removal(pred, source_metrics, outlier_sources):
    source_frame = source_metrics[["Source_Group", "Outer_Fold", "Rows"]].copy()
    source_frame["Source_Group"] = source_frame["Source_Group"].astype(str)
    source_frame["Outer_Fold"] = source_frame["Outer_Fold"].astype(int)
    source_frame["Rows"] = source_frame["Rows"].astype(int)
    source_frame["Size_Bin"] = source_frame["Rows"].map(size_bin)
    source_frame["Audit_Outlier"] = source_frame["Source_Group"].isin(outlier_sources)
    required = (
        source_frame.loc[source_frame["Audit_Outlier"]]
        .groupby(["Outer_Fold", "Size_Bin"])
        .size()
        .to_dict()
    )
    candidates = source_frame.loc[~source_frame["Audit_Outlier"]].copy()
    rng = np.random.default_rng(SEED)
    samples = []
    for iteration in range(N_RANDOM_REMOVALS):
        removed = []
        for (fold, group_bin), count in required.items():
            pool = candidates.loc[
                candidates["Outer_Fold"].eq(fold)
                & candidates["Size_Bin"].eq(group_bin),
                "Source_Group",
            ].to_numpy(dtype=object)
            if len(pool) < count:
                raise AssertionError(f"Insufficient matched pool: fold={fold}, bin={group_bin}")
            removed.extend(rng.choice(pool, size=count, replace=False).tolist())
        kept = pred.loc[~pred["Source_Group"].astype(str).isin(removed)]
        values = metrics(kept)
        samples.append(
            {
                "Iteration": iteration,
                "Removed_Sources": len(removed),
                "Removed_Rows": len(pred) - len(kept),
                **values,
            }
        )
    return pd.DataFrame(samples), required


def comparison_table(pred, outlier_sources, retrained_metrics, random_samples):
    original = metrics(pred)
    filtered = metrics(pred.loc[~pred["Source_Group"].astype(str).isin(outlier_sources)])
    rows = [
        {"Configuration": "Original_refined5_OOF", **original},
        {
            "Configuration": "Posthoc_filter_14_high_error_sources_no_retrain",
            **filtered,
        },
        {
            "Configuration": "Fixed_model_retrain_after_excluding_14_sources",
            **retrained_metrics,
        },
    ]
    comparison = pd.DataFrame(rows)
    random_summary_rows = []
    for metric in ("R2", "RMSE", "MAE", "Removed_Rows"):
        q = random_samples[metric].quantile([0.025, 0.5, 0.975])
        observed = filtered[metric] if metric in filtered else len(pred) - filtered["Rows"]
        if metric == "Removed_Rows":
            observed = len(pred) - filtered["Rows"]
        random_summary_rows.append(
            {
                "Metric": metric,
                "Observed_After_HighError_Removal": observed,
                "Matched_Random_Median": q.loc[0.5],
                "Matched_Random_CI95_Lower": q.loc[0.025],
                "Matched_Random_CI95_Upper": q.loc[0.975],
                "Observed_Percentile_Among_Random": (
                    random_samples[metric].le(observed).mean()
                ),
            }
        )
    return comparison, pd.DataFrame(random_summary_rows)


def doi_collision_audit(data):
    valid = data.loc[data["DOI"].notna() & data["DOI"].astype(str).str.strip().ne("")].copy()
    rows = []
    for doi, part in valid.groupby("DOI", sort=False):
        source_count = part["Source_Group"].nunique()
        fold_count = part["Outer_Fold"].nunique()
        if source_count <= 1:
            continue
        rows.append(
            {
                "DOI": doi,
                "Rows": len(part),
                "Source_Groups": source_count,
                "Outer_Folds": fold_count,
                "Datasets": "|".join(sorted(part["Dataset"].dropna().astype(str).unique())),
                "Source_Group_List": "|".join(sorted(part["Source_Group"].astype(str).unique())),
                "Fold_List": "|".join(sorted(part["Outer_Fold"].astype(str).unique())),
                "Requires_Hierarchical_Validation": fold_count > 1,
            }
        )
    return pd.DataFrame(rows).sort_values("Rows", ascending=False)


def save_figure(out, comparison, random_samples):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ("R2", "RMSE", "MAE")):
        ax.hist(random_samples[metric], bins=35, color="#4C72B0", alpha=0.8)
        observed = comparison.loc[
            comparison["Configuration"].eq(
                "Posthoc_filter_14_high_error_sources_no_retrain"
            ),
            metric,
        ].iloc[0]
        original = comparison.loc[
            comparison["Configuration"].eq("Original_refined5_OOF"), metric
        ].iloc[0]
        ax.axvline(observed, color="#C44E52", linewidth=2, label="remove high-error")
        ax.axvline(original, color="black", linestyle="--", linewidth=1.5, label="original")
        ax.set_xlabel(metric)
        ax.set_ylabel("Matched random removals")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Post-hoc high-error removal versus matched random source removal")
    fig.tight_layout()
    fig.savefig(out / "high_error_removal_selection_bias.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    data, pred, source_metrics, ad, outlier_sources = load_inputs()
    model_module = load_module(
        "model_builders",
        config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py",
    )
    audit_rows, audit_summary = build_audit_tables(
        data, pred, source_metrics, ad, outlier_sources
    )
    retrained_oof, retrained_folds, retrained_metrics = retrain_without_audit_sources(
        data, outlier_sources, model_module
    )
    random_samples, matching_rule = matched_random_removal(
        pred, source_metrics, outlier_sources
    )
    comparison, random_summary = comparison_table(
        pred, outlier_sources, retrained_metrics, random_samples
    )
    doi_collisions = doi_collision_audit(data)

    out = config.PROJECT_ROOT / "results" / "uts_outlier_source_audit"
    out.mkdir(parents=True, exist_ok=True)
    audit_rows.to_csv(out / "audit_14_sources_rows.csv", index=False, encoding="utf-8-sig")
    audit_summary.to_csv(out / "audit_14_sources_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out / "exclusion_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    retrained_oof.to_csv(out / "retrained_excluding14_oof_predictions.csv", index=False, encoding="utf-8-sig")
    retrained_folds.to_csv(out / "retrained_excluding14_fold_metrics.csv", index=False, encoding="utf-8-sig")
    random_samples.to_csv(out / "matched_random_source_removal_samples.csv", index=False, encoding="utf-8-sig")
    random_summary.to_csv(out / "matched_random_source_removal_summary.csv", index=False, encoding="utf-8-sig")
    doi_collisions.to_csv(out / "doi_source_group_collision_audit.csv", index=False, encoding="utf-8-sig")
    save_figure(out, comparison, random_samples)

    cfg = {
        "task": TASK,
        "features": FEATURES,
        "audit_source_count": len(outlier_sources),
        "audit_sources": outlier_sources,
        "automatic_exclusion": False,
        "retrain_hyperparameters": "frozen from original outer folds; no retuning",
        "matched_random_iterations": N_RANDOM_REMOVALS,
        "matched_random_rule": {
            f"fold_{fold}_{bin_name}": count
            for (fold, bin_name), count in matching_rule.items()
        },
        "interpretation": "post-hoc removal selected by OOF error is diagnostic only and cannot be the primary model",
        "seed": SEED,
    }
    (out / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("AUDIT SUMMARY")
    print(
        audit_summary[
            [
                "Source_Group",
                "Rows",
                "Source_MAE_MPa",
                "Audit_Flags",
                "Automatic_Exclusion_Justified",
                "Recommended_Action",
            ]
        ].to_string(index=False)
    )
    print("\nEXCLUSION SENSITIVITY")
    print(comparison.to_string(index=False))
    print("\nMATCHED RANDOM REMOVAL")
    print(random_summary.to_string(index=False))
    print("\nDOI COLLISIONS")
    print(doi_collisions[["DOI", "Rows", "Source_Groups", "Outer_Folds", "Datasets"]].to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
