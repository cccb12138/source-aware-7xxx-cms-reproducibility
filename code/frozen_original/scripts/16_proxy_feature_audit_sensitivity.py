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

TASKS = ("YS", "UTS", "EL")
SEED = 20260725
N_BOOTSTRAP = 1500
MG_DENOMINATOR_MIN = 0.05
CU_DENOMINATOR_MIN = 0.05

PROXY_SETS = {
    "core_direct": [],
    "core_plus_znmg": ["Proxy_Zn_Mg_Ratio"],
    "core_plus_cumg": ["Proxy_Cu_Mg_Ratio"],
    "core_plus_literature_ratios": [
        "Proxy_Zn_Mg_Ratio", "Proxy_Cu_Mg_Ratio", "Proxy_Zn_Cu_Ratio_Safe",
    ],
    "core_plus_sums_share": [
        "Proxy_Zn_Mg_Sum", "Proxy_Zn_Mg_Cu_Sum", "Proxy_Zn_Share_ZnMgCu",
    ],
    "core_plus_total_solute": ["Proxy_Reported_Solute_Sum"],
    "core_plus_all_safe": [
        "Proxy_Zn_Mg_Ratio", "Proxy_Cu_Mg_Ratio", "Proxy_Zn_Cu_Ratio_Safe",
        "Proxy_Zn_Mg_Sum", "Proxy_Zn_Mg_Cu_Sum", "Proxy_Zn_Share_ZnMgCu",
        "Proxy_Reported_Solute_Sum",
    ],
    "core_plus_legacy_raw_ratios": [
        "Legacy_Zn_Mg_Ratio", "Legacy_Zn_Cu_Ratio", "Legacy_Mg_Cu_Ratio",
    ],
}


def load_model_module():
    path = config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    spec = importlib.util.spec_from_file_location("selected_model_builders", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def source_macro(part: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, group in part.groupby("Source_Group"):
        err = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        rows.append((np.abs(err).mean(), np.sqrt(np.square(err).mean())))
    values = np.asarray(rows)
    return {
        "Source_Macro_MAE": float(values[:, 0].mean()),
        "Source_Macro_RMSE": float(values[:, 1].mean()),
    }


def add_recomputed_proxies(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    composition = [feature for feature in config.COMPOSITION_FEATURES if feature in result]
    result[composition] = result[composition].apply(pd.to_numeric, errors="coerce")
    zn, mg, cu = result["Zn"], result["Mg"], result["Cu"]

    result["Proxy_Zn_Mg_Ratio"] = zn / mg.where(mg.ge(MG_DENOMINATOR_MIN))
    result["Proxy_Cu_Mg_Ratio"] = cu / mg.where(mg.ge(MG_DENOMINATOR_MIN))
    result["Proxy_Zn_Cu_Ratio_Safe"] = zn / cu.where(cu.ge(CU_DENOMINATOR_MIN))
    result["Proxy_Mg_Cu_Ratio_Safe"] = mg / cu.where(cu.ge(CU_DENOMINATOR_MIN))
    result["Proxy_Zn_Mg_Sum"] = zn + mg
    result["Proxy_Zn_Mg_Cu_Sum"] = zn + mg + cu
    result["Proxy_Zn_Share_ZnMgCu"] = zn / (zn + mg + cu).where((zn + mg + cu).gt(0))
    non_al = [feature for feature in composition if feature != "Al"]
    result["Proxy_Reported_Solute_Sum"] = result[non_al].fillna(0).sum(axis=1)

    result["Legacy_Zn_Mg_Ratio"] = pd.to_numeric(result["Zn_Mg_Ratio"], errors="coerce")
    result["Legacy_Zn_Cu_Ratio"] = pd.to_numeric(result["Zn_Cu_Ratio"], errors="coerce")
    result["Legacy_Mg_Cu_Ratio"] = pd.to_numeric(result["Mg_Cu_Ratio"], errors="coerce")
    return result


def audit_proxies(all_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    correlation_rows = []
    comparisons = {
        "Zn_Mg_Ratio": "Proxy_Zn_Mg_Ratio",
        "Zn_Mg_Sum": "Proxy_Zn_Mg_Sum",
        "Zn_Mg_Cu_Sum": "Proxy_Zn_Mg_Cu_Sum",
        "Zn_Share_ZnMgCu": "Proxy_Zn_Share_ZnMgCu",
        "Reported_Solute_Sum": "Proxy_Reported_Solute_Sum",
    }
    proxy_columns = sorted({feature for values in PROXY_SETS.values() for feature in values})
    proxy_columns += ["Proxy_Mg_Cu_Ratio_Safe"]

    for task, data in all_data.items():
        target = config.TARGET_COLUMNS[task]
        for proxy in proxy_columns:
            values = pd.to_numeric(data[proxy], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row = {
                "Task": task,
                "Proxy": proxy,
                "Rows": len(data),
                "Observed": values.notna().sum(),
                "Missing": values.isna().sum(),
                "Missing_Pct": values.isna().mean() * 100,
                "Unique": values.nunique(dropna=True),
                "Min": values.min(),
                "Median": values.median(),
                "Max": values.max(),
                "Nonfinite_Before_Cleanup": int(
                    np.isinf(pd.to_numeric(data[proxy], errors="coerce").to_numpy()).sum()
                ),
                "Mg_Below_Threshold_Rows": int(data["Mg"].lt(MG_DENOMINATOR_MIN).sum()),
                "Cu_Below_Threshold_Rows": int(data["Cu"].lt(CU_DENOMINATOR_MIN).sum()),
            }
            original = next((old for old, new in comparisons.items() if new == proxy), None)
            if original is not None:
                expected = values
                actual = pd.to_numeric(data[original], errors="coerce").replace([np.inf, -np.inf], np.nan)
                comparable = expected.notna() & actual.notna()
                diff = (expected[comparable] - actual[comparable]).abs()
                row["Existing_Column"] = original
                row["Comparable_to_Existing"] = comparable.sum()
                row["Existing_Mismatch_gt_1e6"] = diff.gt(1e-6).sum()
                row["Existing_Max_Abs_Error"] = diff.max() if len(diff) else np.nan
            audit_rows.append(row)
            correlation_rows.append({
                "Task": task,
                "Proxy": proxy,
                "Target_Spearman": values.corr(pd.to_numeric(data[target], errors="coerce"), method="spearman"),
                "Zn_Spearman": values.corr(data["Zn"], method="spearman"),
                "Mg_Spearman": values.corr(data["Mg"], method="spearman"),
                "Cu_Spearman": values.corr(data["Cu"], method="spearman"),
            })
    return pd.DataFrame(audit_rows), pd.DataFrame(correlation_rows)


def load_parameters() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = config.OUTPUT_DIRS["single"]
    rf = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    selected = pd.read_csv(root / "baseline_strict" / "selected_features_by_fold.csv")
    return rf, xgb, selected


def run_sensitivity(model_module, all_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf_params, xgb_params, selected = load_parameters()
    predictions = []
    fold_rows = []

    for task, data in all_data.items():
        target = config.TARGET_COLUMNS[task]
        for fold in sorted(data["Outer_Fold"].unique()):
            fold = int(fold)
            train = data.loc[data["Outer_Fold"].ne(fold)].copy()
            test = data.loc[data["Outer_Fold"].eq(fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError("Source leakage")
            if task == "UTS":
                core_features = list(config.PRIMARY_FIXED_FEATURES)
                rf_row = rf_params.loc[rf_params["Task"].eq(task) & rf_params["Outer_Fold"].eq(fold)].iloc[0]
                xgb_row = xgb_params.loc[xgb_params["Task"].eq(task) & xgb_params["Outer_Fold"].eq(fold)].iloc[0]
            else:
                row = selected.loc[
                    selected["Task"].eq(task)
                    & selected["Model"].eq("RandomForest")
                    & selected["Feature_Set"].eq("composition_core")
                    & selected["Outer_Fold"].eq(fold)
                ].iloc[0]
                core_features = row["Selected_Features"].split("|")

            for feature_set, extras in PROXY_SETS.items():
                features = core_features + extras
                x_train = train[features].replace([np.inf, -np.inf], np.nan)
                x_test = test[features].replace([np.inf, -np.inf], np.nan)
                y_train = train[target].to_numpy()
                if task == "UTS":
                    rf_model = model_module.rf_tuned(rf_row, config.RANDOM_SEED + fold)
                    xgb_model = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
                    rf_model.fit(x_train, y_train)
                    xgb_model.fit(x_train, y_train)
                    pred_rf = rf_model.predict(x_test)
                    pred_xgb = xgb_model.predict(x_test)
                    y_pred = (pred_rf + pred_xgb) / 2.0
                    model_name = "RandomForest_XGBoost_OOF_Mean"
                else:
                    model = model_module.rf_baseline(config.RANDOM_SEED + fold)
                    model.fit(x_train, y_train)
                    pred_rf = model.predict(x_test)
                    pred_xgb = np.full(len(test), np.nan)
                    y_pred = pred_rf
                    model_name = "RandomForest"

                fold_score = score(test[target].to_numpy(), y_pred)
                fold_rows.append({
                    "Task": task,
                    "Feature_Set": feature_set,
                    "Outer_Fold": fold,
                    "Model": model_name,
                    "Core_Features": "|".join(core_features),
                    "Added_Proxies": "|".join(extras),
                    "N_Features": len(features),
                    "Test_Rows": len(test),
                    "Test_Sources": test["Source_Group"].nunique(),
                    **fold_score,
                })
                predictions.append(pd.DataFrame({
                    "Task": task,
                    "Feature_Set": feature_set,
                    "Outer_Fold": fold,
                    "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                    "Source_Group": test["Source_Group"].to_numpy(),
                    "y_true": test[target].to_numpy(),
                    "pred_rf": pred_rf,
                    "pred_xgb": pred_xgb,
                    "y_pred": y_pred,
                }))
                print(f"{task} fold={fold} {feature_set}: R2={fold_score['R2']:.3f}")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def summarize(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, feature_set), part in predictions.groupby(["Task", "Feature_Set"], sort=False):
        fold_part = folds.loc[folds["Task"].eq(task) & folds["Feature_Set"].eq(feature_set)]
        rows.append({
            "Task": task,
            "Feature_Set": feature_set,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **score(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            **source_macro(part),
            "Fold_R2_Mean": fold_part["R2"].mean(),
            "Fold_R2_SD": fold_part["R2"].std(ddof=1),
            "Worst_Fold_R2": fold_part["R2"].min(),
        })
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["Feature_Set"].eq("core_direct")].set_index("Task")
    for metric in ("R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE", "Worst_Fold_R2"):
        summary[f"Delta_{metric}_vs_Core"] = summary.apply(
            lambda row: row[metric] - baseline.loc[row["Task"], metric], axis=1
        )
    wins = []
    for (task, feature_set), part in folds.groupby(["Task", "Feature_Set"]):
        base = folds.loc[
            folds["Task"].eq(task) & folds["Feature_Set"].eq("core_direct")
        ].set_index("Outer_Fold")
        wins.append({
            "Task": task,
            "Feature_Set": feature_set,
            "Fold_RMSE_Wins_vs_Core": int(sum(
                row["RMSE"] < base.loc[row["Outer_Fold"], "RMSE"] - 1e-12
                for _, row in part.iterrows()
            )),
        })
    return summary.merge(pd.DataFrame(wins), on=["Task", "Feature_Set"], how="left")


def paired_source_bootstrap(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 50000)
    samples = []
    for task in TASKS:
        base = predictions.loc[
            predictions["Task"].eq(task) & predictions["Feature_Set"].eq("core_direct"),
            ["Model_Row_ID", "Source_Group", "y_true", "y_pred"],
        ].rename(columns={"y_pred": "pred_base"})
        for feature_set in PROXY_SETS:
            if feature_set == "core_direct":
                continue
            candidate = predictions.loc[
                predictions["Task"].eq(task) & predictions["Feature_Set"].eq(feature_set),
                ["Model_Row_ID", "y_pred"],
            ].rename(columns={"y_pred": "pred_candidate"})
            paired = base.merge(candidate, on="Model_Row_ID", validate="one_to_one")
            grouped = {
                source: (
                    part["y_true"].to_numpy(),
                    part["pred_base"].to_numpy(),
                    part["pred_candidate"].to_numpy(),
                )
                for source, part in paired.groupby("Source_Group")
            }
            sources = np.asarray(list(grouped), dtype=object)
            macro_base = np.asarray([np.abs(grouped[s][1] - grouped[s][0]).mean() for s in sources])
            macro_candidate = np.asarray([np.abs(grouped[s][2] - grouped[s][0]).mean() for s in sources])
            for iteration in range(N_BOOTSTRAP):
                draw = rng.integers(0, len(sources), size=len(sources))
                chosen = sources[draw]
                y = np.concatenate([grouped[s][0] for s in chosen])
                p0 = np.concatenate([grouped[s][1] for s in chosen])
                p1 = np.concatenate([grouped[s][2] for s in chosen])
                m0, m1 = score(y, p0), score(y, p1)
                samples.append({
                    "Task": task,
                    "Feature_Set": feature_set,
                    "Iteration": iteration,
                    "Delta_R2": m1["R2"] - m0["R2"],
                    "Delta_RMSE": m1["RMSE"] - m0["RMSE"],
                    "Delta_MAE": m1["MAE"] - m0["MAE"],
                    "Delta_Source_Macro_MAE": (macro_candidate[draw] - macro_base[draw]).mean(),
                })
    samples = pd.DataFrame(samples)
    summary_rows = []
    for (task, feature_set), part in samples.groupby(["Task", "Feature_Set"], sort=False):
        for metric in ("Delta_R2", "Delta_RMSE", "Delta_MAE", "Delta_Source_Macro_MAE"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary_rows.append({
                "Task": task,
                "Feature_Set": feature_set,
                "Metric": metric,
                "Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
                "Probability_Improvement": (
                    part[metric].gt(0).mean() if metric == "Delta_R2"
                    else part[metric].lt(0).mean()
                ),
            })
    return samples, pd.DataFrame(summary_rows)


def validate_core_reproduction(predictions: pd.DataFrame) -> pd.DataFrame:
    root = config.OUTPUT_DIRS["single"]
    base = pd.read_csv(root / "baseline_strict" / "oof_predictions.csv")
    base = base.loc[
        base["Model"].eq("RandomForest")
        & base["Feature_Set"].eq("composition_core")
        & base["Task"].isin(["YS", "EL"]),
        ["Task", "Model_Row_ID", "y_pred"],
    ].rename(columns={"y_pred": "saved_pred"})
    uts = pd.read_csv(root / "rf_xgb_ensemble_strict" / "oof_predictions.csv")
    uts = uts.loc[uts["Task"].eq("UTS"), ["Task", "Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "saved_pred"})
    saved = pd.concat([base, uts], ignore_index=True)
    rerun = predictions.loc[
        predictions["Feature_Set"].eq("core_direct"), ["Task", "Model_Row_ID", "y_pred"]
    ]
    check = saved.merge(rerun, on=["Task", "Model_Row_ID"], validate="one_to_one")
    check["abs_diff"] = (check["saved_pred"] - check["y_pred"]).abs()
    result = check.groupby("Task").agg(Rows=("Model_Row_ID", "size"), Max_Abs_Diff=("abs_diff", "max")).reset_index()
    if result["Max_Abs_Diff"].max() > 1e-4:
        raise AssertionError(f"Core model reproduction failed:\n{result}")
    return result


def save_figure(out: Path, summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    order = [name for name in PROXY_SETS if name != "core_direct"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for column, task in enumerate(TASKS):
        part = summary.loc[summary["Task"].eq(task) & summary["Feature_Set"].isin(order)].set_index("Feature_Set").loc[order]
        axes[0, column].bar(range(len(part)), part["Delta_R2_vs_Core"], color="#4C72B0")
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].set_title(f"{task}: proxy ΔR² vs direct composition")
        axes[0, column].set_xticks(range(len(part)))
        axes[0, column].set_xticklabels([name.replace("core_plus_", "").replace("_", "\n") for name in order], fontsize=7)

        boot = bootstrap.loc[
            bootstrap["Task"].eq(task)
            & bootstrap["Feature_Set"].isin(order)
            & bootstrap["Metric"].eq("Delta_RMSE")
        ].set_index("Feature_Set").loc[order]
        error = np.vstack([boot["Median"] - boot["CI95_Lower"], boot["CI95_Upper"] - boot["Median"]])
        axes[1, column].errorbar(range(len(boot)), boot["Median"], yerr=error, fmt="o", capsize=3, color="#C44E52")
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set_title(f"{task}: paired source-bootstrap ΔRMSE")
        axes[1, column].set_xticks(range(len(boot)))
        axes[1, column].set_xticklabels([name.replace("core_plus_", "").replace("_", "\n") for name in order], fontsize=7)
        axes[1, column].set_ylabel("Negative is improvement")
    fig.tight_layout()
    fig.savefig(out / "proxy_feature_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "proxy_feature_sensitivity_strict"
    out.mkdir(parents=True, exist_ok=True)
    model_module = load_model_module()
    all_data = {
        task: add_recomputed_proxies(pd.read_csv(
            config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv"
        ))
        for task in TASKS
    }
    audit, correlations = audit_proxies(all_data)
    predictions, folds = run_sensitivity(model_module, all_data)
    reproduction = validate_core_reproduction(predictions)
    summary = summarize(predictions, folds)
    boot_samples, boot_summary = paired_source_bootstrap(predictions)

    audit.to_csv(out / "proxy_definition_numeric_audit.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(out / "proxy_correlations.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "oof_predictions_all_proxy_sets.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    reproduction.to_csv(out / "core_reproduction_check.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(out, summary, boot_summary)

    cfg = {
        "analysis": "proxy feature add/remove sensitivity",
        "outer_split": "fixed Source_Group folds",
        "models": "frozen selected models; no augmentation; no proxy retuning",
        "proxy_sets": PROXY_SETS,
        "mg_denominator_min_wt_pct": MG_DENOMINATOR_MIN,
        "cu_denominator_min_wt_pct": CU_DENOMINATOR_MIN,
        "nonfinite_handling": "replace with missing; training-fold median imputation",
        "bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCORE REPRODUCTION")
    print(reproduction.to_string(index=False))
    print("\nPROXY OOF SUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED SOURCE BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
