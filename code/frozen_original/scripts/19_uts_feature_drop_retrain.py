from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = list(config.PRIMARY_FIXED_FEATURES)
SEED = 20260728
N_BOOTSTRAP = 1500


def load_model_module():
    path = config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    spec = importlib.util.spec_from_file_location("final_model_builders", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def source_macro(part: pd.DataFrame):
    values = []
    for _, group in part.groupby("Source_Group"):
        error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        values.append((np.abs(error).mean(), np.sqrt(np.square(error).mean())))
    array = np.asarray(values)
    return {
        "Source_Macro_MAE": float(array[:, 0].mean()),
        "Source_Macro_RMSE": float(array[:, 1].mean()),
    }


def run_drop_retrain(model_module):
    data = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv")
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    root = config.OUTPUT_DIRS["single"]
    rf_params = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb_params = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    predictions = []
    fold_rows = []

    configurations = [("Full", "None", FEATURES)] + [
        (f"Drop_{feature}", feature, [item for item in FEATURES if item != feature])
        for feature in FEATURES
    ]
    for fold in sorted(data["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = data.loc[data["Outer_Fold"].eq(fold)].copy()
        rf_row = rf_params.loc[rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(fold)].iloc[0]
        xgb_row = xgb_params.loc[xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(fold)].iloc[0]
        for configuration, dropped, features in configurations:
            rf = model_module.rf_tuned(rf_row, config.RANDOM_SEED + fold)
            xgb = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
            rf.fit(train[features], train[TARGET])
            xgb.fit(train[features], train[TARGET])
            pred_rf = rf.predict(test[features])
            pred_xgb = xgb.predict(test[features])
            prediction = (pred_rf + pred_xgb) / 2.0
            fold_score = metrics(test[TARGET], prediction)
            fold_rows.append({
                "Outer_Fold": fold,
                "Configuration": configuration,
                "Dropped_Feature": dropped,
                "N_Features": len(features),
                "Test_Rows": len(test),
                "Test_Sources": test["Source_Group"].nunique(),
                **fold_score,
            })
            predictions.append(pd.DataFrame({
                "Outer_Fold": fold,
                "Configuration": configuration,
                "Dropped_Feature": dropped,
                "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                "Source_Group": test["Source_Group"].to_numpy(),
                "y_true": test[TARGET].to_numpy(),
                "y_pred": prediction,
            }))
        print(f"fold={fold}: completed full + {len(FEATURES)} drop-feature models")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def summarize(predictions: pd.DataFrame, folds: pd.DataFrame):
    rows = []
    for (configuration, dropped), part in predictions.groupby(["Configuration", "Dropped_Feature"], sort=False):
        fold_part = folds.loc[folds["Configuration"].eq(configuration)]
        rows.append({
            "Configuration": configuration,
            "Dropped_Feature": dropped,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"], part["y_pred"]),
            **source_macro(part),
            "Worst_Fold_R2": fold_part["R2"].min(),
            "Fold_R2_SD": fold_part["R2"].std(ddof=1),
        })
    summary = pd.DataFrame(rows)
    base = summary.loc[summary["Configuration"].eq("Full")].iloc[0]
    for metric in ("R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE", "Worst_Fold_R2"):
        summary[f"Delta_{metric}_Drop_minus_Full"] = summary[metric] - base[metric]
    losses = []
    full_folds = folds.loc[folds["Configuration"].eq("Full")].set_index("Outer_Fold")
    for configuration, part in folds.groupby("Configuration"):
        losses.append({
            "Configuration": configuration,
            "Folds_RMSE_Worse_after_Drop": int(sum(
                row["RMSE"] > full_folds.loc[row["Outer_Fold"], "RMSE"] + 1e-12
                for _, row in part.iterrows()
            )),
        })
    return summary.merge(pd.DataFrame(losses), on="Configuration", how="left")


def paired_bootstrap(predictions: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    full = predictions.loc[
        predictions["Configuration"].eq("Full"),
        ["Model_Row_ID", "Source_Group", "y_true", "y_pred"],
    ].rename(columns={"y_pred": "pred_full"})
    samples = []
    for feature in FEATURES:
        dropped = predictions.loc[
            predictions["Configuration"].eq(f"Drop_{feature}"), ["Model_Row_ID", "y_pred"]
        ].rename(columns={"y_pred": "pred_drop"})
        paired = full.merge(dropped, on="Model_Row_ID", validate="one_to_one")
        grouped = {
            source: (
                part["y_true"].to_numpy(),
                part["pred_full"].to_numpy(),
                part["pred_drop"].to_numpy(),
            )
            for source, part in paired.groupby("Source_Group")
        }
        sources = np.asarray(list(grouped), dtype=object)
        macro_full = np.asarray([np.abs(grouped[s][1] - grouped[s][0]).mean() for s in sources])
        macro_drop = np.asarray([np.abs(grouped[s][2] - grouped[s][0]).mean() for s in sources])
        for iteration in range(N_BOOTSTRAP):
            draw = rng.integers(0, len(sources), size=len(sources))
            chosen = sources[draw]
            y = np.concatenate([grouped[s][0] for s in chosen])
            p0 = np.concatenate([grouped[s][1] for s in chosen])
            p1 = np.concatenate([grouped[s][2] for s in chosen])
            m0, m1 = metrics(y, p0), metrics(y, p1)
            samples.append({
                "Dropped_Feature": feature,
                "Iteration": iteration,
                "Delta_R2_Drop_minus_Full": m1["R2"] - m0["R2"],
                "Delta_RMSE_Drop_minus_Full": m1["RMSE"] - m0["RMSE"],
                "Delta_MAE_Drop_minus_Full": m1["MAE"] - m0["MAE"],
                "Delta_Source_Macro_MAE_Drop_minus_Full": (macro_drop[draw] - macro_full[draw]).mean(),
            })
    samples = pd.DataFrame(samples)
    summary_rows = []
    for feature, part in samples.groupby("Dropped_Feature", sort=False):
        for metric in (
            "Delta_R2_Drop_minus_Full", "Delta_RMSE_Drop_minus_Full",
            "Delta_MAE_Drop_minus_Full", "Delta_Source_Macro_MAE_Drop_minus_Full",
        ):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            # A feature is useful when dropping it lowers R2 or raises an error metric.
            probability_useful = (
                part[metric].lt(0).mean() if metric == "Delta_R2_Drop_minus_Full"
                else part[metric].gt(0).mean()
            )
            summary_rows.append({
                "Dropped_Feature": feature,
                "Metric": metric,
                "Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
                "Probability_Feature_Useful": probability_useful,
            })
    return samples, pd.DataFrame(summary_rows)


def compare_with_shap(summary: pd.DataFrame):
    shap_importance = pd.read_csv(
        config.PROJECT_ROOT / "results" / "uts_oof_shap_strict" / "global_importance_by_model.csv"
    )
    shap_importance = shap_importance.loc[shap_importance["Model"].eq("Ensemble"), ["Feature", "Mean_Abs_SHAP", "Rank"]]
    drops = summary.loc[summary["Configuration"].ne("Full")].copy()
    drops["Drop_RMSE_Harm"] = drops["Delta_RMSE_Drop_minus_Full"]
    drops["Drop_R2_Harm"] = -drops["Delta_R2_Drop_minus_Full"]
    result = drops.merge(shap_importance, left_on="Dropped_Feature", right_on="Feature", validate="one_to_one")
    result["Drop_RMSE_Harm_Rank"] = result["Drop_RMSE_Harm"].rank(method="min", ascending=False).astype(int)
    result["Drop_R2_Harm_Rank"] = result["Drop_R2_Harm"].rank(method="min", ascending=False).astype(int)
    agreement = pd.DataFrame([{
        "Comparison": "MeanAbsSHAP_vs_DropRMSEHarm",
        "Spearman": spearmanr(result["Mean_Abs_SHAP"], result["Drop_RMSE_Harm"]).statistic,
    }, {
        "Comparison": "MeanAbsSHAP_vs_DropR2Harm",
        "Spearman": spearmanr(result["Mean_Abs_SHAP"], result["Drop_R2_Harm"]).statistic,
    }])
    return result.sort_values("Rank"), agreement


def save_figure(out: Path, comparison: pd.DataFrame, bootstrap_summary: pd.DataFrame):
    order = comparison.sort_values("Rank")["Dropped_Feature"].tolist()
    boot = bootstrap_summary.loc[
        bootstrap_summary["Metric"].eq("Delta_RMSE_Drop_minus_Full")
    ].set_index("Dropped_Feature").loc[order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(order))
    errors = np.vstack([boot["Median"] - boot["CI95_Lower"], boot["CI95_Upper"] - boot["Median"]])
    ax.errorbar(boot["Median"], y, xerr=errors, fmt="o", capsize=4, color="#C44E52")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE after dropping feature (MPa); positive = feature useful")
    ax.set_title("UTS drop-feature retraining with source-bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "uts_drop_feature_retrain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    out = config.PROJECT_ROOT / "results" / "uts_feature_drop_retrain_strict"
    out.mkdir(parents=True, exist_ok=True)
    model_module = load_model_module()
    predictions, folds = run_drop_retrain(model_module)
    summary = summarize(predictions, folds)
    boot_samples, boot_summary = paired_bootstrap(predictions)
    comparison, agreement = compare_with_shap(summary)

    predictions.to_csv(out / "oof_predictions_full_and_feature_drops.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out / "shap_drop_retrain_comparison.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(out / "shap_drop_rank_agreement.csv", index=False, encoding="utf-8-sig")
    save_figure(out, comparison, boot_summary)

    cfg = {
        "task": TASK,
        "model": "frozen 0.5 RF + 0.5 XGBoost, retrained after dropping one feature",
        "outer_split": "fixed Source_Group folds",
        "augmentation": False,
        "purpose": "cross-check SHAP with incremental predictive contribution; correlated features may substitute",
        "bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDROP-FEATURE OOF SUMMARY")
    print(summary.to_string(index=False))
    print("\nSHAP VS DROP-RETRAIN")
    print(comparison.to_string(index=False))
    print("\nAGREEMENT")
    print(agreement.to_string(index=False))
    print("\nBOOTSTRAP ΔRMSE")
    print(boot_summary.loc[boot_summary["Metric"].eq("Delta_RMSE_Drop_minus_Full")].to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
