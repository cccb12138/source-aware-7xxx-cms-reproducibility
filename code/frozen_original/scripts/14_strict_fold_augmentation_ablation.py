from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import config


warnings.filterwarnings("ignore")

TASKS = ("YS", "UTS", "EL")
SEED = 20260723
N_BOOTSTRAP = 1500
FEATURES = list(config.PRIMARY_FIXED_FEATURES)

# Absolute one-standard-deviation measurement perturbations in wt.% at nominal scale.
# These match the conservative analytical-error assumptions used in the previous pipeline.
BASE_SIGMA = {
    "Zn": 0.05,
    "Mg": 0.05,
    "Cu": 0.05,
    "Si": 0.01,
    "Fe": 0.01,
    "Mn": 0.01,
    "Cr": 0.01,
    "Ti": 0.005,
    "Zr": 0.01,
    "Sc": 0.005,
    "Ni": 0.01,
}

# Broad 7xxx-compatible guards; they prevent impossible synthetic values without
# clipping every fold to its observed extrema.
PHYSICAL_BOUNDS = {
    "Zn": (0.0, 12.0),
    "Mg": (0.0, 6.0),
    "Cu": (0.0, 4.0),
    "Si": (0.0, 2.0),
    "Fe": (0.0, 2.0),
    "Mn": (0.0, 1.5),
    "Cr": (0.0, 1.0),
    "Ti": (0.0, 0.5),
    "Zr": (0.0, 0.5),
    "Sc": (0.0, 0.5),
    "Ni": (0.0, 2.0),
}

STRATEGIES = {
    "no_augmentation": {"copies": 0, "sigma_scale": 0.0},
    "half_sigma_2copies": {"copies": 2, "sigma_scale": 0.5},
    # Pre-specified confirmatory policy. Other strategies are sensitivity checks.
    "nominal_sigma_2copies": {"copies": 2, "sigma_scale": 1.0},
    "nominal_sigma_4copies": {"copies": 4, "sigma_scale": 1.0},
    "double_sigma_2copies": {"copies": 2, "sigma_scale": 2.0},
}
PRIMARY_AUGMENTATION = "nominal_sigma_2copies"


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


def augment_composition(
    x: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    copies: int,
    sigma_scale: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    original = x.apply(pd.to_numeric, errors="coerce").copy()
    if copies == 0:
        return original, np.asarray(y), pd.DataFrame()

    rng = np.random.default_rng(seed)
    generated = []
    for _ in range(copies):
        synthetic = original.copy()
        for feature in features:
            values = original[feature]
            observed_positive = values.notna() & values.gt(0)
            noise = rng.normal(0.0, BASE_SIGMA[feature] * sigma_scale, size=len(original))
            perturbed = values.to_numpy(dtype=float) + noise
            lower, upper = PHYSICAL_BOUNDS[feature]
            perturbed = np.clip(perturbed, lower, upper)
            # Preserve missing values and structural/undetected zeros.
            synthetic.loc[observed_positive, feature] = perturbed[observed_positive.to_numpy()]
            synthetic.loc[values.eq(0), feature] = 0.0
            synthetic.loc[values.isna(), feature] = np.nan

        # Enforce a conservative total of modeled non-Al solutes <= 25 wt.%.
        totals = synthetic[features].fillna(0).sum(axis=1)
        scale_rows = totals.gt(25.0)
        if scale_rows.any():
            factors = 25.0 / totals.loc[scale_rows]
            synthetic.loc[scale_rows, features] = synthetic.loc[scale_rows, features].mul(factors, axis=0)
        generated.append(synthetic)

    augmented_only = pd.concat(generated, ignore_index=True)
    x_augmented = pd.concat([original, augmented_only], ignore_index=True)
    y_augmented = np.tile(np.asarray(y), copies + 1)

    audit_rows = []
    for feature in features:
        orig = original[feature].dropna()
        aug = augmented_only[feature].dropna()
        orig_sd = orig.std(ddof=1)
        shift = aug.mean() - orig.mean()
        audit_rows.append({
            "Feature": feature,
            "Original_N": len(orig),
            "Augmented_Only_N": len(aug),
            "Original_Mean": orig.mean(),
            "Original_SD": orig_sd,
            "Augmented_Only_Mean": aug.mean(),
            "Augmented_Only_SD": aug.std(ddof=1),
            "Mean_Shift": shift,
            "Standardized_Mean_Shift": shift / orig_sd if pd.notna(orig_sd) and orig_sd > 0 else np.nan,
            "Augmented_Min": aug.min(),
            "Augmented_Max": aug.max(),
            "Lower_Bound": PHYSICAL_BOUNDS[feature][0],
            "Upper_Bound": PHYSICAL_BOUNDS[feature][1],
            "Bound_Violations": int(
                (aug.lt(PHYSICAL_BOUNDS[feature][0]) | aug.gt(PHYSICAL_BOUNDS[feature][1])).sum()
            ),
        })
    return x_augmented, y_augmented, pd.DataFrame(audit_rows)


def rf_baseline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=4,
        )),
    ])


def rf_tuned(row: pd.Series, seed: int) -> Pipeline:
    depth = None if pd.isna(row["Param_max_depth"]) else int(row["Param_max_depth"])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=int(row["Param_min_samples_leaf"]),
            min_samples_split=int(row["Param_min_samples_split"]),
            max_features=float(row["Param_max_features"]),
            random_state=seed,
            n_jobs=4,
        )),
    ])


def xgb_tuned(row: pd.Series, seed: int) -> Pipeline:
    params = {
        "n_estimators": int(row["Param_n_estimators"]),
        "max_depth": int(row["Param_max_depth"]),
        "learning_rate": float(row["Param_learning_rate"]),
        "min_child_weight": int(row["Param_min_child_weight"]),
        "subsample": float(row["Param_subsample"]),
        "colsample_bytree": float(row["Param_colsample_bytree"]),
        "reg_alpha": float(row["Param_reg_alpha"]),
        "reg_lambda": float(row["Param_reg_lambda"]),
    }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
            verbosity=0,
            **params,
        )),
    ])


def load_tuned_parameters() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.OUTPUT_DIRS["single"]
    rf = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    return rf, xgb


def run_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rf_params, xgb_params = load_tuned_parameters()
    selected_table = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "baseline_strict" / "selected_features_by_fold.csv"
    )
    prediction_parts = []
    fold_rows = []
    audit_parts = []

    for task_index, task in enumerate(TASKS):
        target = config.TARGET_COLUMNS[task]
        data = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv")
        candidate_features = sorted(set(FEATURES + list(config.MODEL_COMPOSITION_FEATURES)))
        data[candidate_features + [target]] = data[candidate_features + [target]].apply(pd.to_numeric, errors="coerce")

        for fold in sorted(data["Outer_Fold"].unique()):
            fold = int(fold)
            train = data.loc[data["Outer_Fold"].ne(fold)].copy()
            test = data.loc[data["Outer_Fold"].eq(fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError(f"Source leakage: {task}, fold {fold}")
            base_seed = SEED + task_index * 1000 + fold * 10
            if task == "UTS":
                model_features = FEATURES
            else:
                feature_row = selected_table.loc[
                    selected_table["Task"].eq(task)
                    & selected_table["Model"].eq("RandomForest")
                    & selected_table["Feature_Set"].eq("composition_core")
                    & selected_table["Outer_Fold"].eq(fold)
                ].iloc[0]
                model_features = feature_row["Selected_Features"].split("|")

            for strategy, policy in STRATEGIES.items():
                x_train, y_train, audit = augment_composition(
                    train[model_features],
                    train[target].to_numpy(),
                    features=model_features,
                    copies=policy["copies"],
                    sigma_scale=policy["sigma_scale"],
                    seed=base_seed,
                )
                if not audit.empty:
                    audit.insert(0, "Strategy", strategy)
                    audit.insert(0, "Outer_Fold", fold)
                    audit.insert(0, "Task", task)
                    audit_parts.append(audit)

                if task == "UTS":
                    rf_row = rf_params.loc[rf_params["Task"].eq(task) & rf_params["Outer_Fold"].eq(fold)].iloc[0]
                    xgb_row = xgb_params.loc[xgb_params["Task"].eq(task) & xgb_params["Outer_Fold"].eq(fold)].iloc[0]
                    model_rf = rf_tuned(rf_row, config.RANDOM_SEED + fold)
                    model_xgb = xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
                    model_rf.fit(x_train, y_train)
                    model_xgb.fit(x_train, y_train)
                    pred_rf = model_rf.predict(test[model_features])
                    pred_xgb = model_xgb.predict(test[model_features])
                    y_pred = (pred_rf + pred_xgb) / 2.0
                    model_name = "RandomForest_XGBoost_OOF_Mean"
                else:
                    model = rf_baseline(config.RANDOM_SEED + fold)
                    model.fit(x_train, y_train)
                    pred_rf = model.predict(test[model_features])
                    pred_xgb = np.full(len(test), np.nan)
                    y_pred = pred_rf
                    model_name = "RandomForest"

                fold_score = score(test[target].to_numpy(), y_pred)
                fold_rows.append({
                    "Task": task,
                    "Strategy": strategy,
                    "Primary_Augmentation": strategy == PRIMARY_AUGMENTATION,
                    "Selected_Model": model_name,
                    "Selected_Features": "|".join(model_features),
                    "Outer_Fold": fold,
                    "Train_Original_Rows": len(train),
                    "Train_After_Augmentation": len(x_train),
                    "Test_Original_Rows": len(test),
                    "Test_Sources": test["Source_Group"].nunique(),
                    **fold_score,
                })
                prediction_parts.append(pd.DataFrame({
                    "Task": task,
                    "Strategy": strategy,
                    "Primary_Augmentation": strategy == PRIMARY_AUGMENTATION,
                    "Selected_Model": model_name,
                    "Selected_Features": "|".join(model_features),
                    "Outer_Fold": fold,
                    "Model_Row_ID": test["Model_Row_ID"].to_numpy(),
                    "Source_Group": test["Source_Group"].to_numpy(),
                    "y_true": test[target].to_numpy(),
                    "pred_rf": pred_rf,
                    "pred_xgb": pred_xgb,
                    "y_pred": y_pred,
                }))
                print(f"{task} fold={fold} {strategy}: R2={fold_score['R2']:.3f}, RMSE={fold_score['RMSE']:.3f}")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    audit = pd.concat(audit_parts, ignore_index=True)
    return predictions, folds, audit


def summarize(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, strategy), part in predictions.groupby(["Task", "Strategy"], sort=False):
        fold_part = folds.loc[folds["Task"].eq(task) & folds["Strategy"].eq(strategy)]
        row = {
            "Task": task,
            "Strategy": strategy,
            "Primary_Augmentation": strategy == PRIMARY_AUGMENTATION,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **score(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            **source_macro(part),
            "Fold_R2_Mean": fold_part["R2"].mean(),
            "Fold_R2_SD": fold_part["R2"].std(ddof=1),
            "Worst_Fold_R2": fold_part["R2"].min(),
            "Fold_RMSE_Mean": fold_part["RMSE"].mean(),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["Strategy"].eq("no_augmentation")].set_index("Task")
    for metric in ("R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE", "Worst_Fold_R2"):
        summary[f"Delta_{metric}_vs_NoAug"] = summary.apply(
            lambda row: row[metric] - baseline.loc[row["Task"], metric], axis=1
        )

    wins = []
    for (task, strategy), part in folds.groupby(["Task", "Strategy"]):
        base = folds.loc[folds["Task"].eq(task) & folds["Strategy"].eq("no_augmentation")].set_index("Outer_Fold")
        wins.append({
            "Task": task,
            "Strategy": strategy,
            "Fold_RMSE_Wins_vs_NoAug": int(sum(
                row["RMSE"] < base.loc[row["Outer_Fold"], "RMSE"] - 1e-12
                for _, row in part.iterrows()
            )),
        })
    return summary.merge(pd.DataFrame(wins), on=["Task", "Strategy"], how="left")


def paired_source_bootstrap(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 9999)
    sample_rows = []
    for task in TASKS:
        baseline = predictions.loc[
            predictions["Task"].eq(task) & predictions["Strategy"].eq("no_augmentation"),
            ["Model_Row_ID", "Source_Group", "y_true", "y_pred"],
        ].rename(columns={"y_pred": "pred_base"})
        for strategy in STRATEGIES:
            if strategy == "no_augmentation":
                continue
            augmented = predictions.loc[
                predictions["Task"].eq(task) & predictions["Strategy"].eq(strategy),
                ["Model_Row_ID", "y_pred"],
            ].rename(columns={"y_pred": "pred_aug"})
            paired = baseline.merge(augmented, on="Model_Row_ID", validate="one_to_one")
            grouped = {
                source: (
                    group["y_true"].to_numpy(),
                    group["pred_base"].to_numpy(),
                    group["pred_aug"].to_numpy(),
                )
                for source, group in paired.groupby("Source_Group")
            }
            sources = np.asarray(list(grouped), dtype=object)
            macro_base = np.asarray([np.abs(grouped[s][1] - grouped[s][0]).mean() for s in sources])
            macro_aug = np.asarray([np.abs(grouped[s][2] - grouped[s][0]).mean() for s in sources])
            for iteration in range(N_BOOTSTRAP):
                draw = rng.integers(0, len(sources), size=len(sources))
                chosen = sources[draw]
                y = np.concatenate([grouped[s][0] for s in chosen])
                p0 = np.concatenate([grouped[s][1] for s in chosen])
                p1 = np.concatenate([grouped[s][2] for s in chosen])
                m0 = score(y, p0)
                m1 = score(y, p1)
                sample_rows.append({
                    "Task": task,
                    "Strategy": strategy,
                    "Iteration": iteration,
                    "Delta_R2": m1["R2"] - m0["R2"],
                    "Delta_RMSE": m1["RMSE"] - m0["RMSE"],
                    "Delta_MAE": m1["MAE"] - m0["MAE"],
                    "Delta_Source_Macro_MAE": (macro_aug[draw] - macro_base[draw]).mean(),
                })
    samples = pd.DataFrame(sample_rows)
    summary_rows = []
    for (task, strategy), part in samples.groupby(["Task", "Strategy"], sort=False):
        for metric in ("Delta_R2", "Delta_RMSE", "Delta_MAE", "Delta_Source_Macro_MAE"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary_rows.append({
                "Task": task,
                "Strategy": strategy,
                "Metric": metric,
                "Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
                "Probability_Improvement": (
                    float(part[metric].gt(0).mean()) if metric == "Delta_R2"
                    else float(part[metric].lt(0).mean())
                ),
            })
    return samples, pd.DataFrame(summary_rows)


def validate_no_augmentation(predictions: pd.DataFrame) -> pd.DataFrame:
    root = config.OUTPUT_DIRS["single"]
    base = pd.read_csv(root / "baseline_strict" / "oof_predictions.csv")
    base = base.loc[
        base["Model"].eq("RandomForest")
        & base["Feature_Set"].eq("composition_core")
        & base["Task"].isin(["YS", "EL"]),
        ["Task", "Model_Row_ID", "y_pred"],
    ].rename(columns={"y_pred": "saved_pred"})
    ensemble = pd.read_csv(root / "rf_xgb_ensemble_strict" / "oof_predictions.csv")
    ensemble = ensemble.loc[ensemble["Task"].eq("UTS"), ["Task", "Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "saved_pred"})
    saved = pd.concat([base, ensemble], ignore_index=True)
    rerun = predictions.loc[predictions["Strategy"].eq("no_augmentation"), ["Task", "Model_Row_ID", "y_pred"]]
    check = saved.merge(rerun, on=["Task", "Model_Row_ID"], validate="one_to_one")
    check["Absolute_Prediction_Difference"] = (check["saved_pred"] - check["y_pred"]).abs()
    result = check.groupby("Task").agg(
        Rows=("Model_Row_ID", "size"),
        Max_Absolute_Prediction_Difference=("Absolute_Prediction_Difference", "max"),
        Mean_Absolute_Prediction_Difference=("Absolute_Prediction_Difference", "mean"),
    ).reset_index()
    if result["Max_Absolute_Prediction_Difference"].max() > 1e-4:
        raise AssertionError(f"No-augmentation rerun failed to reproduce selected predictions:\n{result}")
    return result


def save_figure(out: Path, summary: pd.DataFrame, bootstrap_summary: pd.DataFrame) -> None:
    strategies = [s for s in STRATEGIES if s != "no_augmentation"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for col, task in enumerate(TASKS):
        part = summary.loc[summary["Task"].eq(task) & summary["Strategy"].isin(strategies)].set_index("Strategy").loc[strategies]
        axes[0, col].bar(range(len(part)), part["Delta_R2_vs_NoAug"], color="#4C72B0")
        axes[0, col].axhline(0, color="black", linewidth=0.8)
        axes[0, col].set_title(f"{task}: ΔR² vs no augmentation")
        axes[0, col].set_xticks(range(len(part)))
        axes[0, col].set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=7)

        b = bootstrap_summary.loc[
            bootstrap_summary["Task"].eq(task)
            & bootstrap_summary["Strategy"].isin(strategies)
            & bootstrap_summary["Metric"].eq("Delta_RMSE")
        ].set_index("Strategy").loc[strategies]
        yerr = np.vstack([b["Median"] - b["CI95_Lower"], b["CI95_Upper"] - b["Median"]])
        axes[1, col].errorbar(range(len(b)), b["Median"], yerr=yerr, fmt="o", capsize=4, color="#C44E52")
        axes[1, col].axhline(0, color="black", linewidth=0.8)
        axes[1, col].set_title(f"{task}: paired source-bootstrap ΔRMSE")
        axes[1, col].set_xticks(range(len(b)))
        axes[1, col].set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=7)
        axes[1, col].set_ylabel("Negative is improvement")
    fig.tight_layout()
    fig.savefig(out / "augmentation_ablation_deltas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "augmentation_ablation_strict"
    out.mkdir(parents=True, exist_ok=True)

    predictions, folds, audit = run_models()
    reproduction = validate_no_augmentation(predictions)
    summary = summarize(predictions, folds)
    boot_samples, boot_summary = paired_source_bootstrap(predictions)

    predictions.to_csv(out / "oof_predictions_all_strategies.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "augmentation_distribution_audit.csv", index=False, encoding="utf-8-sig")
    reproduction.to_csv(out / "no_augmentation_reproduction_check.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(out, summary, boot_summary)

    run_config = {
        "outer_split": "fixed Source_Group folds",
        "test_rows": "original only; never augmented",
        "features": FEATURES,
        "targets": list(TASKS),
        "strategies": STRATEGIES,
        "pre_specified_primary_augmentation": PRIMARY_AUGMENTATION,
        "base_sigma_wt_pct": BASE_SIGMA,
        "physical_bounds_wt_pct": PHYSICAL_BOUNDS,
        "preserve_missing": True,
        "preserve_structural_zero": True,
        "modeled_solute_sum_upper_bound_wt_pct": 25.0,
        "target_perturbation": False,
        "paired_source_bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nNO-AUGMENTATION REPRODUCTION CHECK")
    print(reproduction.to_string(index=False))
    print("\nOOF AUGMENTATION SUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED SOURCE BOOTSTRAP SUMMARY")
    print(boot_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
