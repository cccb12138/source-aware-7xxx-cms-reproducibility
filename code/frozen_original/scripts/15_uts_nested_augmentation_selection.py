from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

import config


warnings.filterwarnings("ignore")

SEED = 20260724
N_INNER_SPLITS = 3
N_AUGMENTATION_SEEDS = 2
N_BOOTSTRAP = 3000
TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = list(config.PRIMARY_FIXED_FEATURES)
STRATEGY_ORDER = [
    "no_augmentation",
    "half_sigma_2copies",
    "nominal_sigma_2copies",
    "double_sigma_2copies",
]


def load_augmentation_module():
    path = config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    spec = importlib.util.spec_from_file_location("strict_augmentation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def source_macro(y_true, y_pred, groups) -> dict[str, float]:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "Source_Group": groups})
    rows = []
    for _, part in frame.groupby("Source_Group"):
        err = part["y_pred"].to_numpy() - part["y_true"].to_numpy()
        rows.append((np.abs(err).mean(), np.sqrt(np.square(err).mean())))
    values = np.asarray(rows)
    return {
        "Source_Macro_MAE": float(values[:, 0].mean()),
        "Source_Macro_RMSE": float(values[:, 1].mean()),
    }


def get_fold_parameter_rows(outer_fold: int) -> tuple[pd.Series, pd.Series]:
    root = config.OUTPUT_DIRS["single"]
    rf = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    rf_row = rf.loc[rf["Task"].eq(TASK) & rf["Outer_Fold"].eq(outer_fold)].iloc[0]
    xgb_row = xgb.loc[xgb["Task"].eq(TASK) & xgb["Outer_Fold"].eq(outer_fold)].iloc[0]
    return rf_row, xgb_row


def fit_predict_ensemble(
    aug,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    strategy: str,
    rf_row: pd.Series,
    xgb_row: pd.Series,
    augmentation_seed_base: int,
    model_seed: int,
) -> np.ndarray:
    policy = aug.STRATEGIES[strategy]
    repeat_seeds = 1 if policy["copies"] == 0 else N_AUGMENTATION_SEEDS
    predictions = []
    for repeat in range(repeat_seeds):
        x_train, y_train, _ = aug.augment_composition(
            train[FEATURES],
            train[TARGET].to_numpy(),
            features=FEATURES,
            copies=policy["copies"],
            sigma_scale=policy["sigma_scale"],
            seed=augmentation_seed_base + repeat,
        )
        rf = aug.rf_tuned(rf_row, model_seed)
        xgb = aug.xgb_tuned(xgb_row, model_seed)
        rf.fit(x_train, y_train)
        xgb.fit(x_train, y_train)
        predictions.append((rf.predict(valid[FEATURES]) + xgb.predict(valid[FEATURES])) / 2.0)
    return np.mean(predictions, axis=0)


def select_with_one_se(summary: pd.DataFrame) -> tuple[str, float, str]:
    ranked = summary.sort_values("Inner_Source_Macro_RMSE_Mean")
    best = ranked.iloc[0]
    threshold = best["Inner_Source_Macro_RMSE_Mean"] + best["Inner_Source_Macro_RMSE_SE"]
    eligible = summary.loc[summary["Inner_Source_Macro_RMSE_Mean"].le(threshold), "Strategy"].tolist()
    selected = next(strategy for strategy in STRATEGY_ORDER if strategy in eligible)
    reason = (
        f"best={best['Strategy']}; one_se_threshold={threshold:.6f}; "
        f"eligible={'|'.join(eligible)}; selected_simplest={selected}"
    )
    return selected, float(threshold), reason


def nested_selection(aug) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv")
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    inner_rows = []
    strategy_summary_rows = []
    selected_rows = []
    outer_predictions = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        outer_train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy().reset_index(drop=True)
        outer_test = data.loc[data["Outer_Fold"].eq(outer_fold)].copy().reset_index(drop=True)
        if set(outer_train["Source_Group"]) & set(outer_test["Source_Group"]):
            raise AssertionError("Outer source leakage")
        rf_row, xgb_row = get_fold_parameter_rows(outer_fold)
        splitter = GroupKFold(n_splits=N_INNER_SPLITS)

        for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(outer_train, outer_train[TARGET], groups=outer_train["Source_Group"])
        ):
            inner_train = outer_train.iloc[train_idx].copy()
            inner_valid = outer_train.iloc[valid_idx].copy()
            if set(inner_train["Source_Group"]) & set(inner_valid["Source_Group"]):
                raise AssertionError("Inner source leakage")

            for strategy_index, strategy in enumerate(STRATEGY_ORDER):
                prediction = fit_predict_ensemble(
                    aug,
                    inner_train,
                    inner_valid,
                    strategy,
                    rf_row,
                    xgb_row,
                    augmentation_seed_base=SEED + outer_fold * 10000 + inner_fold * 100 + strategy_index * 10,
                    model_seed=config.RANDOM_SEED + outer_fold,
                )
                row = {
                    "Outer_Fold": outer_fold,
                    "Inner_Fold": inner_fold,
                    "Strategy": strategy,
                    "Train_Rows": len(inner_train),
                    "Train_Sources": inner_train["Source_Group"].nunique(),
                    "Valid_Rows": len(inner_valid),
                    "Valid_Sources": inner_valid["Source_Group"].nunique(),
                    **metrics(inner_valid[TARGET].to_numpy(), prediction),
                    **source_macro(
                        inner_valid[TARGET].to_numpy(),
                        prediction,
                        inner_valid["Source_Group"].to_numpy(),
                    ),
                }
                inner_rows.append(row)

        outer_inner = pd.DataFrame(inner_rows).loc[lambda x: x["Outer_Fold"].eq(outer_fold)]
        summaries = []
        for strategy, part in outer_inner.groupby("Strategy"):
            n = len(part)
            summaries.append({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Inner_Folds": n,
                "Inner_RMSE_Mean": part["RMSE"].mean(),
                "Inner_RMSE_SD": part["RMSE"].std(ddof=1),
                "Inner_MAE_Mean": part["MAE"].mean(),
                "Inner_Source_Macro_MAE_Mean": part["Source_Macro_MAE"].mean(),
                "Inner_Source_Macro_RMSE_Mean": part["Source_Macro_RMSE"].mean(),
                "Inner_Source_Macro_RMSE_SD": part["Source_Macro_RMSE"].std(ddof=1),
                "Inner_Source_Macro_RMSE_SE": part["Source_Macro_RMSE"].std(ddof=1) / np.sqrt(n),
            })
        summary = pd.DataFrame(summaries)
        selected, threshold, reason = select_with_one_se(summary)
        summary["One_SE_Threshold"] = threshold
        summary["Within_One_SE"] = summary["Inner_Source_Macro_RMSE_Mean"].le(threshold)
        summary["Selected"] = summary["Strategy"].eq(selected)
        strategy_summary_rows.extend(summary.to_dict("records"))

        final_prediction = fit_predict_ensemble(
            aug,
            outer_train,
            outer_test,
            selected,
            rf_row,
            xgb_row,
            augmentation_seed_base=SEED + 500000 + outer_fold * 100,
            model_seed=config.RANDOM_SEED + outer_fold,
        )
        outer_score = metrics(outer_test[TARGET].to_numpy(), final_prediction)
        selected_rows.append({
            "Outer_Fold": outer_fold,
            "Selected_Strategy": selected,
            "One_SE_Threshold": threshold,
            "Selection_Reason": reason,
            "Outer_Test_Rows": len(outer_test),
            "Outer_Test_Sources": outer_test["Source_Group"].nunique(),
            **outer_score,
        })
        outer_predictions.append(pd.DataFrame({
            "Task": TASK,
            "Outer_Fold": outer_fold,
            "Selected_Strategy": selected,
            "Model_Row_ID": outer_test["Model_Row_ID"].to_numpy(),
            "Source_Group": outer_test["Source_Group"].to_numpy(),
            "y_true": outer_test[TARGET].to_numpy(),
            "y_pred": final_prediction,
        }))
        print(
            f"outer={outer_fold}: selected={selected}, "
            f"R2={outer_score['R2']:.3f}, RMSE={outer_score['RMSE']:.3f}"
        )

    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(strategy_summary_rows),
        pd.DataFrame(selected_rows),
        pd.concat(outer_predictions, ignore_index=True),
    )


def comparison_table(nested_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.OUTPUT_DIRS["single"]
    baseline = pd.read_csv(root / "rf_xgb_ensemble_strict" / "oof_predictions.csv")
    baseline = baseline.loc[baseline["Task"].eq(TASK)].copy()
    forced = pd.read_csv(
        config.PROJECT_ROOT / "results" / "augmentation_ablation_strict" / "oof_predictions_all_strategies.csv"
    )
    forced = forced.loc[
        forced["Task"].eq(TASK) & forced["Strategy"].eq("half_sigma_2copies")
    ].copy()

    rows = []
    for name, part in (
        ("No_Augmentation", baseline),
        ("Forced_HalfSigma_2Copies_SingleSeed", forced),
        ("Nested_OneSE_Selection", nested_pred),
    ):
        rows.append({
            "Configuration": name,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            **source_macro(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    comparison = pd.DataFrame(rows)

    base = baseline[["Model_Row_ID", "Source_Group", "y_true", "y_pred"]].rename(columns={"y_pred": "pred_base"})
    nested = nested_pred[["Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "pred_nested"})
    paired = base.merge(nested, on="Model_Row_ID", validate="one_to_one")
    return comparison, paired


def paired_source_bootstrap(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 900000)
    grouped = {
        source: (
            part["y_true"].to_numpy(),
            part["pred_base"].to_numpy(),
            part["pred_nested"].to_numpy(),
        )
        for source, part in paired.groupby("Source_Group")
    }
    sources = np.asarray(list(grouped), dtype=object)
    macro_base = np.asarray([np.abs(grouped[s][1] - grouped[s][0]).mean() for s in sources])
    macro_nested = np.asarray([np.abs(grouped[s][2] - grouped[s][0]).mean() for s in sources])
    rows = []
    for iteration in range(N_BOOTSTRAP):
        draw = rng.integers(0, len(sources), size=len(sources))
        selected = sources[draw]
        y = np.concatenate([grouped[s][0] for s in selected])
        p0 = np.concatenate([grouped[s][1] for s in selected])
        p1 = np.concatenate([grouped[s][2] for s in selected])
        m0, m1 = metrics(y, p0), metrics(y, p1)
        rows.append({
            "Iteration": iteration,
            "Delta_R2": m1["R2"] - m0["R2"],
            "Delta_RMSE": m1["RMSE"] - m0["RMSE"],
            "Delta_MAE": m1["MAE"] - m0["MAE"],
            "Delta_Source_Macro_MAE": (macro_nested[draw] - macro_base[draw]).mean(),
        })
    samples = pd.DataFrame(rows)
    summary = []
    for metric in ("Delta_R2", "Delta_RMSE", "Delta_MAE", "Delta_Source_Macro_MAE"):
        q = samples[metric].quantile([0.025, 0.5, 0.975])
        summary.append({
            "Metric": metric,
            "Median": q.loc[0.5],
            "CI95_Lower": q.loc[0.025],
            "CI95_Upper": q.loc[0.975],
            "Probability_Improvement": (
                samples[metric].gt(0).mean() if metric == "Delta_R2"
                else samples[metric].lt(0).mean()
            ),
        })
    return samples, pd.DataFrame(summary)


def save_figure(out: Path, strategy_summary: pd.DataFrame, outer: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = strategy_summary.pivot(index="Outer_Fold", columns="Strategy", values="Inner_Source_Macro_RMSE_Mean")
    pivot = pivot[STRATEGY_ORDER]
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Inner source-macro RMSE used for selection")
    axes[0].set_ylabel("RMSE (MPa)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(outer["Outer_Fold"].astype(str), outer["R2"], color="#4C72B0")
    axes[1].set_title("Nested-selected outer-fold R²")
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("R²")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "uts_nested_augmentation_selection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "augmentation_nested_selection_uts"
    out.mkdir(parents=True, exist_ok=True)
    aug = load_augmentation_module()
    inner, strategy_summary, outer, predictions = nested_selection(aug)
    comparison, paired = comparison_table(predictions)
    boot_samples, boot_summary = paired_source_bootstrap(paired)

    inner.to_csv(out / "inner_fold_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(out / "inner_strategy_summary_and_selection.csv", index=False, encoding="utf-8-sig")
    outer.to_csv(out / "outer_fold_selected_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out / "configuration_comparison.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(out, strategy_summary, outer)

    cfg = {
        "task": TASK,
        "outer_split": "fixed Source_Group folds",
        "inner_split": f"{N_INNER_SPLITS}-fold GroupKFold(Source_Group)",
        "selection_metric": "mean inner Source_Macro_RMSE",
        "selection_rule": "one-standard-error; simplest eligible strategy",
        "strategy_complexity_order": STRATEGY_ORDER,
        "augmentation_seeds_averaged": N_AUGMENTATION_SEEDS,
        "model_parameters": "frozen per outer fold from prior nested no-augmentation tuning",
        "outer_test_augmentation": False,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSELECTED STRATEGY BY OUTER FOLD")
    print(outer.to_string(index=False))
    print("\nCONFIGURATION COMPARISON")
    print(comparison.to_string(index=False))
    print("\nPAIRED SOURCE BOOTSTRAP: NESTED SELECTION VS NO AUGMENTATION")
    print(boot_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
