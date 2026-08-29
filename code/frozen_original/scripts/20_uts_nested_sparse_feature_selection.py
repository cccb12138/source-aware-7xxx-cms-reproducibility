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

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
ALL_FEATURES = list(config.PRIMARY_FIXED_FEATURES)
MAJOR_FEATURES = ["Zn", "Mg", "Cu"]
SEED = 20260729
N_INNER_SPLITS = 3
N_BOOTSTRAP = 2000

# Ordered from simplest to most complex for the one-standard-error rule.
STRATEGY_ORDER = [
    "major3",
    "drop_si_sparse50",
    "drop_si_sparse40",
    "drop_si_sparse20",
    "sparse20_keep_si",
    "drop_si",
    "full10",
]


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


def source_macro(y_true, y_pred, groups):
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "Source_Group": groups})
    values = []
    for _, part in frame.groupby("Source_Group"):
        error = part["y_pred"].to_numpy() - part["y_true"].to_numpy()
        values.append((np.abs(error).mean(), np.sqrt(np.square(error).mean())))
    array = np.asarray(values)
    return {
        "Source_Macro_MAE": float(array[:, 0].mean()),
        "Source_Macro_RMSE": float(array[:, 1].mean()),
    }


def nonzero_rate(training: pd.DataFrame, feature: str) -> float:
    values = pd.to_numeric(training[feature], errors="coerce").fillna(0)
    return float(values.ne(0).mean())


def strategy_features(strategy: str, training: pd.DataFrame) -> list[str]:
    if strategy == "full10":
        return ALL_FEATURES.copy()
    if strategy == "drop_si":
        return [feature for feature in ALL_FEATURES if feature != "Si"]
    if strategy == "major3":
        return MAJOR_FEATURES.copy()

    if strategy == "sparse20_keep_si":
        threshold, drop_si = 0.20, False
    elif strategy == "drop_si_sparse20":
        threshold, drop_si = 0.20, True
    elif strategy == "drop_si_sparse40":
        threshold, drop_si = 0.40, True
    elif strategy == "drop_si_sparse50":
        threshold, drop_si = 0.50, True
    else:
        raise KeyError(strategy)

    selected = []
    for feature in ALL_FEATURES:
        if feature in MAJOR_FEATURES:
            selected.append(feature)
            continue
        if drop_si and feature == "Si":
            continue
        if nonzero_rate(training, feature) >= threshold:
            selected.append(feature)
    return selected


def fit_predict(model_module, train, valid, features, rf_row, xgb_row, model_seed):
    rf = model_module.rf_tuned(rf_row, model_seed)
    xgb = model_module.xgb_tuned(xgb_row, model_seed)
    rf.fit(train[features], train[TARGET])
    xgb.fit(train[features], train[TARGET])
    return (rf.predict(valid[features]) + xgb.predict(valid[features])) / 2.0


def select_one_se(summary: pd.DataFrame):
    best = summary.sort_values("Inner_Source_Macro_RMSE_Mean").iloc[0]
    threshold = best["Inner_Source_Macro_RMSE_Mean"] + best["Inner_Source_Macro_RMSE_SE"]
    eligible = summary.loc[summary["Inner_Source_Macro_RMSE_Mean"].le(threshold), "Strategy"].tolist()
    selected = next(strategy for strategy in STRATEGY_ORDER if strategy in eligible)
    reason = (
        f"best={best['Strategy']}; one_se_threshold={threshold:.6f}; "
        f"eligible={'|'.join(eligible)}; selected_simplest={selected}"
    )
    return selected, float(threshold), reason


def run_nested(model_module):
    data = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv")
    data[ALL_FEATURES + [TARGET]] = data[ALL_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    root = config.OUTPUT_DIRS["single"]
    rf_params = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb_params = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")

    inner_rows = []
    inner_summary_rows = []
    outer_selection_rows = []
    outer_strategy_predictions = []
    outer_strategy_metrics = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        outer_train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy().reset_index(drop=True)
        outer_test = data.loc[data["Outer_Fold"].eq(outer_fold)].copy().reset_index(drop=True)
        if set(outer_train["Source_Group"]) & set(outer_test["Source_Group"]):
            raise AssertionError("Outer source leakage")
        rf_row = rf_params.loc[rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(outer_fold)].iloc[0]
        xgb_row = xgb_params.loc[xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(outer_fold)].iloc[0]
        splitter = GroupKFold(n_splits=N_INNER_SPLITS)

        for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(outer_train, outer_train[TARGET], groups=outer_train["Source_Group"])
        ):
            inner_train = outer_train.iloc[train_idx].copy()
            inner_valid = outer_train.iloc[valid_idx].copy()
            if set(inner_train["Source_Group"]) & set(inner_valid["Source_Group"]):
                raise AssertionError("Inner source leakage")
            for strategy in STRATEGY_ORDER:
                features = strategy_features(strategy, inner_train)
                prediction = fit_predict(
                    model_module, inner_train, inner_valid, features, rf_row, xgb_row,
                    config.RANDOM_SEED + outer_fold,
                )
                inner_rows.append({
                    "Outer_Fold": outer_fold,
                    "Inner_Fold": inner_fold,
                    "Strategy": strategy,
                    "Selected_Features": "|".join(features),
                    "N_Features": len(features),
                    "Train_Rows": len(inner_train),
                    "Train_Sources": inner_train["Source_Group"].nunique(),
                    "Valid_Rows": len(inner_valid),
                    "Valid_Sources": inner_valid["Source_Group"].nunique(),
                    **metrics(inner_valid[TARGET], prediction),
                    **source_macro(inner_valid[TARGET], prediction, inner_valid["Source_Group"]),
                })

        outer_inner = pd.DataFrame(inner_rows).loc[lambda x: x["Outer_Fold"].eq(outer_fold)]
        summaries = []
        for strategy, part in outer_inner.groupby("Strategy"):
            count = len(part)
            summaries.append({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Inner_Folds": count,
                "Feature_Sets_Seen": ";".join(sorted(part["Selected_Features"].unique())),
                "Inner_N_Features_Median": part["N_Features"].median(),
                "Inner_RMSE_Mean": part["RMSE"].mean(),
                "Inner_MAE_Mean": part["MAE"].mean(),
                "Inner_Source_Macro_MAE_Mean": part["Source_Macro_MAE"].mean(),
                "Inner_Source_Macro_RMSE_Mean": part["Source_Macro_RMSE"].mean(),
                "Inner_Source_Macro_RMSE_SD": part["Source_Macro_RMSE"].std(ddof=1),
                "Inner_Source_Macro_RMSE_SE": part["Source_Macro_RMSE"].std(ddof=1) / np.sqrt(count),
            })
        summary = pd.DataFrame(summaries)
        selected_strategy, threshold, reason = select_one_se(summary)
        summary["One_SE_Threshold"] = threshold
        summary["Within_One_SE"] = summary["Inner_Source_Macro_RMSE_Mean"].le(threshold)
        summary["Selected"] = summary["Strategy"].eq(selected_strategy)
        inner_summary_rows.extend(summary.to_dict("records"))

        outer_features_by_strategy = {
            strategy: strategy_features(strategy, outer_train) for strategy in STRATEGY_ORDER
        }
        fold_predictions = {}
        for strategy in STRATEGY_ORDER:
            features = outer_features_by_strategy[strategy]
            prediction = fit_predict(
                model_module, outer_train, outer_test, features, rf_row, xgb_row,
                config.RANDOM_SEED + outer_fold,
            )
            fold_predictions[strategy] = prediction
            fold_score = metrics(outer_test[TARGET], prediction)
            outer_strategy_metrics.append({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Selected_Features": "|".join(features),
                "N_Features": len(features),
                "Outer_Test_Rows": len(outer_test),
                "Outer_Test_Sources": outer_test["Source_Group"].nunique(),
                **fold_score,
            })
            outer_strategy_predictions.append(pd.DataFrame({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Selected_Features": "|".join(features),
                "Model_Row_ID": outer_test["Model_Row_ID"].to_numpy(),
                "Source_Group": outer_test["Source_Group"].to_numpy(),
                "y_true": outer_test[TARGET].to_numpy(),
                "y_pred": prediction,
            }))

        selected_features = outer_features_by_strategy[selected_strategy]
        selected_prediction = fold_predictions[selected_strategy]
        selected_score = metrics(outer_test[TARGET], selected_prediction)
        outer_selection_rows.append({
            "Outer_Fold": outer_fold,
            "Selected_Strategy": selected_strategy,
            "Selected_Features": "|".join(selected_features),
            "N_Selected_Features": len(selected_features),
            "One_SE_Threshold": threshold,
            "Selection_Reason": reason,
            "Outer_Test_Rows": len(outer_test),
            "Outer_Test_Sources": outer_test["Source_Group"].nunique(),
            **selected_score,
        })
        print(
            f"outer={outer_fold}: selected={selected_strategy} "
            f"features={selected_features}, R2={selected_score['R2']:.3f}"
        )

    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(inner_summary_rows),
        pd.DataFrame(outer_selection_rows),
        pd.concat(outer_strategy_predictions, ignore_index=True),
        pd.DataFrame(outer_strategy_metrics),
    )


def nested_predictions(outer_selection: pd.DataFrame, all_predictions: pd.DataFrame):
    parts = []
    for _, selection in outer_selection.iterrows():
        part = all_predictions.loc[
            all_predictions["Outer_Fold"].eq(selection["Outer_Fold"])
            & all_predictions["Strategy"].eq(selection["Selected_Strategy"])
        ].copy()
        part["Selected_Strategy"] = selection["Selected_Strategy"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def configuration_summary(all_predictions: pd.DataFrame, nested: pd.DataFrame):
    rows = []
    for strategy, part in all_predictions.groupby("Strategy", sort=False):
        rows.append({
            "Configuration": strategy,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            "Feature_Sets_Across_Folds": ";".join(sorted(part["Selected_Features"].unique())),
            **metrics(part["y_true"], part["y_pred"]),
            **source_macro(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    rows.append({
        "Configuration": "nested_one_se_selection",
        "Rows": len(nested),
        "Sources": nested["Source_Group"].nunique(),
        "Feature_Sets_Across_Folds": ";".join(sorted(nested["Selected_Features"].unique())),
        **metrics(nested["y_true"], nested["y_pred"]),
        **source_macro(nested["y_true"], nested["y_pred"], nested["Source_Group"]),
    })
    return pd.DataFrame(rows)


def paired_source_bootstrap(all_predictions: pd.DataFrame, nested: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    full = all_predictions.loc[
        all_predictions["Strategy"].eq("full10"),
        ["Model_Row_ID", "Source_Group", "y_true", "y_pred"],
    ].rename(columns={"y_pred": "pred_full"})
    candidates = {
        strategy: all_predictions.loc[all_predictions["Strategy"].eq(strategy), ["Model_Row_ID", "y_pred"]]
        for strategy in STRATEGY_ORDER if strategy != "full10"
    }
    candidates["nested_one_se_selection"] = nested[["Model_Row_ID", "y_pred"]]
    samples = []
    for name, candidate in candidates.items():
        paired = full.merge(candidate.rename(columns={"y_pred": "pred_candidate"}), on="Model_Row_ID", validate="one_to_one")
        grouped = {
            source: (
                part["y_true"].to_numpy(),
                part["pred_full"].to_numpy(),
                part["pred_candidate"].to_numpy(),
            )
            for source, part in paired.groupby("Source_Group")
        }
        sources = np.asarray(list(grouped), dtype=object)
        macro_full = np.asarray([np.abs(grouped[s][1] - grouped[s][0]).mean() for s in sources])
        macro_candidate = np.asarray([np.abs(grouped[s][2] - grouped[s][0]).mean() for s in sources])
        for iteration in range(N_BOOTSTRAP):
            draw = rng.integers(0, len(sources), size=len(sources))
            chosen = sources[draw]
            y = np.concatenate([grouped[s][0] for s in chosen])
            p0 = np.concatenate([grouped[s][1] for s in chosen])
            p1 = np.concatenate([grouped[s][2] for s in chosen])
            m0, m1 = metrics(y, p0), metrics(y, p1)
            samples.append({
                "Configuration": name,
                "Iteration": iteration,
                "Delta_R2_vs_Full": m1["R2"] - m0["R2"],
                "Delta_RMSE_vs_Full": m1["RMSE"] - m0["RMSE"],
                "Delta_MAE_vs_Full": m1["MAE"] - m0["MAE"],
                "Delta_Source_Macro_MAE_vs_Full": (macro_candidate[draw] - macro_full[draw]).mean(),
            })
    samples = pd.DataFrame(samples)
    summary_rows = []
    for configuration, part in samples.groupby("Configuration", sort=False):
        for metric in ("Delta_R2_vs_Full", "Delta_RMSE_vs_Full", "Delta_MAE_vs_Full", "Delta_Source_Macro_MAE_vs_Full"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary_rows.append({
                "Configuration": configuration,
                "Metric": metric,
                "Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
                "Probability_Improvement": (
                    part[metric].gt(0).mean() if metric == "Delta_R2_vs_Full"
                    else part[metric].lt(0).mean()
                ),
            })
    return samples, pd.DataFrame(summary_rows)


def save_figure(out: Path, summary: pd.DataFrame, bootstrap_summary: pd.DataFrame):
    order = STRATEGY_ORDER[:-1] + ["nested_one_se_selection"]
    table = summary.set_index("Configuration").loc[order]
    boot = bootstrap_summary.loc[
        bootstrap_summary["Configuration"].isin(order)
        & bootstrap_summary["Metric"].eq("Delta_RMSE_vs_Full")
    ].set_index("Configuration").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    axes[0].bar(range(len(order)), table["R2"], color="#4C72B0")
    axes[0].axhline(summary.loc[summary["Configuration"].eq("full10"), "R2"].iloc[0], color="black", linestyle="--")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([item.replace("_", "\n") for item in order], fontsize=8)
    axes[0].set_ylabel("OOF R²")
    axes[0].set_title("UTS sparse feature configurations")

    error = np.vstack([boot["Median"] - boot["CI95_Lower"], boot["CI95_Upper"] - boot["Median"]])
    axes[1].errorbar(range(len(order)), boot["Median"], yerr=error, fmt="o", capsize=4, color="#C44E52")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([item.replace("_", "\n") for item in order], fontsize=8)
    axes[1].set_ylabel("ΔRMSE vs full10 (negative = improvement)")
    axes[1].set_title("Paired source-bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(out / "uts_nested_sparse_feature_selection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    out = config.PROJECT_ROOT / "results" / "uts_nested_sparse_feature_selection"
    out.mkdir(parents=True, exist_ok=True)
    model_module = load_model_module()
    inner, inner_summary, outer_selection, all_predictions, outer_metrics = run_nested(model_module)
    nested = nested_predictions(outer_selection, all_predictions)
    summary = configuration_summary(all_predictions, nested)
    boot_samples, boot_summary = paired_source_bootstrap(all_predictions, nested)

    inner.to_csv(out / "inner_fold_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    inner_summary.to_csv(out / "inner_strategy_summary_and_selection.csv", index=False, encoding="utf-8-sig")
    outer_selection.to_csv(out / "outer_fold_selected_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(out / "outer_oof_predictions_all_strategies.csv", index=False, encoding="utf-8-sig")
    outer_metrics.to_csv(out / "outer_fold_metrics_all_strategies.csv", index=False, encoding="utf-8-sig")
    nested.to_csv(out / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "configuration_comparison.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(out, summary, boot_summary)

    cfg = {
        "task": TASK,
        "base_features": ALL_FEATURES,
        "strategies_in_complexity_order": STRATEGY_ORDER,
        "zero_definition": "numeric missing treated as zero only for sparsity-rate filtering",
        "major_features_always_kept": MAJOR_FEATURES,
        "outer_split": "fixed Source_Group folds",
        "inner_split": f"{N_INNER_SPLITS}-fold GroupKFold(Source_Group)",
        "selection_metric": "mean inner Source_Macro_RMSE",
        "selection_rule": "one-standard-error; simplest eligible strategy",
        "model": "frozen fold-specific RF+XGBoost ensemble; no augmentation",
        "bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOUTER SELECTION")
    print(outer_selection.to_string(index=False))
    print("\nCONFIGURATION COMPARISON")
    print(summary.to_string(index=False))
    print("\nBOOTSTRAP ΔRMSE")
    print(boot_summary.loc[boot_summary["Metric"].eq("Delta_RMSE_vs_Full")].to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
