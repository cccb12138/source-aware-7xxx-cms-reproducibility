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

TASK = "YS"
TARGET = config.TARGET_COLUMNS[TASK]
SEED = 20260726
N_INNER_SPLITS = 3
N_BOOTSTRAP = 3000
FEATURE_SET_ORDER = [
    "core_direct",
    "core_plus_cumg",
    "core_plus_literature_ratios",
    "core_plus_all_safe",
]


def load_module(filename: str, name: str):
    path = config.PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
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
    values = []
    for _, part in frame.groupby("Source_Group"):
        error = part["y_pred"].to_numpy() - part["y_true"].to_numpy()
        values.append((np.abs(error).mean(), np.sqrt(np.square(error).mean())))
    array = np.asarray(values)
    return {
        "Source_Macro_MAE": float(array[:, 0].mean()),
        "Source_Macro_RMSE": float(array[:, 1].mean()),
    }


def core_features_for_outer_fold(selected: pd.DataFrame, outer_fold: int) -> list[str]:
    row = selected.loc[
        selected["Task"].eq(TASK)
        & selected["Model"].eq("RandomForest")
        & selected["Feature_Set"].eq("composition_core")
        & selected["Outer_Fold"].eq(outer_fold)
    ].iloc[0]
    return row["Selected_Features"].split("|")


def select_one_se(summary: pd.DataFrame) -> tuple[str, float, str]:
    best = summary.sort_values("Inner_Source_Macro_RMSE_Mean").iloc[0]
    threshold = best["Inner_Source_Macro_RMSE_Mean"] + best["Inner_Source_Macro_RMSE_SE"]
    eligible = summary.loc[summary["Inner_Source_Macro_RMSE_Mean"].le(threshold), "Feature_Set"].tolist()
    selected = next(feature_set for feature_set in FEATURE_SET_ORDER if feature_set in eligible)
    reason = (
        f"best={best['Feature_Set']}; one_se_threshold={threshold:.6f}; "
        f"eligible={'|'.join(eligible)}; selected_simplest={selected}"
    )
    return selected, float(threshold), reason


def nested_selection(model_module, proxy_module):
    raw = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "YS_with_outer_folds.csv")
    data = proxy_module.add_recomputed_proxies(raw)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    selected_table = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "baseline_strict" / "selected_features_by_fold.csv"
    )
    inner_rows = []
    strategy_rows = []
    outer_rows = []
    prediction_parts = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        outer_train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy().reset_index(drop=True)
        outer_test = data.loc[data["Outer_Fold"].eq(outer_fold)].copy().reset_index(drop=True)
        if set(outer_train["Source_Group"]) & set(outer_test["Source_Group"]):
            raise AssertionError("Outer source leakage")
        core = core_features_for_outer_fold(selected_table, outer_fold)
        splitter = GroupKFold(n_splits=N_INNER_SPLITS)

        for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(outer_train, outer_train[TARGET], groups=outer_train["Source_Group"])
        ):
            inner_train = outer_train.iloc[train_idx].copy()
            inner_valid = outer_train.iloc[valid_idx].copy()
            if set(inner_train["Source_Group"]) & set(inner_valid["Source_Group"]):
                raise AssertionError("Inner source leakage")
            for feature_set in FEATURE_SET_ORDER:
                extras = proxy_module.PROXY_SETS[feature_set]
                features = core + extras
                model = model_module.rf_baseline(config.RANDOM_SEED + outer_fold)
                model.fit(
                    inner_train[features].replace([np.inf, -np.inf], np.nan),
                    inner_train[TARGET],
                )
                prediction = model.predict(inner_valid[features].replace([np.inf, -np.inf], np.nan))
                inner_rows.append({
                    "Outer_Fold": outer_fold,
                    "Inner_Fold": inner_fold,
                    "Feature_Set": feature_set,
                    "Added_Proxies": "|".join(extras),
                    "N_Features": len(features),
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
                })

        inner_outer = pd.DataFrame(inner_rows).loc[lambda x: x["Outer_Fold"].eq(outer_fold)]
        summaries = []
        for feature_set, part in inner_outer.groupby("Feature_Set"):
            count = len(part)
            summaries.append({
                "Outer_Fold": outer_fold,
                "Feature_Set": feature_set,
                "Inner_Folds": count,
                "Inner_RMSE_Mean": part["RMSE"].mean(),
                "Inner_RMSE_SD": part["RMSE"].std(ddof=1),
                "Inner_MAE_Mean": part["MAE"].mean(),
                "Inner_Source_Macro_MAE_Mean": part["Source_Macro_MAE"].mean(),
                "Inner_Source_Macro_RMSE_Mean": part["Source_Macro_RMSE"].mean(),
                "Inner_Source_Macro_RMSE_SD": part["Source_Macro_RMSE"].std(ddof=1),
                "Inner_Source_Macro_RMSE_SE": part["Source_Macro_RMSE"].std(ddof=1) / np.sqrt(count),
            })
        summary = pd.DataFrame(summaries)
        selected_set, threshold, reason = select_one_se(summary)
        summary["One_SE_Threshold"] = threshold
        summary["Within_One_SE"] = summary["Inner_Source_Macro_RMSE_Mean"].le(threshold)
        summary["Selected"] = summary["Feature_Set"].eq(selected_set)
        strategy_rows.extend(summary.to_dict("records"))

        selected_extras = proxy_module.PROXY_SETS[selected_set]
        selected_features = core + selected_extras
        model = model_module.rf_baseline(config.RANDOM_SEED + outer_fold)
        model.fit(
            outer_train[selected_features].replace([np.inf, -np.inf], np.nan),
            outer_train[TARGET],
        )
        outer_prediction = model.predict(
            outer_test[selected_features].replace([np.inf, -np.inf], np.nan)
        )
        outer_score = metrics(outer_test[TARGET].to_numpy(), outer_prediction)
        outer_rows.append({
            "Outer_Fold": outer_fold,
            "Selected_Feature_Set": selected_set,
            "Selected_Proxies": "|".join(selected_extras),
            "One_SE_Threshold": threshold,
            "Selection_Reason": reason,
            "Outer_Test_Rows": len(outer_test),
            "Outer_Test_Sources": outer_test["Source_Group"].nunique(),
            **outer_score,
        })
        prediction_parts.append(pd.DataFrame({
            "Task": TASK,
            "Outer_Fold": outer_fold,
            "Selected_Feature_Set": selected_set,
            "Model_Row_ID": outer_test["Model_Row_ID"].to_numpy(),
            "Source_Group": outer_test["Source_Group"].to_numpy(),
            "y_true": outer_test[TARGET].to_numpy(),
            "y_pred": outer_prediction,
        }))
        print(
            f"outer={outer_fold}: selected={selected_set}, "
            f"R2={outer_score['R2']:.3f}, RMSE={outer_score['RMSE']:.3f}"
        )

    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(strategy_rows),
        pd.DataFrame(outer_rows),
        pd.concat(prediction_parts, ignore_index=True),
    )


def compare_configurations(nested: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensitivity = pd.read_csv(
        config.PROJECT_ROOT / "results" / "proxy_feature_sensitivity_strict" / "oof_predictions_all_proxy_sets.csv"
    )
    rows = []
    configurations = [
        ("Direct_Composition", sensitivity.loc[
            sensitivity["Task"].eq(TASK) & sensitivity["Feature_Set"].eq("core_direct")
        ]),
        ("Forced_CuMg", sensitivity.loc[
            sensitivity["Task"].eq(TASK) & sensitivity["Feature_Set"].eq("core_plus_cumg")
        ]),
        ("Forced_Literature_Ratios", sensitivity.loc[
            sensitivity["Task"].eq(TASK) & sensitivity["Feature_Set"].eq("core_plus_literature_ratios")
        ]),
        ("Forced_All_Safe", sensitivity.loc[
            sensitivity["Task"].eq(TASK) & sensitivity["Feature_Set"].eq("core_plus_all_safe")
        ]),
        ("Nested_OneSE_Selection", nested),
    ]
    for name, part in configurations:
        rows.append({
            "Configuration": name,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            **source_macro(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    comparison = pd.DataFrame(rows)
    base = configurations[0][1][["Model_Row_ID", "Source_Group", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_base"}
    )
    paired = base.merge(
        nested[["Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "pred_nested"}),
        on="Model_Row_ID",
        validate="one_to_one",
    )
    return comparison, paired


def paired_bootstrap(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 100000)
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
        chosen = sources[draw]
        y = np.concatenate([grouped[s][0] for s in chosen])
        p0 = np.concatenate([grouped[s][1] for s in chosen])
        p1 = np.concatenate([grouped[s][2] for s in chosen])
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


def save_figure(out: Path, inner_summary: pd.DataFrame, outer: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = inner_summary.pivot(
        index="Outer_Fold", columns="Feature_Set", values="Inner_Source_Macro_RMSE_Mean"
    )[FEATURE_SET_ORDER]
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("YS inner source-macro RMSE")
    axes[0].set_ylabel("RMSE (MPa)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.2)

    colors = ["#4C72B0" if name == "core_direct" else "#55A868" for name in outer["Selected_Feature_Set"]]
    axes[1].bar(outer["Outer_Fold"].astype(str), outer["R2"], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("YS nested-selected outer-fold R²")
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("R²")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "ys_nested_proxy_selection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "proxy_nested_selection_ys"
    out.mkdir(parents=True, exist_ok=True)
    model_module = load_module("14_strict_fold_augmentation_ablation.py", "model_builders")
    proxy_module = load_module("16_proxy_feature_audit_sensitivity.py", "proxy_definitions")
    inner, inner_summary, outer, nested = nested_selection(model_module, proxy_module)
    comparison, paired = compare_configurations(nested)
    boot_samples, boot_summary = paired_bootstrap(paired)

    inner.to_csv(out / "inner_fold_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    inner_summary.to_csv(out / "inner_proxy_summary_and_selection.csv", index=False, encoding="utf-8-sig")
    outer.to_csv(out / "outer_fold_selected_proxy_metrics.csv", index=False, encoding="utf-8-sig")
    nested.to_csv(out / "nested_selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out / "configuration_comparison.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(out, inner_summary, outer)

    cfg = {
        "task": TASK,
        "outer_split": "fixed Source_Group folds",
        "inner_split": f"{N_INNER_SPLITS}-fold GroupKFold(Source_Group)",
        "selection_metric": "mean inner Source_Macro_RMSE",
        "selection_rule": "one-standard-error; simplest eligible feature set",
        "feature_set_complexity_order": FEATURE_SET_ORDER,
        "model": "frozen RandomForest baseline; no augmentation",
        "outer_test_used_for_selection": False,
        "bootstrap_iterations": N_BOOTSTRAP,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSELECTED FEATURE SET BY OUTER FOLD")
    print(outer.to_string(index=False))
    print("\nCONFIGURATION COMPARISON")
    print(comparison.to_string(index=False))
    print("\nPAIRED SOURCE BOOTSTRAP: NESTED VS DIRECT")
    print(boot_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
