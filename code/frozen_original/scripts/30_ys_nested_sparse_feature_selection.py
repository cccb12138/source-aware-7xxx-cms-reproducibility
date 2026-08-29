from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
INPUT_ROOT = Path(r"F:\CC\outputs\ys_el_scope_audit")
ABLATION_ROOT = Path(r"F:\CC\outputs\ys_el_sparse_feature_ablation")
OUTPUT_ROOT = Path(r"F:\CC\outputs\ys_nested_sparse_selection")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASK = "YS"
TARGET = "YS_0.2pct_MPa"
INNER_SPLITS = 3
N_BOOTSTRAP = 5000
SEED = 20260804
FEATURE_SETS = {
    "major3": ["Zn", "Mg", "Cu"],
    "refined4": ["Zn", "Mg", "Cu", "Zr"],
    "refined5": ["Zn", "Mg", "Cu", "Fe", "Zr"],
    "dense7": ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Zr"],
    "drop_ni_sc": ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr"],
}
SIMPLICITY_ORDER = ["major3", "refined4", "refined5", "dense7", "drop_ni_sc", "original_fold_core"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def original_features(fold: int) -> list[str]:
    selected = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "baseline_strict" / "selected_features_by_fold.csv"
    )
    row = selected.loc[
        selected["Task"].eq(TASK)
        & selected["Feature_Set"].eq("composition_core")
        & selected["Model"].eq("RandomForest")
        & selected["Outer_Fold"].eq(fold)
    ].iloc[0]
    return str(row["Selected_Features"]).split("|")


def metric_values(y_true, y_pred, groups) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "Source_Group": groups})
    source_values = []
    for _, part in frame.groupby("Source_Group"):
        error = part["y_pred"] - part["y_true"]
        source_values.append((error.abs().mean(), np.sqrt(np.square(error).mean())))
    source_values = np.asarray(source_values)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Source_Macro_MAE": float(source_values[:, 0].mean()),
        "Source_Macro_RMSE": float(source_values[:, 1].mean()),
    }


def fit_predict(train, valid, features, fold, model_module, xgb_row, seed_offset):
    rf = model_module.rf_baseline(config.RANDOM_SEED + fold + seed_offset)
    xgb = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold + seed_offset)
    rf.fit(train[features], train[TARGET])
    xgb.fit(train[features], train[TARGET])
    return (rf.predict(valid[features]) + xgb.predict(valid[features])) / 2.0


def nested_selection(data, model_module, xgb_params):
    inner_rows = []
    selection_rows = []
    outer_predictions = []
    outer_metrics = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        outer_train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy().reset_index(drop=True)
        outer_test = data.loc[data["Outer_Fold"].eq(outer_fold)].copy().reset_index(drop=True)
        xgb_row = xgb_params.loc[
            xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(outer_fold)
        ].iloc[0]
        candidates = {**FEATURE_SETS, "original_fold_core": original_features(outer_fold)}
        splitter = GroupKFold(n_splits=INNER_SPLITS)
        groups = outer_train["Source_Group"].astype(str).to_numpy()

        for inner_fold, (fit_idx, valid_idx) in enumerate(splitter.split(outer_train, groups=groups)):
            fit = outer_train.iloc[fit_idx]
            valid = outer_train.iloc[valid_idx]
            if set(fit["Source_Group"]) & set(valid["Source_Group"]):
                raise AssertionError("Inner source leakage")
            for feature_set, features in candidates.items():
                pred = fit_predict(
                    fit,
                    valid,
                    features,
                    outer_fold,
                    model_module,
                    xgb_row,
                    seed_offset=inner_fold * 100,
                )
                inner_rows.append(
                    {
                        "Outer_Fold": outer_fold,
                        "Inner_Fold": inner_fold,
                        "Feature_Set": feature_set,
                        "N_Features": len(features),
                        "Features": "|".join(features),
                        "Train_Rows": len(fit),
                        "Train_Sources": fit["Source_Group"].nunique(),
                        "Valid_Rows": len(valid),
                        "Valid_Sources": valid["Source_Group"].nunique(),
                        **metric_values(valid[TARGET], pred, valid["Source_Group"]),
                    }
                )

        inner_frame = pd.DataFrame(inner_rows).loc[lambda x: x["Outer_Fold"].eq(outer_fold)]
        summaries = []
        for feature_set, part in inner_frame.groupby("Feature_Set"):
            count = len(part)
            summaries.append(
                {
                    "Outer_Fold": outer_fold,
                    "Feature_Set": feature_set,
                    "N_Features": int(part["N_Features"].iloc[0]),
                    "Features": part["Features"].iloc[0],
                    "Inner_RMSE_Mean": part["RMSE"].mean(),
                    "Inner_MAE_Mean": part["MAE"].mean(),
                    "Inner_Source_Macro_RMSE_Mean": part["Source_Macro_RMSE"].mean(),
                    "Inner_Source_Macro_RMSE_SD": part["Source_Macro_RMSE"].std(ddof=1),
                    "Inner_Source_Macro_RMSE_SE": part["Source_Macro_RMSE"].std(ddof=1) / np.sqrt(count),
                }
            )
        summary = pd.DataFrame(summaries)
        best_row = summary.sort_values("Inner_Source_Macro_RMSE_Mean").iloc[0]
        threshold = best_row["Inner_Source_Macro_RMSE_Mean"] + best_row["Inner_Source_Macro_RMSE_SE"]
        eligible = set(summary.loc[summary["Inner_Source_Macro_RMSE_Mean"].le(threshold), "Feature_Set"])
        one_se = next(name for name in SIMPLICITY_ORDER if name in eligible)
        best = str(best_row["Feature_Set"])
        summary["Best_Selected"] = summary["Feature_Set"].eq(best)
        summary["One_SE_Threshold"] = threshold
        summary["Within_One_SE"] = summary["Inner_Source_Macro_RMSE_Mean"].le(threshold)
        summary["One_SE_Selected"] = summary["Feature_Set"].eq(one_se)
        selection_rows.extend(summary.to_dict("records"))

        for strategy, selected_set in (("Nested_Best", best), ("Nested_OneSE", one_se)):
            features = candidates[selected_set]
            pred = fit_predict(
                outer_train,
                outer_test,
                features,
                outer_fold,
                model_module,
                xgb_row,
                seed_offset=0,
            )
            part = pd.DataFrame(
                {
                    "Task": TASK,
                    "Strategy": strategy,
                    "Outer_Fold": outer_fold,
                    "Selected_Feature_Set": selected_set,
                    "Selected_Features": "|".join(features),
                    "Model_Row_ID": outer_test["Model_Row_ID"].to_numpy(),
                    "Source_Group": outer_test["Source_Group"].to_numpy(),
                    "y_true": outer_test[TARGET].to_numpy(),
                    "y_pred": pred,
                }
            )
            outer_predictions.append(part)
            outer_metrics.append(
                {
                    "Task": TASK,
                    "Strategy": strategy,
                    "Outer_Fold": outer_fold,
                    "Selected_Feature_Set": selected_set,
                    "Selected_Features": "|".join(features),
                    "Outer_Test_Rows": len(outer_test),
                    "Outer_Test_Sources": outer_test["Source_Group"].nunique(),
                    **metric_values(outer_test[TARGET], pred, outer_test["Source_Group"]),
                }
            )
    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(selection_rows),
        pd.concat(outer_predictions, ignore_index=True),
        pd.DataFrame(outer_metrics),
    )


def comparison_table(nested_predictions):
    forced = pd.read_csv(ABLATION_ROOT / "forced_feature_set_oof_predictions.csv")
    rows = []
    frames = {
        "Original_Fold_Core": forced.loc[
            forced["Task"].eq(TASK) & forced["Feature_Set"].eq("original_fold_core")
        ],
        "Forced_Refined5_Exploratory": forced.loc[
            forced["Task"].eq(TASK) & forced["Feature_Set"].eq("refined5")
        ],
        "Nested_Best": nested_predictions.loc[nested_predictions["Strategy"].eq("Nested_Best")],
        "Nested_OneSE": nested_predictions.loc[nested_predictions["Strategy"].eq("Nested_OneSE")],
    }
    for name, part in frames.items():
        rows.append(
            {
                "Configuration": name,
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                **metric_values(part["y_true"], part["y_pred"], part["Source_Group"]),
            }
        )
    return pd.DataFrame(rows), frames


def paired_bootstrap(frames):
    base = frames["Original_Fold_Core"][["Model_Row_ID", "Source_Group", "y_true", "y_pred"]].rename(
        columns={"y_pred": "pred_base"}
    )
    rows = []
    summaries = []
    rng = np.random.default_rng(SEED)
    for strategy in ["Nested_Best", "Nested_OneSE"]:
        paired = base.merge(
            frames[strategy][["Model_Row_ID", "y_pred"]].rename(columns={"y_pred": "pred_new"}),
            on="Model_Row_ID",
            validate="one_to_one",
        )
        grouped = {
            source: (
                part["y_true"].to_numpy(dtype=float),
                part["pred_base"].to_numpy(dtype=float),
                part["pred_new"].to_numpy(dtype=float),
            )
            for source, part in paired.groupby("Source_Group")
        }
        sources = np.asarray(list(grouped), dtype=object)
        macro_base = np.asarray(
            [np.abs(grouped[source][1] - grouped[source][0]).mean() for source in sources]
        )
        macro_new = np.asarray(
            [np.abs(grouped[source][2] - grouped[source][0]).mean() for source in sources]
        )
        for iteration in range(N_BOOTSTRAP):
            draw_index = rng.integers(0, len(sources), size=len(sources))
            draw = sources[draw_index]
            y = np.concatenate([grouped[source][0] for source in draw])
            p0 = np.concatenate([grouped[source][1] for source in draw])
            p1 = np.concatenate([grouped[source][2] for source in draw])
            base_r2 = r2_score(y, p0)
            new_r2 = r2_score(y, p1)
            base_rmse = mean_squared_error(y, p0, squared=False)
            new_rmse = mean_squared_error(y, p1, squared=False)
            base_mae = mean_absolute_error(y, p0)
            new_mae = mean_absolute_error(y, p1)
            rows.append(
                {
                    "Strategy": strategy,
                    "Iteration": iteration,
                    "Delta_R2": new_r2 - base_r2,
                    "Delta_RMSE": new_rmse - base_rmse,
                    "Delta_MAE": new_mae - base_mae,
                    "Delta_Source_Macro_MAE": (macro_new[draw_index] - macro_base[draw_index]).mean(),
                }
            )
    samples = pd.DataFrame(rows)
    for strategy, part in samples.groupby("Strategy"):
        for metric in ["Delta_R2", "Delta_RMSE", "Delta_MAE", "Delta_Source_Macro_MAE"]:
            q = part[metric].quantile([0.025, 0.5, 0.975])
            improvement = part[metric].gt(0).mean() if metric == "Delta_R2" else part[metric].lt(0).mean()
            summaries.append(
                {
                    "Strategy": strategy,
                    "Metric": metric,
                    "Median": q.loc[0.5],
                    "CI95_Lower": q.loc[0.025],
                    "CI95_Upper": q.loc[0.975],
                    "Probability_Improvement": improvement,
                }
            )
    return samples, pd.DataFrame(summaries)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_ROOT / "YS_scope_clean_with_outer_folds.csv")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    model_module = load_module(
        "ys_nested_sparse_models", PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    )
    xgb_params = pd.read_csv(
        config.OUTPUT_DIRS["single"] / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv"
    )
    inner, selection, predictions, fold_metrics = nested_selection(data, model_module, xgb_params)
    comparison, frames = comparison_table(predictions)
    boot_samples, boot_summary = paired_bootstrap(frames)

    inner.to_csv(OUTPUT_ROOT / "inner_source_group_metrics.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUTPUT_ROOT / "inner_selection_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_ROOT / "nested_outer_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUTPUT_ROOT / "nested_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_ROOT / "configuration_comparison.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(OUTPUT_ROOT / "paired_source_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUTPUT_ROOT / "paired_source_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "candidate_sets": FEATURE_SETS,
                "selection": "3-fold inner GroupKFold by source-macro RMSE; best and one-SE",
                "outer": "unchanged source-exclusive five folds",
                "model": "RandomForest baseline + frozen nested XGBoost",
                "parameter_tuning": False,
                "bootstrap_iterations": N_BOOTSTRAP,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("INNER SELECTION BY OUTER FOLD")
    print(
        selection.loc[selection["Best_Selected"] | selection["One_SE_Selected"], [
            "Outer_Fold", "Feature_Set", "N_Features", "Inner_Source_Macro_RMSE_Mean",
            "One_SE_Threshold", "Best_Selected", "One_SE_Selected"
        ]].sort_values(["Outer_Fold", "Best_Selected"], ascending=[True, False]).to_string(index=False)
    )
    print("\nCONFIGURATION COMPARISON")
    print(comparison.to_string(index=False))
    print("\nPAIRED SOURCE BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
