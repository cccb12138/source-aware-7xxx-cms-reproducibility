from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
INPUT = PROJECT_ROOT / "results" / "uts_systematic_scope_audit" / "UTS_scope_clean_with_outer_folds.csv"
OUT = Path(r"F:\CC\outputs\paper_scope_clean_final") / "model_decisions"

TARGET = "UTS_MPa"
FULL10 = ["Zn", "Mg", "Cu", "Si", "Fe", "Mn", "Cr", "Ti", "Zr", "Sc"]
REFINED5 = ["Zn", "Mg", "Cu", "Fe", "Zr"]
FEATURE_CONFIGS = {
    "major3": ["Zn", "Mg", "Cu"],
    "refined5": REFINED5,
    "drop_si": ["Zn", "Mg", "Cu", "Fe", "Mn", "Cr", "Ti", "Zr", "Sc"],
    "full10": FULL10,
}
FEATURE_COMPLEXITY = ["major3", "refined5", "drop_si", "full10"]
SEED = 20260810
N_TRIALS = 8
N_INNER_SPLITS = 3
N_AUGMENTATION_SEEDS = 1
AUGMENTATION_ORDER = [
    "no_augmentation",
    "half_sigma_2copies",
    "nominal_sigma_2copies",
    "double_sigma_2copies",
]


def load_augmentation_module():
    path = PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    spec = importlib.util.spec_from_file_location("scope_clean_augmentation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    data[FULL10 + [TARGET]] = data[FULL10 + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if len(data) != 675:
        raise AssertionError(f"Expected 675 UTS rows, found {len(data)}")
    if data["Source_Group"].nunique() != 258:
        raise AssertionError("Expected 258 UTS sources")
    if data.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("A source occurs in more than one outer fold")
    return data


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def source_macro_rmse(y_true, y_pred, groups) -> float:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "Source_Group": groups})
    values = []
    for _, part in frame.groupby("Source_Group"):
        values.append(np.sqrt(np.mean(np.square(part["y_pred"] - part["y_true"]))))
    return float(np.mean(values))


def rf_model(params, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=180,
            random_state=seed,
            n_jobs=4,
            **params,
        )),
    ])


def xgb_model(params, seed):
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


def tune_rf(train: pd.DataFrame, outer_fold: int):
    inner = GroupKFold(n_splits=N_INNER_SPLITS)
    x = train[REFINED5]
    y = train[TARGET].to_numpy()
    groups = train["Source_Group"].to_numpy()

    def objective(trial):
        params = {
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        }
        fold_scores = []
        for inner_train, inner_valid in inner.split(x, y, groups):
            model = rf_model(params, SEED + outer_fold)
            model.fit(x.iloc[inner_train], y[inner_train])
            prediction = model.predict(x.iloc[inner_valid])
            fold_scores.append(np.sqrt(mean_squared_error(y[inner_valid], prediction)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED + outer_fold),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value), study


def tune_xgb(train: pd.DataFrame, outer_fold: int):
    inner = GroupKFold(n_splits=N_INNER_SPLITS)
    x = train[REFINED5]
    y = train[TARGET].to_numpy()
    groups = train["Source_Group"].to_numpy()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 420, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        fold_scores = []
        for inner_train, inner_valid in inner.split(x, y, groups):
            model = xgb_model(params, SEED + outer_fold)
            model.fit(x.iloc[inner_train], y[inner_train])
            prediction = model.predict(x.iloc[inner_valid])
            fold_scores.append(np.sqrt(mean_squared_error(y[inner_valid], prediction)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED + 100 + outer_fold),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value), study


def fit_ensemble(train, valid, features, rf_params, xgb_params, seed):
    rf = rf_model(rf_params, seed)
    xgb = xgb_model(xgb_params, seed)
    rf.fit(train[features], train[TARGET])
    xgb.fit(train[features], train[TARGET])
    pred_rf = rf.predict(valid[features])
    pred_xgb = xgb.predict(valid[features])
    return pred_rf, pred_xgb, (pred_rf + pred_xgb) / 2.0


def select_one_se(summary, order, metric="Source_Macro_RMSE_Mean"):
    best = summary.sort_values(metric).iloc[0]
    threshold = best[metric] + best[f"{metric}_SE"]
    eligible = summary.loc[summary[metric].le(threshold), "Strategy"].tolist()
    selected = next(item for item in order if item in eligible)
    return selected, float(threshold), "|".join(eligible)


def fit_predict_augmented(aug, train, valid, strategy, rf_params, xgb_params, seed):
    policy = aug.STRATEGIES[strategy]
    repeats = 1 if policy["copies"] == 0 else N_AUGMENTATION_SEEDS
    predictions = []
    for repeat in range(repeats):
        x_train, y_train, _ = aug.augment_composition(
            train[REFINED5],
            train[TARGET].to_numpy(),
            features=REFINED5,
            copies=policy["copies"],
            sigma_scale=policy["sigma_scale"],
            seed=seed + repeat,
        )
        rf = rf_model(rf_params, seed)
        xgb = xgb_model(xgb_params, seed)
        rf.fit(x_train, y_train)
        xgb.fit(x_train, y_train)
        predictions.append((rf.predict(valid[REFINED5]) + xgb.predict(valid[REFINED5])) / 2.0)
    return np.mean(predictions, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    aug = load_augmentation_module()

    rf_param_rows = []
    xgb_param_rows = []
    trial_rows = []
    model_prediction_parts = []
    feature_prediction_parts = []
    feature_selection_rows = []
    augmentation_prediction_parts = []
    augmentation_selection_rows = []

    for outer_fold in sorted(data["Outer_Fold"].unique()):
        outer_fold = int(outer_fold)
        train = data.loc[data["Outer_Fold"].ne(outer_fold)].copy().reset_index(drop=True)
        test = data.loc[data["Outer_Fold"].eq(outer_fold)].copy().reset_index(drop=True)
        if set(train["Source_Group"]) & set(test["Source_Group"]):
            raise AssertionError("Outer source leakage")

        rf_params, rf_inner_rmse, rf_study = tune_rf(train, outer_fold)
        xgb_params, xgb_inner_rmse, xgb_study = tune_xgb(train, outer_fold)
        rf_param_rows.append({
            "Task": "UTS", "Outer_Fold": outer_fold, "Features": "|".join(REFINED5),
            "Inner_Best_RMSE": rf_inner_rmse,
            **{f"Param_{key}": value for key, value in rf_params.items()},
        })
        xgb_param_rows.append({
            "Task": "UTS", "Outer_Fold": outer_fold, "Features": "|".join(REFINED5),
            "Inner_Best_RMSE": xgb_inner_rmse,
            **{f"Param_{key}": value for key, value in xgb_params.items()},
        })
        for model_name, study in (("RandomForest", rf_study), ("XGBoost", xgb_study)):
            for trial in study.trials:
                trial_rows.append({
                    "Model": model_name,
                    "Outer_Fold": outer_fold,
                    "Trial": trial.number,
                    "Inner_RMSE": trial.value,
                    **trial.params,
                })

        # Model comparison on the same five features and the same outer test rows.
        pred_rf, pred_xgb, pred_ensemble = fit_ensemble(
            train, test, REFINED5, rf_params, xgb_params, SEED + outer_fold
        )
        comparison_models = {
            "DummyMedian": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="median")),
            ]),
            "Ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
            "ExtraTrees": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=SEED + outer_fold,
                    n_jobs=4,
                )),
            ]),
        }
        prediction_map = {"RandomForest": pred_rf, "XGBoost": pred_xgb, "RF+XGB ensemble": pred_ensemble}
        for name, model in comparison_models.items():
            model.fit(train[REFINED5], train[TARGET])
            prediction_map[name] = model.predict(test[REFINED5])
        for name, prediction in prediction_map.items():
            model_prediction_parts.append(pd.DataFrame({
                "Outer_Fold": outer_fold,
                "Model": name,
                "Model_Row_ID": test["Model_Row_ID"],
                "Source_Group": test["Source_Group"],
                "y_true": test[TARGET],
                "y_pred": prediction,
            }))

        # Nested feature decision; all selection occurs within the outer training set.
        feature_inner_rows = []
        splitter = GroupKFold(n_splits=N_INNER_SPLITS)
        for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(train, train[TARGET], groups=train["Source_Group"])
        ):
            inner_train = train.iloc[train_idx].copy()
            inner_valid = train.iloc[valid_idx].copy()
            for strategy, features in FEATURE_CONFIGS.items():
                _, _, prediction = fit_ensemble(
                    inner_train, inner_valid, features, rf_params, xgb_params,
                    SEED + outer_fold * 100 + inner_fold,
                )
                feature_inner_rows.append({
                    "Strategy": strategy,
                    "Inner_Fold": inner_fold,
                    "RMSE": metrics(inner_valid[TARGET], prediction)["RMSE"],
                    "Source_Macro_RMSE": source_macro_rmse(
                        inner_valid[TARGET], prediction, inner_valid["Source_Group"]
                    ),
                })
        feature_inner = pd.DataFrame(feature_inner_rows)
        feature_summary = feature_inner.groupby("Strategy", as_index=False).agg(
            Source_Macro_RMSE_Mean=("Source_Macro_RMSE", "mean"),
            Source_Macro_RMSE_SD=("Source_Macro_RMSE", "std"),
        )
        feature_summary["Source_Macro_RMSE_Mean_SE"] = (
            feature_summary["Source_Macro_RMSE_SD"] / np.sqrt(N_INNER_SPLITS)
        )
        selected_feature, threshold, eligible = select_one_se(feature_summary, FEATURE_COMPLEXITY)
        feature_selection_rows.append({
            "Outer_Fold": outer_fold,
            "Selected_Strategy": selected_feature,
            "Selected_Features": "|".join(FEATURE_CONFIGS[selected_feature]),
            "One_SE_Threshold": threshold,
            "Eligible_Strategies": eligible,
        })
        for strategy, features in FEATURE_CONFIGS.items():
            _, _, prediction = fit_ensemble(
                train, test, features, rf_params, xgb_params, SEED + outer_fold
            )
            feature_prediction_parts.append(pd.DataFrame({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Features": "|".join(features),
                "Model_Row_ID": test["Model_Row_ID"],
                "Source_Group": test["Source_Group"],
                "y_true": test[TARGET],
                "y_pred": prediction,
            }))

        # Nested augmentation decision on the final five-feature representation.
        augmentation_inner_rows = []
        for inner_fold, (train_idx, valid_idx) in enumerate(
            splitter.split(train, train[TARGET], groups=train["Source_Group"])
        ):
            inner_train = train.iloc[train_idx].copy()
            inner_valid = train.iloc[valid_idx].copy()
            for strategy_index, strategy in enumerate(AUGMENTATION_ORDER):
                prediction = fit_predict_augmented(
                    aug, inner_train, inner_valid, strategy, rf_params, xgb_params,
                    SEED + outer_fold * 10000 + inner_fold * 100 + strategy_index * 10,
                )
                augmentation_inner_rows.append({
                    "Strategy": strategy,
                    "Inner_Fold": inner_fold,
                    "RMSE": metrics(inner_valid[TARGET], prediction)["RMSE"],
                    "Source_Macro_RMSE": source_macro_rmse(
                        inner_valid[TARGET], prediction, inner_valid["Source_Group"]
                    ),
                })
        augmentation_inner = pd.DataFrame(augmentation_inner_rows)
        augmentation_summary = augmentation_inner.groupby("Strategy", as_index=False).agg(
            Source_Macro_RMSE_Mean=("Source_Macro_RMSE", "mean"),
            Source_Macro_RMSE_SD=("Source_Macro_RMSE", "std"),
        )
        augmentation_summary["Source_Macro_RMSE_Mean_SE"] = (
            augmentation_summary["Source_Macro_RMSE_SD"] / np.sqrt(N_INNER_SPLITS)
        )
        selected_aug, threshold, eligible = select_one_se(
            augmentation_summary, AUGMENTATION_ORDER
        )
        augmentation_selection_rows.append({
            "Outer_Fold": outer_fold,
            "Selected_Strategy": selected_aug,
            "One_SE_Threshold": threshold,
            "Eligible_Strategies": eligible,
        })
        for strategy_index, strategy in enumerate(AUGMENTATION_ORDER):
            prediction = fit_predict_augmented(
                aug, train, test, strategy, rf_params, xgb_params,
                SEED + 500000 + outer_fold * 100 + strategy_index * 10,
            )
            augmentation_prediction_parts.append(pd.DataFrame({
                "Outer_Fold": outer_fold,
                "Strategy": strategy,
                "Model_Row_ID": test["Model_Row_ID"],
                "Source_Group": test["Source_Group"],
                "y_true": test[TARGET],
                "y_pred": prediction,
            }))
        print(
            f"fold={outer_fold}: RF inner={rf_inner_rmse:.2f}, XGB inner={xgb_inner_rmse:.2f}, "
            f"feature={selected_feature}, augmentation={selected_aug}"
        )

    model_predictions = pd.concat(model_prediction_parts, ignore_index=True)
    feature_predictions = pd.concat(feature_prediction_parts, ignore_index=True)
    augmentation_predictions = pd.concat(augmentation_prediction_parts, ignore_index=True)

    model_summary = []
    for model, part in model_predictions.groupby("Model", sort=False):
        model_summary.append({
            "Model": model,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"], part["y_pred"]),
            "Source_Macro_RMSE": source_macro_rmse(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    model_summary = pd.DataFrame(model_summary).sort_values("R2")

    feature_summary_rows = []
    for strategy, part in feature_predictions.groupby("Strategy", sort=False):
        feature_summary_rows.append({
            "Configuration": strategy,
            "Features": part["Features"].iloc[0],
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"], part["y_pred"]),
            "Source_Macro_RMSE": source_macro_rmse(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    feature_summary = pd.DataFrame(feature_summary_rows)

    augmentation_summary_rows = []
    for strategy, part in augmentation_predictions.groupby("Strategy", sort=False):
        augmentation_summary_rows.append({
            "Configuration": strategy,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"], part["y_pred"]),
            "Source_Macro_RMSE": source_macro_rmse(part["y_true"], part["y_pred"], part["Source_Group"]),
        })
    augmentation_summary = pd.DataFrame(augmentation_summary_rows)

    final_oof = model_predictions.loc[
        model_predictions["Model"].eq("RF+XGB ensemble")
    ].copy()
    final_metrics = pd.DataFrame([{
        "Task": "UTS",
        "Selected_Model": "ScopeClean_Refined5_RF_XGBoost_Retuned",
        "Rows": len(final_oof),
        "Sources": final_oof["Source_Group"].nunique(),
        "Features": "|".join(REFINED5),
        **metrics(final_oof["y_true"], final_oof["y_pred"]),
        "Source_Macro_RMSE": source_macro_rmse(
            final_oof["y_true"], final_oof["y_pred"], final_oof["Source_Group"]
        ),
    }])

    pd.DataFrame(rf_param_rows).to_csv(OUT / "rf_params_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(xgb_param_rows).to_csv(OUT / "xgb_params_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trial_rows).to_csv(OUT / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    model_predictions.to_csv(OUT / "model_comparison_oof_predictions.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    feature_predictions.to_csv(OUT / "feature_configuration_oof_predictions.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(OUT / "feature_configuration_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_selection_rows).to_csv(OUT / "feature_nested_selection.csv", index=False, encoding="utf-8-sig")
    augmentation_predictions.to_csv(OUT / "augmentation_configuration_oof_predictions.csv", index=False, encoding="utf-8-sig")
    augmentation_summary.to_csv(OUT / "augmentation_configuration_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(augmentation_selection_rows).to_csv(OUT / "augmentation_nested_selection.csv", index=False, encoding="utf-8-sig")
    final_oof.to_csv(OUT / "final_oof_predictions.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(OUT / "final_metrics.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "input": str(INPUT),
        "rows": len(data),
        "sources": int(data["Source_Group"].nunique()),
        "main_features": REFINED5,
        "main_model": "equal-weight RF + XGBoost ensemble",
        "outer_validation": "five fixed source-exclusive folds",
        "inner_tuning": "three-fold GroupKFold(Source_Group)",
        "trials_per_model_per_outer_fold": N_TRIALS,
        "feature_selection": "nested one-standard-error source-macro RMSE audit",
        "augmentation_selection": "nested one-standard-error source-macro RMSE audit",
        "old_689_row_results_used": False,
        "seed": SEED,
    }
    (OUT / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFINAL")
    print(final_metrics.to_string(index=False))
    print("\nMODEL COMPARISON")
    print(model_summary.to_string(index=False))
    print("\nFEATURE COMPARISON")
    print(feature_summary.to_string(index=False))
    print("\nAUGMENTATION COMPARISON")
    print(augmentation_summary.to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
