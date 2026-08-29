from __future__ import annotations

import json
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import config
from src.data_utils import ensure_output_dirs, write_json


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TASKS = ["YS", "UTS", "EL"]
N_TRIALS = 15


def build_model(params, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=4,
            **params,
        )),
    ])


def score_dict(y, pred):
    return {
        "R2": r2_score(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": mean_absolute_error(y, pred),
    }


def main():
    ensure_output_dirs()
    out = config.OUTPUT_DIRS["single"] / "nested_optuna_strict"
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    fold_rows = []
    trial_rows = []

    for task in TASKS:
        df = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv")
        target = config.TARGET_COLUMNS[task]
        features = [c for c in config.PRIMARY_FIXED_FEATURES if c in df]
        df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors="coerce")

        for outer_fold in sorted(df["Outer_Fold"].unique()):
            train = df.loc[df["Outer_Fold"].ne(outer_fold)].copy()
            test = df.loc[df["Outer_Fold"].eq(outer_fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError("Outer-fold source leakage")

            X = train[features]
            y = train[target].to_numpy()
            groups = train["Source_Group"].to_numpy()
            inner = GroupKFold(n_splits=3)

            def objective(trial):
                params = {
                    "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
                    "max_features": trial.suggest_float("max_features", 0.4, 1.0),
                }
                rmses = []
                for inner_train, inner_valid in inner.split(X, y, groups):
                    model = build_model(params, config.RANDOM_SEED + int(outer_fold))
                    model.fit(X.iloc[inner_train], y[inner_train])
                    pred = model.predict(X.iloc[inner_valid])
                    rmses.append(np.sqrt(mean_squared_error(y[inner_valid], pred)))
                return float(np.mean(rmses))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED + int(outer_fold)),
            )
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            best = study.best_params
            model = build_model(best, config.RANDOM_SEED + int(outer_fold))
            model.fit(train[features], train[target])
            pred = model.predict(test[features])
            fold_rows.append({
                "Task": task,
                "Outer_Fold": int(outer_fold),
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "Train_Sources": train["Source_Group"].nunique(),
                "Test_Sources": test["Source_Group"].nunique(),
                "Inner_Best_RMSE": study.best_value,
                **score_dict(test[target], pred),
                **{f"Param_{k}": v for k, v in best.items()},
            })
            predictions.append(pd.DataFrame({
                "Task": task,
                "Outer_Fold": int(outer_fold),
                "Model_Row_ID": test["Model_Row_ID"].values,
                "Source_Group": test["Source_Group"].values,
                "y_true": test[target].values,
                "y_pred": pred,
            }))
            for tr in study.trials:
                trial_rows.append({
                    "Task": task,
                    "Outer_Fold": int(outer_fold),
                    "Trial": tr.number,
                    "Inner_RMSE": tr.value,
                    **tr.params,
                })
            print(f"{task} outer fold {outer_fold}: inner={study.best_value:.3f}, outer={fold_rows[-1]['RMSE']:.3f}")

    pred = pd.concat(predictions, ignore_index=True)
    summary = []
    for task, part in pred.groupby("Task"):
        summary.append({
            "Task": task,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **score_dict(part["y_true"], part["y_pred"]),
        })
    summary = pd.DataFrame(summary)
    pd.DataFrame(fold_rows).to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trial_rows).to_csv(out / "optuna_inner_trials.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "run_config.json", {
        "model": "RandomForestRegressor",
        "features": config.PRIMARY_FIXED_FEATURES,
        "outer_split": "fixed Source_Group folds",
        "inner_split": "3-fold GroupKFold(Source_Group)",
        "trials_per_outer_fold": N_TRIALS,
        "augmentation": False,
    })
    print("\nNESTED SOURCE-GROUP OPTUNA SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
