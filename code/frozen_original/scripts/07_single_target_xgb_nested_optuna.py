from __future__ import annotations

import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

import config
from src.data_utils import ensure_output_dirs, write_json


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
TASKS = ["YS", "UTS", "EL"]
N_TRIALS = 15


def model_from(params, seed):
    return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=4, verbosity=0, **params)


def scores(y, p):
    return {"R2": r2_score(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)),
            "MAE": mean_absolute_error(y, p)}


def main():
    ensure_output_dirs()
    out = config.OUTPUT_DIRS["single"] / "nested_optuna_xgb_strict"
    out.mkdir(parents=True, exist_ok=True)
    pred_parts, fold_rows, trial_rows = [], [], []
    for task in TASKS:
        df = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv")
        target = config.TARGET_COLUMNS[task]
        features = config.PRIMARY_FIXED_FEATURES
        df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors="coerce")
        for fold in sorted(df.Outer_Fold.unique()):
            train = df[df.Outer_Fold != fold].copy(); test = df[df.Outer_Fold == fold].copy()
            groups = train.Source_Group.to_numpy(); y = train[target].to_numpy(); X = train[features]
            inner = GroupKFold(n_splits=3)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                }
                rmses = []
                for itr, iva in inner.split(X, y, groups):
                    imputer = SimpleImputer(strategy="median")
                    xtr = imputer.fit_transform(X.iloc[itr]); xva = imputer.transform(X.iloc[iva])
                    model = model_from(params, config.RANDOM_SEED + int(fold))
                    model.fit(xtr, y[itr])
                    rmses.append(np.sqrt(mean_squared_error(y[iva], model.predict(xva))))
                return float(np.mean(rmses))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED + int(fold)))
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            imputer = SimpleImputer(strategy="median")
            xtr = imputer.fit_transform(train[features]); xte = imputer.transform(test[features])
            model = model_from(study.best_params, config.RANDOM_SEED + int(fold))
            model.fit(xtr, train[target]); pred = model.predict(xte)
            fold_rows.append({"Task": task, "Outer_Fold": int(fold), "Inner_Best_RMSE": study.best_value,
                              "Train_Rows": len(train), "Test_Rows": len(test), **scores(test[target], pred),
                              **{f"Param_{k}": v for k, v in study.best_params.items()}})
            pred_parts.append(pd.DataFrame({"Task": task, "Outer_Fold": int(fold),
                                            "Model_Row_ID": test.Model_Row_ID.values,
                                            "Source_Group": test.Source_Group.values,
                                            "y_true": test[target].values, "y_pred": pred}))
            for tr in study.trials:
                trial_rows.append({"Task": task, "Outer_Fold": int(fold), "Trial": tr.number,
                                   "Inner_RMSE": tr.value, **tr.params})
            print(f"{task} fold {fold}: inner={study.best_value:.3f}, outer={fold_rows[-1]['RMSE']:.3f}")

    pred = pd.concat(pred_parts, ignore_index=True)
    summary = []
    for task, part in pred.groupby("Task"):
        summary.append({"Task": task, "Rows": len(part), "Sources": part.Source_Group.nunique(),
                        **scores(part.y_true, part.y_pred)})
    summary = pd.DataFrame(summary)
    pred.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trial_rows).to_csv(out / "optuna_inner_trials.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "run_config.json", {"model": "XGBoost", "features": config.PRIMARY_FIXED_FEATURES,
                                          "outer": "fixed Source_Group folds", "inner": "3-fold GroupKFold",
                                          "n_trials": N_TRIALS, "augmentation": False})
    print("\nXGBOOST NESTED OOF SUMMARY\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
