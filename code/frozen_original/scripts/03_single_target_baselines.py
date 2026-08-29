from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
from src.data_utils import ensure_output_dirs, write_json


warnings.filterwarnings("ignore")

ANALYSIS_MODE = "strict"
TASKS = ["YS", "UTS", "EL"]
FEATURE_SETS = {
    "composition_core": config.MODEL_COMPOSITION_FEATURES,
    "composition_plus_znmg": config.MODEL_COMPOSITION_FEATURES + config.MODEL_DERIVED_FEATURES,
}


def select_features_on_training_fold(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    selected = []
    min_nonzero = max(5, math.ceil(len(df) * 0.01))
    for col in candidates:
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().mean() >= 0.80 and s.nunique(dropna=True) >= 2 and s.fillna(0).ne(0).sum() >= min_nonzero:
            selected.append(col)
    if not selected:
        raise ValueError("No features survived the training-fold filter")
    return selected


def make_models(seed: int):
    return {
        "DummyMedian": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_features=0.8, min_samples_leaf=2,
                random_state=seed, n_jobs=4,
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300, max_features=0.8, min_samples_leaf=2,
                random_state=seed, n_jobs=4,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=4,
                min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, objective="reg:squarederror",
                random_state=seed, n_jobs=4, verbosity=0,
            )),
        ]),
    }


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def source_macro_metrics(pred: pd.DataFrame) -> dict:
    per_source = []
    for _, part in pred.groupby("Source_Group"):
        err = part["y_pred"] - part["y_true"]
        per_source.append({
            "MAE": np.abs(err).mean(),
            "RMSE": np.sqrt(np.square(err).mean()),
        })
    summary = pd.DataFrame(per_source)
    return {
        "Source_Macro_MAE": summary["MAE"].mean(),
        "Source_Macro_RMSE": summary["RMSE"].mean(),
    }


def main():
    ensure_output_dirs()
    out = config.OUTPUT_DIRS["single"] / "baseline_strict"
    out.mkdir(parents=True, exist_ok=True)
    fold_records = []
    prediction_parts = []
    feature_records = []

    for task in TASKS:
        path = config.OUTPUT_DIRS["processed"] / ANALYSIS_MODE / f"{task}_with_outer_folds.csv"
        df = pd.read_csv(path)
        target = config.TARGET_COLUMNS[task]
        df[target] = pd.to_numeric(df[target], errors="coerce")
        if df[target].isna().any():
            raise ValueError(f"{task}: target contains missing values")

        for feature_set, candidates in FEATURE_SETS.items():
            for fold in sorted(df["Outer_Fold"].unique()):
                train = df.loc[df["Outer_Fold"].ne(fold)].copy()
                test = df.loc[df["Outer_Fold"].eq(fold)].copy()
                overlap = set(train["Source_Group"]).intersection(test["Source_Group"])
                if overlap:
                    raise AssertionError(f"Source leakage in {task}, fold={fold}: {sorted(overlap)[:5]}")

                selected = select_features_on_training_fold(train, candidates)
                X_train = train[selected].apply(pd.to_numeric, errors="coerce")
                X_test = test[selected].apply(pd.to_numeric, errors="coerce")
                y_train = train[target].to_numpy()
                y_test = test[target].to_numpy()

                for model_name, model in make_models(config.RANDOM_SEED + int(fold)).items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    fold_records.append({
                        "Task": task,
                        "Feature_Set": feature_set,
                        "Model": model_name,
                        "Outer_Fold": int(fold),
                        "Train_Rows": len(train),
                        "Test_Rows": len(test),
                        "Train_Sources": train["Source_Group"].nunique(),
                        "Test_Sources": test["Source_Group"].nunique(),
                        "N_Features": len(selected),
                        **regression_metrics(y_test, y_pred),
                    })
                    prediction_parts.append(pd.DataFrame({
                        "Task": task,
                        "Feature_Set": feature_set,
                        "Model": model_name,
                        "Outer_Fold": int(fold),
                        "Model_Row_ID": test["Model_Row_ID"].values,
                        "Source_Group": test["Source_Group"].values,
                        "y_true": y_test,
                        "y_pred": y_pred,
                    }))
                    feature_records.append({
                        "Task": task,
                        "Feature_Set": feature_set,
                        "Model": model_name,
                        "Outer_Fold": int(fold),
                        "Selected_Features": "|".join(selected),
                    })

    folds = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selected_features = pd.DataFrame(feature_records)

    summary_rows = []
    for keys, part in predictions.groupby(["Task", "Feature_Set", "Model"]):
        task, feature_set, model = keys
        summary_rows.append({
            "Task": task,
            "Feature_Set": feature_set,
            "Model": model,
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **regression_metrics(part["y_true"], part["y_pred"]),
            **source_macro_metrics(part),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["Task", "RMSE"])

    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    selected_features.to_csv(out / "selected_features_by_fold.csv", index=False, encoding="utf-8-sig")
    write_json(out / "run_config.json", {
        "analysis_mode": ANALYSIS_MODE,
        "random_seed": config.RANDOM_SEED,
        "feature_sets": FEATURE_SETS,
        "preprocessing": "training-fold feature filter; median imputation; Ridge scaling",
        "data_augmentation": False,
        "hyperparameter_tuning": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, task in zip(axes, TASKS):
        part = summary.loc[summary["Task"].eq(task)].copy()
        labels = part["Model"] + "\n" + part["Feature_Set"].str.replace("composition_", "")
        order = np.arange(len(part))
        ax.bar(order, part["RMSE"], color="#4472C4")
        ax.set_xticks(order)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_title(f"{task}: source-group OOF RMSE")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "baseline_oof_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nSTRICT SINGLE-TARGET BASELINE - OOF SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
