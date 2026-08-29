from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.data_utils import ensure_output_dirs, write_json


warnings.filterwarnings("ignore")


def models(seed):
    rf = dict(n_estimators=300, max_features=0.8, min_samples_leaf=2, random_state=seed, n_jobs=4)
    return {
        "Independent_RF": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("model", MultiOutputRegressor(RandomForestRegressor(**rf), n_jobs=1))]),
        "Joint_RF": Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("model", RandomForestRegressor(**rf))]),
        "Joint_ExtraTrees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                      ("model", ExtraTreesRegressor(**rf))]),
        "PLS_5": Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler()), ("model", PLSRegression(n_components=5, scale=False))]),
        "MultiTask_ElasticNet": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                          ("scaler", StandardScaler()),
                                          ("model", MultiTaskElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20000,
                                                                         random_state=seed))]),
    }


def main():
    ensure_output_dirs()
    out = config.OUTPUT_DIRS["multi"] / "baseline_strict_triple"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "TRIPLE_with_outer_folds.csv")
    features = [c for c in config.PRIMARY_FIXED_FEATURES if c in df]
    target_map = {t: config.TARGET_COLUMNS[t] for t in ("YS", "UTS", "EL")}
    targets = list(target_map.values())
    df[features + targets] = df[features + targets].apply(pd.to_numeric, errors="coerce")
    preds, fold_metrics = [], []

    for fold in sorted(df.Outer_Fold.unique()):
        train = df[df.Outer_Fold != fold].copy(); test = df[df.Outer_Fold == fold].copy()
        if set(train.Source_Group) & set(test.Source_Group):
            raise AssertionError("Source leakage")
        y_mean = train[targets].mean().to_numpy(); y_std = train[targets].std().replace(0, 1).to_numpy()
        y_train_z = (train[targets].to_numpy() - y_mean) / y_std
        for name, model in models(config.RANDOM_SEED + int(fold)).items():
            model.fit(train[features], y_train_z)
            pred = np.asarray(model.predict(test[features])) * y_std + y_mean
            part = test[["Model_Row_ID", "Source_Group", "Outer_Fold"]].copy()
            part["Model"] = name
            for j, task in enumerate(("YS", "UTS", "EL")):
                part[f"true_{task}"] = test[target_map[task]].to_numpy()
                part[f"pred_{task}"] = pred[:, j]
                fold_metrics.append({"Model": name, "Outer_Fold": int(fold), "Task": task,
                                     "R2": r2_score(part[f"true_{task}"], part[f"pred_{task}"]),
                                     "RMSE": np.sqrt(mean_squared_error(part[f"true_{task}"], part[f"pred_{task}"])),
                                     "MAE": mean_absolute_error(part[f"true_{task}"], part[f"pred_{task}"])})
            preds.append(part)

    pred = pd.concat(preds, ignore_index=True)
    summary = []
    for model, part in pred.groupby("Model"):
        model_r2 = []
        for task in ("YS", "UTS", "EL"):
            r2 = r2_score(part[f"true_{task}"], part[f"pred_{task}"])
            model_r2.append(r2)
            summary.append({"Model": model, "Task": task, "Rows": len(part),
                            "R2": r2,
                            "RMSE": np.sqrt(mean_squared_error(part[f"true_{task}"], part[f"pred_{task}"])),
                            "MAE": mean_absolute_error(part[f"true_{task}"], part[f"pred_{task}"])})
        summary.append({"Model": model, "Task": "MEAN", "Rows": len(part), "R2": np.mean(model_r2),
                        "RMSE": np.nan, "MAE": np.nan})
    summary = pd.DataFrame(summary).sort_values(["Task", "R2"], ascending=[True, False])
    pred.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metrics).to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "run_config.json", {"rows": len(df), "features": features,
                                          "targets_standardized_within_outer_train": True,
                                          "augmentation": False, "tuning": False})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
