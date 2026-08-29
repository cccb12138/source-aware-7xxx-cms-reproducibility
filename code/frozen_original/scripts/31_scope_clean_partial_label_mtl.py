from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
torch.set_num_threads(4)

PROJECT_ROOT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
YS_EL_ROOT = Path(r"F:\CC\outputs\ys_el_scope_audit")
UTS_ROOT = PROJECT_ROOT / "results" / "uts_systematic_scope_audit"
OUTPUT_ROOT = Path(r"F:\CC\outputs\scope_clean_partial_label_mtl")
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


TASKS = ["YS", "UTS", "EL"]
TARGETS = [config.TARGET_COLUMNS[task] for task in TASKS]
MASKS = [config.MASK_COLUMNS[task] for task in TASKS]
FEATURE_SETS = {
    "core10": list(config.PRIMARY_FIXED_FEATURES),
    "refined5": ["Zn", "Mg", "Cu", "Fe", "Zr"],
}
SEEDS = [20260805, 20260806, 20260807]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def key_series(frame: pd.DataFrame) -> pd.Series:
    return frame[["Dataset", "Original_Sample_ID"]].fillna("").astype(str).agg("||".join, axis=1)


def build_clean_mtl() -> tuple[pd.DataFrame, pd.DataFrame]:
    mtl = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "strict" / "MTL_with_outer_folds.csv")
    mtl_key = key_series(mtl)
    clean_paths = {
        "YS": YS_EL_ROOT / "YS_scope_clean_with_outer_folds.csv",
        "UTS": UTS_ROOT / "UTS_scope_clean_with_outer_folds.csv",
        "EL": YS_EL_ROOT / "EL_scope_clean_with_outer_folds.csv",
    }
    checks = []
    for task, target, mask in zip(TASKS, TARGETS, MASKS):
        clean = pd.read_csv(clean_paths[task])
        clean_key = key_series(clean)
        if clean_key.duplicated().any():
            raise AssertionError(f"{task}: duplicate clean sample keys")
        key_to_target = pd.Series(
            pd.to_numeric(clean[target], errors="coerce").to_numpy(), index=clean_key
        )
        observed = mtl_key.isin(set(clean_key))
        mtl[mask] = observed.astype(int)
        mtl[target] = np.where(observed, mtl_key.map(key_to_target), np.nan)
        matched_values = pd.to_numeric(mtl.loc[observed, target], errors="coerce")
        checks.append(
            {
                "Task": task,
                "Expected_Clean_Labels": len(clean),
                "MTL_Observed_Labels": int(observed.sum()),
                "Unique_Clean_Keys": int(clean_key.nunique()),
                "Missing_Target_After_Merge": int(matched_values.isna().sum()),
                "Exact_Count_Match": bool(observed.sum() == len(clean)),
            }
        )
    mtl["Observed_Target_Count"] = mtl[MASKS].sum(axis=1).astype(int)
    mtl = mtl.loc[mtl["Observed_Target_Count"].gt(0)].copy()
    mtl["Triple_Target_Complete"] = mtl["Observed_Target_Count"].eq(3).astype(int)
    if mtl.groupby("Source_Group")["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("Source leakage in clean MTL data")
    if not all(row["Exact_Count_Match"] for row in checks):
        raise AssertionError("Clean target counts do not match MTL masks")
    return mtl, pd.DataFrame(checks)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def source_macro(part: pd.DataFrame) -> dict[str, float]:
    values = []
    for _, source_part in part.groupby("Source_Group"):
        error = source_part["y_pred"] - source_part["y_true"]
        values.append((error.abs().mean(), np.sqrt(np.square(error).mean())))
    values = np.asarray(values)
    return {
        "Source_Macro_MAE": float(values[:, 0].mean()),
        "Source_Macro_RMSE": float(values[:, 1].mean()),
    }


def run_models(data: pd.DataFrame, mtl_module):
    prediction_rows = []
    training_rows = []
    for feature_set, features in FEATURE_SETS.items():
        work = data.copy()
        work[features + TARGETS + MASKS] = work[features + TARGETS + MASKS].apply(
            pd.to_numeric, errors="coerce"
        )
        for outer in sorted(work["Outer_Fold"].unique()):
            outer = int(outer)
            train = work.loc[work["Outer_Fold"].ne(outer)].copy()
            test = work.loc[work["Outer_Fold"].eq(outer)].copy()
            remaining_folds = sorted(train["Outer_Fold"].unique())
            val_fold = remaining_folds[outer % len(remaining_folds)]
            fit = train.loc[train["Outer_Fold"].ne(val_fold)].copy()
            val = train.loc[train["Outer_Fold"].eq(val_fold)].copy()
            if set(train["Source_Group"]) & set(test["Source_Group"]):
                raise AssertionError("Outer source leakage")

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(imputer.fit_transform(fit[features]))
            x_val = scaler.transform(imputer.transform(val[features]))
            y_mean = np.asarray(
                [fit.loc[fit[MASKS[j]].eq(1), TARGETS[j]].mean() for j in range(3)]
            )
            y_std = np.asarray(
                [fit.loc[fit[MASKS[j]].eq(1), TARGETS[j]].std() for j in range(3)]
            )
            y_std = np.where(y_std > 0, y_std, 1.0)

            def tensors(part, x_values, local_mean, local_std):
                fill = pd.Series(local_mean, index=TARGETS)
                y_values = part[TARGETS].fillna(fill).to_numpy(dtype=float)
                return (
                    torch.tensor(x_values, dtype=torch.float32),
                    torch.tensor((y_values - local_mean) / local_std, dtype=torch.float32),
                    torch.tensor(part[MASKS].to_numpy(dtype=float), dtype=torch.float32),
                )

            fit_tensors = tensors(fit, x_fit, y_mean, y_std)
            val_tensors = tensors(val, x_val, y_mean, y_std)
            epochs = []
            for seed in SEEDS:
                epoch, validation_loss = mtl_module.train_with_early_stop(
                    *fit_tensors, *val_tensors, seed
                )
                epochs.append(epoch)
                training_rows.append(
                    {
                        "Feature_Set": feature_set,
                        "Outer_Fold": outer,
                        "Seed": seed,
                        "Best_Epoch": epoch,
                        "Validation_Loss": validation_loss,
                    }
                )

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
            x_test = scaler.transform(imputer.transform(test[features]))
            y_mean = np.asarray(
                [train.loc[train[MASKS[j]].eq(1), TARGETS[j]].mean() for j in range(3)]
            )
            y_std = np.asarray(
                [train.loc[train[MASKS[j]].eq(1), TARGETS[j]].std() for j in range(3)]
            )
            y_std = np.where(y_std > 0, y_std, 1.0)
            train_tensors = tensors(train, x_train, y_mean, y_std)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            ensemble = []
            for seed, epoch in zip(SEEDS, epochs):
                model = mtl_module.train_full(*train_tensors, epoch, seed)
                model.eval()
                with torch.no_grad():
                    ensemble.append(model(x_test_tensor).numpy() * y_std + y_mean)
            mtl_pred = np.mean(ensemble, axis=0)

            for task_index, task in enumerate(TASKS):
                valid_test = test[MASKS[task_index]].eq(1).to_numpy()
                valid_indices = np.where(valid_test)[0]
                for index in valid_indices:
                    prediction_rows.append(
                        {
                            "Feature_Set": feature_set,
                            "Model": "Masked_MTL",
                            "Task": task,
                            "Outer_Fold": outer,
                            "Model_Row_ID": test.iloc[index]["Model_Row_ID"],
                            "Source_Group": test.iloc[index]["Source_Group"],
                            "y_true": test.iloc[index][TARGETS[task_index]],
                            "y_pred": mtl_pred[index, task_index],
                        }
                    )

                observed_train = train[MASKS[task_index]].eq(1)
                rf_imputer = SimpleImputer(strategy="median")
                x_rf_train = rf_imputer.fit_transform(train.loc[observed_train, features])
                x_rf_test = rf_imputer.transform(test.loc[valid_test, features])
                rf = RandomForestRegressor(
                    n_estimators=300,
                    max_features=0.8,
                    min_samples_leaf=2,
                    random_state=config.RANDOM_SEED + outer,
                    n_jobs=4,
                )
                rf.fit(x_rf_train, train.loc[observed_train, TARGETS[task_index]])
                rf_pred = rf.predict(x_rf_test)
                for position, index in enumerate(valid_indices):
                    prediction_rows.append(
                        {
                            "Feature_Set": feature_set,
                            "Model": "Independent_RF",
                            "Task": task,
                            "Outer_Fold": outer,
                            "Model_Row_ID": test.iloc[index]["Model_Row_ID"],
                            "Source_Group": test.iloc[index]["Source_Group"],
                            "y_true": test.iloc[index][TARGETS[task_index]],
                            "y_pred": rf_pred[position],
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for (feature_set, model, task), part in predictions.groupby(
        ["Feature_Set", "Model", "Task"]
    ):
        summaries.append(
            {
                "Feature_Set": feature_set,
                "Model": model,
                "Task": task,
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                **metrics(part["y_true"], part["y_pred"]),
                **source_macro(part),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["Task", "Feature_Set", "R2"], ascending=[True, True, False])
    return predictions, pd.DataFrame(training_rows), summary


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data, label_checks = build_clean_mtl()
    mtl_module = load_module("scope_clean_mtl_base", PROJECT_ROOT / "08_partial_label_mtl.py")
    predictions, training, summary = run_models(data, mtl_module)
    data.to_csv(OUTPUT_ROOT / "MTL_scope_clean_with_outer_folds.csv", index=False, encoding="utf-8-sig")
    label_checks.to_csv(OUTPUT_ROOT / "label_count_checks.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_ROOT / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    training.to_csv(OUTPUT_ROOT / "training_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "run_config.json").write_text(
        json.dumps(
            {
                "rows": len(data),
                "sources": int(data["Source_Group"].nunique()),
                "label_counts": {task: int(data[mask].sum()) for task, mask in zip(TASKS, MASKS)},
                "feature_sets": FEATURE_SETS,
                "model": "masked shared MLP [64, 32] with three heads",
                "comparison": "independent RandomForest using identical folds and feature set",
                "seeds": SEEDS,
                "augmentation": False,
                "outer_folds": "existing source-exclusive folds",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("LABEL CHECKS")
    print(label_checks.to_string(index=False))
    print("\nCLEAN MTL DATA")
    print(
        pd.DataFrame(
            [
                {
                    "Rows": len(data),
                    "Sources": data["Source_Group"].nunique(),
                    "YS": int(data["Mask_YS"].sum()),
                    "UTS": int(data["Mask_UTS"].sum()),
                    "EL": int(data["Mask_EL"].sum()),
                    "Triple_Complete": int(data["Triple_Target_Complete"].sum()),
                }
            ]
        ).to_string(index=False)
    )
    print("\nMTL VS INDEPENDENT RF")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
