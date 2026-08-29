from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = ["Zn", "Mg", "Cu", "Fe", "Zr"]
SEED = 20260731
PARENT_DOIS = [
    "10.1016/j.commatsci.2025.114121",
    "10.1016/j.actamat.2024.119873",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 and np.ptp(y_true) > 0 else np.nan
    return {
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def aggregate_parameters():
    root = config.OUTPUT_DIRS["single"]
    rf = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    rf = rf.loc[rf["Task"].eq(TASK)].copy()
    xgb = xgb.loc[xgb["Task"].eq(TASK)].copy()
    rf_row = rf.iloc[0].copy()
    xgb_row = xgb.iloc[0].copy()
    for column in [c for c in rf.columns if c.startswith("Param_")]:
        rf_row[column] = pd.to_numeric(rf[column], errors="coerce").median()
    for column in [c for c in xgb.columns if c.startswith("Param_")]:
        xgb_row[column] = pd.to_numeric(xgb[column], errors="coerce").median()
    for column in ("Param_max_depth", "Param_min_samples_leaf", "Param_min_samples_split"):
        rf_row[column] = int(round(float(rf_row[column])))
    for column in ("Param_n_estimators", "Param_max_depth", "Param_min_child_weight"):
        xgb_row[column] = int(round(float(xgb_row[column])))
    return rf_row, xgb_row


def fit_predict(train, test, model_module, rf_row, xgb_row, seed):
    rf = model_module.rf_tuned(rf_row, seed)
    xgb = model_module.xgb_tuned(xgb_row, seed)
    rf.fit(train[FEATURES], train[TARGET])
    xgb.fit(train[FEATURES], train[TARGET])
    pred_rf = rf.predict(test[FEATURES])
    pred_xgb = xgb.predict(test[FEATURES])
    return pred_rf, pred_xgb, (pred_rf + pred_xgb) / 2.0


def range_diagnostics(train, test):
    train_values = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    test_values = test[FEATURES].apply(pd.to_numeric, errors="coerce")
    mins = train_values.min()
    maxs = train_values.max()
    outside = pd.DataFrame(
        {
            feature: test_values[feature].lt(mins[feature])
            | test_values[feature].gt(maxs[feature])
            for feature in FEATURES
        }
    )
    return outside.any(axis=1).mean(), "|".join(
        feature for feature in FEATURES if outside[feature].any()
    )


def evaluate_holdouts(data, model_module, rf_row, xgb_row):
    metric_rows = []
    prediction_rows = []
    datasets = sorted(data["Dataset"].dropna().astype(str).unique())
    blocks = [
        {
            "Scheme": "Leave_One_Dataset_Out",
            "Block": dataset,
            "Mask": data["Dataset"].astype(str).eq(dataset),
        }
        for dataset in datasets
    ]
    blocks.extend(
        {
            "Scheme": "Leave_Parent_DOI_Out",
            "Block": doi,
            "Mask": data["DOI"].fillna("").astype(str).eq(doi),
        }
        for doi in PARENT_DOIS
    )

    for block_index, spec in enumerate(blocks):
        test = data.loc[spec["Mask"]].copy()
        train = data.loc[~spec["Mask"]].copy()
        if len(test) == 0:
            continue
        source_overlap = set(train["Source_Group"]) & set(test["Source_Group"])
        doi_overlap = (
            set(train["DOI"].dropna().astype(str))
            & set(test["DOI"].dropna().astype(str))
        )
        pred_rf, pred_xgb, y_pred = fit_predict(
            train, test, model_module, rf_row, xgb_row, SEED + block_index
        )
        outside_fraction, outside_features = range_diagnostics(train, test)
        values = metric_values(test[TARGET], y_pred)
        baseline_mean = np.repeat(pd.to_numeric(train[TARGET]).mean(), len(test))
        baseline = metric_values(test[TARGET], baseline_mean)
        metric_rows.append(
            {
                "Scheme": spec["Scheme"],
                "Holdout_Block": spec["Block"],
                "Train_Rows": len(train),
                "Train_Sources": train["Source_Group"].nunique(),
                "Test_Rows": len(test),
                "Test_Sources": test["Source_Group"].nunique(),
                "Test_UTS_Mean": pd.to_numeric(test[TARGET]).mean(),
                "Test_UTS_SD": pd.to_numeric(test[TARGET]).std(ddof=1),
                "Source_Group_Overlap": len(source_overlap),
                "DOI_Overlap_Count": len(doi_overlap),
                "Outside_Train_MinMax_Fraction": outside_fraction,
                "Outside_Train_MinMax_Features": outside_features,
                **values,
                "Mean_Baseline_R2": baseline["R2"],
                "Mean_Baseline_RMSE": baseline["RMSE"],
                "Mean_Baseline_MAE": baseline["MAE"],
            }
        )
        for position, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "Scheme": spec["Scheme"],
                    "Holdout_Block": spec["Block"],
                    "Model_Row_ID": row["Model_Row_ID"],
                    "Source_Group": row["Source_Group"],
                    "Dataset": row["Dataset"],
                    "DOI": row["DOI"],
                    "y_true": row[TARGET],
                    "pred_rf": pred_rf[position],
                    "pred_xgb": pred_xgb[position],
                    "y_pred": y_pred[position],
                    "Absolute_Error": abs(y_pred[position] - row[TARGET]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def pooled_dataset_metrics(predictions):
    part = predictions.loc[predictions["Scheme"].eq("Leave_One_Dataset_Out")]
    if part["Model_Row_ID"].nunique() != len(part):
        raise AssertionError("Each row must appear once in leave-one-dataset-out predictions")
    overall = metric_values(part["y_true"], part["y_pred"])
    source_mae = (
        part.assign(Error=lambda x: (x["y_pred"] - x["y_true"]).abs())
        .groupby("Source_Group")["Error"]
        .mean()
    )
    dataset_mae = (
        part.assign(Error=lambda x: (x["y_pred"] - x["y_true"]).abs())
        .groupby("Dataset")["Error"]
        .mean()
    )
    return pd.DataFrame(
        [
            {
                "Scheme": "Leave_One_Dataset_Out_Pooled",
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                "Datasets": part["Dataset"].nunique(),
                **overall,
                "Source_Macro_MAE": source_mae.mean(),
                "Dataset_Macro_MAE": dataset_mae.mean(),
            }
        ]
    )


def source_group_context(data):
    rows = []
    for dataset, part in data.groupby("Dataset", sort=False):
        rows.append(
            {
                "Dataset": dataset,
                "Rows": len(part),
                "Source_Groups": part["Source_Group"].nunique(),
                "DOIs": part.loc[
                    part["DOI"].notna() & part["DOI"].astype(str).str.strip().ne(""),
                    "DOI",
                ].nunique(),
                "Outer_Folds_Current": part["Outer_Fold"].nunique(),
                "UTS_Mean": pd.to_numeric(part[TARGET]).mean(),
                "UTS_SD": pd.to_numeric(part[TARGET]).std(ddof=1),
                "Current_Source_Group_Testing_Context": (
                    "within-dataset generalization"
                    if part["Outer_Fold"].nunique() > 1
                    else "entire dataset already confined to one fold"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Rows", ascending=False)


def save_figure(out, metrics_frame):
    part = metrics_frame.loc[
        metrics_frame["Scheme"].eq("Leave_One_Dataset_Out")
    ].sort_values("Test_Rows", ascending=False)
    labels = part["Holdout_Block"].astype(str).tolist()
    y = np.arange(len(part))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(y, part["R2"], color=["#C44E52" if x < 0 else "#4C72B0" for x in part["R2"]])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("R²")
    axes[0].set_title("Leave-one-dataset-out R²")
    axes[1].barh(y, part["MAE"], color="#4C72B0")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE (MPa)")
    axes[1].set_title("Leave-one-dataset-out MAE")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "leave_one_dataset_out_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    data = pd.read_csv(
        config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv"
    )
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(
        pd.to_numeric, errors="coerce"
    )
    model_module = load_module(
        "model_builders",
        config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py",
    )
    rf_row, xgb_row = aggregate_parameters()
    metrics_frame, predictions = evaluate_holdouts(
        data, model_module, rf_row, xgb_row
    )
    pooled = pooled_dataset_metrics(predictions)
    context = source_group_context(data)

    out = config.PROJECT_ROOT / "results" / "uts_hierarchical_source_validation"
    out.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(out / "hierarchical_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "hierarchical_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(out / "leave_one_dataset_out_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    context.to_csv(out / "dataset_source_group_context.csv", index=False, encoding="utf-8-sig")
    save_figure(out, metrics_frame)

    cfg = {
        "task": TASK,
        "features": FEATURES,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "hyperparameters": "median of five pre-existing source-fold optimized configurations; frozen for all hierarchical holdouts",
        "purpose": "dataset/publication hierarchy stress test, not replacement primary performance estimate",
        "parent_dois": PARENT_DOIS,
        "seed": SEED,
    }
    (out / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("DATASET CONTEXT")
    print(context.to_string(index=False))
    print("\nHIERARCHICAL HOLDOUT METRICS")
    print(metrics_frame.to_string(index=False))
    print("\nPOOLED LEAVE-ONE-DATASET-OUT")
    print(pooled.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
