from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config


warnings.filterwarnings("ignore")

TASKS = ("YS", "UTS", "EL")
N_BOOTSTRAP = 3000
SEED = 20260722


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 and np.ptp(y_true) > 0 else np.nan
    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def load_selected_predictions() -> pd.DataFrame:
    base_dir = config.OUTPUT_DIRS["single"]
    baseline = pd.read_csv(base_dir / "baseline_strict" / "oof_predictions.csv")
    baseline = baseline.loc[
        baseline["Model"].eq("RandomForest")
        & baseline["Feature_Set"].eq("composition_core")
        & baseline["Task"].isin(["YS", "EL"])
    ].copy()
    baseline["Selected_Model"] = "RandomForest"

    ensemble = pd.read_csv(base_dir / "rf_xgb_ensemble_strict" / "oof_predictions.csv")
    ensemble = ensemble.loc[ensemble["Task"].eq("UTS")].copy()
    ensemble["Selected_Model"] = "RandomForest_XGBoost_OOF_Mean"

    keep = [
        "Task", "Outer_Fold", "Model_Row_ID", "Source_Group",
        "y_true", "y_pred", "Selected_Model",
    ]
    pred = pd.concat([baseline[keep], ensemble[keep]], ignore_index=True)
    pred["Outer_Fold"] = pred["Outer_Fold"].astype(int)
    pred["Residual"] = pred["y_pred"] - pred["y_true"]
    pred["Absolute_Error"] = pred["Residual"].abs()
    pred["Squared_Error"] = pred["Residual"].pow(2)

    expected = {"YS": 307, "UTS": 689, "EL": 550}
    observed = pred.groupby("Task").size().to_dict()
    if observed != expected:
        raise AssertionError(f"Unexpected selected prediction rows: {observed}")
    if pred.duplicated(["Task", "Model_Row_ID"]).any():
        raise AssertionError("Duplicate task/row identifiers in selected OOF predictions")
    if pred[["y_true", "y_pred", "Source_Group", "Outer_Fold"]].isna().any().any():
        raise AssertionError("Missing values in selected OOF prediction metadata")
    if pred.groupby(["Task", "Source_Group"])["Outer_Fold"].nunique().max() != 1:
        raise AssertionError("A source occurs in more than one outer fold")
    return pred


def pooled_and_fold_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    pooled_rows = []
    for task, part in pred.groupby("Task", sort=False):
        pooled_rows.append({
            "Task": task,
            "Selected_Model": part["Selected_Model"].iloc[0],
            "Rows": len(part),
            "Sources": part["Source_Group"].nunique(),
            **metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
        })
        for fold, fold_part in part.groupby("Outer_Fold"):
            fold_rows.append({
                "Task": task,
                "Outer_Fold": int(fold),
                "Rows": len(fold_part),
                "Sources": fold_part["Source_Group"].nunique(),
                **metrics(fold_part["y_true"].to_numpy(), fold_part["y_pred"].to_numpy()),
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def fold_stability(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, part in folds.groupby("Task", sort=False):
        row = {"Task": task, "Folds": len(part)}
        for metric in ("R2", "RMSE", "MAE"):
            values = part[metric].to_numpy(dtype=float)
            row[f"{metric}_Mean"] = np.nanmean(values)
            row[f"{metric}_SD"] = np.nanstd(values, ddof=1)
            row[f"{metric}_Min"] = np.nanmin(values)
            row[f"{metric}_Max"] = np.nanmax(values)
        rows.append(row)
    return pd.DataFrame(rows)


def source_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, source), part in pred.groupby(["Task", "Source_Group"], sort=False):
        err = part["Residual"].to_numpy(dtype=float)
        rows.append({
            "Task": task,
            "Source_Group": source,
            "Outer_Fold": int(part["Outer_Fold"].iloc[0]),
            "Rows": len(part),
            "y_true_mean": part["y_true"].mean(),
            "y_true_sd": part["y_true"].std(ddof=1),
            "y_pred_mean": part["y_pred"].mean(),
            "Bias_pred_minus_true": err.mean(),
            "MAE": np.abs(err).mean(),
            "RMSE": np.sqrt(np.square(err).mean()),
            "Max_Absolute_Error": np.abs(err).max(),
            "Absolute_Error_Sum": np.abs(err).sum(),
        })
    result = pd.DataFrame(rows)
    result["Error_Share_within_Task"] = result["Absolute_Error_Sum"] / result.groupby("Task")["Absolute_Error_Sum"].transform("sum")
    result["MAE_Outlier"] = False
    for task, idx in result.groupby("Task").groups.items():
        values = result.loc[idx, "MAE"]
        q1, q3 = values.quantile([0.25, 0.75])
        threshold = q3 + 1.5 * (q3 - q1)
        result.loc[idx, "MAE_Outlier"] = values.gt(threshold)
        result.loc[idx, "MAE_Outlier_Threshold"] = threshold
    return result.sort_values(["Task", "MAE"], ascending=[True, False])


def source_cluster_bootstrap(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    samples = []
    for task, part in pred.groupby("Task", sort=False):
        grouped = {
            source: (
                g["y_true"].to_numpy(dtype=float),
                g["y_pred"].to_numpy(dtype=float),
            )
            for source, g in part.groupby("Source_Group")
        }
        sources = np.asarray(list(grouped), dtype=object)
        source_mae = np.asarray([
            np.abs(grouped[source][1] - grouped[source][0]).mean() for source in sources
        ])
        source_rmse = np.asarray([
            np.sqrt(np.square(grouped[source][1] - grouped[source][0]).mean()) for source in sources
        ])
        for iteration in range(N_BOOTSTRAP):
            draw_idx = rng.integers(0, len(sources), size=len(sources))
            draw_sources = sources[draw_idx]
            y_true = np.concatenate([grouped[source][0] for source in draw_sources])
            y_pred = np.concatenate([grouped[source][1] for source in draw_sources])
            samples.append({
                "Task": task,
                "Iteration": iteration,
                **metrics(y_true, y_pred),
                "Source_Macro_MAE": source_mae[draw_idx].mean(),
                "Source_Macro_RMSE": source_rmse[draw_idx].mean(),
            })
    boot = pd.DataFrame(samples)
    summary = []
    for task, part in boot.groupby("Task", sort=False):
        for metric in ("R2", "RMSE", "MAE", "Source_Macro_MAE", "Source_Macro_RMSE"):
            q = part[metric].quantile([0.025, 0.5, 0.975])
            summary.append({
                "Task": task,
                "Metric": metric,
                "Bootstrap_Median": q.loc[0.5],
                "CI95_Lower": q.loc[0.025],
                "CI95_Upper": q.loc[0.975],
            })
    return boot, pd.DataFrame(summary)


def applicability_domain(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = []
    features = list(config.PRIMARY_FIXED_FEATURES)
    for task, task_pred in pred.groupby("Task", sort=False):
        data_path = config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv"
        data = pd.read_csv(data_path)
        use_features = [feature for feature in features if feature in data]
        merged = data.merge(
            task_pred[["Model_Row_ID", "y_true", "y_pred"]],
            on="Model_Row_ID",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_oof"),
        )
        if len(merged) != len(task_pred):
            raise AssertionError(f"{task}: failed to merge all OOF predictions for AD")

        for fold in sorted(merged["Outer_Fold"].unique()):
            train = data.loc[data["Outer_Fold"].ne(fold)].copy()
            test = merged.loc[merged["Outer_Fold"].eq(fold)].copy()
            x_train = train[use_features].apply(pd.to_numeric, errors="coerce")
            x_test = test[use_features].apply(pd.to_numeric, errors="coerce")

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
            test_scaled = scaler.transform(imputer.transform(x_test))

            k = min(5, max(1, len(train_scaled) - 1))
            train_nn = NearestNeighbors(n_neighbors=k + 1).fit(train_scaled)
            train_dist = train_nn.kneighbors(train_scaled, return_distance=True)[0][:, 1:].mean(axis=1)
            threshold = float(np.quantile(train_dist, 0.95))

            test_nn = NearestNeighbors(n_neighbors=k).fit(train_scaled)
            test_dist = test_nn.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
            threshold_safe = max(threshold, np.finfo(float).eps)

            for i, (_, row) in enumerate(test.iterrows()):
                diagnostics.append({
                    "Task": task,
                    "Outer_Fold": int(fold),
                    "Model_Row_ID": row["Model_Row_ID"],
                    "Source_Group": row["Source_Group"],
                    "y_true": row["y_true"],
                    "y_pred": row["y_pred"],
                    "Absolute_Error": abs(row["y_pred"] - row["y_true"]),
                    "AD_Distance": test_dist[i],
                    "AD_Threshold": threshold,
                    "AD_Distance_Ratio": test_dist[i] / threshold_safe,
                    "AD_Status": "Inside" if test_dist[i] <= threshold else "Outside",
                    "AD_Features": "|".join(use_features),
                })

    diag = pd.DataFrame(diagnostics)
    summary = []
    for task, part in diag.groupby("Task", sort=False):
        corr = part[["AD_Distance_Ratio", "Absolute_Error"]].corr(method="spearman").iloc[0, 1]
        for status in ("All", "Inside", "Outside"):
            subset = part if status == "All" else part.loc[part["AD_Status"].eq(status)]
            row = {
                "Task": task,
                "AD_Status": status,
                "Rows": len(subset),
                "Row_Fraction": len(subset) / len(part),
                "Sources": subset["Source_Group"].nunique() if len(subset) else 0,
                "Distance_Error_Spearman_All": corr if status == "All" else np.nan,
            }
            row.update(metrics(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy()))
            summary.append(row)
    return diag, pd.DataFrame(summary)


def save_figures(out: Path, folds: pd.DataFrame, ad: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task in zip(axes, TASKS):
        part = folds.loc[folds["Task"].eq(task)].sort_values("Outer_Fold")
        colors = ["#C44E52" if value < 0 else "#4C72B0" for value in part["R2"]]
        ax.bar(part["Outer_Fold"].astype(str), part["R2"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{task}: R² by source fold")
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("R²")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fold_r2_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, task in zip(axes, TASKS):
        part = ad.loc[ad["Task"].eq(task)]
        colors = part["AD_Status"].map({"Inside": "#4C72B0", "Outside": "#C44E52"})
        ax.scatter(part["AD_Distance_Ratio"], part["Absolute_Error"], c=colors, s=18, alpha=0.65)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{task}: error vs composition AD")
        ax.set_xlabel("Distance / fold threshold")
        ax.set_ylabel("Absolute error")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "applicability_distance_vs_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out: Path,
    pooled: pd.DataFrame,
    stability: pd.DataFrame,
    boot_summary: pd.DataFrame,
    per_source: pd.DataFrame,
    ad_summary: pd.DataFrame,
) -> None:
    def text_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        numeric = display.select_dtypes(include=[np.number]).columns
        display[numeric] = display[numeric].round(4)
        return "```text\n" + display.to_string(index=False) + "\n```"

    lines = [
        "# Final candidate model credibility assessment",
        "",
        "All predictions are source-group outer-fold OOF predictions on original, non-augmented rows.",
        "The applicability-domain flag is a preliminary composition-space k-nearest-neighbor distance diagnostic, not a calibrated prediction interval.",
        "",
        "## Pooled OOF metrics",
        "",
        text_table(pooled),
        "",
        "## Fold stability",
        "",
        text_table(stability),
        "",
        "## Source-cluster bootstrap 95% intervals",
        "",
        text_table(boot_summary),
        "",
        "## Source error flags",
        "",
        text_table(per_source.groupby("Task")["MAE_Outlier"].agg(["sum", "count"]).reset_index()),
        "",
        "## Applicability domain summary",
        "",
        text_table(ad_summary),
        "",
    ]
    (out / "CREDIBILITY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "model_credibility_strict"
    out.mkdir(parents=True, exist_ok=True)

    pred = load_selected_predictions()
    pooled, folds = pooled_and_fold_metrics(pred)
    stability = fold_stability(folds)
    per_source = source_metrics(pred)
    boot, boot_summary = source_cluster_bootstrap(pred)
    ad, ad_summary = applicability_domain(pred)

    pred.to_csv(out / "selected_final_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(out / "pooled_oof_metrics.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out / "metrics_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "fold_stability_summary.csv", index=False, encoding="utf-8-sig")
    per_source.to_csv(out / "per_source_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(out / "source_cluster_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "source_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    ad.to_csv(out / "applicability_row_diagnostics.csv", index=False, encoding="utf-8-sig")
    ad_summary.to_csv(out / "applicability_summary.csv", index=False, encoding="utf-8-sig")
    save_figures(out, folds, ad)
    write_summary(out, pooled, stability, boot_summary, per_source, ad_summary)

    print("\nPOOLED OOF METRICS")
    print(pooled.to_string(index=False))
    print("\nSOURCE-CLUSTER BOOTSTRAP")
    print(boot_summary.to_string(index=False))
    print("\nAPPLICABILITY DOMAIN")
    print(ad_summary.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
