from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from src.data_utils import ensure_output_dirs


def metrics(y, p):
    return r2_score(y, p), np.sqrt(mean_squared_error(y, p)), mean_absolute_error(y, p)


def source_bootstrap(part, n_boot=2000, seed=20260721):
    rng = np.random.default_rng(seed)
    groups = {g: x for g, x in part.groupby("Source_Group")}
    names = np.array(list(groups))
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(names, size=len(names), replace=True)
        sample = pd.concat([groups[g] for g in chosen], ignore_index=True)
        values.append(metrics(sample.y_true, sample.y_pred))
    arr = np.asarray(values)
    return {
        "R2_CI_Low": np.quantile(arr[:, 0], 0.025), "R2_CI_High": np.quantile(arr[:, 0], 0.975),
        "RMSE_CI_Low": np.quantile(arr[:, 1], 0.025), "RMSE_CI_High": np.quantile(arr[:, 1], 0.975),
        "MAE_CI_Low": np.quantile(arr[:, 2], 0.025), "MAE_CI_High": np.quantile(arr[:, 2], 0.975),
    }


def main():
    ensure_output_dirs()
    out = config.PROJECT_ROOT / "reports" / "eda_v2"
    out.mkdir(parents=True, exist_ok=True)
    base = config.OUTPUT_DIRS["single"]
    baseline = pd.read_csv(base / "baseline_strict" / "oof_predictions.csv")
    tuned = pd.read_csv(base / "nested_optuna_strict" / "oof_predictions.csv")
    chosen = pd.concat([
        baseline.query("Task == 'YS' and Feature_Set == 'composition_core' and Model == 'RandomForest'"),
        tuned.query("Task == 'UTS'"),
        baseline.query("Task == 'EL' and Feature_Set == 'composition_core' and Model == 'RandomForest'"),
    ], ignore_index=True)
    perf = []
    for task, part in chosen.groupby("Task"):
        r2, rmse, mae = metrics(part.y_true, part.y_pred)
        perf.append({"Task": task, "Rows": len(part), "Sources": part.Source_Group.nunique(),
                     "R2": r2, "RMSE": rmse, "MAE": mae, **source_bootstrap(part)})
    pd.DataFrame(perf).to_csv(out / "single_target_source_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    targets = [config.TARGET_COLUMNS[t] for t in ("YS", "UTS", "EL")]
    triple = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "TRIPLE_with_outer_folds.csv")
    triple[targets] = triple[targets].apply(pd.to_numeric, errors="coerce")
    triple[targets].describe().T.to_csv(out / "triple_target_descriptive_stats.csv", encoding="utf-8-sig")
    triple[targets].corr("pearson").to_csv(out / "triple_target_pearson.csv", encoding="utf-8-sig")
    triple[targets].corr("spearman").to_csv(out / "triple_target_spearman.csv", encoding="utf-8-sig")
    features = [c for c in config.PRIMARY_FIXED_FEATURES if c in triple]
    corr = triple[features + targets].corr("spearman").loc[features, targets]
    corr.to_csv(out / "feature_target_spearman.csv", encoding="utf-8-sig")
    triple.groupby("Outer_Fold")[targets].agg(["count", "mean", "std"]).to_csv(
        out / "triple_target_by_outer_fold.csv", encoding="utf-8-sig"
    )
    triple.groupby("Source_Group").size().describe().to_csv(out / "triple_source_size_stats.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(triple[targets].corr("pearson"), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Triple-complete target correlations")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1])
    axes[1].set_title("Feature-target Spearman correlations")
    fig.tight_layout()
    fig.savefig(out / "eda_prior_correlations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame(perf).to_string(index=False))
    print("\nTriple Pearson correlation:\n", triple[targets].corr("pearson").to_string())


if __name__ == "__main__":
    main()
