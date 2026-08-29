from __future__ import annotations

import importlib.util
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import config


warnings.filterwarnings("ignore")

TASK = "UTS"
TARGET = config.TARGET_COLUMNS[TASK]
FEATURES = list(config.PRIMARY_FIXED_FEATURES)
SEED = 20260727
N_BOOTSTRAP = 2000


def load_model_module():
    path = config.PROJECT_ROOT / "14_strict_fold_augmentation_ablation.py"
    spec = importlib.util.spec_from_file_location("final_model_builders", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def scalar_expected_value(explainer) -> float:
    value = np.asarray(explainer.expected_value).reshape(-1)
    return float(value[0])


def shap_array(explainer, values: np.ndarray) -> np.ndarray:
    result = explainer.shap_values(values, check_additivity=False)
    if isinstance(result, list):
        result = result[0]
    result = np.asarray(result)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[:, :, 0]
    return result


def rebuild_and_explain(model_module):
    data = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "UTS_with_outer_folds.csv")
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    root = config.OUTPUT_DIRS["single"]
    rf_params = pd.read_csv(root / "nested_optuna_strict" / "metrics_by_outer_fold.csv")
    xgb_params = pd.read_csv(root / "nested_optuna_xgb_strict" / "metrics_by_outer_fold.csv")
    saved_oof = pd.read_csv(root / "rf_xgb_ensemble_strict" / "oof_predictions.csv")
    saved_oof = saved_oof.loc[saved_oof["Task"].eq(TASK), ["Model_Row_ID", "y_pred"]].rename(
        columns={"y_pred": "saved_y_pred"}
    )

    rows = []
    validation_rows = []
    for fold in sorted(data["Outer_Fold"].unique()):
        fold = int(fold)
        train = data.loc[data["Outer_Fold"].ne(fold)].copy()
        test = data.loc[data["Outer_Fold"].eq(fold)].copy()
        if set(train["Source_Group"]) & set(test["Source_Group"]):
            raise AssertionError("Source leakage")
        rf_row = rf_params.loc[rf_params["Task"].eq(TASK) & rf_params["Outer_Fold"].eq(fold)].iloc[0]
        xgb_row = xgb_params.loc[xgb_params["Task"].eq(TASK) & xgb_params["Outer_Fold"].eq(fold)].iloc[0]

        rf = model_module.rf_tuned(rf_row, config.RANDOM_SEED + fold)
        xgb = model_module.xgb_tuned(xgb_row, config.RANDOM_SEED + fold)
        rf.fit(train[FEATURES], train[TARGET])
        xgb.fit(train[FEATURES], train[TARGET])

        x_rf = rf.named_steps["imputer"].transform(test[FEATURES])
        x_xgb = xgb.named_steps["imputer"].transform(test[FEATURES])
        if not np.allclose(x_rf, x_xgb, equal_nan=True):
            raise AssertionError("RF and XGBoost imputed test matrices differ")
        pred_rf = rf.named_steps["model"].predict(x_rf)
        pred_xgb = xgb.named_steps["model"].predict(x_xgb)
        pred_ensemble = (pred_rf + pred_xgb) / 2.0

        explain_rf = shap.TreeExplainer(rf.named_steps["model"])
        explain_xgb = shap.TreeExplainer(xgb.named_steps["model"])
        sv_rf = shap_array(explain_rf, x_rf)
        sv_xgb = shap_array(explain_xgb, x_xgb)
        base_rf = scalar_expected_value(explain_rf)
        base_xgb = scalar_expected_value(explain_xgb)
        sv_ensemble = (sv_rf + sv_xgb) / 2.0
        base_ensemble = (base_rf + base_xgb) / 2.0

        reconstructed_rf = base_rf + sv_rf.sum(axis=1)
        reconstructed_xgb = base_xgb + sv_xgb.sum(axis=1)
        reconstructed_ensemble = base_ensemble + sv_ensemble.sum(axis=1)
        validation_rows.append({
            "Outer_Fold": fold,
            "Rows": len(test),
            "RF_Max_Additivity_Error": float(np.max(np.abs(reconstructed_rf - pred_rf))),
            "XGB_Max_Additivity_Error": float(np.max(np.abs(reconstructed_xgb - pred_xgb))),
            "Ensemble_Max_Additivity_Error": float(np.max(np.abs(reconstructed_ensemble - pred_ensemble))),
        })

        for index, (_, row) in enumerate(test.iterrows()):
            record = {
                "Task": TASK,
                "Outer_Fold": fold,
                "Model_Row_ID": row["Model_Row_ID"],
                "Source_Group": row["Source_Group"],
                "y_true": row[TARGET],
                "pred_rf": pred_rf[index],
                "pred_xgb": pred_xgb[index],
                "y_pred": pred_ensemble[index],
                "Base_RF": base_rf,
                "Base_XGB": base_xgb,
                "Base_Ensemble": base_ensemble,
            }
            for feature_index, feature in enumerate(FEATURES):
                record[f"Value_{feature}"] = x_rf[index, feature_index]
                record[f"SHAP_RF_{feature}"] = sv_rf[index, feature_index]
                record[f"SHAP_XGB_{feature}"] = sv_xgb[index, feature_index]
                record[f"SHAP_Ensemble_{feature}"] = sv_ensemble[index, feature_index]
            rows.append(record)
        print(
            f"fold={fold}: rows={len(test)}, "
            f"ensemble additivity max={validation_rows[-1]['Ensemble_Max_Additivity_Error']:.6g}"
        )

    explained = pd.DataFrame(rows)
    validation = pd.DataFrame(validation_rows)
    explained = explained.merge(saved_oof, on="Model_Row_ID", validate="one_to_one")
    explained["Saved_Prediction_Abs_Diff"] = (explained["y_pred"] - explained["saved_y_pred"]).abs()
    if explained["Saved_Prediction_Abs_Diff"].max() > 1e-4:
        raise AssertionError("Rebuilt ensemble predictions do not reproduce saved OOF predictions")
    if validation[["RF_Max_Additivity_Error", "XGB_Max_Additivity_Error", "Ensemble_Max_Additivity_Error"]].max().max() > 1e-2:
        raise AssertionError("SHAP additivity error exceeds tolerance")
    return explained, validation


def model_importance(explained: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    fold_rows = []
    for model_label in ("RF", "XGB", "Ensemble"):
        shap_columns = [f"SHAP_{model_label}_{feature}" for feature in FEATURES]
        total = explained[shap_columns].abs().to_numpy().mean(axis=0).sum()
        for feature in FEATURES:
            values = explained[f"Value_{feature}"]
            shap_values = explained[f"SHAP_{model_label}_{feature}"]
            global_rows.append({
                "Model": model_label,
                "Feature": feature,
                "Mean_Abs_SHAP": shap_values.abs().mean(),
                "Mean_Signed_SHAP": shap_values.mean(),
                "Importance_Share": shap_values.abs().mean() / total if total else np.nan,
                "Value_SHAP_Spearman": values.corr(shap_values, method="spearman"),
            })
        for fold, part in explained.groupby("Outer_Fold"):
            fold_total = sum(part[f"SHAP_{model_label}_{feature}"].abs().mean() for feature in FEATURES)
            temp = []
            for feature in FEATURES:
                sv = part[f"SHAP_{model_label}_{feature}"]
                temp.append({
                    "Model": model_label,
                    "Outer_Fold": int(fold),
                    "Feature": feature,
                    "Mean_Abs_SHAP": sv.abs().mean(),
                    "Mean_Signed_SHAP": sv.mean(),
                    "Importance_Share": sv.abs().mean() / fold_total if fold_total else np.nan,
                    "Value_SHAP_Spearman": part[f"Value_{feature}"].corr(sv, method="spearman"),
                })
            frame = pd.DataFrame(temp)
            frame["Rank"] = frame["Mean_Abs_SHAP"].rank(method="min", ascending=False).astype(int)
            fold_rows.extend(frame.to_dict("records"))
    global_frame = pd.DataFrame(global_rows)
    global_frame["Rank"] = global_frame.groupby("Model")["Mean_Abs_SHAP"].rank(
        method="min", ascending=False
    ).astype(int)
    return global_frame.sort_values(["Model", "Rank"]), pd.DataFrame(fold_rows)


def rank_agreement(global_importance: pd.DataFrame, fold_importance: pd.DataFrame):
    rows = []
    rf = global_importance.loc[global_importance["Model"].eq("RF")].set_index("Feature")
    xgb = global_importance.loc[global_importance["Model"].eq("XGB")].set_index("Feature")
    rows.append({
        "Comparison": "RF_vs_XGB_Global_Importance",
        "Fold_A": np.nan,
        "Fold_B": np.nan,
        "Spearman": spearmanr(rf.loc[FEATURES, "Mean_Abs_SHAP"], xgb.loc[FEATURES, "Mean_Abs_SHAP"]).statistic,
    })
    for fold in sorted(fold_importance["Outer_Fold"].unique()):
        part = fold_importance.loc[fold_importance["Outer_Fold"].eq(fold)]
        rf_fold = part.loc[part["Model"].eq("RF")].set_index("Feature")
        xgb_fold = part.loc[part["Model"].eq("XGB")].set_index("Feature")
        rows.append({
            "Comparison": "RF_vs_XGB_Within_Fold",
            "Fold_A": int(fold),
            "Fold_B": int(fold),
            "Spearman": spearmanr(rf_fold.loc[FEATURES, "Mean_Abs_SHAP"], xgb_fold.loc[FEATURES, "Mean_Abs_SHAP"]).statistic,
        })
    ensemble = fold_importance.loc[fold_importance["Model"].eq("Ensemble")]
    for fold_a, fold_b in combinations(sorted(ensemble["Outer_Fold"].unique()), 2):
        a = ensemble.loc[ensemble["Outer_Fold"].eq(fold_a)].set_index("Feature")
        b = ensemble.loc[ensemble["Outer_Fold"].eq(fold_b)].set_index("Feature")
        rows.append({
            "Comparison": "Ensemble_Fold_vs_Fold",
            "Fold_A": int(fold_a),
            "Fold_B": int(fold_b),
            "Spearman": spearmanr(a.loc[FEATURES, "Mean_Abs_SHAP"], b.loc[FEATURES, "Mean_Abs_SHAP"]).statistic,
        })
    return pd.DataFrame(rows)


def direction_stability(global_importance: pd.DataFrame, fold_importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_label in ("RF", "XGB", "Ensemble"):
        global_part = global_importance.loc[global_importance["Model"].eq(model_label)].set_index("Feature")
        for feature in FEATURES:
            fold_values = fold_importance.loc[
                fold_importance["Model"].eq(model_label)
                & fold_importance["Feature"].eq(feature),
                "Value_SHAP_Spearman",
            ].dropna()
            positive = int(fold_values.gt(0.1).sum())
            negative = int(fold_values.lt(-0.1).sum())
            weak = int(fold_values.between(-0.1, 0.1).sum())
            rows.append({
                "Model": model_label,
                "Feature": feature,
                "Global_Value_SHAP_Spearman": global_part.loc[feature, "Value_SHAP_Spearman"],
                "Fold_Spearman_Median": fold_values.median(),
                "Fold_Spearman_Min": fold_values.min(),
                "Fold_Spearman_Max": fold_values.max(),
                "Positive_Folds_gt_0p1": positive,
                "Negative_Folds_lt_minus0p1": negative,
                "Weak_Folds": weak,
                "Direction_Consistent_4of5": positive >= 4 or negative >= 4,
                "Monotonic_Strength_At_Least_0p2": abs(fold_values.median()) >= 0.2,
            })
    return pd.DataFrame(rows)


def source_cluster_bootstrap(explained: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    grouped = {
        source: part[[f"SHAP_Ensemble_{feature}" for feature in FEATURES]].abs().to_numpy()
        for source, part in explained.groupby("Source_Group")
    }
    sources = np.asarray(list(grouped), dtype=object)
    rows = []
    for iteration in range(N_BOOTSTRAP):
        draw = sources[rng.integers(0, len(sources), size=len(sources))]
        values = np.concatenate([grouped[source] for source in draw], axis=0).mean(axis=0)
        ranks = pd.Series(values, index=FEATURES).rank(method="min", ascending=False).astype(int)
        for index, feature in enumerate(FEATURES):
            rows.append({
                "Iteration": iteration,
                "Feature": feature,
                "Mean_Abs_SHAP": values[index],
                "Rank": int(ranks.loc[feature]),
            })
    samples = pd.DataFrame(rows)
    summary_rows = []
    point = explained[[f"SHAP_Ensemble_{feature}" for feature in FEATURES]].abs().mean()
    for feature in FEATURES:
        part = samples.loc[samples["Feature"].eq(feature)]
        q_importance = part["Mean_Abs_SHAP"].quantile([0.025, 0.5, 0.975])
        q_rank = part["Rank"].quantile([0.025, 0.5, 0.975])
        summary_rows.append({
            "Feature": feature,
            "Point_Mean_Abs_SHAP": point[f"SHAP_Ensemble_{feature}"],
            "Bootstrap_Median": q_importance.loc[0.5],
            "CI95_Lower": q_importance.loc[0.025],
            "CI95_Upper": q_importance.loc[0.975],
            "Rank_Median": q_rank.loc[0.5],
            "Rank_CI95_Lower": q_rank.loc[0.025],
            "Rank_CI95_Upper": q_rank.loc[0.975],
            "Top3_Probability": part["Rank"].le(3).mean(),
            "Top5_Probability": part["Rank"].le(5).mean(),
        })
    summary = pd.DataFrame(summary_rows).sort_values("Point_Mean_Abs_SHAP", ascending=False)
    summary["Point_Rank"] = np.arange(1, len(summary) + 1)
    return samples, summary


def applicability_sensitivity(explained: pd.DataFrame):
    ad_path = config.PROJECT_ROOT / "results" / "model_credibility_strict" / "applicability_row_diagnostics.csv"
    ad = pd.read_csv(ad_path)
    ad = ad.loc[ad["Task"].eq(TASK), ["Model_Row_ID", "AD_Status", "AD_Distance_Ratio"]]
    merged = explained.merge(ad, on="Model_Row_ID", validate="one_to_one")
    rows = []
    for status in ("All", "Inside", "Outside"):
        part = merged if status == "All" else merged.loc[merged["AD_Status"].eq(status)]
        temp = []
        for feature in FEATURES:
            sv = part[f"SHAP_Ensemble_{feature}"]
            temp.append({
                "AD_Status": status,
                "Rows": len(part),
                "Sources": part["Source_Group"].nunique(),
                "Feature": feature,
                "Mean_Abs_SHAP": sv.abs().mean(),
                "Value_SHAP_Spearman": part[f"Value_{feature}"].corr(sv, method="spearman"),
            })
        frame = pd.DataFrame(temp)
        frame["Rank"] = frame["Mean_Abs_SHAP"].rank(method="min", ascending=False).astype(int)
        rows.extend(frame.to_dict("records"))
    return merged, pd.DataFrame(rows)


def leave_one_source_out(explained: pd.DataFrame):
    overall = pd.Series({
        feature: explained[f"SHAP_Ensemble_{feature}"].abs().mean() for feature in FEATURES
    })
    overall_rank = overall.rank(method="min", ascending=False).astype(int)
    rows = []
    for source in explained["Source_Group"].unique():
        part = explained.loc[explained["Source_Group"].ne(source)]
        importance = pd.Series({
            feature: part[f"SHAP_Ensemble_{feature}"].abs().mean() for feature in FEATURES
        })
        ranks = importance.rank(method="min", ascending=False).astype(int)
        for feature in FEATURES:
            rows.append({
                "Excluded_Source": source,
                "Feature": feature,
                "Mean_Abs_SHAP": importance[feature],
                "Rank": int(ranks[feature]),
                "Overall_Rank": int(overall_rank[feature]),
                "Rank_Change": int(ranks[feature] - overall_rank[feature]),
            })
    values = pd.DataFrame(rows)
    summary = values.groupby("Feature").agg(
        Overall_Rank=("Overall_Rank", "first"),
        LOO_Rank_Min=("Rank", "min"),
        LOO_Rank_Max=("Rank", "max"),
        Max_Abs_Rank_Change=("Rank_Change", lambda x: x.abs().max()),
        LOO_Importance_Min=("Mean_Abs_SHAP", "min"),
        LOO_Importance_Max=("Mean_Abs_SHAP", "max"),
    ).reset_index().sort_values("Overall_Rank")
    return values, summary


def save_figures(out: Path, explained: pd.DataFrame, bootstrap_summary: pd.DataFrame, fold_importance: pd.DataFrame, global_importance: pd.DataFrame):
    ordered = bootstrap_summary.sort_values("Point_Rank")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(ordered))
    error = np.vstack([
        ordered["Point_Mean_Abs_SHAP"] - ordered["CI95_Lower"],
        ordered["CI95_Upper"] - ordered["Point_Mean_Abs_SHAP"],
    ])
    ax.barh(y, ordered["Point_Mean_Abs_SHAP"], color="#4C72B0")
    ax.errorbar(ordered["Point_Mean_Abs_SHAP"], y, xerr=error, fmt="none", ecolor="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (MPa), source-bootstrap 95% CI")
    ax.set_title("UTS OOF ensemble SHAP importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "uts_oof_shap_global_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    shap_values = explained[[f"SHAP_Ensemble_{feature}" for feature in FEATURES]].to_numpy()
    feature_values = explained[[f"Value_{feature}" for feature in FEATURES]].copy()
    feature_values.columns = FEATURES
    shap.summary_plot(shap_values, feature_values, feature_names=FEATURES, show=False, max_display=len(FEATURES))
    plt.title("UTS OOF ensemble SHAP summary")
    plt.tight_layout()
    plt.savefig(out / "uts_oof_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    ensemble = fold_importance.loc[fold_importance["Model"].eq("Ensemble")]
    rank_matrix = ensemble.pivot(index="Feature", columns="Outer_Fold", values="Rank")
    order = global_importance.loc[global_importance["Model"].eq("Ensemble")].sort_values("Rank")["Feature"]
    rank_matrix = rank_matrix.loc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(rank_matrix.to_numpy(), cmap="YlGnBu_r", aspect="auto", vmin=1, vmax=len(FEATURES))
    ax.set_xticks(range(rank_matrix.shape[1]))
    ax.set_xticklabels(rank_matrix.columns)
    ax.set_yticks(range(rank_matrix.shape[0]))
    ax.set_yticklabels(rank_matrix.index)
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, int(rank_matrix.iloc[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xlabel("Outer fold")
    ax.set_title("UTS ensemble SHAP rank by source fold")
    fig.colorbar(image, ax=ax, label="Rank (1 = highest)")
    fig.tight_layout()
    fig.savefig(out / "uts_oof_shap_fold_rank_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = config.PROJECT_ROOT / "results" / "uts_oof_shap_strict"
    out.mkdir(parents=True, exist_ok=True)
    model_module = load_model_module()
    explained, validation = rebuild_and_explain(model_module)
    global_importance, fold_importance = model_importance(explained)
    agreement = rank_agreement(global_importance, fold_importance)
    directions = direction_stability(global_importance, fold_importance)
    boot_samples, boot_summary = source_cluster_bootstrap(explained)
    explained_ad, ad_summary = applicability_sensitivity(explained)
    loo_values, loo_summary = leave_one_source_out(explained)

    explained.to_csv(out / "uts_oof_shap_values_wide.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(out / "shap_additivity_and_prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    global_importance.to_csv(out / "global_importance_by_model.csv", index=False, encoding="utf-8-sig")
    fold_importance.to_csv(out / "fold_importance_and_direction.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(out / "model_and_fold_rank_agreement.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(out / "direction_stability.csv", index=False, encoding="utf-8-sig")
    boot_samples.to_csv(out / "source_bootstrap_importance_samples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(out / "source_bootstrap_importance_summary.csv", index=False, encoding="utf-8-sig")
    explained_ad.to_csv(out / "uts_oof_shap_with_applicability_domain.csv", index=False, encoding="utf-8-sig")
    ad_summary.to_csv(out / "applicability_domain_shap_sensitivity.csv", index=False, encoding="utf-8-sig")
    loo_values.to_csv(out / "leave_one_source_out_importance.csv", index=False, encoding="utf-8-sig")
    loo_summary.to_csv(out / "leave_one_source_out_summary.csv", index=False, encoding="utf-8-sig")
    save_figures(out, explained, boot_summary, fold_importance, global_importance)

    cfg = {
        "task": TASK,
        "model": "0.5 RandomForest + 0.5 XGBoost",
        "features": FEATURES,
        "explanation_rows": "original outer-fold test rows only",
        "training_augmentation": False,
        "shap_method": "TreeExplainer per component; component SHAP averaged with ensemble weights",
        "source_bootstrap_iterations": N_BOOTSTRAP,
        "interpretation_scope": "model associations, not causal material mechanisms",
        "seed": SEED,
    }
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSHAP VALIDATION")
    print(validation.to_string(index=False))
    print("\nENSEMBLE GLOBAL IMPORTANCE")
    print(global_importance.loc[global_importance["Model"].eq("Ensemble")].to_string(index=False))
    print("\nSOURCE BOOTSTRAP IMPORTANCE")
    print(boot_summary.to_string(index=False))
    print("\nRANK AGREEMENT")
    print(agreement.groupby("Comparison")["Spearman"].agg(["mean", "min", "max"]).to_string())
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
