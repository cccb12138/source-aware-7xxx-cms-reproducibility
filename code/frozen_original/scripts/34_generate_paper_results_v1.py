from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
RESULTS = PROJECT / "results"
OUT = Path(r"F:\CC\outputs\paper_results_v1")
FIG = OUT / "figures"
FINAL_UTS = Path(r"F:\CC\outputs\uts_scope_clean_final")

BLUE = "#2F6B9A"
ORANGE = "#D97935"
GREEN = "#3A8D78"
RED = "#B84A4A"
GRAY = "#6B7280"
LIGHT = "#EAF1F6"
TASK_COLORS = {"UTS": BLUE, "YS": ORANGE, "EL": GREEN}
FEATURES = ["Zn", "Fe", "Cu", "Zr", "Mg"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_figure(fig, stem: str):
    fig.savefig(FIG / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style():
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def workflow_figure():
    fig, ax = plt.subplots(figsize=(12.2, 6.7))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    stages = [
        (0.25, 5.05, 2.05, 0.95, "Traceable inputs", "Literature + public data\nSample ID + source"),
        (2.75, 5.05, 2.05, 0.95, "Scope audit", "Study object + test\ncondition review"),
        (5.25, 5.05, 2.05, 0.95, "Source-blocked CV", "No source shared across\ntrain and test"),
        (7.75, 5.05, 2.05, 0.95, "Nested decisions", "Models, augmentation,\nproxies, sparse features"),
        (10.25, 5.05, 1.7, 0.95, "Credibility", "SHAP + AD +\nuncertainty"),
    ]
    for i, (x, y, w, h, title, subtitle) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.08",
                                    facecolor=LIGHT if i < 4 else "#E9F3EF", edgecolor=BLUE if i < 4 else GREEN, linewidth=1.2))
        ax.text(x + w / 2, y + 0.63, title, ha="center", va="center", fontweight="bold", fontsize=9.5)
        ax.text(x + w / 2, y + 0.27, subtitle, ha="center", va="center", fontsize=7.7, color="#374151")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.08, y + h / 2), (stages[i + 1][0] - 0.08, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=10, color=GRAY, linewidth=1.1))

    ax.text(0.25, 4.45, "Clean partial-label dataset", fontsize=10.5, fontweight="bold", color="#1F2937")
    cards = [
        (0.35, 2.75, 2.25, 1.15, "UTS", "675 rows | 258 sources", "Primary prediction target", BLUE),
        (2.9, 2.75, 2.25, 1.15, "YS", "307 rows | 63 sources", "Exploratory target", ORANGE),
        (5.45, 2.75, 2.25, 1.15, "EL", "537 rows | 164 sources", "Exploratory target", GREEN),
        (8.0, 2.75, 3.85, 1.15, "Matched robustness subset", "266 rows | 59 sources | 4 datasets", "YS–UTS–EL on the same samples", "#7768AE"),
    ]
    for x, y, w, h, title, count, role, color in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.07",
                                    facecolor="white", edgecolor=color, linewidth=1.6))
        ax.text(x + 0.16, y + 0.82, title, ha="left", va="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x + 0.16, y + 0.49, count, ha="left", va="center", fontsize=8.5)
        ax.text(x + 0.16, y + 0.20, role, ha="left", va="center", fontsize=7.8, color=GRAY)

    ax.add_patch(FancyBboxPatch((0.35, 0.7), 11.5, 1.15, boxstyle="round,pad=0.035,rounding_size=0.07",
                                facecolor="#F8FAFC", edgecolor="#9CA3AF", linewidth=1.0))
    ax.text(0.55, 1.52, "Final evidence hierarchy", fontweight="bold", fontsize=9.5)
    ax.text(0.55, 1.18, "Confirmatory: UTS source-blocked ensemble, cross-dataset validation, SHAP stability, applicability domain and conformal intervals",
            fontsize=8.2)
    ax.text(0.55, 0.86, "Exploratory/supplementary: YS, EL, HV gap, data augmentation ablation, derived descriptors, partial-label MTL and matched multi-output models",
            fontsize=8.2)
    ax.set_title("Traceable and source-aware machine-learning workflow for heterogeneous 7xxx aluminium-alloy data",
                 fontsize=13, fontweight="bold", pad=12)
    save_figure(fig, "Fig1_workflow_and_evidence_hierarchy")


def uts_performance_figure(oof, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    palette = sns.color_palette("deep", 5)
    for fold in range(5):
        part = oof.loc[oof["Outer_Fold"].eq(fold)]
        ax.scatter(part["y_true"], part["y_pred"], s=17, alpha=0.68, color=palette[fold], edgecolor="none", label=f"Fold {fold + 1}")
    lo = min(oof["y_true"].min(), oof["y_pred"].min())
    hi = max(oof["y_true"].max(), oof["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#111827", linewidth=1.1)
    ax.set_xlim(lo - 15, hi + 15)
    ax.set_ylim(lo - 15, hi + 15)
    ax.set_xlabel("Measured UTS (MPa)")
    ax.set_ylabel("OOF-predicted UTS (MPa)")
    ax.set_title("Source-blocked out-of-fold predictions")
    overall = metrics.loc[metrics["Scope"].eq("All_OOF")].iloc[0]
    ax.text(0.04, 0.96, f"n = {int(overall.Rows)}\nR² = {overall.R2:.3f}\nRMSE = {overall.RMSE:.1f} MPa\nMAE = {overall.MAE:.1f} MPa",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95))
    ax.legend(loc="lower right", frameon=True, ncol=1)
    panel_label(ax, "a")

    ax = axes[1]
    folds = metrics.loc[metrics["Scope"].eq("Outer_Fold")].copy()
    folds["Fold_Label"] = [f"Fold {int(x) + 1}" for x in folds["Fold"]]
    bars = ax.bar(folds["Fold_Label"], folds["R2"], color=palette, edgecolor="white", linewidth=0.7)
    ax.axhline(overall.R2, color=BLUE, linestyle="--", linewidth=1.2, label=f"Pooled OOF R² = {overall.R2:.3f}")
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("R²")
    ax.set_title("Performance remains positive in every source fold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper left")
    for bar, rmse in zip(bars, folds["RMSE"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.018, f"{bar.get_height():.2f}\n({rmse:.0f})",
                ha="center", va="bottom", fontsize=7.3)
    panel_label(ax, "b")
    fig.suptitle("UTS is the primary model with moderate, source-aware predictive performance", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Fig2_UTS_source_blocked_performance")


def shap_figure(shap_wide, importance):
    ensemble = importance.loc[importance["Model"].eq("Ensemble")].sort_values("Rank")
    ordered = ensemble["Feature"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"width_ratios": [0.82, 1.55]})

    ax = axes[0]
    y = np.arange(len(ordered))[::-1]
    shares = ensemble.set_index("Feature").loc[ordered, "Importance_Share"].to_numpy() * 100
    colors = [BLUE, "#8A6D5A", "#C56B3C", "#6F8F72", "#7B6D9D"]
    ax.barh(y, shares, color=colors, edgecolor="white")
    ax.set_yticks(y, ordered)
    ax.set_xlabel("Mean |SHAP| share (%)")
    ax.set_title("Global importance")
    ax.set_xlim(0, max(shares) * 1.2)
    for yi, value in zip(y, shares):
        ax.text(value + 0.6, yi, f"{value:.1f}%", va="center", fontsize=8)
    panel_label(ax, "a")

    ax = axes[1]
    rng = np.random.default_rng(20260814)
    cmap = mpl.colormaps["coolwarm"]
    for row, feature in enumerate(ordered):
        shap_values = shap_wide[f"SHAP_Ensemble_{feature}"].to_numpy(float)
        feature_values = shap_wide[f"Value_{feature}"].to_numpy(float)
        finite = np.isfinite(feature_values)
        lo, hi = np.nanpercentile(feature_values[finite], [2, 98]) if finite.any() else (0, 1)
        normed = np.clip((feature_values - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.normal(0, 0.075, len(shap_values))
        ax.scatter(shap_values, (len(ordered)-1-row) + jitter, c=cmap(normed), s=10, alpha=0.72, edgecolor="none", rasterized=True)
    ax.axvline(0, color="#4B5563", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered))[::-1], ordered)
    ax.set_xlabel("Ensemble SHAP value (MPa)")
    ax.set_title("Feature effects across 675 OOF samples")
    sm = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_ticks([])
    cbar.set_label("")
    cbar.ax.tick_params(length=0)
    panel_label(ax, "b")
    fig.suptitle("Direct-composition SHAP interpretation of the final UTS model", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "Fig3_UTS_SHAP_importance_and_effects")


def credibility_figure(boot_samples, boot_summary, ad, intervals, lodo, matched_fair):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))

    ax = axes[0, 0]
    sns.kdeplot(data=boot_samples, x="R2", fill=True, color=BLUE, alpha=0.28, linewidth=1.2, ax=ax)
    s = boot_summary.loc[boot_summary["Metric"].eq("R2")].iloc[0]
    ax.axvline(s.Bootstrap_Median, color=BLUE, linewidth=1.2)
    ax.axvspan(s.CI95_Lower, s.CI95_Upper, color=BLUE, alpha=0.12)
    ax.text(0.04, 0.92, f"Median = {s.Bootstrap_Median:.3f}\n95% CI = [{s.CI95_Lower:.3f}, {s.CI95_Upper:.3f}]",
            transform=ax.transAxes, va="top", fontsize=8.2)
    ax.set_xlabel("Source-cluster bootstrap R²")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty respects source clustering")
    panel_label(ax, "a")

    ax = axes[0, 1]
    part = ad.loc[ad["AD_Status"].isin(["Inside", "Outside"])].copy()
    x = np.arange(len(part))
    width = 0.34
    ax.bar(x - width/2, part["RMSE"], width, label="RMSE", color=BLUE)
    ax.bar(x + width/2, part["MAE"], width, label="MAE", color="#9CC0D9")
    ax.set_xticks(x, [f"{status}\n(n={int(rows)})" for status, rows in zip(part["AD_Status"], part["Rows"])])
    ax.set_ylabel("Error (MPa)")
    ax.set_title("Errors increase outside the applicability domain")
    ax.legend()
    panel_label(ax, "b")

    ax = axes[1, 0]
    all_intervals = intervals.loc[intervals["Scope"].eq("All")].copy()
    methods = ["RowCrossConformal", "SourceMaxCrossConformal"]
    labels = ["Row-wise", "Source-max"]
    x = np.arange(2)
    for i, (method, label, color) in enumerate(zip(methods, labels, [BLUE, GREEN])):
        p = all_intervals.loc[all_intervals["Method"].eq(method)].sort_values("Nominal_Coverage")
        ax.plot(x, p["Row_Coverage"].to_numpy(), marker="o", label=label, color=color, linewidth=1.4)
    ax.plot(x, [0.90, 0.95], "--", color="#111827", label="Nominal")
    ax.set_xticks(x, ["90%", "95%"])
    ax.set_ylim(0.84, 0.99)
    ax.set_ylabel("Observed row coverage")
    ax.set_title("Conformal intervals approach nominal coverage")
    ax.legend(loc="lower right")
    panel_label(ax, "c")

    ax = axes[1, 1]
    matched_uts = matched_fair.loc[(matched_fair["Task"].eq("UTS")) & matched_fair["Model"].str.startswith("Full_Label")].iloc[0]
    values = [0.5468056654484905, float(lodo.iloc[0]["R2"]), matched_uts.R2]
    labels = ["Source-blocked\nOOF", "Leave-one-\ndataset-out", "Same model on\nmatched subset"]
    colors = [BLUE, "#5C88A8", "#7768AE"]
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("R²")
    ax.set_title("Performance decreases in harder transfer domains")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)
    panel_label(ax, "d")

    fig.suptitle("Credibility assessment of the final UTS model", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "Fig4_UTS_credibility_and_transfer")


def multitarget_figure(mtl, matched_corr, ys_el_boot, uts_boot, matched_multi):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))

    ax = axes[0, 0]
    counts = pd.DataFrame({"Target": ["UTS", "EL", "YS", "Matched\nYS–UTS–EL"], "Rows": [675, 537, 307, 266]})
    colors = [BLUE, GREEN, ORANGE, "#7768AE"]
    bars = ax.bar(counts["Target"], counts["Rows"], color=colors)
    for bar, value in zip(bars, counts["Rows"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 15, str(value), ha="center", fontsize=8)
    ax.set_ylim(0, 750)
    ax.set_ylabel("Usable rows")
    ax.set_title("Partial labels preserve substantially more data")
    panel_label(ax, "a")

    ax = axes[0, 1]
    rows = []
    for task, boot in (("UTS", uts_boot), ("YS", ys_el_boot.loc[ys_el_boot["Task"].eq("YS")]), ("EL", ys_el_boot.loc[ys_el_boot["Task"].eq("EL")])):
        r = boot.loc[boot["Metric"].eq("R2")].iloc[0]
        rows.append((task, r.Bootstrap_Median, r.CI95_Lower, r.CI95_Upper))
    x = np.arange(3)
    med = np.array([r[1] for r in rows])
    lower = med - np.array([r[2] for r in rows])
    upper = np.array([r[3] for r in rows]) - med
    ax.errorbar(x, med, yerr=np.vstack([lower, upper]), fmt="o", markersize=7, capsize=4,
                color="#374151", ecolor="#6B7280", linewidth=1.2)
    for i, (task, value, _, _) in enumerate(rows):
        ax.scatter(i, value, s=65, color=TASK_COLORS[task], zorder=3)
    ax.axhline(0, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("Source-bootstrap R²")
    ax.set_title("Only UTS has a clearly positive R² interval")
    panel_label(ax, "b")

    ax = axes[1, 0]
    p = mtl.loc[mtl["Feature_Set"].eq("refined5")].copy()
    task_order = ["UTS", "YS", "EL"]
    x = np.arange(3)
    width = 0.34
    independent = [p.loc[(p["Task"].eq(t)) & p["Model"].eq("Independent_RF"), "R2"].iloc[0] for t in task_order]
    masked = [p.loc[(p["Task"].eq(t)) & p["Model"].eq("Masked_MTL"), "R2"].iloc[0] for t in task_order]
    ax.bar(x - width/2, independent, width, label="Independent RF", color="#587D9A")
    ax.bar(x + width/2, masked, width, label="Partial-label MTL", color="#B7A6D8")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, task_order)
    ax.set_ylabel("R²")
    ax.set_title("Partial-label MTL shows negative transfer")
    ax.legend()
    panel_label(ax, "c")

    ax = axes[1, 1]
    matrix = np.eye(3)
    labels = ["YS", "UTS", "EL"]
    for _, row in matched_corr.loc[matched_corr["Level"].eq("Row")].iterrows():
        i, j = labels.index(row.Target_A), labels.index(row.Target_B)
        matrix[i, j] = matrix[j, i] = row.Spearman_r
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
                xticklabels=labels, yticklabels=labels, square=True, cbar_kws={"ticks": []}, ax=ax)
    heatmap_cbar = ax.collections[0].colorbar
    heatmap_cbar.set_label("")
    heatmap_cbar.ax.tick_params(length=0)
    ax.set_title("Matched-sample target relationships")
    panel_label(ax, "d")

    fig.suptitle("Evidence-based positioning of the three prediction targets", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "Fig5_target_positioning_and_MTL")


def matched_figure(matched, fair):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7))
    pairs = [
        ("YS_0.2pct_MPa", "UTS_MPa", "YS (MPa)", "UTS (MPa)", "a"),
        ("YS_0.2pct_MPa", "EL_pct", "YS (MPa)", "EL (%)", "b"),
        ("UTS_MPa", "EL_pct", "UTS (MPa)", "EL (%)", "c"),
    ]
    dataset_colors = {name: color for name, color in zip(sorted(matched["Dataset"].unique()), sns.color_palette("colorblind", matched["Dataset"].nunique()))}
    dataset_labels = {
        "Aged可追溯194": "Aged-forged dataset",
        "四篇论文12": "Four-study dataset",
        "基础合并147": "Literature/public merged",
        "材料信息学24": "Materials-informatics dataset",
    }
    for ax, (xcol, ycol, xlabel, ylabel, label) in zip(axes.flat[:3], pairs):
        for dataset, part in matched.groupby("Dataset"):
            ax.scatter(part[xcol], part[ycol], s=20, alpha=0.7, color=dataset_colors[dataset], edgecolor="none", label=dataset_labels.get(dataset, str(dataset)))
        rho = matched[[xcol, ycol]].corr(method="spearman").iloc[0, 1]
        annotation_x = 0.025 if label == "b" else 0.04
        annotation_y = 0.92 if label == "b" else 0.94
        ax.text(annotation_x, annotation_y, f"Spearman r = {rho:.3f}", transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB"), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        panel_label(ax, label)
    axes[0, 0].legend(loc="lower right", fontsize=6.6, frameon=True)

    ax = axes[1, 1]
    task_order = ["UTS", "YS", "EL"]
    x = np.arange(3)
    width = 0.34
    full = [fair.loc[(fair["Task"].eq(t)) & fair["Model"].str.startswith("Full_Label"), "R2"].iloc[0] for t in task_order]
    matched_only = [fair.loc[(fair["Task"].eq(t)) & fair["Model"].str.startswith("Matched_Independent"), "R2"].iloc[0] for t in task_order]
    ax.bar(x - width/2, full, width, label="All labels trained; matched scored", color="#668EAA")
    ax.bar(x + width/2, matched_only, width, label="Matched-only trained", color="#9B88C4")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, task_order)
    ax.set_ylabel("R² on the same 266 samples")
    ax.set_title("Matched-subset difficulty is not only a sample-size effect")
    ax.legend(fontsize=6.8)
    panel_label(ax, "d")

    fig.suptitle("Robustness analysis on 266 samples with complete YS–UTS–EL labels", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "Fig6_matched_subset_robustness")


def records(frame: pd.DataFrame):
    clean = frame.copy()
    clean = clean.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def build_payload(data):
    oof, pred_metrics, importance, direction, ad_summary, intervals, boot_summary, lodo, ys_el_overall, ys_el_boot, mtl, matched_counts, matched_fair, matched_multi, matched_corr, exclusions = data
    main_metrics = pd.DataFrame([
        {"Target": "UTS", "Role": "Primary", "Rows": 675, "Sources": 258, "R2": 0.526821, "RMSE": 69.125167, "MAE": 49.701704, "Unit": "MPa", "Evidence": "Moderate; source-bootstrap R2 CI excludes zero"},
        {"Target": "YS", "Role": "Exploratory", "Rows": 307, "Sources": 63, "R2": 0.16078931010480002, "RMSE": 98.98868113165638, "MAE": 69.60009194949133, "Unit": "MPa", "Evidence": "Weak; source-bootstrap R2 CI crosses zero"},
        {"Target": "EL", "Role": "Exploratory", "Rows": 537, "Sources": 164, "R2": 0.140361, "RMSE": 4.20638, "MAE": 2.84515, "Unit": "%", "Evidence": "Weak; source-bootstrap R2 CI crosses zero"},
        {"Target": "HV", "Role": "Small-sample only", "Rows": None, "Sources": None, "R2": None, "RMSE": None, "MAE": None, "Unit": "HV", "Evidence": "Not included in the main rebuilt workflow"},
    ])
    workflow = pd.DataFrame([
        (1, "Traceable data assembly", "Unify literature and public-database fields; preserve Sample_ID, source and DOI", "Completed"),
        (2, "Cross-target matching", "Match rows by Dataset + Original_Sample_ID; retain partial labels", "Completed"),
        (3, "Scope audit", "Exclude only clearly out-of-scope materials/tests; do not delete by residual size", "Completed"),
        (4, "EDA and sparsity audit", "Check counts, missingness, anomalies, zero prevalence and target comparability", "Completed"),
        (5, "Source-exclusive folds", "Keep each Source_Group in one outer fold", "Completed"),
        (6, "Independent model comparison", "RF, XGBoost and equal-weight ensemble", "Completed"),
        (7, "Physical-noise augmentation ablation", "Generate only within training folds; nested selection retained no augmentation", "Completed"),
        (8, "Derived-descriptor audit", "Safe ratio definitions, literature basis, sensitivity and absorption tests", "Completed"),
        (9, "Sparse-feature selection", "Nested source-group selection; lock UTS to Zn, Mg, Cu, Fe and Zr", "Completed"),
        (10, "UTS explanation and credibility", "OOF SHAP, fold stability, LODO, applicability domain, bootstrap and conformal intervals", "Completed"),
        (11, "Partial-label MTL benchmark", "Compare shared MLP with independent RF under identical folds/features", "Completed"),
        (12, "Matched-subset robustness", "266 complete samples; independent vs joint models and target correlations", "Completed"),
        (13, "Paper outputs", "Final tables and candidate figures; manuscript writing follows separately", "In progress"),
    ], columns=["Step", "Stage", "Rule_or_Output", "Status"])
    figure_index = pd.DataFrame([
        ("Fig. 1", "Workflow and evidence hierarchy", "Main text", "Shows traceability, source-aware validation and target roles", "Fig1_workflow_and_evidence_hierarchy.png"),
        ("Fig. 2", "UTS source-blocked performance", "Main text", "Shows moderate OOF prediction and positive fold R2", "Fig2_UTS_source_blocked_performance.png"),
        ("Fig. 3", "UTS SHAP interpretation", "Main text", "Shows direct-composition importance and effect distributions", "Fig3_UTS_SHAP_importance_and_effects.png"),
        ("Fig. 4", "UTS credibility and transfer", "Main text", "Shows bootstrap, AD, conformal coverage and transfer domains", "Fig4_UTS_credibility_and_transfer.png"),
        ("Fig. 5", "Target positioning and MTL", "Main/Supplement decision", "Shows why only UTS is confirmatory and MTL is supplementary", "Fig5_target_positioning_and_MTL.png"),
        ("Fig. 6", "Matched-subset robustness", "Supplementary candidate", "Shows same-sample correlations and model difficulty", "Fig6_matched_subset_robustness.png"),
    ], columns=["Figure", "Title", "Recommended_Location", "Takeaway", "File"])

    return {
        "README": records(pd.DataFrame([
            ("Purpose", "Audit-ready result tables and source data for the approved paper positioning"),
            ("Primary claim", "UTS has moderate source-aware predictive performance; YS and EL are exploratory"),
            ("Validation unit", "Source_Group, not random rows"),
            ("Main UTS features", "Zn | Mg | Cu | Fe | Zr"),
            ("Augmentation", "Not used in the final model after nested selection"),
            ("Derived descriptors", "Sensitivity only; not used in main prediction or SHAP"),
            ("MTL", "Supplementary negative-transfer benchmark"),
            ("Matched subset", "266 rows, 59 sources, 4 datasets; robustness only"),
            ("Manuscript status", "Writing intentionally deferred until tables and figures are approved"),
        ], columns=["Item", "Decision"])),
        "Workflow": records(workflow),
        "Main_Metrics": records(main_metrics),
        "UTS_Fold_Metrics": records(pred_metrics),
        "UTS_Bootstrap": records(boot_summary),
        "UTS_LODO": records(lodo),
        "UTS_SHAP": records(importance),
        "SHAP_Direction": records(direction),
        "Applicability_Domain": records(ad_summary),
        "Conformal_Intervals": records(intervals),
        "YS_EL_Variants": records(ys_el_overall),
        "YS_EL_Bootstrap": records(ys_el_boot),
        "MTL_Comparison": records(mtl),
        "Matched_Datasets": records(matched_counts),
        "Matched_Fair_Models": records(matched_fair),
        "Matched_Joint_Models": records(matched_multi),
        "Matched_Correlations": records(matched_corr),
        "Excluded_Sources": records(exclusions),
        "Figure_Index": records(figure_index),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()

    oof = read(FINAL_UTS / "oof_shap" / "scope_clean_oof_shap_values_wide.csv")
    pred_metrics = read(FINAL_UTS / "oof_shap" / "prediction_metrics.csv")
    importance = read(FINAL_UTS / "oof_shap" / "global_importance_by_model.csv")
    direction = read(FINAL_UTS / "oof_shap" / "direction_stability.csv")
    cred = FINAL_UTS / "credibility"
    ad_summary = read(cred / "applicability_summary.csv")
    intervals = read(cred / "prediction_interval_summary.csv")
    boot_samples = read(cred / "source_cluster_bootstrap_samples.csv")
    boot_summary = read(cred / "source_cluster_bootstrap_summary.csv")
    lodo = read(RESULTS / "uts_systematic_scope_audit" / "scope_clean_leave_one_dataset_out_pooled_metrics.csv")
    ys_el = RESULTS / "ys_el_scope_audit"
    ys_el_overall = read(ys_el / "ys_el_variant_overall_metrics.csv")
    ys_el_boot = read(ys_el / "scope_clean_source_bootstrap_summary.csv")
    mtl = read(RESULTS / "scope_clean_partial_label_mtl" / "metrics_oof_summary.csv")
    matched_root = RESULTS / "matched_subset_final_robustness"
    matched = read(matched_root / "matched_complete_266_with_outer_folds.csv")
    matched_counts = read(matched_root / "matched_dataset_counts.csv")
    matched_fair = read(matched_root / "fair_comparison_full_vs_matched_only_metrics.csv")
    matched_multi = read(matched_root / "matched_multioutput_vs_independent_rf_metrics.csv")
    matched_corr = read(matched_root / "target_correlations_row_and_source_mean.csv")
    exclusions = read(RESULTS / "uts_systematic_scope_audit" / "excluded_sources_5.csv")

    workflow_figure()
    uts_performance_figure(oof, pred_metrics)
    shap_figure(oof, importance)
    credibility_figure(boot_samples, boot_summary, ad_summary, intervals, lodo, matched_fair)
    multitarget_figure(mtl, matched_corr, ys_el_boot, boot_summary, matched_multi)
    matched_figure(matched, matched_fair)

    payload = build_payload((oof, pred_metrics, importance, direction, ad_summary, intervals, boot_summary, lodo,
                             ys_el_overall, ys_el_boot, mtl, matched_counts, matched_fair, matched_multi,
                             matched_corr, exclusions))
    (OUT / "paper_results_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_files = sorted(FIG.glob("*.png"))
    audit = pd.DataFrame({"Figure": [p.name for p in figure_files], "Bytes": [p.stat().st_size for p in figure_files]})
    audit.to_csv(OUT / "figure_file_audit.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))
    print(f"Payload: {OUT / 'paper_results_payload.json'}")


if __name__ == "__main__":
    main()
