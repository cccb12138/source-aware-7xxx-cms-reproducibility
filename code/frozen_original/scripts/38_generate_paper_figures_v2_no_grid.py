from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(r"D:\Jupyter\Al7xxx_Traceable_Modeling")
RESULTS = PROJECT / "results"
OUT = Path(r"F:\CC\outputs\paper_results_v2")
FIG = OUT / "figures"
DATA = OUT / "figure_data"
FINAL_UTS = Path(r"F:\CC\outputs\uts_scope_clean_final")
DECISIONS = Path(r"F:\CC\outputs\paper_scope_clean_final\model_decisions")
BASE_SCRIPT = Path(r"F:\CC\stage_al7xxx\34_generate_paper_results_v1.py")

spec = importlib.util.spec_from_file_location("paper_figures_v1", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
base.OUT = OUT
base.FIG = FIG

BLUE = base.BLUE
ORANGE = base.ORANGE
GREEN = base.GREEN
RED = base.RED
GRAY = base.GRAY
PURPLE = "#7768AE"

DATASET_LABELS = {
    "Acta_UTS461": "Acta dataset",
    "Aged可追溯194": "Aged-forged dataset",
    "四篇论文12": "Four-study dataset",
    "基础合并147": "Literature/public merged",
    "材料信息学24": "Materials-informatics dataset",
    "补充文献260": "Supplementary literature",
}


def save(fig, stem):
    for ax in fig.axes:
        ax.grid(False)
    fig.savefig(FIG / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, label):
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def fig1_data_and_validation(mtl):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.25), gridspec_kw={"width_ratios": [1.0, 1.1, 1.25]})

    ax = axes[0]
    audit = pd.DataFrame({
        "Target": ["YS", "UTS", "EL"],
        "Before_scope_audit": [307, 689, 550],
        "Scope_clean": [307, 675, 537],
    })
    y = np.arange(3)
    h = 0.34
    ax.barh(y + h/2, audit["Before_scope_audit"], h, color="#CBD5E1", label="Before scope audit")
    ax.barh(y - h/2, audit["Scope_clean"], h, color=[ORANGE, BLUE, GREEN], label="Scope-clean")
    ax.set_yticks(y, audit["Target"])
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("Target-specific scope audit")
    for i, row in audit.iterrows():
        removed = row.Before_scope_audit - row.Scope_clean
        ax.text(row.Before_scope_audit + 10, i + h/2, str(row.Before_scope_audit), va="center", fontsize=7.5, color=GRAY)
        ax.text(row.Scope_clean + 10, i - h/2, str(row.Scope_clean), va="center", fontsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.2, frameon=False)
    panel(ax, "a")

    ax = axes[1]
    combos = (
        mtl.assign(
            Combination=mtl[["Mask_YS", "Mask_UTS", "Mask_EL"]].astype(int).apply(
                lambda r: "+".join([task for task, mask in zip(["YS", "UTS", "EL"], r) if mask == 1]), axis=1
            )
        )
        .groupby("Combination", as_index=False)
        .size()
        .sort_values("size", ascending=True)
    )
    colors = [PURPLE if x == "YS+UTS+EL" else "#7FA6BF" for x in combos["Combination"]]
    ax.barh(combos["Combination"], combos["size"], color=colors)
    ax.set_xlabel("Rows")
    ax.set_title("Exact partial-label combinations")
    for i, value in enumerate(combos["size"]):
        ax.text(value + 5, i, str(value), va="center", fontsize=7.7)
    ax.text(0.98, 0.05, "Triple-complete subset: 266 rows", transform=ax.transAxes, ha="right", fontsize=7.8, color=PURPLE)
    panel(ax, "b")

    ax = axes[2]
    n_sources = 25
    n_folds = 5
    matrix = np.ones((n_sources, n_folds))
    for source in range(n_sources):
        matrix[source, source // 5] = 0
    cmap = mpl.colors.ListedColormap([ORANGE, "#DCEAF3"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    for split in range(1, 5):
        ax.axhline(split * 5 - 0.5, color="white", linewidth=1.6)
    ax.set_xticks(np.arange(5), [f"Fold {i+1}" for i in range(5)], rotation=0, ha="center", fontsize=7.5)
    ax.set_yticks([2, 7, 12, 17, 22], ["Source block 1", "Source block 2", "Source block 3", "Source block 4", "Source block 5"])
    ax.set_title("Source-exclusive outer validation (schematic)")
    ax.set_xlabel("Evaluation split")
    ax.legend(handles=[Patch(facecolor=ORANGE, label="Test sources"), Patch(facecolor="#DCEAF3", label="Training sources")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=7.5)
    panel(ax, "c")

    fig.suptitle("Data structure and leakage-resistant validation design", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "Fig1_data_structure_and_validation_design")
    audit.to_csv(DATA / "Fig1a_scope_audit_counts.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(DATA / "Fig1b_partial_label_combinations.csv", index=False, encoding="utf-8-sig")


def fig7_model_decisions():
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))

    comparison = pd.read_csv(DECISIONS / "model_comparison.csv")
    model_order = ["DummyMedian", "Ridge", "XGBoost", "ExtraTrees", "RandomForest", "RF+XGB ensemble"]
    model_values = [comparison.loc[comparison["Model"].eq(name), "R2"].iloc[0] for name in model_order]
    ax = axes[0, 0]
    colors = ["#BFC7CE"] * 5 + [BLUE]
    bars = ax.barh(model_order, model_values, color=colors)
    ax.set_xlim(-0.05, 0.57)
    ax.set_xlabel("OOF R²")
    ax.set_title("Model comparison under source-blocked CV")
    for bar, value in zip(bars, model_values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=7.5)
    panel(ax, "a")

    sparse = pd.read_csv(DECISIONS / "feature_configuration_comparison.csv")
    mapping = {
        "full10": "Full 10",
        "drop_si": "Drop Si",
        "refined5": "Refined 5",
        "major3": "Major 3",
    }
    p = sparse.loc[sparse["Configuration"].isin(mapping)].copy()
    p["Label"] = p["Configuration"].map(mapping)
    order = ["Major 3", "Full 10", "Drop Si", "Refined 5"]
    p["Label"] = pd.Categorical(p["Label"], order, ordered=True)
    p = p.sort_values("Label")
    ax = axes[0, 1]
    colors = [BLUE if label == "Refined 5" else "#91AFC3" for label in p["Label"].astype(str)]
    bars = ax.bar(p["Label"].astype(str), p["R2"], color=colors)
    ax.set_ylim(0.40, 0.56)
    ax.set_ylabel("OOF R²")
    ax.set_title("Sparse-feature decision")
    ax.set_xticks(
        np.arange(len(p)),
        ["Major 3", "Full 10", "Drop Si", "Refined 5"],
        rotation=0,
    )
    for bar, value in zip(bars, p["R2"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.004, f"{value:.3f}", ha="center", fontsize=7.3)
    ax.text(0.03, 0.96, "Final: Zn, Mg, Cu, Fe, Zr", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=BLUE)
    panel(ax, "b")

    augmentation = pd.read_csv(DECISIONS / "augmentation_configuration_comparison.csv")
    augmentation = augmentation.set_index("Configuration").loc[
        ["no_augmentation", "half_sigma_2copies", "nominal_sigma_2copies", "double_sigma_2copies"]
    ].reset_index()
    aug_labels = ["None", "Half σ", "Nominal σ", "Double σ"]
    ax = axes[1, 0]
    bars = ax.bar(aug_labels, augmentation["R2"], color=[BLUE, "#8EABC0", "#D5A16C", "#C98E8E"])
    ax.set_ylim(0.49, 0.54)
    ax.set_ylabel("OOF R²")
    ax.set_title("Augmentation decision")
    ax.set_xticks(
        np.arange(len(augmentation)),
        aug_labels,
        rotation=0,
    )
    for bar, value in zip(bars, augmentation["R2"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.0015, f"{value:.3f}", ha="center", fontsize=7.4)
    ax.text(0.03, 0.95, "Nested selection retained no augmentation", transform=ax.transAxes, va="top", fontsize=7.2, color=BLUE)
    panel(ax, "c")

    proxy = pd.read_csv(RESULTS / "proxy_nested_selection_ys" / "configuration_comparison.csv")
    proxy_map = {
        "Direct_Composition": "Direct",
        "Forced_Literature_Ratios": "Forced ratios",
        "Forced_All_Safe": "Forced safe set",
        "Nested_OneSE_Selection": "Nested selection",
    }
    pp = proxy.loc[proxy["Configuration"].isin(proxy_map)].copy()
    pp["Label"] = pp["Configuration"].map(proxy_map)
    pp["Label"] = pd.Categorical(pp["Label"], ["Direct", "Forced ratios", "Forced safe set", "Nested selection"], ordered=True)
    pp = pp.sort_values("Label")
    ax = axes[1, 1]
    bars = ax.bar(pp["Label"].astype(str), pp["R2"], color=["#8EABC0", "#D5A16C", "#D5A16C", BLUE])
    ax.set_ylim(0.15, 0.26)
    ax.set_ylabel("YS OOF R²")
    ax.set_title("Derived-descriptor decision")
    ax.set_xticks(
        np.arange(len(pp)),
        ["Direct", "Forced\nratios", "Forced\nsafe set", "Nested\nselection"],
        rotation=0,
    )
    for bar, value in zip(bars, pp["R2"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.003, f"{value:.3f}", ha="center", fontsize=7.4)
    ax.text(0.03, 0.95, "Forced gains did not survive nested selection", transform=ax.transAxes, va="top", fontsize=7.1, color=BLUE)
    panel(ax, "d")

    fig.suptitle("Nested ablations distinguish stable decisions from optimistic sensitivity results", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "Fig7_nested_model_and_feature_decisions")

    pd.DataFrame({"Model": model_order, "R2": model_values}).to_csv(DATA / "Fig7a_model_comparison.csv", index=False, encoding="utf-8-sig")
    p.to_csv(DATA / "Fig7b_sparse_feature_decision.csv", index=False, encoding="utf-8-sig")
    augmentation.assign(Label=aug_labels).to_csv(DATA / "Fig7c_augmentation_decision.csv", index=False, encoding="utf-8-sig")
    pp.to_csv(DATA / "Fig7d_descriptor_decision.csv", index=False, encoding="utf-8-sig")


def fig8_source_heterogeneity(oof):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))

    dataset_counts = oof.groupby("Dataset").agg(Rows=("Model_Row_ID", "size"), Sources=("Source_Group", "nunique")).reset_index()
    dataset_counts["Label"] = dataset_counts["Dataset"].map(DATASET_LABELS).fillna(dataset_counts["Dataset"].astype(str))
    dataset_counts = dataset_counts.sort_values("Rows")
    ax = axes[0, 0]
    bars = ax.barh(dataset_counts["Label"], dataset_counts["Rows"], color="#7FA6BF")
    ax.set_xlabel("UTS rows")
    ax.set_title("Unequal representation across datasets")
    for bar, rows, sources in zip(bars, dataset_counts["Rows"], dataset_counts["Sources"]):
        ax.text(rows + 4, bar.get_y() + bar.get_height()/2, f"{rows} ({sources} src)", va="center", fontsize=7.2)
    panel(ax, "a")

    holdout = pd.read_csv(RESULTS / "uts_systematic_scope_audit" / "scope_clean_hierarchical_holdout_metrics.csv")
    lodo = holdout.loc[holdout["Scheme"].eq("Leave_One_Dataset_Out")].copy()
    lodo["Label"] = lodo["Holdout_Block"].map(DATASET_LABELS).fillna(lodo["Holdout_Block"].astype(str))
    lodo = lodo.sort_values("R2")
    ax = axes[0, 1]
    colors = [RED if value < 0 else BLUE for value in lodo["R2"]]
    bars = ax.barh(lodo["Label"], lodo["R2"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Leave-one-dataset-out R²")
    ax.set_title("Transfer performance is dataset-dependent")
    for bar, value in zip(bars, lodo["R2"]):
        if value < -0.15:
            x, ha, color = value + 0.04, "left", "white"
        elif value < 0:
            x, ha, color = value - 0.02, "right", RED
        else:
            x, ha, color = value + 0.02, "left", "#111827"
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", ha=ha, fontsize=7.2, color=color)
    panel(ax, "b")

    per_source = pd.read_csv(RESULTS / "uts_scope_clean_final" / "credibility" / "per_source_metrics.csv")
    ax = axes[1, 0]
    ax.scatter(per_source["Rows"], per_source["MAE"], s=18, alpha=0.65,
               c=np.where(per_source["MAE_Outlier"], RED, "#6E9CB8"), edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Rows per source (log scale)")
    ax.set_ylabel("Source MAE (MPa)")
    ax.set_title("Source-level errors remain heterogeneous")
    ax.legend(handles=[Patch(facecolor="#6E9CB8", label="Typical source"), Patch(facecolor=RED, label="MAE diagnostic outlier")],
              frameon=False, fontsize=7.3)
    panel(ax, "c")

    plot_oof = oof.copy()
    plot_oof["Absolute_Error"] = (plot_oof["y_pred"] - plot_oof["y_true"]).abs()
    plot_oof["Dataset_Label"] = plot_oof["Dataset"].map(DATASET_LABELS).fillna(plot_oof["Dataset"].astype(str))
    order = dataset_counts.sort_values("Rows", ascending=False)["Label"].tolist()
    ax = axes[1, 1]
    sns.boxplot(data=plot_oof, x="Absolute_Error", y="Dataset_Label", order=order, color="#98B8CB",
                showfliers=False, linewidth=0.8, ax=ax)
    ax.set_xlabel("Absolute OOF error (MPa)")
    ax.set_ylabel("")
    ax.set_title("Error distributions differ across data domains")
    panel(ax, "d")

    fig.suptitle("Source and dataset heterogeneity define the practical boundary of UTS prediction", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "Fig8_source_and_dataset_heterogeneity")

    dataset_counts.to_csv(DATA / "Fig8a_dataset_counts.csv", index=False, encoding="utf-8-sig")
    lodo.to_csv(DATA / "Fig8b_dataset_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    per_source.to_csv(DATA / "Fig8c_per_source_metrics.csv", index=False, encoding="utf-8-sig")
    plot_oof[["Model_Row_ID", "Source_Group", "Dataset", "Dataset_Label", "Absolute_Error"]].to_csv(
        DATA / "Fig8d_dataset_absolute_errors.csv", index=False, encoding="utf-8-sig"
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    base.style()
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update({"axes.grid": False, "axes.spines.top": False, "axes.spines.right": False})

    mtl = pd.read_csv(RESULTS / "scope_clean_partial_label_mtl" / "MTL_scope_clean_with_outer_folds.csv")
    oof = pd.read_csv(FINAL_UTS / "oof_shap" / "scope_clean_oof_shap_values_wide.csv")
    metrics = pd.read_csv(FINAL_UTS / "oof_shap" / "prediction_metrics.csv")
    importance = pd.read_csv(FINAL_UTS / "oof_shap" / "global_importance_by_model.csv")
    cred = FINAL_UTS / "credibility"
    boot_samples = pd.read_csv(cred / "source_cluster_bootstrap_samples.csv")
    boot_summary = pd.read_csv(cred / "source_cluster_bootstrap_summary.csv")
    ad = pd.read_csv(cred / "applicability_summary.csv")
    intervals = pd.read_csv(cred / "prediction_interval_summary.csv")
    lodo = pd.read_csv(RESULTS / "uts_systematic_scope_audit" / "scope_clean_leave_one_dataset_out_pooled_metrics.csv")
    mtl_metrics = pd.read_csv(RESULTS / "scope_clean_partial_label_mtl" / "metrics_oof_summary.csv")
    matched_root = RESULTS / "matched_subset_final_robustness"
    matched = pd.read_csv(matched_root / "matched_complete_266_with_outer_folds.csv")
    matched_corr = pd.read_csv(matched_root / "target_correlations_row_and_source_mean.csv")
    matched_fair = pd.read_csv(matched_root / "fair_comparison_full_vs_matched_only_metrics.csv")
    matched_multi = pd.read_csv(matched_root / "matched_multioutput_vs_independent_rf_metrics.csv")
    ys_el_boot = pd.read_csv(RESULTS / "ys_el_scope_audit" / "scope_clean_source_bootstrap_summary.csv")

    fig1_data_and_validation(mtl)
    base.uts_performance_figure(oof, metrics)
    base.shap_figure(oof, importance)
    base.credibility_figure(boot_samples, boot_summary, ad, intervals, lodo, matched_fair)
    base.multitarget_figure(mtl_metrics, matched_corr, ys_el_boot, boot_summary, matched_multi)
    base.matched_figure(matched, matched_fair)
    fig7_model_decisions()
    fig8_source_heterogeneity(oof)

    # Rename the unchanged-content figures into the V2 eight-figure sequence.
    old_to_new = {
        "Fig2_UTS_source_blocked_performance": "Fig2_UTS_source_blocked_performance",
        "Fig3_UTS_SHAP_importance_and_effects": "Fig3_UTS_SHAP_importance_and_effects",
        "Fig4_UTS_credibility_and_transfer": "Fig4_UTS_credibility_and_transfer",
        "Fig5_target_positioning_and_MTL": "Fig5_target_positioning_and_MTL",
        "Fig6_matched_subset_robustness": "Fig6_matched_subset_robustness",
    }
    index = pd.DataFrame([
        (1, "Data structure and validation design", "Methods / end of Introduction", "New academic-style data/validation figure"),
        (2, "UTS source-blocked predictive performance", "Results", "Accepted content; background grids removed"),
        (3, "UTS SHAP importance and effects", "Results", "Accepted content; background grids removed"),
        (4, "UTS credibility and transfer", "Results", "Accepted content; background grids removed"),
        (5, "Target positioning and MTL", "Results", "Accepted content; background grids removed"),
        (6, "Matched-subset robustness", "Results or Supplement", "Same-sample robustness evidence"),
        (7, "Nested model and feature decisions", "Results / ablation subsection", "New: model, sparse feature, augmentation and descriptor decisions"),
        (8, "Source and dataset heterogeneity", "Discussion / limitations", "New: dataset transfer and source-level error boundary"),
    ], columns=["Figure", "Title", "Recommended_Location", "Revision_or_Takeaway"])
    index.to_csv(OUT / "figure_index_v2.csv", index=False, encoding="utf-8-sig")
    print(index.to_string(index=False))
    print(f"Saved to {FIG}")


if __name__ == "__main__":
    main()
