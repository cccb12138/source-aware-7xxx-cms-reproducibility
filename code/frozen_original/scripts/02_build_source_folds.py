from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data_utils import ensure_output_dirs, expected_targets, read_sheet, write_json


MAIN_TASKS = ["YS", "UTS", "EL", "MTL", "TRIPLE"]
HV_TASKS = ["HV", "FOUR"]
MODES = ["strict", "expanded"]


def add_target_bins(work: pd.DataFrame, targets: list[str], n_bins: int = 5) -> tuple[pd.DataFrame, list[str]]:
    metric_cols = ["row_count"]
    work = work.copy()
    work["row_count"] = 1
    for target in targets:
        observed_col = f"observed__{target}"
        work[observed_col] = work[target].notna().astype(int)
        metric_cols.append(observed_col)
        valid = work[target].notna()
        if valid.sum() < n_bins:
            continue
        bins = pd.qcut(work.loc[valid, target].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
        for b in sorted(bins.dropna().unique()):
            col = f"bin__{target}__{int(b)}"
            work[col] = 0
            work.loc[bins.index[bins.eq(b)], col] = 1
            metric_cols.append(col)
    return work, metric_cols


def greedy_group_folds(
    df: pd.DataFrame,
    targets: list[str],
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df["Source_Group"].nunique() < n_splits:
        raise ValueError("Not enough Source_Group values for requested folds")
    work, metrics = add_target_bins(df[["Source_Group"] + targets], targets)
    grouped = work.groupby("Source_Group", sort=False)[metrics].sum()
    totals = grouped.sum(axis=0).to_numpy(dtype=float)
    desired = np.where(totals > 0, totals / n_splits, 1.0)

    rng = np.random.default_rng(seed)
    order = grouped.copy()
    order["_priority"] = grouped["row_count"] + grouped.drop(columns=["row_count"]).sum(axis=1) * 0.05
    order["_jitter"] = rng.random(len(order)) * 1e-6
    order = order.sort_values(["_priority", "_jitter"], ascending=False)

    state = np.zeros((n_splits, len(metrics)), dtype=float)
    assignments = {}
    for i, group in enumerate(order.index):
        vector = grouped.loc[group, metrics].to_numpy(dtype=float)
        if i < n_splits:
            chosen = i
        else:
            scores = []
            for fold in range(n_splits):
                trial = state.copy()
                trial[fold] += vector
                normalized = trial / desired
                score = float(np.square(normalized).sum())
                scores.append(score)
            chosen = int(np.argmin(scores))
        state[chosen] += vector
        assignments[group] = chosen

    mapping = pd.DataFrame({"Source_Group": list(assignments), "Outer_Fold": list(assignments.values())})
    assigned = df.merge(mapping, on="Source_Group", how="left", validate="many_to_one")
    if assigned["Outer_Fold"].isna().any():
        raise AssertionError("Some rows did not receive a fold")
    assigned["Outer_Fold"] = assigned["Outer_Fold"].astype(int)

    fold_stats = []
    for fold, part in assigned.groupby("Outer_Fold"):
        row = {
            "Outer_Fold": int(fold),
            "rows": len(part),
            "source_groups": int(part["Source_Group"].nunique()),
        }
        for target in targets:
            row[f"observed_{target}"] = int(part[target].notna().sum())
            row[f"mean_{target}"] = pd.to_numeric(part[target], errors="coerce").mean()
        fold_stats.append(row)
    return mapping.sort_values(["Outer_Fold", "Source_Group"]), pd.DataFrame(fold_stats).sort_values("Outer_Fold")


def save_task_with_mapping(task: str, mode: str, mapping: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    df = read_sheet(task, mode, exclude_critical=True)
    for target in expected_targets(task):
        if target in df and task != "MTL":
            df = df.loc[df[target].notna()].copy()
    merged = df.merge(mapping, on="Source_Group", how="left", validate="many_to_one")
    if merged["Outer_Fold"].isna().any():
        missing = merged.loc[merged["Outer_Fold"].isna(), "Source_Group"].drop_duplicates().tolist()
        raise ValueError(f"{task}/{mode}: groups missing from fold map: {missing[:10]}")
    merged["Outer_Fold"] = merged["Outer_Fold"].astype(int)

    # Explicit leakage check: a source may occur in exactly one outer fold.
    leakage = merged.groupby("Source_Group")["Outer_Fold"].nunique().gt(1)
    if leakage.any():
        raise AssertionError(f"Source leakage detected in {task}/{mode}")
    if merged["Outer_Fold"].nunique() != n_folds:
        raise AssertionError(f"{task}/{mode} does not cover all {n_folds} folds")

    out_dir = config.OUTPUT_DIRS["processed"] / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"{task}_with_outer_folds.csv", index=False, encoding="utf-8-sig")
    return merged


def main():
    ensure_output_dirs()
    report_rows = []
    all_mapping_rows = []

    for mode in MODES:
        exclusion_rows = []
        for task in ["YS", "UTS", "EL", "HV", "MTL", "TRIPLE", "FOUR"]:
            before = read_sheet(task, mode, exclude_critical=False)
            after = read_sheet(task, mode, exclude_critical=True)
            removed_ids = sorted(set(before["Model_Row_ID"]) - set(after["Model_Row_ID"]))
            exclusion_rows.append({
                "mode": mode,
                "task": task,
                "rows_before": len(before),
                "rows_excluded": len(removed_ids),
                "rows_after": len(after),
                "excluded_model_row_ids": "|".join(removed_ids),
            })
        pd.DataFrame(exclusion_rows).to_csv(
            config.OUTPUT_DIRS["reports"] / f"{mode}_critical_exclusions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        mtl = read_sheet("MTL", mode, exclude_critical=True)
        main_targets = [config.TARGET_COLUMNS[t] for t in ("YS", "UTS", "EL")]
        main_map, main_stats = greedy_group_folds(
            mtl, main_targets, config.N_OUTER_FOLDS, config.RANDOM_SEED
        )
        main_map.insert(0, "mode", mode)
        main_map.insert(1, "scheme", "MAIN_5FOLD")
        all_mapping_rows.append(main_map)
        main_stats.insert(0, "mode", mode)
        main_stats.insert(1, "scheme", "MAIN_5FOLD")
        main_stats.to_csv(
            config.OUTPUT_DIRS["folds"] / f"{mode}_main_fold_balance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plain_main_map = main_map[["Source_Group", "Outer_Fold"]]
        for task in MAIN_TASKS:
            part = save_task_with_mapping(task, mode, plain_main_map, config.N_OUTER_FOLDS)
            report_rows.append({
                "mode": mode,
                "task": task,
                "rows": len(part),
                "sources": int(part["Source_Group"].nunique()),
                "folds": int(part["Outer_Fold"].nunique()),
            })

        hv = read_sheet("HV", mode, exclude_critical=True)
        four = read_sheet("FOUR", mode, exclude_critical=True)
        hv_target = config.TARGET_COLUMNS["HV"]
        # Target-level and four-target de-duplication use different keys.  A source
        # can therefore survive in FOUR while an equivalent HV row from another
        # source is retained in the target-level HV sheet.  Build the fold universe
        # from HV plus only the source groups that are unique to FOUR.
        hv_groups = set(hv["Source_Group"])
        four_only = four.loc[~four["Source_Group"].isin(hv_groups)].copy()
        hv_fold_base = pd.concat([hv, four_only], ignore_index=True, sort=False)
        source_universe_report = pd.DataFrame({
            "mode": [mode],
            "hv_source_groups": [hv["Source_Group"].nunique()],
            "four_source_groups": [four["Source_Group"].nunique()],
            "four_only_source_groups": [four_only["Source_Group"].nunique()],
            "four_only_sources": ["|".join(sorted(four_only["Source_Group"].unique()))],
        })
        source_universe_report.to_csv(
            config.OUTPUT_DIRS["reports"] / f"{mode}_hv_four_source_universe.csv",
            index=False,
            encoding="utf-8-sig",
        )
        hv_map, hv_stats = greedy_group_folds(
            hv_fold_base, [hv_target], config.N_HV_FOLDS, config.RANDOM_SEED + 1
        )
        hv_map.insert(0, "mode", mode)
        hv_map.insert(1, "scheme", "HV_4FOLD")
        all_mapping_rows.append(hv_map)
        hv_stats.insert(0, "mode", mode)
        hv_stats.insert(1, "scheme", "HV_4FOLD")
        hv_stats.to_csv(
            config.OUTPUT_DIRS["folds"] / f"{mode}_hv_fold_balance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plain_hv_map = hv_map[["Source_Group", "Outer_Fold"]]
        for task in HV_TASKS:
            part = save_task_with_mapping(task, mode, plain_hv_map, config.N_HV_FOLDS)
            report_rows.append({
                "mode": mode,
                "task": task,
                "rows": len(part),
                "sources": int(part["Source_Group"].nunique()),
                "folds": int(part["Outer_Fold"].nunique()),
            })

    mappings = pd.concat(all_mapping_rows, ignore_index=True)
    mappings.to_csv(config.OUTPUT_DIRS["folds"] / "source_group_outer_folds.csv", index=False, encoding="utf-8-sig")
    report = pd.DataFrame(report_rows)
    report.to_csv(config.OUTPUT_DIRS["reports"] / "fold_assignment_summary.csv", index=False, encoding="utf-8-sig")
    write_json(config.OUTPUT_DIRS["reports"] / "fold_assignment_summary.json", report.to_dict("records"))

    print("\nSOURCE-GROUP OUTER FOLDS")
    print(report.to_string(index=False))
    print(f"\nProcessed data: {config.OUTPUT_DIRS['processed']}")
    print(f"Fold maps:      {config.OUTPUT_DIRS['folds']}")


if __name__ == "__main__":
    main()
