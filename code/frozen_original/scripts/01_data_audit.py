from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data_utils import (
    ensure_output_dirs,
    expected_targets,
    feature_columns,
    numeric_feature_summary,
    read_sheet,
    validate_mtl_masks,
    write_json,
)


TASKS = ["YS", "UTS", "EL", "HV", "MTL", "TRIPLE", "FOUR"]
MODES = ["strict", "expanded"]


def composition_plausibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious values for manual review; never deletes or overwrites rows."""
    records = []
    for _, row in df.iterrows():
        critical = []
        review = []
        ratio = []

        def val(name):
            return pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]

        al, zn, mg, cu = val("Al"), val("Zn"), val("Mg"), val("Cu")
        ti, zr, mn, cr = val("Ti"), val("Zr"), val("Mn"), val("Cr")
        solute = val("Reported_Solute_Sum")
        grade = str(row.get("Alloy_Grade", ""))

        if pd.notna(zn) and zn <= 0:
            critical.append("Zn_zero_or_negative_in_7xxx")
        if pd.notna(al) and (al < 75 or al > 99.9):
            critical.append("Al_outside_75_to_99.9")
        if pd.notna(cu) and cu > 6:
            critical.append("Cu_gt_6wtpct")
        if pd.notna(ti) and ti > 1:
            critical.append("Ti_gt_1wtpct")
        if pd.notna(zr) and zr > 1:
            critical.append("Zr_gt_1wtpct")

        if pd.notna(zn) and zn > 12:
            review.append("Zn_gt_12wtpct")
        if pd.notna(mg) and mg > 4.5:
            review.append("Mg_gt_4.5wtpct")
        if pd.notna(cu) and 4 < cu <= 6:
            review.append("Cu_4_to_6wtpct")
        if pd.notna(ti) and 0.3 < ti <= 1:
            review.append("Ti_0.3_to_1wtpct")
        if pd.notna(zr) and 0.5 < zr <= 1:
            review.append("Zr_0.5_to_1wtpct")
        if pd.notna(mn) and mn > 1.5:
            review.append("Mn_gt_1.5wtpct")
        if pd.notna(cr) and cr > 0.5:
            review.append("Cr_gt_0.5wtpct")
        if pd.notna(solute) and solute > 20:
            review.append("Reported_solute_gt_20wtpct")

        if pd.notna(mg) and mg < 0.05:
            ratio.append("Zn_Mg_ratio_unstable_Mg_lt_0.05")
        if pd.notna(cu) and cu < 0.05:
            ratio.append("Zn_Cu_and_Mg_Cu_unstable_Cu_lt_0.05")

        if critical or review or ratio:
            records.append({
                "Model_Row_ID": row.get("Model_Row_ID"),
                "Source_Group": row.get("Source_Group"),
                "DOI": row.get("DOI"),
                "Original_Row_ID": row.get("Original_Row_ID"),
                "Original_Sample_ID": row.get("Original_Sample_ID"),
                "Alloy_Grade": grade,
                "Al": al, "Zn": zn, "Mg": mg, "Cu": cu,
                "Ti": ti, "Zr": zr, "Mn": mn, "Cr": cr,
                "Reported_Solute_Sum": solute,
                "Critical_Flag": "|".join(critical),
                "Review_Flag": "|".join(review),
                "Ratio_Stability_Flag": "|".join(ratio),
            })
    return pd.DataFrame(records)


def target_summary(df: pd.DataFrame, target: str) -> dict:
    s = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        return {"target": target, "observed": 0}
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    return {
        "target": target,
        "observed": int(valid.size),
        "missing": int(s.isna().sum()),
        "min": valid.min(),
        "q1": q1,
        "median": valid.median(),
        "q3": q3,
        "max": valid.max(),
        "mean": valid.mean(),
        "std": valid.std(),
        "iqr_3x_flag_count": int(((valid < low) | (valid > high)).sum()),
        "iqr_3x_low": low,
        "iqr_3x_high": high,
    }


def formula_checks(df: pd.DataFrame) -> list[dict]:
    checks = {
        "Zn_Mg_Ratio": (df["Zn"] / df["Mg"].replace(0, np.nan)),
        "Zn_Cu_Ratio": (df["Zn"] / df["Cu"].replace(0, np.nan)),
        "Mg_Cu_Ratio": (df["Mg"] / df["Cu"].replace(0, np.nan)),
        "Zn_Mg_Sum": df["Zn"] + df["Mg"],
        "Zn_Mg_Cu_Sum": df["Zn"] + df["Mg"] + df["Cu"],
        "Zn_Share_ZnMgCu": df["Zn"] / (df["Zn"] + df["Mg"] + df["Cu"]).replace(0, np.nan),
    }
    records = []
    for col, expected in checks.items():
        actual = pd.to_numeric(df[col], errors="coerce")
        comparable = actual.notna() & expected.notna()
        diff = (actual[comparable] - expected[comparable]).abs()
        records.append({
            "feature": col,
            "comparable": int(comparable.sum()),
            "mismatch_gt_1e_6": int((diff > 1e-6).sum()),
            "max_abs_error": diff.max() if len(diff) else np.nan,
        })
    return records


def main():
    ensure_output_dirs()
    report_dir = config.OUTPUT_DIRS["reports"]
    overview_rows = []
    target_rows = []
    formula_rows = []
    plausibility_summary = []

    for mode in MODES:
        mode_dir = report_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            df = read_sheet(task, mode)
            targets = expected_targets(task)
            duplicate_ids = int(df["Model_Row_ID"].duplicated().sum())
            exact_duplicate_rows = int(df.duplicated().sum())
            overview_rows.append({
                "mode": mode,
                "task": task,
                "rows": len(df),
                "source_groups": int(df["Source_Group"].nunique()),
                "duplicate_model_ids": duplicate_ids,
                "exact_duplicate_rows": exact_duplicate_rows,
                "largest_source_n": int(df["Source_Group"].value_counts().max()),
            })

            for target in targets:
                if target in df:
                    target_rows.append({"mode": mode, "task": task, **target_summary(df, target)})

            candidate_features = feature_columns(df, "composition_derived_process")
            feature_report = numeric_feature_summary(df, candidate_features)
            feature_report.insert(0, "task", task)
            feature_report.insert(0, "mode", mode)
            feature_report.to_csv(mode_dir / f"{task}_feature_coverage.csv", index=False, encoding="utf-8-sig")

            source_counts = df.groupby(["Source_Group", "Sensitivity_Group"], dropna=False).size().rename("rows").reset_index()
            source_counts.to_csv(mode_dir / f"{task}_source_counts.csv", index=False, encoding="utf-8-sig")

            for record in formula_checks(df):
                formula_rows.append({"mode": mode, "task": task, **record})

            if task == "MTL":
                mask_report = validate_mtl_masks(df)
                mask_report.insert(0, "mode", mode)
                mask_report.to_csv(mode_dir / "MTL_mask_validation.csv", index=False, encoding="utf-8-sig")
                if mask_report["mask_mismatch"].sum() != 0:
                    raise ValueError(f"MTL mask mismatch found in {mode} data")

                flags = composition_plausibility_flags(df)
                flags.to_csv(
                    mode_dir / "MTL_composition_plausibility_flags.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                plausibility_summary.append({
                    "mode": mode,
                    "flagged_rows": len(flags),
                    "critical_rows": int(flags["Critical_Flag"].ne("").sum()) if len(flags) else 0,
                    "review_rows": int(flags["Review_Flag"].ne("").sum()) if len(flags) else 0,
                    "ratio_stability_rows": int(flags["Ratio_Stability_Flag"].ne("").sum()) if len(flags) else 0,
                })

            if duplicate_ids or exact_duplicate_rows:
                raise ValueError(f"Unexpected duplicates found: mode={mode}, task={task}")

    overview = pd.DataFrame(overview_rows)
    targets = pd.DataFrame(target_rows)
    formulas = pd.DataFrame(formula_rows)
    overview.to_csv(report_dir / "dataset_overview.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(report_dir / "target_summary.csv", index=False, encoding="utf-8-sig")
    formulas.to_csv(report_dir / "derived_formula_validation.csv", index=False, encoding="utf-8-sig")

    payload = {
        "workbook": config.WORKBOOK_PATH,
        "overview": overview.to_dict("records"),
        "target_summary": targets.to_dict("records"),
        "derived_formula_mismatch_total": int(formulas["mismatch_gt_1e_6"].sum()),
        "composition_plausibility": plausibility_summary,
        "status": "PASS" if int(formulas["mismatch_gt_1e_6"].sum()) == 0 else "REVIEW",
    }
    write_json(report_dir / "audit_summary.json", payload)

    print("\nDATA AUDIT OVERVIEW")
    print(overview.to_string(index=False))
    print("\nDERIVED FORMULA CHECK")
    print(formulas.groupby(["mode", "task"])["mismatch_gt_1e_6"].sum().to_string())
    print(f"\nSaved Stage-1 audit reports to: {report_dir}")


if __name__ == "__main__":
    main()
