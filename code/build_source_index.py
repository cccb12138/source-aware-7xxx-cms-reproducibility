"""Build a provenance-only source index from approved local modelling tables.

Run this script locally before a release. It intentionally exports no chemical
composition, processing parameter, or mechanical-property value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COHORT_FILES = {
    "UTS": "UTS_scope_clean_675.csv",
    "YS": "YS_scope_clean_307.csv",
    "EL": "EL_scope_clean_537.csv",
    "MTL": "Partial_label_MTL_689.csv",
    "Matched": "Matched_complete_266.csv",
}

SAFE_COLUMNS = [
    "Dataset",
    "Source_Group",
    "DOI",
    "Source_Reference",
    "Source_Type",
    "Evidence_Level",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    records: list[pd.DataFrame] = []
    for cohort, filename in COHORT_FILES.items():
        path = args.input_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        available = [column for column in SAFE_COLUMNS if column in frame.columns]
        source = frame[available].copy()
        source["Cohort"] = cohort
        source["Rows_in_cohort"] = 1
        records.append(source)

    if not records:
        raise FileNotFoundError("No recognised cohort CSV files were found.")

    combined = pd.concat(records, ignore_index=True)
    group_columns = [column for column in SAFE_COLUMNS if column in combined.columns]
    index = (
        combined.groupby(group_columns + ["Cohort"], dropna=False, as_index=False)
        ["Rows_in_cohort"]
        .sum()
    )
    index["Redistribution_Status"] = "Review_required"
    index["Licence_or_Terms_URL"] = ""
    index["Review_Notes"] = ""
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(index)} source-cohort rows to {args.output_csv}")


if __name__ == "__main__":
    main()
