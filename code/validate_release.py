"""Validate the public CMS reproducibility release without restricted data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(paths: list[Path]) -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release files: " + ", ".join(missing))


def validate_frozen_scripts() -> None:
    manifest_path = ROOT / "code" / "frozen_original" / "SHA256SUMS.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 53:
        raise AssertionError(f"Expected 53 frozen code files, found {len(manifest)}")
    for row in manifest.itertuples(index=False):
        path = ROOT / row.relative_path
        if not path.is_file():
            raise FileNotFoundError(row.relative_path)
        if path.stat().st_size != int(row.bytes):
            raise AssertionError(f"Size mismatch: {row.relative_path}")
        if sha256(path) != str(row.sha256).upper():
            raise AssertionError(f"SHA-256 mismatch: {row.relative_path}")
    print(f"PASS frozen scripts: {len(manifest)} files and hashes")


def validate_folds() -> None:
    strict = pd.read_csv(ROOT / "data" / "folds" / "source_group_outer_folds_strict_main.csv")
    final = pd.read_csv(ROOT / "data" / "folds" / "final_analysis_source_folds.csv")
    balance = pd.read_csv(ROOT / "data" / "folds" / "fold_balance.csv")

    if len(strict) != 265 or strict["Source_Group"].nunique() != 265:
        raise AssertionError("Strict mapping must contain 265 unique source groups")
    if len(final) != 260 or final["Source_Group"].nunique() != 260:
        raise AssertionError("Final mapping must contain 260 unique source groups")
    if set(strict["Outer_Fold"].astype(int)) != set(range(5)):
        raise AssertionError("Strict mapping must use folds 0-4")
    if set(final["Outer_Fold"].astype(int)) != set(range(5)):
        raise AssertionError("Final mapping must use folds 0-4")

    merged = final.merge(
        strict[["Source_Group", "Outer_Fold"]],
        on="Source_Group",
        how="left",
        suffixes=("_final", "_strict"),
        validate="one_to_one",
    )
    if merged["Outer_Fold_strict"].isna().any():
        raise AssertionError("Final sources missing from strict mapping")
    if not (merged["Outer_Fold_final"] == merged["Outer_Fold_strict"]).all():
        raise AssertionError("Final and strict fold assignments disagree")

    expected = {0: 47, 1: 49, 2: 55, 3: 55, 4: 54}
    observed = final.groupby("Outer_Fold")["Source_Group"].nunique().astype(int).to_dict()
    if observed != expected:
        raise AssertionError(f"Unexpected final fold balance: {observed}")
    if balance["Sources"].sum() != 260:
        raise AssertionError("fold_balance.csv does not sum to 260 sources")
    print("PASS fixed folds: 265 strict sources, 260 final sources, zero conflicts")


def validate_source_index() -> None:
    index = pd.read_csv(ROOT / "data" / "source_index" / "source_index.csv", dtype=str).fillna("")
    if len(index) != 260 or index["Source_Group"].nunique() != 260:
        raise AssertionError("Source index must contain 260 unique source groups")
    if set(index["Redistribution_Status"]) != {"Review_required"}:
        raise AssertionError("Unexpected redistribution status in release-preparation index")
    corrected = {
        "ACTASRC_151": "10.1023/A:1018676501368",
        "ACTASRC_161": "10.1007/BF02817276",
        "ACTASRC_173": "10.1007/s11661-002-0214-2",
    }
    actual = index.set_index("Source_Group")["DOI_for_access"].to_dict()
    for source, doi in corrected.items():
        if actual.get(source) != doi:
            raise AssertionError(f"Corrected DOI missing for {source}")
    for column in index.columns:
        if index[column].str.contains(r"[A-Z]:\\", regex=True).any():
            raise AssertionError(f"Local absolute path found in source index column {column}")
    print("PASS source index: 260 sources and corrected DOI records")


def validate_results_and_figures() -> None:
    metrics = pd.read_csv(ROOT / "results" / "summary" / "UTS_final_metrics.csv")
    if len(metrics) != 1:
        raise AssertionError("UTS_final_metrics.csv must contain one final row")
    r2 = float(metrics.iloc[0]["R2"])
    expected = 0.5268211291102216
    if abs(r2 - expected) > 1e-12:
        raise AssertionError(f"Final UTS R2 mismatch: {r2}")
    for number in range(1, 8):
        matches = list((ROOT / "figures").glob(f"Fig{number}_*.png"))
        if len(matches) != 1:
            raise AssertionError(f"Expected exactly one PNG for Fig{number}, found {len(matches)}")
    if (ROOT / "results" / "summary" / "UTS_final_oof_predictions.csv").exists():
        raise AssertionError("Restricted row-level OOF predictions must not be public")
    print("PASS metrics and figures: final UTS R2=0.5268211291102216 and Fig1-Fig7 present")


def validate_restricted_exclusions() -> None:
    forbidden_names = {
        "UTS_scope_clean_675.csv",
        "YS_scope_clean_307.csv",
        "EL_scope_clean_537.csv",
        "Partial_label_MTL_689.csv",
        "Matched_complete_266.csv",
        "UTS_final_oof_predictions.csv",
    }
    hits = [p for p in ROOT.rglob("*") if p.is_file() and p.name in forbidden_names]
    spreadsheets = [p for p in ROOT.rglob("*.xlsx") if ".git" not in p.parts]
    if hits:
        raise AssertionError("Restricted files present: " + ", ".join(str(p) for p in hits))
    if spreadsheets:
        raise AssertionError("Spreadsheet files require explicit review: " + ", ".join(str(p) for p in spreadsheets))
    public_files = [p for p in (ROOT / "data" / "public").glob("*") if p.is_file() and p.name != "README.md"]
    if public_files:
        raise AssertionError("Unreviewed public row-level files are present")
    print("PASS restricted-data exclusions")


def main() -> None:
    require([
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "DATA_LICENSES.md",
        ROOT / "environment.yml",
        ROOT / "requirements-lock.txt",
        ROOT / "code" / "frozen_original" / "SHA256SUMS.csv",
        ROOT / "data" / "folds" / "source_group_outer_folds_strict_main.csv",
        ROOT / "data" / "folds" / "final_analysis_source_folds.csv",
        ROOT / "data" / "source_index" / "source_index.csv",
        ROOT / "data" / "source_index" / "reconstruction_instructions.md",
        ROOT / "results" / "summary" / "UTS_final_metrics.csv",
    ])
    validate_frozen_scripts()
    validate_folds()
    validate_source_index()
    validate_results_and_figures()
    validate_restricted_exclusions()
    print("PASS public CMS reproducibility release")


if __name__ == "__main__":
    main()
