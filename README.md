# Source-aware, leakage-resistant materials informatics for 7xxx aluminium alloys

This repository accompanies the manuscript prepared for *Computational Materials Science* on source-aware validation of heterogeneous, partially labelled 7xxx-series aluminium-alloy data.

The repository separates three release layers:

1. an immutable snapshot of the scripts used during the final analysis;
2. openly shareable aggregate results, fold assignments, figures, and provenance metadata;
3. row-level data, which are released only where source-specific redistribution terms permit it.

## Analysis scope

| Cohort | Rows | Source groups | Role |
|---|---:|---:|---|
| UTS | 675 | 258 | Primary confirmatory modelling |
| YS | 307 | 63 | Secondary exploratory modelling |
| EL | 537 | 164 | Secondary exploratory modelling |
| Partial-label MTL | 689 | 260 | Complementary multi-task comparison |
| Complete matched subset | 266 | 59 | Same-sample robustness analysis |

The final UTS RF–XGBoost ensemble achieved source-blocked out-of-fold R² = 0.5268211291102216, reported as 0.527 in the manuscript.

## Repository contents

- `code/frozen_original/`: byte-preserving snapshot of the complete numbered analysis chain (00–48), configuration/entry files, shared utilities, and SHA-256 checksums.
- `code/`: release validation and source-index utilities.
- `data/folds/`: the fixed 265-source strict mapping and the 260-source mapping used by the final manuscript analyses.
- `data/source_index/`: a 260-source citation/access index and reconstruction instructions.
- `data/public/`: row-level records approved for redistribution. This directory currently contains documentation only because the source-level review is not complete.
- `data/schema/`: cohort summary and data dictionary.
- `results/summary/`: aggregate metrics and decisions; row-level OOF predictions are deliberately excluded.
- `figure_data/`: aggregate data used by the released figures.
- `figures/`: CMS manuscript figures with final Fig. 1–Fig. 7 numbering and corrected Fig. 4.
- `docs/`: reproducibility, data provenance, licensing, and release instructions.

## Environment

The confirmed modelling environment used Python 3.9.25 with NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.5.1, XGBoost 2.0.3, Optuna 4.8.0, SHAP 0.47.2, SciPy 1.13.1, Matplotlib 3.9.2, Seaborn 0.13.2, and OpenPyXL 3.1.5. The current verified MTL environment has PyTorch 2.6.0.

The historical stage-1 environment snapshot predates installation of PyTorch. This limitation is stated explicitly in `docs/REPRODUCIBILITY.md`.

Create the pinned environment with:

```bash
conda env create -f environment.yml
conda activate source-aware-7xxx
```

## Validate the release

From the repository root:

```bash
python code/validate_release.py
```

The validator checks script hashes, fixed-fold coverage and conflicts, source-index coverage, required aggregate results, corrected final metrics, and the absence of prohibited row-level release files.

## Frozen scripts and portability

The files in `code/frozen_original/scripts/` preserve the scripts that generated the final analysis and submission freeze. Some contain historical Windows paths. They are retained unchanged for provenance and are not presented as a platform-independent software package. See `code/frozen_original/README.md` and `docs/REPRODUCIBILITY.md` for the execution order, path mapping, and inputs required to reconstruct a runnable workspace.

## Data availability

The integrated row-level tables contain values curated from journal articles, theses, and database-derived collections. Access to an article or database does not by itself establish permission to republish extracted records. At this release-preparation stage, all 260 source groups remain subject to source-level redistribution review. Consequently, no mixed-license row-level modelling table is committed here.

The repository instead provides the complete source-group index, source-level row counts, fixed folds, schema, reconstruction procedure, aggregate results, and code. Approved row-level subsets will be added only after their licences or author rights are documented. See `docs/DATA_AVAILABILITY.md`.

## Citation and release status

This repository is prepared for GitHub release v1.1.0 dated 2026-08-29. Zenodo–GitHub integration is enabled for this repository; Zenodo will archive the tagged source when the GitHub release is published and assign a new version-specific DOI. That DOI will then be added to this README, `CITATION.cff`, and the manuscript. The earlier repository archive at <https://doi.org/10.5281/zenodo.21840483> is historical and is not the archive to cite for the CMS submission.

## Licensing

Original code in this repository is licensed under the BSD 3-Clause License. That licence does not cover third-party-derived data. Data and figure terms are described separately in `DATA_LICENSES.md`.
