# CMS reproducibility release v1.1.0

This release is the public reproducibility package prepared for the associated *Computational Materials Science* submission.

Zenodo DOI: <https://doi.org/10.5281/zenodo.22162478>

## Included

- Complete frozen 00–48 analysis and manuscript-asset script chain, shared utilities, and SHA-256 checksums.
- Pinned Python 3.9.25 modelling environment, including NumPy, pandas, scikit-learn, XGBoost, Optuna, SHAP, PyTorch, and supporting packages.
- Fixed source-group outer-fold mappings: 265 sources in the strict mapping and 260 sources in the final manuscript union.
- A populated 260-source provenance/access index and reconstruction instructions.
- Aggregate modelling results and figure data.
- Final CMS manuscript figures Fig.1–Fig.7 and FigS1.
- BSD 3-Clause code licence and separate data-licensing statement.

## Data boundary

The complete mixed-license row-level modelling tables and row-level OOF predictions are not included. All 260 source groups remain subject to source-specific redistribution review. The release provides source citations, access routes, cohort-specific counts, fixed folds, schema, and reconstruction instructions without asserting redistribution rights over third-party-derived values.

## Corrections and manuscript alignment

- Replaced the obsolete Fig.4 annotation of 0.547 with the frozen final UTS source-blocked OOF result of R² = 0.5268211291102216, reported as 0.527.
- Aligned figure filenames with the CMS manuscript numbering: historical Fig7 and Fig8 are released as manuscript Fig6 and Fig7.
- Updated repository metadata for `source-aware-7xxx-cms-reproducibility`.

## Validation

The release validator confirms 53 frozen code-file hashes, fixed-fold coverage with zero conflicts, 260-source index coverage, the final UTS metric, final Fig.1–Fig.7 presence, and exclusion of restricted row-level files.

The frozen scripts preserve historical machine-specific paths for provenance. Full retraining requires legally reconstructing the row-level inputs and adapting a copy of the scripts as described in `docs/REPRODUCIBILITY.md`.
