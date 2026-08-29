# Frozen original analysis scripts

This directory is an immutable provenance snapshot of the code used to construct the processed cohorts, train and validate the models, assemble paper results, generate figures, and freeze submission assets.

## What is included

- the full numbered chain `00_...` through `48_...`;
- `config.py` and `run_stage1.py`;
- shared `src/data_utils.py` utilities;
- `SHA256SUMS.csv` with path, size, role, release tier, and SHA-256.

The core final-result stages are 25–34, 38, 40, and 42. Earlier numbered stages are included because they created strict processed inputs, folds, selected features, or model decisions consumed downstream. Stages 46–48 assemble supplementary material.

## Historical path roots

The original scripts reference these machine-specific roots:

| Historical root | Purpose | Reconstructed equivalent |
|---|---|---|
| `D:\Jupyter\Al7xxx_Traceable_Modeling` | project, processed inputs, early results | a local analysis workspace containing reconstructed inputs and `results/` |
| `F:\CC\stage_al7xxx` | late-stage script source | this directory's `scripts/` folder |
| `F:\CC\outputs` | late-stage outputs | a writable output root |
| `C:\Users\dell\OneDrive\Desktop\论文优化数据` | author-only source workbooks | legally obtained source files reconstructed from the public index |

Do not edit these frozen files in place. For a new run, copy them to a separate working directory and replace path constants in that copy, or mount equivalent paths. Record every change in a patch file.

## Core execution sequence

The historical numbering records development order. A reconstruction run should use the following dependency order:

1. `00`–`02`: environment, data audit, and source folds.
2. `03`–`24`: baseline models, nested tuning, MTL prototypes, feature/augmentation studies, SHAP, credibility, outlier, and hierarchical checks.
3. `25`: systematic UTS scope audit.
4. `28`–`30`: YS/EL scope audit and sparse feature decisions.
5. `40`: final 675-row UTS rebuild with frozen nested decisions.
6. `26`–`27`: final UTS OOF SHAP and credibility outputs using the frozen decisions.
7. `31`: scope-clean partial-label MTL.
8. `32`–`33`: matched-subset robustness and verification.
9. `34`, `37`–`39`: paper tables/figures and verification.
10. `42`: submission freeze assembly.
11. `46`–`48`: supplementary-table assembly where required.

Because some historical stages were rerun after their numerical filename was assigned, use the dependency order above rather than assuming simple lexical order.

## Expected final checks

- UTS cohort: 675 rows, 258 source groups.
- Final source-blocked RF–XGBoost ensemble: R² = 0.5268211291102216.
- Final union: 260 unique source groups assigned to exactly one of five outer folds.
- Final fold source counts: 47, 49, 55, 55, 54.
