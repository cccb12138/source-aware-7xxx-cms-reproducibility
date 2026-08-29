# Reproducibility notes

## Environment evidence

The archived `original_environment_stage1.json` records Python 3.9.25 and the core tabular/modelling package versions used in the early pipeline. That snapshot reports PyTorch as unavailable because it predates the final MTL run. The currently verified ML environment contains PyTorch 2.6.0. This repository therefore distinguishes historical evidence from the current verified reconstruction environment instead of claiming that the old JSON captured the final MTL state.

## Randomisation and validation

- Final UTS rebuild seed: 20260810.
- Five source-exclusive outer folds.
- Three-fold GroupKFold inner validation.
- Eight Optuna trials per outer fold for the final UTS rebuild.
- Final refined composition set: Zn, Mg, Cu, Fe, Zr.
- Partial-label MTL seeds: 20260805, 20260806, 20260807.
- MTL shared hidden layers: 64 and 32 units.

Run configurations and additional seeds are embedded in the frozen scripts and generated result metadata.

## What can be reproduced immediately

Repository validation, fold checks, hashes, aggregate-table checks, and figure-asset checks can be run without restricted row-level data.

Full model retraining requires reconstructing or legally obtaining the row-level inputs described by `data/source_index/reconstruction_instructions.md`. The frozen scripts preserve historical paths and must be copied to a separate workspace for path adaptation.

## Acceptance criteria

The reconstructed run should reproduce the frozen cohort sizes, source counts, fixed fold assignments, selected feature/augmentation decisions, and aggregate metrics. The main checkpoint is UTS ensemble source-blocked OOF R² = 0.5268211291102216.
