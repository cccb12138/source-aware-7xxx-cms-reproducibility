# Code layout

## Frozen analysis snapshot

`frozen_original/scripts/` contains the complete numbered 00–48 analysis and manuscript-asset chain, `config.py`, `run_stage1.py`, and shared `src` utilities. `frozen_original/SHA256SUMS.csv` records every released file hash.

These files are deliberately preserved byte-for-byte. Historical absolute Windows paths remain visible because changing them would destroy the exact provenance snapshot. See `frozen_original/README.md` before trying to execute them.

## Release utilities

- `validate_release.py`: checks the public release structure, hashes, folds, source index, metrics, and restricted-data exclusions.
- `build_source_index.py`: legacy helper retained for audit history.
- `generate_manifest.py`: regenerates a general file manifest.

The repository does not claim that the historical script snapshot runs unchanged on another computer. Reproduction requires mapping the documented historical roots to legally reconstructed inputs, then running the stages in the documented order.
