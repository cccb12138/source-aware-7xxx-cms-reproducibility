# Reconstruction instructions for non-redistributable records

These instructions describe how a qualified researcher can reconstruct the modelling tables from legally obtained sources without the repository redistributing restricted row-level values.

## 1. Resolve and acquire each source

1. Start from `source_index.csv` and select the source groups required for the intended cohort.
2. If `DOI_for_access` is present, resolve the DOI and obtain the article, thesis, supplementary information, or linked dataset through an authorised route.
3. If `DOI_for_access` is blank, use `Source_Reference` and `DOI_as_recorded` as search evidence, then manually confirm the source identity before extraction.
4. Record the source URL, access date, licence/terms URL, and exact document or dataset version.

## 2. Extract source records

For every sample or condition, transcribe the reported alloy grade, composition, solution treatment, quench/cooling condition, ageing schedule, test temperature, YS, UTS, elongation, and hardness when available. Preserve the original units, table/figure/sheet identifier, sample identifier, and any qualifiers before conversion.

Do not infer an unreported processing value as zero. Retain it as missing. Do not merge records solely because they share an alloy designation.

## 3. Standardise variables

Use `data/schema/data_dictionary.csv` as the canonical variable definition. Convert composition to the stated units, temperatures to degrees Celsius, time to hours, strength to MPa, and elongation to percent. Derive ratio and solute-sum descriptors only after the elemental fields have been standardised.

Preserve `Source_Group` at publication/dataset level. All records derived from one source group must remain in the same outer fold.

## 4. Apply scope and quality decisions

Use the frozen scripts in `code/frozen_original/scripts/` to reproduce the documented audit sequence. Apply explicit grade/evidence requirements, duplicate checks, implausible-value checks, and the scope-clean decisions before constructing target-specific cohorts.

Expected cohort sizes after applying the frozen decisions are:

| Cohort | Rows | Source groups |
|---|---:|---:|
| UTS | 675 | 258 |
| YS | 307 | 63 |
| EL | 537 | 164 |
| Partial-label MTL | 689 | 260 |
| Matched subset | 266 | 59 |

## 5. Apply fixed folds

Join records to `data/folds/source_group_outer_folds_strict_main.csv` by exact `Source_Group`. For the final manuscript cohorts, verify the result against `data/folds/final_analysis_source_folds.csv`. Never assign folds at random at record level.

## 6. Validate the reconstruction

Run `python code/validate_release.py`, then compare regenerated aggregate outputs with `results/summary/`. The primary numerical checkpoint is the UTS RF–XGBoost ensemble source-blocked OOF R² of 0.5268211291102216.

If the reconstructed cohort differs, first inspect source identity, unit conversion, missing-value handling, duplicate decisions, scope exclusions, and the exact fixed-fold join. Document any irreducible difference rather than silently altering the frozen mapping.
