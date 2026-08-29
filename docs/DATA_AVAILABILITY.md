# Data availability

## Publicly included in this repository

- the full frozen analysis-script snapshot and checksums;
- exact software-environment specifications;
- data schema and cohort summaries;
- fixed source-group fold assignments;
- a 260-source citation, access, and redistribution-status index;
- reconstruction instructions for records that cannot be redistributed;
- aggregate modelling results and figure data;
- the final CMS manuscript figures.

## Not publicly included

The five complete row-level modelling tables and the row-level UTS OOF prediction file are not committed. They integrate records from sources whose redistribution terms were not consistently captured during the early curation workflow. At the time of this release preparation, all 260 source groups remain `Review_required`.

These files are therefore specifically excluded:

- `UTS_scope_clean_675.csv`
- `YS_scope_clean_307.csv`
- `EL_scope_clean_537.csv`
- `Partial_label_MTL_689.csv`
- `Matched_complete_266.csv`
- `UTS_final_oof_predictions.csv`
- the integrated master workbook
- third-party reference PDFs

## Future open subset

After source-level review, records may be released only when the source is classified `Approved_open` and the supporting licence, terms URL, or author ownership is documented. Restricted and unresolved sources will continue to be represented by the source index and reconstruction instructions rather than by redistributed values.

## Manuscript wording

The manuscript Data availability statement must name the exact GitHub release/tag and version-specific Zenodo DOI that contain these files. It must not claim that the full row-level dataset is open unless the repository contents later change and the licence review supports that statement.
