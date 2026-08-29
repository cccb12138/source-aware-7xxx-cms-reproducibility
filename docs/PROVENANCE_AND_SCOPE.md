# Provenance and modelling scope

Each publication, thesis, or database source is represented by `Source_Group`. All records from one source group are assigned to a single outer fold. This prevents closely related records from the same source entering training and testing partitions simultaneously.

The fixed strict mapping contains 265 source groups. After the final scope audit, 260 groups remain in at least one manuscript cohort. Their five-fold source counts are 47, 49, 55, 55, and 54, with no cross-cohort fold conflict.

The primary confirmatory analysis uses 675 UTS-labelled records from 258 source groups. YS (307 records, 63 source groups) and EL (537 records, 164 source groups) are secondary exploratory targets. Partial-label MTL uses 689 records from 260 source groups. A 266-record, 59-source matched subset supports same-sample comparisons.

The source index exposes provenance and access information without redistributing the full feature/target values. Full local paths, author workstation locations, integrated workbooks, and reference PDFs are excluded from the public release.
