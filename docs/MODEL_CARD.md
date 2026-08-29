# Model card

## Intended use

Research-level prediction and interpretation of mechanical properties within the represented 7xxx aluminium-alloy composition and processing domain.

## Primary endpoint

Ultimate tensile strength (UTS), evaluated with nested, source-blocked cross-validation.

## Secondary analyses

Yield strength, elongation, partial-label multi-task learning, applicability-domain diagnostics, conformal prediction intervals, source-cluster bootstrap uncertainty, leave-one-dataset-out transfer, and complete matched-subset robustness.

## Important limitations

- Source and dataset heterogeneity materially affect transfer performance.
- Missing processing variables limit mechanistic interpretation.
- SHAP values describe model behaviour and are not causal effects.
- Predictions should not replace qualification testing or engineering certification.
- Performance outside the represented composition/process domain is not established.
