## Decision: EDA Summary & Modeling Implications
- Date: 2026-01-13
- Owner: Junhyeong
- Status: accepted

## Context
- Problem: Understand data structure and risks before modeling
- Data sources: train.csv, test.csv
- Constraint: Kaggle-style offline CV, no label leakage

## Options Considered
- Option A: Standard StratifiedKFold
- Option B: Group-aware CV using PassengerId
- Option C: No grouping

## Decision
- Chosen option: Group-aware CV (PassengerId group)
- Rationale: 77.3% of groups show perfect within-group target consistency

## Evidence
- Artifact: EDA output (01_eda)
- Key findings:
  - Strong group signal (77.3% within-group consistency)
  - Spending features extremely skewed (skewness > 6, ~60% zeros)
  - Cabin high cardinality (6560 unique values)

## Risks and Mitigations
- Risk: Overfitting via group leakage
- Mitigation: GroupKFold, no group-level target encoding

## Validation Plan
- Checks:
  - Compare GroupKFold vs StratifiedKFold CV
  - Monitor CV ↔ LB gap
- Expected outcomes:
  - Reduced leakage, more stable CV
