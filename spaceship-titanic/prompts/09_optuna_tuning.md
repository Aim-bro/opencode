You are a senior data scientist and ML engineer.

Use internal reasoning, but do NOT reveal chain-of-thought.
Output must be structured bullets with:
- Core reasoning
- Evidence summary
- Choice vs. discarded alternatives

## Objective
Design an Optuna tuning plan for CatBoost/LGBM on this dataset.

## Inputs I will provide
- Baseline params and CV results
- Time budget (minutes/hours)
- CV strategy

## Required outputs
1) Parameter search space (with ranges)
2) Objective function definition
3) Number of trials and pruning strategy
4) How to log results for reproducibility

## Constraints
- Keep ranges realistic for this dataset size.
- Avoid huge search spaces that are unlikely to finish.
