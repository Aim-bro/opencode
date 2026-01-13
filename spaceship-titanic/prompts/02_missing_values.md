# Prompt: Missing Values

Role: senior data analyst + practical ML engineer.

Goal
- Diagnose missingness patterns and propose handling strategies that can be validated.

Data
- data/raw/train.csv
- data/raw/test.csv

Constraints
- Treat all ideas as hypotheses until confirmed with evidence.
- Avoid leakage; do not use target information in imputation rules.

Tasks
1) Compute missing rate per column for train/test and compare.
2) Identify co-missingness groups (columns missing together).
3) Check if missingness is correlated with the target (Transported) using simple tests.
4) Propose imputation strategies per feature type, each with a validation step.

Output format
- Evidence: numeric tables or summaries.
- Hypotheses: for each imputation idea, add a verification method.
- Risks: list at least 3 data leakage or bias risks.
