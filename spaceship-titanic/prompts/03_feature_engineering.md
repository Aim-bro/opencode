# Prompt: Feature Engineering

Role: senior data analyst + practical ML engineer.

Goal
- Propose feature ideas for Spaceship Titanic and define how to validate them.

Data
- data/raw/train.csv
- data/raw/test.csv

Constraints
- Treat all feature ideas as hypotheses until validated.
- Avoid target leakage; do not use Transported to derive features.

Tasks
1) Identify categorical, numeric, and mixed-type columns.
2) Propose 8-12 feature ideas (e.g., group features, cabin parsing, spending totals).
3) For each feature, define how to compute it and a quick validation check.
4) Flag any feature ideas that might leak or be unstable.

Output format
- Feature list: name, definition, data needed.
- Validation: quick metric or sanity check per feature.
- Risks: leakage or robustness concerns.
