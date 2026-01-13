# Prompt: EDA

Role: senior data analyst + practical ML engineer.

Goal
- Produce a minimal, verifiable EDA summary for Spaceship Titanic.

Data
- data/raw/train.csv
- data/raw/test.csv

Constraints
- Treat all suggestions as hypotheses until validated.
- No external data or leakage from test labels.

Tasks
1) Load train/test and report shapes + columns.
2) Identify target distribution and class balance (Transported).
3) Summarize missingness (rate per column, top 10).
4) Provide 3-5 quick sanity checks (duplicates, ID uniqueness, obvious data issues).
5) List 3-5 candidate features or interactions as hypotheses.

Output format
- Evidence: bullet list of numeric findings with sources.
- Hypotheses: bullet list, each with a validation step.
- Open questions: bullet list.
