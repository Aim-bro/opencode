# 🧠 Project Bootstrap

This repository contains an LLM-assisted Kaggle-style ML pipeline for the
Spaceship Titanic dataset.


## Current state

Data:
- data/raw/train.csv
- data/raw/test.csv
- data/raw/sample_submission.csv

Prompt pipeline:
1. 01_eda.md – dataset summary, target balance, missing patterns
2. 02_missing_values.md – imputation hypotheses
3. 03_feature_engineering.md – feature ideas and validation logic
4. 04_modeling_baseline.md – preprocessing + baseline model plan
5. 05_cv_report.md – CV strategy + reporting checklist

Automation:
- src/opencode_runs/run_prompt.py is used to run prompts through OpenCode
- outputs/ stores LLM outputs
- docs/decisions.md stores human-approved decisions


