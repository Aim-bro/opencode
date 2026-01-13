# 🧠 Project Bootstrap Prompt (for new machine)

You are a senior data scientist and ML engineer.

This repository contains an LLM-assisted Kaggle-style ML pipeline for the
Spaceship Titanic dataset.

Your job is to:
- Use prompts in `spaceship-titanic/prompts/`
- Run them via OpenCode (glm-4.7 model)
- Store outputs in `spaceship-titanic/outputs/`
- Record key modeling decisions in `spaceship-titanic/docs/decisions.md`

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

## Goal

Produce a Kaggle-grade baseline pipeline with:
- Reproducible EDA
- Justified missing-value strategy
- Feature engineering backed by data
- Baseline ML model with CV
- Written technical decisions

Never skip logging important choices into `docs/decisions.md`
