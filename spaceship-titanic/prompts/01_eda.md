# 01 – Exploratory Data Analysis (EDA)

You are a senior data scientist performing reproducible EDA for the
Kaggle **Spaceship Titanic** dataset.

Your goal is to produce a **concise, verifiable, and modeling-oriented EDA**
that will directly inform downstream decisions.

---

## Inputs
- data/raw/train.csv
- data/raw/test.csv

Target column (train only):
- Transported (binary)

---

## Task 1. Dataset Snapshot

For **train** and **test** separately, report:

- Number of rows and columns
- Column list grouped by:
  - numerical
  - categorical
  - boolean
  - identifier-like
- Data types (pandas dtype)
- Memory usage (optional, if cheap)

Output format:
- Table or bullet list
- Clearly label train vs test

---

## Task 2. Target Analysis (train only)

- Class distribution of `Transported`
  - counts
  - percentages
- Simple balance assessment
  - Is stratified CV required? (yes/no + reason)

---

## Task 3. Missing Value Analysis

For **each dataset (train, test)**:

- Missing rate per column (percentage)
- Top 10 columns by missing rate

Additionally:
- Compute **difference in missing rate between train and test**
  - (train_missing_rate − test_missing_rate)
  - Report Top 10 absolute differences

Highlight:
- Columns with >20% missing
- Columns with large train/test mismatch

---

## Task 4. Data Quality & Sanity Checks

Explicitly verify and report:

- PassengerId uniqueness (train / test)
- Number of fully duplicated rows
- Target column sanity:
  - missing values in `Transported`
  - invalid values (non {0,1})
- Basic string hygiene checks on categorical columns:
  - leading/trailing spaces
  - inconsistent casing
- Any obvious data leakage signals

---

## Task 5. Numerical Feature Overview

For numerical columns (train only unless stated):

- Summary statistics:
  - mean, std, min, max
- Identify potential outlier-prone columns
  - extremely skewed
  - heavy tails
- Flag columns likely requiring:
  - log transform
  - clipping / winsorization

---

## Task 6. Categorical Feature Overview

For categorical columns:

- Cardinality (number of unique values)
- Top 10 columns by highest cardinality
- Columns with:
  - very low cardinality (binary-like)
  - suspiciously high cardinality (ID leakage risk)

---

## Task 7. Structural / Group Signals

Check for implicit structure:

- Can PassengerId be decomposed into:
  - group identifier
  - individual index
- Distribution of group sizes (if applicable)
- Any evidence of group-level correlation with target

---

## Task 8. Initial Feature Hypotheses

Based strictly on EDA evidence:

- List **5–10 candidate feature ideas**
- For each feature:
  - intuition
  - supporting EDA signal
  - **how it should be validated**
    (e.g. univariate AUC, CV delta, ablation)

Do NOT implement features here — hypothesis only.

---

## Task 9. Key Takeaways (Actionable)

Summarize:

- Top data issues to handle before modeling
- High-risk columns (missingness / leakage / noise)
- Immediate implications for:
  - missing value strategy
  - CV design
  - baseline model choice

---

## Output Rules

- Use clear section headers
- Prefer tables over prose
- Avoid speculation not supported by EDA output
- When stating conclusions, briefly cite the evidence
  (e.g. “Column X has 45% missing in both train/test”)

End with a short **EDA Summary for Modeling** (5–7 bullets).
