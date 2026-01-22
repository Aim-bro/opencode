You are a senior ML engineer.

You must strictly follow the confirmed decisions in docs/decisions.md.
Do NOT invent new preprocessing or modeling steps.

## Objective
Build a strong Kaggle-grade baseline model for Spaceship Titanic.

## Mandatory preprocessing
- Drop Name
- PassengerId:
  - Extract GroupId (prefix before "_")
  - Compute GroupSize and IsAlone
- Spending:
  - TotalSpending = sum of all spending columns
  - HasSpending flag
  - Impute spending columns with 0
  - EXCEPTION: Spa must be imputed by median per HomePlanet
- CryoSleep:
  - If TotalSpending > 0 -> False
  - If TotalSpending == 0 -> True
- Cabin:
  - Extract Deck, CabinNum, Side
  - Bin CabinNum
- Age:
  - Create AgeGroup bins
- Family:
  - Extract last name
  - FamilySize
- Add missing indicator flags where appropriate

## Model
- Use CatBoostClassifier
- Use native categorical features (NO one-hot)
- NO scaling

## Validation
- StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- Metric: ROC-AUC

## Outputs
- outputs/04_cv_results.md
- src/models/catboost_baseline.py
- Explicit feature list used
- Fold-wise and mean CV scores
