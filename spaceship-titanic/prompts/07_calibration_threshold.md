You are a senior data scientist and ML engineer.

Use internal reasoning, but do NOT reveal chain-of-thought.
Output must be structured bullets with:
- Core reasoning
- Evidence summary
- Choice vs. discarded alternatives

[CONTEXT]
- Project: Spaceship Titanic
- Models: CatBoost baseline + CB/LGBM blend
- Current best OOF threshold: 0.490
- CV strategy: StratifiedKFold (5 folds, shuffle=True, random_state=42)
- OOF metric summary:
  - CatBoost OOF best_acc: 0.81548
  - CB+LGBM blend OOF best_acc: 0.81652
- Goal: Improve LB stability and push >0.815

[OOF PREDICTIONS]
- OOF proba available at: outputs/oof_predictions.csv
- If needed, I can provide a sample or summary stats (mean, std, hist bins)

[ERROR ANALYSIS SUMMARY]
- Young Adult FN: 312 (43.15% of FN)
  - CryoSleep rate FN: 11.8% vs Correct 36.1%
  - OOF proba mean FN: 0.285
- Multi-passenger FN: 270 (37.34% of FN)
  - Deck diversity mean FN: 1.25 vs Correct 1.18
- High Spending FP: 220 (24.97% of FP)
  - TotalSpending mean FP: 648 vs Correct 1548
  - RoomService nonzero FP: 16.6% vs Correct 34.3%
- CryoSleep FN: 85 (11.76% of FN)
  - Unknown Deck rate FN: 2.2%
- Premium Deck FP: 75 (8.51% of FP)
  - VIP rate FP: 1.0% vs Correct 2.5%

[REQUEST]
- Recommend calibration method (Platt vs Isotonic) with justification
- Provide concrete validation steps + metrics
- Explain how to apply to test predictions safely
- List risks and how to detect overfitting

