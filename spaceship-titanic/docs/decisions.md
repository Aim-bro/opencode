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

## Decision: Feature Engineering (03)

Core features:
- GroupSize, IsAlone (PassengerId-based)
- TotalSpending, HasSpending
- Cabin decomposition: Deck, CabinSide, CabinNum (binned)
- AgeGroup

Additional structure:
- FamilySize (from last name)
- Group-level deck consistency flags

Rationale:
- Strong social/group signals validated in EDA
- Spending features consistent with CryoSleep logic
- No target leakage introduced


모델: CatBoost

CV: StratifiedKFold 5

Metric: AUC

Score: 0.9051 ± 0.0064

Features: 21개

핵심 전처리: CryoSleep/Spending/Spa 규칙

Model: CatBoostClassifier

CV: StratifiedKFold 5

Metric: ROC-AUC

Score: 0.9051 ± 0.0064

Submission: reports/submission_catboost.csv 생성 완료
## Decision: Use Fold Ensemble for Test Predictions (CatBoost)
- Date: 2026-01-16
- Owner: Junhyeong
- Status: accepted

## Context
- Problem: Public LB dropped after applying OOF-derived threshold to a differently-parameterized final model
- Observation: OOF threshold was optimized on CV fold models (iterations=1000, lr=0.1, depth=6)
- Risk: Final model probability distribution mismatch (miscalibration)

## Options Considered
- Option A: Use fold-model ensemble for test proba, then apply OOF best threshold
- Option B: Train a single final model with the same params as CV, then apply threshold
- Option C: Keep current final model and threshold

## Decision
- Chosen option: Option A (fold-model ensemble)
- Rationale: Aligns threshold distribution with OOF source, reduces variance, avoids calibration mismatch

## Risks and Mitigations
- Risk: Slightly higher inference time
- Mitigation: Keep fold count fixed (5), simple averaging

## Validation Plan
- Submit fold-ensemble predictions with OOF threshold
- Compare LB against prior submissions (baseline 0.80430, current 0.80196)

## Decision: Blend CatBoost and LGBM with OOF-Optimized Threshold
- Date: 2026-01-16
- Owner: Junhyeong
- Status: proposed

## Context
- Problem: Public LB below target; single-model thresholding is sensitive to distribution shift
- Goal: Improve stability and score by blending diverse models

## Options Considered
- Option A: CatBoost only (baseline + threshold)
- Option B: CatBoost + LGBM blend with OOF weight search
- Option C: Switch to LGBM only

## Decision
- Chosen option: Option B (CatBoost + LGBM blend)
- Rationale: Combines CatBoost categorical handling with LGBM linear/leaf diversity

## Risks and Mitigations
- Risk: Added dependency (lightgbm) and preprocessing complexity
- Mitigation: Keep one-hot alignment fixed to full-train columns; log OOF weight/threshold

## Validation Plan
- Run blend script and submit `reports/submission_cb_lgbm.csv`
- Compare LB against CatBoost-only submissions

## 회고: Spaceship Titanic 모델링 진행 요약
- 기간: 2026-01-16
- Owner: Junhyeong
- 목표: Public LB 0.815+ 달성, 제출 횟수 제한 고려

## 1) 문제 정의 및 초기 베이스라인
- CatBoost baseline 구축, OOF 기반 threshold 탐색 적용
- 결과: baseline LB 0.80430, 이후 튜닝 제출 0.80196로 하락

## 2) 원인 가설 및 교정
- final_model 파라미터와 CV 모델 분포 불일치로 threshold 미스매치 가능성
- 해결: Fold 모델 앙상블로 test proba 생성 후 OOF threshold 적용

## 3) 모델 확장 및 보정
- CB + LGBM 블렌드 도입, OOF weight 탐색
- Isotonic calibration 적용
- LB 개선: 0.80313 -> 0.80360 (소폭)

## 4) 에러 분석 및 피처 실험
- OOF 오류 패턴 기반 가설 도출
- feature_experiments로 조합 검증
- 최적 피처: spending ratios만 켠 ratios_only (best_oof_acc 0.817784)
- group aggregate/diversity는 개선 없음 -> 보류

## 5) CV 전략 비교
- StratifiedKFold vs GroupKFold 비교
- 성능 차이 미미 -> Stratified 유지

## 6) 파라미터 미니 그리드
- CatBoost 최적: iterations=800, lr=0.1, depth=6, l2=3
- LGBM 최적: n_estimators=1200, lr=0.05, num_leaves=31 등
- 블렌드 재실행: best_oof_acc 0.81859, weight(cb)=0.70, threshold=0.480

## 7) 스태킹 및 XGBoost 평가
- 스태킹: AUC는 소폭 개선, OOF acc 하락 -> 기각
- XGBoost 단독: OOF acc 0.8018 -> 기각

## 현재 최선
- CB + LGBM 블렌드 + Isotonic calibration
- ratios_only 피처 고정
- LB 0.80734, OOF best_acc 0.81859

## 제출 전략
- 제출 횟수 제한 고려: OOF 개선 신호가 확실할 때만 제출
