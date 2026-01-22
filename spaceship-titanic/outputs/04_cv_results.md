# CatBoost Baseline - Cross-Validation Results

## Model Configuration
- Algorithm: CatBoostClassifier
- CV Strategy: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- Metric: ROC-AUC
- Native categorical features: Yes (no one-hot encoding)
- Scaling: None

## Data Statistics
- Number of samples: 8693
- Number of features: 23
- Number of categorical features: 7

## Feature List
- HomePlanet
- CryoSleep
- Destination
- Age
- VIP
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck
- GroupId
- GroupSize
- IsAlone
- TotalSpending
- HasSpending
- SpaRatio
- VRDeckRatio
- Deck
- CabinNum
- Side
- CabinNumBin
- AgeGroup
- FamilySize

## Cross-Validation Results
Fold 1: 0.9118
Fold 2: 0.9012
Fold 3: 0.9041
Fold 4: 0.9149
Fold 5: 0.8984

**Mean CV Score**: 0.9061
**Std CV Score**: 0.0063
