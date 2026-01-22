from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.titanic.load import load_train_test, TARGET_COLUMN

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "CabinNumBin", "AgeGroup", "IsAlone"]
FEATURE_COLS = ["GroupId", "GroupSize", "IsAlone", "TotalSpending", "HasSpending", "HomePlanet", "CryoSleep", "Destination", "VIP", "Age", "AgeGroup", "Deck", "CabinNumBin", "Side", "FamilySize"] + SPENDING_COLS
TARGET_COLUMN = "Transported"

def extract_group_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GroupId"] = df["PassengerId"].str.split("_", expand=True)[0]
    group_sizes = df.groupby("GroupId").size()
    df["GroupSize"] = df["GroupId"].map(group_sizes)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    return df


def impute_spa(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spa_median_by_planet = train_df.groupby("HomePlanet")["Spa"].median()
    df["Spa"] = df.apply(
        lambda row: spa_median_by_planet[row["HomePlanet"]] if pd.isna(row["Spa"]) and pd.notna(row["HomePlanet"]) else row["Spa"],
        axis=1
    )
    return df


def preprocess(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    feature_flags: Dict[str, bool] | None = None,
) -> pd.DataFrame:
    if train_df is None:
        train_df = df.copy()
    
    df = df.copy()
    train_df = train_df.copy()

    if feature_flags is None:
        # Default to the best-performing feature set from feature_experiments
        feature_flags = {
            "spending_ratios": True,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": False,
        }
    
    df["LastName"] = df["Name"].str.split().str[-1]
    train_df["LastName"] = train_df["Name"].str.split().str[-1]
    
    df = df.drop(columns=["Name"])
    train_df = train_df.drop(columns=["Name"])
    
    df = extract_group_features(df)
    train_df = extract_group_features(train_df)
    
    # 1) Spa 먼저 planet-median impute (train 기준)
    df = impute_spa(df, train_df)

    # 2) 나머지 spending만 0으로
    zero_cols = ["RoomService", "FoodCourt", "ShoppingMall", "VRDeck"]
    df[zero_cols] = df[zero_cols].fillna(0)

    # 3) TotalSpending/HasSpending은 impute 이후 계산
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
    # Spending ratios (avoid divide-by-zero)
    if feature_flags.get("spending_ratios", False):
        denom = df["TotalSpending"].replace(0, np.nan)
        df["SpaRatio"] = (df["Spa"] / denom).fillna(0.0)
        df["VRDeckRatio"] = (df["VRDeck"] / denom).fillna(0.0)

    mask = df["CryoSleep"].isna()
    df.loc[mask & (df["TotalSpending"] > 0), "CryoSleep"] = False
    df.loc[mask & (df["TotalSpending"] == 0), "CryoSleep"] = True

    
    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
    df["CabinNumBin"] = pd.cut(df["CabinNum"], bins=5, labels=False)
    df["Deck"] = df["Deck"].fillna("Unknown")
    df["Side"] = df["Side"].fillna("Unknown")
    df["CabinNumBin"] = df["CabinNumBin"].fillna(-1).astype(int)
    if feature_flags.get("deck_unknown", False):
        df["DeckUnknown"] = (df["Deck"] == "Unknown").astype(int)
    
    age_groups = pd.cut(df["Age"], bins=[0, 18, 30, 50, 100], labels=["Child", "Young Adult", "Adult", "Senior"], right=False)
    df["AgeGroup"] = age_groups.astype(str).replace("nan", "Unknown")
    
    family_sizes = train_df["LastName"].value_counts()
    df["FamilySize"] = df["LastName"].map(family_sizes).fillna(1).astype(int)

    # Group diversity features
    if feature_flags.get("group_diversity", False):
        df["GroupDeckDiversity"] = df.groupby("GroupId")["Deck"].transform("nunique")
        df["GroupDestinationDiversity"] = df.groupby("GroupId")["Destination"].transform("nunique")
        df["GroupCryoSleepDiversity"] = df.groupby("GroupId")["CryoSleep"].transform("nunique")

    # Interaction features
    if feature_flags.get("interactions", False):
        df["AgeGroup_x_CryoSleep"] = (
            df["AgeGroup"].astype(str) + "_x_" + df["CryoSleep"].astype(str)
        )
        df["AgeGroup_x_Destination"] = (
            df["AgeGroup"].astype(str) + "_x_" + df["Destination"].astype(str)
        )
        df["CryoSleep_x_HasSpending"] = (
            (df["CryoSleep"] == True) & (df["HasSpending"] == 1)
        ).astype(int)
    
    df = df.drop(columns=["Cabin", "PassengerId", "LastName"])
    
    for col in df.columns:
        if col not in SPENDING_COLS + ["Age", "CabinNum"]:
            df[col] = df[col].fillna("Unknown")
    
    for col in SPENDING_COLS:
        df[col] = df[col].astype(float)
    
    return df


def train_baseline_model(train_df: pd.DataFrame) -> Tuple[List[CatBoostClassifier], dict, np.ndarray, pd.DataFrame, pd.Series]:
    df = train_df.copy()
    
    processed_df = preprocess(df)
    
    X = processed_df.drop(columns=[TARGET_COLUMN])
    y = processed_df[TARGET_COLUMN].astype(int)
    
    categorical_features = [col for col in X.columns if X[col].dtype == "object"]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_proba = np.zeros(len(X))  # 또는 len(train_df) 말고, 학습에 쓰는 X 기준이 안전

    cv_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False,
            cat_features=categorical_features,
            eval_metric="AUC"
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_pred 
        fold_score = roc_auc_score(y_val, val_pred)
        cv_scores.append(fold_score)
        fold_models.append(model)
        
        print(f"Fold {fold + 1} ROC-AUC: {fold_score:.4f}")
        
    best_t, best_oof_acc = find_best_threshold_acc(y.values.astype(int), oof_proba)
    print(f"[OOF] best_threshold={best_t:.3f}, best_accuracy={best_oof_acc:.5f}")

    
    results = {
        "fold_scores": cv_scores,
        "mean_score": np.mean(cv_scores),
        "std_score": np.std(cv_scores),
        "features": X.columns.tolist(),
        "categorical_features": categorical_features,
        "n_features": len(X.columns),
        "n_categorical": len(categorical_features),
        "n_samples": len(X),
        "best_threshold": best_t,
        "best_oof_acc": best_oof_acc,
    }
    
    return fold_models, results, oof_proba, X, y


def generate_cv_report(results: dict) -> str:
    report = f"""# CatBoost Baseline - Cross-Validation Results

## Model Configuration
- Algorithm: CatBoostClassifier
- CV Strategy: StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
- Metric: ROC-AUC
- Native categorical features: Yes (no one-hot encoding)
- Scaling: None

## Data Statistics
- Number of samples: {results['n_samples']}
- Number of features: {results['n_features']}
- Number of categorical features: {results['n_categorical']}

## Feature List
{chr(10).join(f"- {f}" for f in results['features'])}

## Cross-Validation Results
"""
    for i, score in enumerate(results['fold_scores'], 1):
        report += f"Fold {i}: {score:.4f}\n"
    
    report += f"""
**Mean CV Score**: {results['mean_score']:.4f}
**Std CV Score**: {results['std_score']:.4f}
"""
    return report

def find_best_threshold_acc(y_true: np.ndarray, proba: np.ndarray):
    ts = np.arange(0.05, 0.951, 0.005)
    best_t, best_acc = 0.5, -1.0
    for t in ts:
        pred = (proba >= t).astype(int)
        acc = (pred == y_true).mean()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, float(best_acc)



def main():
    train_df, _ = load_train_test()
    
    print("Training CatBoost baseline model...")
    fold_models, results, oof_proba, X, y = train_baseline_model(train_df)
    
    print(f"\nMean CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
    print(f"Features used: {results['n_features']}")
    print(f"Categorical features: {results['n_categorical']}")
    
    report = generate_cv_report(results)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "04_cv_results.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\nCV report saved to: {report_path}")
    
    # Save OOF predictions for error analysis
    oof_df = pd.DataFrame({
        "PassengerId": train_df["PassengerId"],
        "oof_proba": oof_proba,
        "true_label": y.values
    })
    oof_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to: {oof_path}")
    
    # === Train on full data & make submission ===
    

    print("Generating test predictions with fold ensemble...")

    processed_train = preprocess(train_df.copy())
    X_full = processed_train.drop(columns=[TARGET_COLUMN])
    
    # cat_features: preprocess 결과에서 object/category/bool 컬럼명을 사용
    # Keep categorical feature list consistent with CV training
    cat_features = results["categorical_features"]


    # test 로드
    test_path = Path("data/raw/test.csv")
    test_df = pd.read_csv(test_path)
    processed_test = preprocess(test_df.copy(), train_df=train_df.copy())

    X_test = processed_test  
    X_test = X_test.reindex(columns=X_full.columns)
    assert list(X_test.columns) == list(X_full.columns)


    # CatBoost Pool
    


    test_pool = Pool(X_test, cat_features=cat_features)

    proba = np.zeros(len(X_test), dtype=float)
    for m in fold_models:
        proba += m.predict_proba(test_pool)[:, 1] / len(fold_models)
    best_t = results["best_threshold"]
    pred = (proba >= best_t).astype(bool)

    sub = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": pred}
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_catboost.csv"
    sub.to_csv(out_path, index=False)

    print(f"Submission saved to: {out_path}")

    
    
    return fold_models, results

    


if __name__ == "__main__":
    main()
    if feature_flags.get("group_aggregates", False):
        age_median = train_df["Age"].median()
        group_aggs = df.groupby("GroupId").agg(
            group_total_spending_mean=("TotalSpending", "mean"),
            group_total_spending_sum=("TotalSpending", "sum"),
            group_total_spending_max=("TotalSpending", "max"),
            group_total_spending_std=("TotalSpending", "std"),
            group_has_spending_mean=("HasSpending", "mean"),
            group_age_mean=("Age", "mean"),
            group_age_std=("Age", "std"),
        )
        group_aggs["group_total_spending_std"] = group_aggs[
            "group_total_spending_std"
        ].fillna(0.0)
        group_aggs["group_age_mean"] = group_aggs["group_age_mean"].fillna(age_median)
        group_aggs["group_age_std"] = group_aggs["group_age_std"].fillna(0.0)
        df = df.merge(group_aggs, left_on="GroupId", right_index=True, how="left")
