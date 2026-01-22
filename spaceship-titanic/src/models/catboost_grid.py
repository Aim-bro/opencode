from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.catboost_baseline import find_best_threshold_acc, preprocess
from src.titanic.load import load_train_test, TARGET_COLUMN


def run_cv(train_df: pd.DataFrame, params: dict) -> dict:
    processed = preprocess(train_df.copy())
    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN].astype(int)
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=42,
            verbose=False,
            cat_features=cat_features,
            eval_metric="AUC",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred
        scores.append(roc_auc_score(y_val, val_pred))

    best_t, best_acc = find_best_threshold_acc(y.values.astype(int), oof)
    return {
        "mean_auc": float(np.mean(scores)),
        "std_auc": float(np.std(scores)),
        "best_threshold": best_t,
        "best_oof_acc": best_acc,
        "n_features": int(X.shape[1]),
        "n_categorical": int(len(cat_features)),
    }


def main():
    train_df, _ = load_train_test()

    grid = [
        {"iterations": 800, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3},
        {"iterations": 1200, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3},
        {"iterations": 1000, "learning_rate": 0.1, "depth": 7, "l2_leaf_reg": 3},
        {"iterations": 1200, "learning_rate": 0.08, "depth": 7, "l2_leaf_reg": 5},
        {"iterations": 1500, "learning_rate": 0.05, "depth": 7, "l2_leaf_reg": 5},
        {"iterations": 1500, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 7},
    ]

    rows = []
    for params in grid:
        print(f"Running: {params}")
        res = run_cv(train_df, params)
        rows.append({**params, **res})

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "catboost_grid.csv"
    out_md = out_dir / "catboost_grid.md"

    df = pd.DataFrame(rows).sort_values(by="best_oof_acc", ascending=False)
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# CatBoost Grid Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
