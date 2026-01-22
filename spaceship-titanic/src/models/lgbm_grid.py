from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.catboost_baseline import find_best_threshold_acc, preprocess
from src.titanic.load import load_train_test, TARGET_COLUMN


def _one_hot_align(
    df: pd.DataFrame, categorical_cols: list[str], columns: pd.Index
) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    aligned = encoded.reindex(columns=columns, fill_value=0)
    return aligned.astype(np.float32)


def run_cv(train_df: pd.DataFrame, params: dict) -> dict:
    import lightgbm as lgb

    processed = preprocess(train_df.copy())
    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN].astype(int)
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

    full_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    lgbm_columns = full_encoded.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = _one_hot_align(X.iloc[train_idx], categorical_cols, lgbm_columns)
        X_val = _one_hot_align(X.iloc[val_idx], categorical_cols, lgbm_columns)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=42,
            objective="binary",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
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
        "n_features": int(full_encoded.shape[1]),
    }


def main():
    train_df, _ = load_train_test()

    grid = [
        {
            "n_estimators": 1200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
        },
        {
            "n_estimators": 1200,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    ]

    rows = []
    for params in grid:
        print(f"Running: {params}")
        res = run_cv(train_df, params)
        rows.append({**params, **res})

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "lgbm_grid.csv"
    out_md = out_dir / "lgbm_grid.md"

    df = pd.DataFrame(rows).sort_values(by="best_oof_acc", ascending=False)
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# LightGBM Grid Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
