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


def _require_xgboost():
    try:
        import xgboost as xgb  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required. Install with: pip install xgboost"
        ) from exc


def _one_hot_align(
    df: pd.DataFrame, categorical_cols: list[str], columns: pd.Index
) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    aligned = encoded.reindex(columns=columns, fill_value=0)
    return aligned.astype(np.float32)


def run_cv(train_df: pd.DataFrame, params: dict) -> dict:
    _require_xgboost()
    import xgboost as xgb

    processed = preprocess(train_df.copy())
    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN].astype(int)
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

    full_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    xgb_columns = full_encoded.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = _one_hot_align(X.iloc[train_idx], categorical_cols, xgb_columns)
        X_val = _one_hot_align(X.iloc[val_idx], categorical_cols, xgb_columns)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
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
        "n_features": int(full_encoded.shape[1]),
    }


def main():
    train_df, _ = load_train_test()

    params = {
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    print(f"Running XGB CV with params: {params}")
    res = run_cv(train_df, params)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "xgb_cv.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# XGBoost CV Results\n\n")
        for k, v in res.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
