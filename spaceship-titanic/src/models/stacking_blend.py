from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
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


def train_catboost_cv(
    X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]
) -> tuple[list[CatBoostClassifier], np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            cat_features=categorical_cols,
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
        models.append(model)

        print(f"[CatBoost] Fold {fold + 1} ROC-AUC: {scores[-1]:.4f}")

    return models, oof, scores


def train_lgbm_cv(
    X: pd.DataFrame, y: pd.Series, categorical_cols: list[str], columns: pd.Index
) -> tuple[list[object], np.ndarray, list[float]]:
    import lightgbm as lgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = _one_hot_align(X.iloc[train_idx], categorical_cols, columns)
        X_val = _one_hot_align(X.iloc[val_idx], categorical_cols, columns)
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
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
        models.append(model)

        print(f"[LGBM] Fold {fold + 1} ROC-AUC: {scores[-1]:.4f}")

    return models, oof, scores


def meta_cv_oof(
    X_meta: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    scores = []

    for train_idx, val_idx in skf.split(X_meta, y):
        model = LogisticRegression(solver="lbfgs", max_iter=2000)
        model.fit(X_meta[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X_meta[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], oof[val_idx]))

    return oof, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Write submission file (no Kaggle upload).",
    )
    args = parser.parse_args()

    train_df, test_df = load_train_test()

    processed_train = preprocess(train_df.copy())
    X = processed_train.drop(columns=[TARGET_COLUMN])
    y = processed_train[TARGET_COLUMN].astype(int)

    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

    print("Training CatBoost CV...")
    cb_models, cb_oof, cb_scores = train_catboost_cv(X, y, categorical_cols)
    print(
        f"[CatBoost] Mean CV AUC: {np.mean(cb_scores):.4f} (+/- {np.std(cb_scores):.4f})"
    )

    print("Training LGBM CV...")
    full_encoded = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    lgbm_columns = full_encoded.columns
    lgbm_models, lgbm_oof, lgbm_scores = train_lgbm_cv(
        X, y, categorical_cols, lgbm_columns
    )
    print(
        f"[LGBM] Mean CV AUC: {np.mean(lgbm_scores):.4f} (+/- {np.std(lgbm_scores):.4f})"
    )

    X_meta = np.column_stack([cb_oof, lgbm_oof])
    meta_oof, meta_scores = meta_cv_oof(X_meta, y.values)
    best_t, best_acc = find_best_threshold_acc(y.values, meta_oof)
    print(
        f"[Stack] Mean CV AUC: {np.mean(meta_scores):.4f} "
        f"(+/- {np.std(meta_scores):.4f}), "
        f"best_threshold={best_t:.3f}, best_oof_acc={best_acc:.5f}"
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "stacking_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Stacking Results (CB + LGBM)\n\n")
        f.write(f"- CatBoost mean AUC: {np.mean(cb_scores):.4f}\n")
        f.write(f"- LGBM mean AUC: {np.mean(lgbm_scores):.4f}\n")
        f.write(f"- Stacking mean AUC: {np.mean(meta_scores):.4f}\n")
        f.write(f"- Stacking best threshold: {best_t:.3f}\n")
        f.write(f"- Stacking best OOF acc: {best_acc:.5f}\n")
    print(f"Saved: {out_path}")

    if not args.submit:
        return

    processed_test = preprocess(test_df.copy(), train_df=train_df.copy())
    X_test = processed_test.reindex(columns=X.columns)

    cb_test_proba = np.zeros(len(X_test))
    test_pool = Pool(X_test, cat_features=categorical_cols)
    for m in cb_models:
        cb_test_proba += m.predict_proba(test_pool)[:, 1] / len(cb_models)

    X_test_lgbm = _one_hot_align(X_test, categorical_cols, lgbm_columns)
    lgbm_test_proba = np.zeros(len(X_test_lgbm))
    for m in lgbm_models:
        lgbm_test_proba += m.predict_proba(X_test_lgbm)[:, 1] / len(lgbm_models)

    meta_model = LogisticRegression(solver="lbfgs", max_iter=2000)
    meta_model.fit(X_meta, y.values)
    test_meta = np.column_stack([cb_test_proba, lgbm_test_proba])
    meta_test_proba = meta_model.predict_proba(test_meta)[:, 1]
    pred = (meta_test_proba >= best_t).astype(bool)

    sub = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": pred}
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission_stacking.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")


if __name__ == "__main__":
    main()
