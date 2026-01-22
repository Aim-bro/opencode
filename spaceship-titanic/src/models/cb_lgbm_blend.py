from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.titanic.load import load_train_test, TARGET_COLUMN
from src.models.catboost_baseline import preprocess, find_best_threshold_acc


def _require_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LightGBM is required. Install with: pip install lightgbm"
        ) from exc


def _one_hot_align(
    df: pd.DataFrame, categorical_cols: List[str], columns: pd.Index
) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    aligned = encoded.reindex(columns=columns, fill_value=0)
    return aligned.astype(np.float32)


def train_catboost_cv(
    X: pd.DataFrame, y: pd.Series, categorical_cols: List[str]
) -> Tuple[List[CatBoostClassifier], np.ndarray, List[float]]:
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
        fold_score = roc_auc_score(y_val, val_pred)
        scores.append(fold_score)
        models.append(model)

        print(f"[CatBoost] Fold {fold + 1} ROC-AUC: {fold_score:.4f}")

    return models, oof, scores


def train_lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    columns: pd.Index,
) -> Tuple[List[object], np.ndarray, List[float]]:
    _require_lightgbm()
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
        fold_score = roc_auc_score(y_val, val_pred)
        scores.append(fold_score)
        models.append(model)

        print(f"[LGBM] Fold {fold + 1} ROC-AUC: {fold_score:.4f}")

    return models, oof, scores


def _fit_calibrator(proba: np.ndarray, y: np.ndarray, method: str):
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(proba, y)
        return calibrator
    if method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(proba.reshape(-1, 1), y)
        return calibrator
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_calibrator(calibrator, proba: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return calibrator.transform(proba)
    if method == "platt":
        return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")


def main():
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

    calibration_method = "isotonic"
    weights = np.arange(0.2, 0.81, 0.05)
    best = {"acc": -1.0, "t": 0.5, "w": 0.5}
    for w in weights:
        blend_oof = w * cb_oof + (1.0 - w) * lgbm_oof
        calibrator = _fit_calibrator(blend_oof, y.values.astype(int), calibration_method)
        calibrated_oof = _apply_calibrator(calibrator, blend_oof, calibration_method)
        t, acc = find_best_threshold_acc(y.values.astype(int), calibrated_oof)
        if acc > best["acc"]:
            best = {"acc": acc, "t": t, "w": float(w)}

    print(
        f"[Blend] best_weight(cb)={best['w']:.2f}, "
        f"best_threshold={best['t']:.3f}, best_oof_acc={best['acc']:.5f}, "
        f"calibration={calibration_method}"
    )

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

    blend_proba = best["w"] * cb_test_proba + (1.0 - best["w"]) * lgbm_test_proba
    final_calibrator = _fit_calibrator(
        best["w"] * cb_oof + (1.0 - best["w"]) * lgbm_oof,
        y.values.astype(int),
        calibration_method,
    )
    calibrated_test = _apply_calibrator(final_calibrator, blend_proba, calibration_method)
    pred = (calibrated_test >= best["t"]).astype(bool)

    sub = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": pred}
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_cb_lgbm.csv"
    sub.to_csv(out_path, index=False)

    print(f"Submission saved to: {out_path}")


if __name__ == "__main__":
    main()
