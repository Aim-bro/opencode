from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.catboost_baseline import find_best_threshold_acc, preprocess
from src.titanic.load import load_train_test, TARGET_COLUMN


def _group_id(series: pd.Series) -> pd.Series:
    return series.str.split("_", expand=True)[0]


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    splitter,
    groups: pd.Series | None = None,
) -> dict:
    oof = np.zeros(len(X))
    scores = []

    for train_idx, val_idx in splitter.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False,
            cat_features=categorical_features,
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
    }


def main():
    train_df, _ = load_train_test()
    processed = preprocess(train_df.copy())
    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN].astype(int)
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    groupk = GroupKFold(n_splits=5)
    groups = _group_id(train_df["PassengerId"])

    stratified_res = run_cv(X, y, cat_features, stratified)
    groupk_res = run_cv(X, y, cat_features, groupk, groups=groups)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "groupkfold_compare.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# CV Strategy Comparison (CatBoost)\n\n")
        f.write("## StratifiedKFold (5 folds)\n")
        f.write(f"- mean_auc: {stratified_res['mean_auc']:.4f}\n")
        f.write(f"- std_auc: {stratified_res['std_auc']:.4f}\n")
        f.write(f"- best_threshold: {stratified_res['best_threshold']:.3f}\n")
        f.write(f"- best_oof_acc: {stratified_res['best_oof_acc']:.5f}\n\n")
        f.write("## GroupKFold (5 folds, group=PassengerId prefix)\n")
        f.write(f"- mean_auc: {groupk_res['mean_auc']:.4f}\n")
        f.write(f"- std_auc: {groupk_res['std_auc']:.4f}\n")
        f.write(f"- best_threshold: {groupk_res['best_threshold']:.3f}\n")
        f.write(f"- best_oof_acc: {groupk_res['best_oof_acc']:.5f}\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
