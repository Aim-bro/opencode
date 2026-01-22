from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.titanic.load import load_train_test, TARGET_COLUMN
from src.models.catboost_baseline import find_best_threshold_acc, preprocess


def run_cv(train_df: pd.DataFrame, feature_flags: dict[str, bool]) -> dict:
    processed = preprocess(train_df.copy(), feature_flags=feature_flags)
    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN].astype(int)

    categorical_features = [col for col in X.columns if X[col].dtype == "object"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = np.zeros(len(X))
    fold_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=800,
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
        oof_proba[val_idx] = val_pred
        fold_scores.append(roc_auc_score(y_val, val_pred))

    best_t, best_acc = find_best_threshold_acc(y.values.astype(int), oof_proba)
    return {
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "best_threshold": best_t,
        "best_oof_acc": best_acc,
        "n_features": int(X.shape[1]),
        "n_categorical": int(len(categorical_features)),
    }


def main():
    train_df, _ = load_train_test()

    experiments: dict[str, dict[str, bool]] = {
        "baseline_only": {
            "spending_ratios": False,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": False,
        },
        "ratios_only": {
            "spending_ratios": True,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": False,
        },
        "deck_unknown_only": {
            "spending_ratios": False,
            "deck_unknown": True,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": False,
        },
        "group_aggregates_only": {
            "spending_ratios": False,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": True,
            "interactions": False,
        },
        "ratios_plus_deck_unknown": {
            "spending_ratios": True,
            "deck_unknown": True,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": False,
        },
        "group_diversity_only": {
            "spending_ratios": False,
            "deck_unknown": False,
            "group_diversity": True,
            "group_aggregates": False,
            "interactions": False,
        },
        "ratios_plus_group_diversity": {
            "spending_ratios": True,
            "deck_unknown": False,
            "group_diversity": True,
            "group_aggregates": False,
            "interactions": False,
        },
        "ratios_plus_group_aggregates": {
            "spending_ratios": True,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": True,
            "interactions": False,
        },
        "interactions_only": {
            "spending_ratios": False,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": True,
        },
        "ratios_plus_interactions": {
            "spending_ratios": True,
            "deck_unknown": False,
            "group_diversity": False,
            "group_aggregates": False,
            "interactions": True,
        },
        "group_diversity_plus_interactions": {
            "spending_ratios": False,
            "deck_unknown": False,
            "group_diversity": True,
            "group_aggregates": False,
            "interactions": True,
        },
        "all_upgrades": {
            "spending_ratios": True,
            "deck_unknown": True,
            "group_diversity": True,
            "group_aggregates": False,
            "interactions": True,
        },
    }

    rows = []
    for name, flags in experiments.items():
        print(f"Running: {name}")
        results = run_cv(train_df, flags)
        rows.append({"experiment": name, **flags, **results})

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "feature_experiments.csv"
    out_md = out_dir / "feature_experiments.md"

    df = pd.DataFrame(rows).sort_values(by="best_oof_acc", ascending=False)
    df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Feature Experiments (CatBoost CV)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
