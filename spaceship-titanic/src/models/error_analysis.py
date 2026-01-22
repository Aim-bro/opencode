from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.catboost_baseline import (
    SPENDING_COLS,
    find_best_threshold_acc,
    preprocess,
)


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _add_group_diversity(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    group_id = raw_df["PassengerId"].str.split("_", expand=True)[0]
    deck = raw_df["Cabin"].str.split("/", expand=True)[0].fillna("Unknown")

    group_stats = pd.DataFrame(
        {
            "GroupId": group_id,
            "Deck": deck,
            "Destination": raw_df["Destination"],
            "CryoSleep": raw_df["CryoSleep"],
        }
    )
    group_stats["Destination"] = group_stats["Destination"].fillna("Unknown")
    group_stats["CryoSleep"] = group_stats["CryoSleep"].fillna("Unknown")

    agg = group_stats.groupby("GroupId").agg(
        group_deck_diversity=("Deck", "nunique"),
        group_destination_diversity=("Destination", "nunique"),
        group_cryosleep_diversity=("CryoSleep", "nunique"),
    )
    processed_df = processed_df.merge(agg, on="GroupId", how="left")
    return processed_df


def load_data() -> tuple[pd.DataFrame, float]:
    oof_df = pd.read_csv("outputs/oof_predictions.csv")
    train_df = pd.read_csv("data/raw/train.csv")

    processed = preprocess(train_df.copy())
    processed["PassengerId"] = train_df["PassengerId"].values

    processed = _add_group_diversity(train_df, processed)

    merged = processed.merge(oof_df, on="PassengerId", how="inner")

    best_t, _ = find_best_threshold_acc(
        merged["true_label"].values.astype(int), merged["oof_proba"].values
    )

    merged["pred_label"] = (merged["oof_proba"] >= best_t).astype(int)
    merged["correct"] = merged["pred_label"] == merged["true_label"]

    merged["error_type"] = "correct"
    merged.loc[
        (merged["true_label"] == 1) & (merged["pred_label"] == 0), "error_type"
    ] = "false_negative"
    merged.loc[
        (merged["true_label"] == 0) & (merged["pred_label"] == 1), "error_type"
    ] = "false_positive"

    return merged, best_t


def analyze_error_distribution(df: pd.DataFrame, error_type: str) -> dict | None:
    errors = df[df["error_type"] == error_type].copy()
    correct = df[df["error_type"] == "correct"].copy()

    if len(errors) == 0:
        return None

    analysis = {
        "count": len(errors),
        "percentage": len(errors) / len(df) * 100,
    }

    numeric_features = [
        "Age",
        "TotalSpending",
        "GroupSize",
        "FamilySize",
        "oof_proba",
        "group_deck_diversity",
        "group_destination_diversity",
        "group_cryosleep_diversity",
    ]
    for feature in numeric_features:
        if feature not in errors.columns:
            continue
        analysis[feature] = {
            "mean": errors[feature].mean(),
            "median": errors[feature].median(),
            "std": errors[feature].std(),
            "q25": errors[feature].quantile(0.25),
            "q75": errors[feature].quantile(0.75),
        }
        if feature in correct.columns:
            analysis[f"{feature}_correct_mean"] = correct[feature].mean()
            analysis[f"{feature}_correct_median"] = correct[feature].median()

    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "Deck",
        "Side",
        "AgeGroup",
    ]
    for feature in categorical_features:
        if feature not in errors.columns:
            continue
        analysis[f"{feature}_dist"] = (
            errors[feature].value_counts(normalize=True).head().to_dict()
        )
        analysis[f"{feature}_correct_dist"] = (
            correct[feature].value_counts(normalize=True).head().to_dict()
        )

    for col in SPENDING_COLS:
        if col not in errors.columns:
            continue
        analysis[f"{col}_nonzero_pct"] = (errors[col] > 0).mean()
        analysis[f"{col}_correct_nonzero_pct"] = (correct[col] > 0).mean()
        analysis[f"{col}_mean"] = errors[col].mean()
        analysis[f"{col}_correct_mean"] = correct[col].mean()

    return analysis


def find_top_patterns(df: pd.DataFrame) -> list[dict]:
    fn = df[df["error_type"] == "false_negative"]
    fp = df[df["error_type"] == "false_positive"]
    correct = df[df["error_type"] == "correct"]

    patterns: list[dict] = []

    young_fn = fn[fn["Age"].between(18, 30)]
    if len(young_fn) > 50:
        patterns.append(
            {
                "name": "Young Adult False Negatives",
                "type": "false_negative",
                "count": len(young_fn),
                "percentage": len(young_fn) / len(fn) * 100,
                "core_reasoning": (
                    "Young adults (18-30) are a large FN segment; interactions with "
                    "CryoSleep or Destination may be under-modeled."
                ),
                "evidence_summary": (
                    f"Age mean FN: {fn['Age'].mean():.1f} vs Correct: "
                    f"{correct['Age'].mean():.1f}. "
                    f"CryoSleep rate FN: {_is_true(fn['CryoSleep']).mean()*100:.1f}% vs "
                    f"Correct: {_is_true(correct['CryoSleep']).mean()*100:.1f}%. "
                    f"OOF proba mean FN: {fn['oof_proba'].mean():.3f}"
                ),
                "hypothesis": (
                    "Age interacts with CryoSleep or Destination in a way not captured "
                    "by current features."
                ),
                "validation": (
                    "Compare Transported by (AgeGroup x CryoSleep) and "
                    "(AgeGroup x Destination) cross-tabs."
                ),
                "feature_change": "Add AgeGroup x CryoSleep and AgeGroup x Destination",
                "discarded_alternatives": (
                    "Refine age bins only (limited), remove Age (information loss)"
                ),
            }
        )

    high_spending_fp = fp[fp["TotalSpending"] > fp["TotalSpending"].quantile(0.75)]
    if len(high_spending_fp) > 50:
        patterns.append(
            {
                "name": "High Spending False Positives",
                "type": "false_positive",
                "count": len(high_spending_fp),
                "percentage": len(high_spending_fp) / len(fp) * 100,
                "core_reasoning": (
                    "High spenders are overpredicted; spending mix may matter more than "
                    "total volume."
                ),
                "evidence_summary": (
                    f"TotalSpending mean FP: {fp['TotalSpending'].mean():.0f} vs "
                    f"Correct: {correct['TotalSpending'].mean():.0f}. "
                    f"RoomService nonzero FP: {(fp['RoomService']>0).mean()*100:.1f}% "
                    f"vs Correct: {(correct['RoomService']>0).mean()*100:.1f}%."
                ),
                "hypothesis": (
                    "Spending ratios (e.g., Spa/Total) are more predictive than totals."
                ),
                "validation": (
                    "Compare spending ratio distributions for FP vs correct groups."
                ),
                "feature_change": "Add spending ratio features (Spa/Total, VRDeck/Total)",
                "discarded_alternatives": (
                    "Only log-transform spending (may be insufficient)"
                ),
            }
        )

    group_fn = fn[fn["GroupSize"] > 1]
    if len(group_fn) > 50:
        patterns.append(
            {
                "name": "Multi-passenger False Negatives",
                "type": "false_negative",
                "count": len(group_fn),
                "percentage": len(group_fn) / len(fn) * 100,
                "core_reasoning": (
                    "Groups with mixed characteristics may have lower transport "
                    "consistency than groups with homogeneous attributes."
                ),
                "evidence_summary": (
                    f"GroupSize mean FN: {fn['GroupSize'].mean():.2f} vs Correct: "
                    f"{correct['GroupSize'].mean():.2f}. "
                    f"Deck diversity mean FN: {fn['group_deck_diversity'].mean():.2f} "
                    f"vs Correct: {correct['group_deck_diversity'].mean():.2f}."
                ),
                "hypothesis": (
                    "Group diversity (deck/destination/cryosleep) affects transport odds."
                ),
                "validation": (
                    "Compare error rates by group diversity buckets (1 vs >1)."
                ),
                "feature_change": "Add group diversity features",
                "discarded_alternatives": "Remove GroupSize (information loss)",
            }
        )

    cryo_fn = fn[_is_true(fn["CryoSleep"])]
    if len(cryo_fn) > 30:
        patterns.append(
            {
                "name": "CryoSleep False Negatives",
                "type": "false_negative",
                "count": len(cryo_fn),
                "percentage": len(cryo_fn) / len(fn) * 100,
                "core_reasoning": (
                    "CryoSleep is strong but not absolute; exceptions may be tied to "
                    "spending and cabin data quality."
                ),
                "evidence_summary": (
                    f"CryoSleep FN count: {len(cryo_fn)}. "
                    f"TotalSpending mean FN: {fn['TotalSpending'].mean():.0f} vs "
                    f"Correct: {correct['TotalSpending'].mean():.0f}. "
                    f"Unknown Deck rate FN: {(fn['Deck']=='Unknown').mean()*100:.1f}%."
                ),
                "hypothesis": (
                    "CryoSleep passengers with non-zero spending or unknown cabins "
                    "behave differently."
                ),
                "validation": (
                    "Compare CryoSleep errors by HasSpending and Deck==Unknown."
                ),
                "feature_change": "Add CryoSleep x HasSpending and DeckUnknown flags",
                "discarded_alternatives": "Hard CryoSleep rule (CV mismatch risk)",
            }
        )

    premium_fp = fp[fp["Deck"].isin(["A", "B", "C"])]
    if len(premium_fp) > 20:
        patterns.append(
            {
                "name": "Premium Deck False Positives",
                "type": "false_positive",
                "count": len(premium_fp),
                "percentage": len(premium_fp) / len(fp) * 100,
                "core_reasoning": (
                    "Premium decks skew positive; some profiles may be overpredicted."
                ),
                "evidence_summary": (
                    f"Premium deck FP: {len(premium_fp)}/{len(fp)} "
                    f"({len(premium_fp)/len(fp)*100:.1f}%). "
                    f"VIP rate FP: {_is_true(fp['VIP']).mean()*100:.1f}% vs "
                    f"Correct: {_is_true(correct['VIP']).mean()*100:.1f}%."
                ),
                "hypothesis": (
                    "Deck effect interacts with VIP and Age; premium deck alone is "
                    "overweighted."
                ),
                "validation": "Cross-tab Deck x VIP x AgeGroup vs Transported.",
                "feature_change": "Add Deck x VIP and Deck x AgeGroup interactions",
                "discarded_alternatives": "Drop Deck (information loss)",
            }
        )

    patterns.sort(key=lambda x: x["count"], reverse=True)
    return patterns


def generate_report(df: pd.DataFrame, threshold: float, patterns: list[dict]) -> str:
    total = len(df)
    correct = (df["correct"]).sum()
    incorrect = total - correct

    fn_count = len(df[df["error_type"] == "false_negative"])
    fp_count = len(df[df["error_type"] == "false_positive"])

    report = f"""# Error Analysis Report

## Threshold: {threshold:.3f}
## Overall Accuracy: {correct/total:.4f}
- Correct predictions: {correct} ({correct/total*100:.2f}%)
- Incorrect predictions: {incorrect} ({incorrect/total*100:.2f}%)
  - False Negatives: {fn_count} ({fn_count/total*100:.2f}%)
  - False Positives: {fp_count} ({fp_count/total*100:.2f}%)

## Top 5 Failure Patterns

"""

    for i, pattern in enumerate(patterns[:5], 1):
        report += f"""
### Pattern {i}: {pattern['name']}

**Type**: {pattern['type']}
**Count**: {pattern['count']} ({pattern['percentage']:.2f}% of {pattern['type']}s)

**Core Reasoning**: {pattern['core_reasoning']}

**Evidence Summary**: {pattern['evidence_summary']}

**Hypothesis**: {pattern['hypothesis']}

**How to Validate**: {pattern['validation']}

**Which Feature Change to Try**: {pattern['feature_change']}

**Choice vs. Discarded Alternatives**: {pattern['discarded_alternatives']}

---
"""

    report += """
## Ranked Experiment List (Max 6)

"""

    for i, pattern in enumerate(patterns[:6], 1):
        feature_part = pattern["feature_change"].replace("Add ", "")
        report += f"{i}. {pattern['name']} - {feature_part}\n"

    return report


def main():
    print("Loading data...")
    df, threshold = load_data()

    print("Analyzing error patterns...")
    patterns = find_top_patterns(df)

    print("Generating report...")
    report = generate_report(df, threshold, patterns)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "06_error_analysis.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Error analysis report saved to: {report_path}")
    return df, threshold, patterns


if __name__ == "__main__":
    main()
