# Error Analysis Report

## Threshold: 0.500
## Overall Accuracy: 0.8165
- Correct predictions: 7098 (81.65%)
- Incorrect predictions: 1595 (18.35%)
  - False Negatives: 758 (8.72%)
  - False Positives: 837 (9.63%)

## Top 5 Failure Patterns


### Pattern 1: Young Adult False Negatives

**Type**: false_negative
**Count**: 339 (44.72% of false_negatives)

**Core Reasoning**: Young adults (18-30) are a large FN segment; interactions with CryoSleep or Destination may be under-modeled.

**Evidence Summary**: Age mean FN: 30.0 vs Correct: 29.1. CryoSleep rate FN: 12.5% vs Correct: 36.1%. OOF proba mean FN: 0.293

**Hypothesis**: Age interacts with CryoSleep or Destination in a way not captured by current features.

**How to Validate**: Compare Transported by (AgeGroup x CryoSleep) and (AgeGroup x Destination) cross-tabs.

**Which Feature Change to Try**: Add AgeGroup x CryoSleep and AgeGroup x Destination

**Choice vs. Discarded Alternatives**: Refine age bins only (limited), remove Age (information loss)

---

### Pattern 2: Multi-passenger False Negatives

**Type**: false_negative
**Count**: 280 (36.94% of false_negatives)

**Core Reasoning**: Groups with mixed characteristics may have lower transport consistency than groups with homogeneous attributes.

**Evidence Summary**: GroupSize mean FN: 1.97 vs Correct: 2.02. Deck diversity mean FN: 1.23 vs Correct: 1.18.

**Hypothesis**: Group diversity (deck/destination/cryosleep) affects transport odds.

**How to Validate**: Compare error rates by group diversity buckets (1 vs >1).

**Which Feature Change to Try**: Add group diversity features

**Choice vs. Discarded Alternatives**: Remove GroupSize (information loss)

---

### Pattern 3: High Spending False Positives

**Type**: false_positive
**Count**: 207 (24.73% of false_positives)

**Core Reasoning**: High spenders are overpredicted; spending mix may matter more than total volume.

**Evidence Summary**: TotalSpending mean FP: 607 vs Correct: 1561. RoomService nonzero FP: 16.6% vs Correct: 34.2%.

**Hypothesis**: Spending ratios (e.g., Spa/Total) are more predictive than totals.

**How to Validate**: Compare spending ratio distributions for FP vs correct groups.

**Which Feature Change to Try**: Add spending ratio features (Spa/Total, VRDeck/Total)

**Choice vs. Discarded Alternatives**: Only log-transform spending (may be insufficient)

---

### Pattern 4: CryoSleep False Negatives

**Type**: false_negative
**Count**: 95 (12.53% of false_negatives)

**Core Reasoning**: CryoSleep is strong but not absolute; exceptions may be tied to spending and cabin data quality.

**Evidence Summary**: CryoSleep FN count: 95. TotalSpending mean FN: 1235 vs Correct: 1561. Unknown Deck rate FN: 1.7%.

**Hypothesis**: CryoSleep passengers with non-zero spending or unknown cabins behave differently.

**How to Validate**: Compare CryoSleep errors by HasSpending and Deck==Unknown.

**Which Feature Change to Try**: Add CryoSleep x HasSpending and DeckUnknown flags

**Choice vs. Discarded Alternatives**: Hard CryoSleep rule (CV mismatch risk)

---

### Pattern 5: Premium Deck False Positives

**Type**: false_positive
**Count**: 62 (7.41% of false_positives)

**Core Reasoning**: Premium decks skew positive; some profiles may be overpredicted.

**Evidence Summary**: Premium deck FP: 62/837 (7.4%). VIP rate FP: 1.4% vs Correct: 2.4%.

**Hypothesis**: Deck effect interacts with VIP and Age; premium deck alone is overweighted.

**How to Validate**: Cross-tab Deck x VIP x AgeGroup vs Transported.

**Which Feature Change to Try**: Add Deck x VIP and Deck x AgeGroup interactions

**Choice vs. Discarded Alternatives**: Drop Deck (information loss)

---

## Ranked Experiment List (Max 6)

1. Young Adult False Negatives - AgeGroup x CryoSleep and AgeGroup x Destination
2. Multi-passenger False Negatives - group diversity features
3. High Spending False Positives - spending ratio features (Spa/Total, VRDeck/Total)
4. CryoSleep False Negatives - CryoSleep x HasSpending and DeckUnknown flags
5. Premium Deck False Positives - Deck x VIP and Deck x AgeGroup interactions
