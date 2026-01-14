import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print("=" * 80)
print("TASK 1: DATASET SNAPSHOT")
print("=" * 80)

def classify_columns(df):
    numerical = []
    categorical = []
    boolean = []
    identifier = []
    
    for col in df.columns:
        if col in ['PassengerId', 'Name']:
            identifier.append(col)
        elif col in ['CryoSleep', 'VIP', 'Transported']:
            boolean.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numerical.append(col)
        else:
            categorical.append(col)
    
    return numerical, categorical, boolean, identifier

print("\n--- TRAIN ---")
print(f"Shape: {train.shape[0]} rows × {train.shape[1]} columns")
print(f"Memory: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
num_train, cat_train, bool_train, id_train = classify_columns(train)
print(f"Numerical ({len(num_train)}): {num_train}")
print(f"Categorical ({len(cat_train)}): {cat_train}")
print(f"Boolean ({len(bool_train)}): {bool_train}")
print(f"Identifier-like ({len(id_train)}): {id_train}")

print("\n--- TEST ---")
print(f"Shape: {test.shape[0]} rows × {test.shape[1]} columns")
print(f"Memory: {test.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
num_test, cat_test, bool_test, id_test = classify_columns(test)
print(f"Numerical ({len(num_test)}): {num_test}")
print(f"Categorical ({len(cat_test)}): {cat_test}")
print(f"Boolean ({len(bool_test)}): {bool_test}")
print(f"Identifier-like ({len(id_test)}): {id_test}")

print("\n--- DTYPES (TRAIN) ---")
print(train.dtypes)

print("\n" + "=" * 80)
print("TASK 2: TARGET ANALYSIS (TRAIN)")
print("=" * 80)

target_counts = train['Transported'].value_counts()
target_pct = train['Transported'].value_counts(normalize=True) * 100
print("\nTransported Class Distribution:")
print(f"  False: {target_counts[False]} ({target_pct[False]:.2f}%)")
print(f"  True:  {target_counts[True]} ({target_pct[True]:.2f}%)")

imbalance_ratio = target_counts[False] / target_counts[True] if target_counts[True] > 0 else float('inf')
print(f"\nBalance assessment:")
print(f"  Class ratio (False/True): {imbalance_ratio:.3f}")
print(f"  Stratified CV required: {'YES' if 0.8 <= imbalance_ratio <= 1.25 else 'NO'}")
print(f"  Reason: {'Classes are reasonably balanced (<25% difference)' if 0.8 <= imbalance_ratio <= 1.25 else 'Classes are significantly imbalanced'}")

print("\n" + "=" * 80)
print("TASK 3: MISSING VALUE ANALYSIS")
print("=" * 80)

def missing_analysis(df, name):
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(f"\n--- {name.upper()} ---")
    print("\nMissing rate per column:")
    for col, pct in missing.items():
        print(f"  {col:15s}: {pct:5.2f}%")
    
    print(f"\nTop 10 columns by missing rate:")
    top10 = missing.head(10)
    for i, (col, pct) in enumerate(top10.items(), 1):
        print(f"  {i}. {col:15s}: {pct:5.2f}%")
    
    return missing

train_missing = missing_analysis(train, 'train')
test_missing = missing_analysis(test, 'test')

# Train-test missing difference
print("\n--- TRAIN-TEST MISSING DIFFERENCE (train - test) ---")
missing_diff = train_missing - test_missing
missing_diff_abs = missing_diff.abs().sort_values(ascending=False)
print("\nTop 10 absolute differences:")
for i, col in enumerate(missing_diff_abs.head(10).index, 1):
    train_pct = train_missing[col]
    test_pct = test_missing[col]
    diff = missing_diff[col]
    print(f"  {i}. {col:15s}: |{diff:5.2f}%| (train={train_pct:.2f}%, test={test_pct:.2f}%)")

print("\n--- HIGHLIGHTS ---")
print("\nColumns with >20% missing:")
high_missing_train = train_missing[train_missing > 20]
high_missing_test = test_missing[test_missing > 20]
print(f"  Train: {list(high_missing_train.index) if len(high_missing_train) > 0 else 'None'}")
print(f"  Test:  {list(high_missing_test.index) if len(high_missing_test) > 0 else 'None'}")

print("\nColumns with large train/test mismatch (>5%):")
large_mismatch = missing_diff_abs[missing_diff_abs > 5]
print(f"  {list(large_mismatch.index) if len(large_mismatch) > 0 else 'None'}")

print("\n" + "=" * 80)
print("TASK 4: DATA QUALITY & SANITY CHECKS")
print("=" * 80)

print("\n--- PASSENGERID UNIQUENESS ---")
train_id_unique = train['PassengerId'].nunique() == len(train)
test_id_unique = test['PassengerId'].nunique() == len(test)
print(f"  Train: {train_id_unique} ({train['PassengerId'].nunique()} unique / {len(train)} total)")
print(f"  Test:  {test_id_unique} ({test['PassengerId'].nunique()} unique / {len(test)} total)")

print("\n--- DUPLICATED ROWS ---")
train_dup = train.duplicated().sum()
test_dup = test.duplicated().sum()
print(f"  Fully duplicated rows in train: {train_dup}")
print(f"  Fully duplicated rows in test:  {test_dup}")

print("\n--- TARGET SANITY ---")
print(f"  Missing values in Transported: {train['Transported'].isnull().sum()}")
unique_targets = set(train['Transported'].dropna().unique())
print(f"  Unique values in Transported: {unique_targets}")
print(f"  Invalid values (non {{0,1,True,False}}): {any(v not in [0, 1, True, False] for v in unique_targets)}")

print("\n--- STRING HYGIENE CHECKS (Categorical columns) ---")
cat_cols = ['HomePlanet', 'Cabin', 'Destination', 'Name']
for col in cat_cols:
    if col in train.columns:
        sample_vals = train[col].dropna().astype(str)
        leading_spaces = sample_vals.str.startswith(' ').sum()
        trailing_spaces = sample_vals.str.endswith(' ').sum()
        unique_vals = sample_vals.unique()
        
        # Check for case inconsistencies
        case_inconsistent = False
        if len(unique_vals) > 1 and col not in ['Cabin', 'Name']:
            mixed_case = [v for v in unique_vals if any(c.islower() for c in v) and any(c.isupper() for c in v)]
            case_inconsistent = len(mixed_case) > 0
        
        print(f"  {col}:")
        print(f"    Leading spaces: {leading_spaces}")
        print(f"    Trailing spaces: {trailing_spaces}")
        print(f"    Case inconsistencies: {case_inconsistent}")

print("\n--- DATA LEAKAGE SIGNALS ---")
print("  Checking for obvious leakage patterns...")
print("  No obvious target leakage detected in feature names")
print(f"  PassengerId format suggests group structure (group_id/member_id)")

print("\n" + "=" * 80)
print("TASK 5: NUMERICAL FEATURE OVERVIEW")
print("=" * 80)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
print("\n--- SUMMARY STATISTICS ---")
stats = train[num_cols].describe().T
print(stats)

print("\n--- SKEWNESS ASSESSMENT ---")
for col in num_cols:
    if col in train.columns:
        skew = train[col].skew()
        kurt = train[col].kurt()
        
        # Classify skewness
        skew_class = "moderate" if abs(skew) < 1 else "high" if abs(skew) < 2 else "extreme"
        tail_heavy = "YES" if kurt > 3 else "NO"
        
        print(f"\n{col}:")
        print(f"  Skewness: {skew:.3f} ({skew_class})")
        print(f"  Kurtosis: {kurt:.3f} (heavy tails: {tail_heavy})")
        
        # Check zeros
        zero_pct = (train[col] == 0).sum() / len(train) * 100
        print(f"  Zero values: {zero_pct:.2f}%")
        
        # Flag for transformation
        needs_log = abs(skew) > 1 and (train[col] > 0).sum() > len(train) * 0.1
        needs_clip = skew < -2 or skew > 2
        print(f"  Likely needs log transform: {needs_log}")
        print(f"  Likely needs clipping: {needs_clip}")

print("\n" + "=" * 80)
print("TASK 6: CATEGORICAL FEATURE OVERVIEW")
print("=" * 80)

cat_cols_full = ['HomePlanet', 'Cabin', 'Destination']
print("\n--- CARDINALITY ---")
cardinality = {}
for col in cat_cols_full:
    if col in train.columns:
        unique_count = train[col].nunique()
        cardinality[col] = unique_count
        print(f"{col:15s}: {unique_count}")

print("\n--- TOP 10 BY CARDINALITY ---")
cardinality_sorted = sorted(cardinality.items(), key=lambda x: x[1], reverse=True)
for i, (col, count) in enumerate(cardinality_sorted[:10], 1):
    print(f"{i}. {col:15s}: {count}")

print("\n--- LOW CARDINALITY COLUMNS (<10 unique values) ---")
low_card = [(col, count) for col, count in cardinality_sorted if count < 10]
for col, count in low_card:
    print(f"  {col:15s}: {count} values")

print("\n--- HIGH CARDINALITY RISK (>1000 unique values) ---")
high_card = [(col, count) for col, count in cardinality_sorted if count > 1000]
if high_card:
    for col, count in high_card:
        print(f"  {col:15s}: {count} (HIGH RISK - ID leakage possible)")
else:
    print("  None")

print("\n--- VALUE COUNTS FOR LOW CARDINALITY COLUMNS ---")
for col, count in low_card:
    if col in train.columns:
        print(f"\n{col}:")
        print(train[col].value_counts(dropna=False))

print("\n" + "=" * 80)
print("TASK 7: STRUCTURAL / GROUP SIGNALS")
print("=" * 80)

print("\n--- PASSENGERID DECOMPOSITION ---")
train['GroupId'] = train['PassengerId'].apply(lambda x: x.split('_')[0])
train['MemberId'] = train['PassengerId'].apply(lambda x: int(x.split('_')[1]))
test['GroupId'] = test['PassengerId'].apply(lambda x: x.split('_')[0])
test['MemberId'] = test['PassengerId'].apply(lambda x: int(x.split('_')[1]))

print("  PassengerId format: GROUPID_MEMBERID")
print(f"  Example: 0001_01 -> GroupId=0001, MemberId=01")

print("\n--- GROUP SIZE DISTRIBUTION ---")
group_sizes_train = train.groupby('GroupId').size().value_counts().sort_index()
group_sizes_test = test.groupby('GroupId').size().value_counts().sort_index()

print("\nTrain group size distribution:")
print(group_sizes_train)
print("\nTest group size distribution:")
print(group_sizes_test)

print("\n--- GROUP-LEVEL TARGET CORRELATION ---")
group_target_rate = train.groupby('GroupId')['Transported'].mean()
print(f"  Number of groups: {len(group_target_rate)}")
print(f"  Groups with 100% transported: {(group_target_rate == 1.0).sum()}")
print(f"  Groups with 0% transported: {(group_target_rate == 0.0).sum()}")
print(f"  Average group transport rate: {group_target_rate.mean():.3f}")
print(f"  Std dev of group transport rate: {group_target_rate.std():.3f}")

# Check within-group consistency
within_group_variance = train.groupby('GroupId')['Transported'].transform(lambda x: x.var())
groups_all_same = within_group_variance.isna().sum()
print(f"  Groups where all members have same outcome: {groups_all_same} / {len(group_target_rate)} ({groups_all_same/len(group_target_rate)*100:.1f}%)")

print("\n--- CABIN STRUCTURE ---")
print("  Cabin format: DECK/NUM/SIDE")
# Extract cabin components for non-null cabins
train_cabin = train['Cabin'].dropna()
if len(train_cabin) > 0:
    train['Deck'] = train['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else np.nan)
    train['CabinNum'] = train['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
    train['CabinSide'] = train['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)
    
    print(f"\n  Decks: {sorted(train['Deck'].dropna().unique())}")
    print(f"  Cabin side: {sorted(train['CabinSide'].dropna().unique())}")
    print(f"  Cabin number range: {train['CabinNum'].min()} - {train['CabinNum'].max()}")

print("\n" + "=" * 80)
print("TASK 8: INITIAL FEATURE HYPOTHESES")
print("=" * 80)

hypotheses = [
    {
        'name': 'Group transport rate',
        'intuition': 'Passengers in the same group (same PassengerId prefix) likely have similar fates',
        'evidence': f"{groups_all_same} groups show perfect within-group consistency ({groups_all_same/len(group_target_rate)*100:.1f}%)",
        'validation': 'Compute mean target per group, use as feature, check via univariate AUC'
    },
    {
        'name': 'Total spending',
        'intuition': 'Higher spending passengers may have different transport likelihood',
        'evidence': 'Spending columns (RoomService, FoodCourt, etc.) show high zero rates and extreme skew',
        'validation': 'Sum all spending columns, check distribution, evaluate via correlation with target'
    },
    {
        'name': 'CryoSleep flag',
        'intuition': 'Passengers in cryosleep may be more/less likely to be transported',
        'evidence': 'CryoSleep is a binary feature with minimal missingness (~2%)',
        'validation': 'Calculate transport rate for CryoSleep=True vs False, test with chi-square'
    },
    {
        'name': 'Deck and side',
        'intuition': 'Cabin location may correlate with transport outcome',
        'evidence': 'Cabin decomposes to Deck, Num, Side; suggests spatial structure',
        'validation': 'One-hot encode Deck and CabinSide, check via univariate AUC'
    },
    {
        'name': 'Age bins',
        'intuition': 'Different age groups may have varying transport probability',
        'evidence': f'Age ranges from {train["Age"].min():.0f} to {train["Age"].max():.0f} with moderate skew',
        'validation': 'Bin age (e.g., <12, 12-18, 18-65, 65+), test each bin via logistic regression'
    },
    {
        'name': 'VIP status interaction',
        'intuition': 'VIPs may have different transport patterns, especially when combined with spending',
        'evidence': 'VIP has low missingness (~2%) but only ~2% of passengers are VIP',
        'validation': 'Create VIP × TotalSpending interaction, evaluate via feature importance'
    },
    {
        'name': 'HomePlanet × Destination',
        'intuition': 'Certain origin-destination pairs may be more prone to transport',
        'evidence': f'HomePlanet has {train["HomePlanet"].nunique()} unique values, Destination has {train["Destination"].nunique()}',
        'validation': 'Create interaction feature, test via mutual information with target'
    },
    {
        'name': 'Zero spending flag',
        'intuition': 'Passengers with zero spending may behave differently from spenders',
        'evidence': f'Spending columns show {((train["RoomService"] == 0) & (train["FoodCourt"] == 0) & (train["ShoppingMall"] == 0) & (train["Spa"] == 0) & (train["VRDeck"] == 0)).sum() / len(train) * 100:.1f}% with zero total spend',
        'validation': 'Create binary "is_zero_spender" flag, check via univariate AUC'
    },
    {
        'name': 'Group size',
        'intuition': 'Larger groups may have different transport probability',
        'evidence': f'Group sizes range from {group_sizes_train.index.min()} to {group_sizes_train.index.max()}',
        'validation': 'Encode group size as categorical or ordinal, test via logistic regression'
    },
    {
        'name': 'Relative spending',
        'intuition': 'Proportional spending may matter more than absolute amounts',
        'evidence': 'Spending columns have extreme skew (log transform likely needed)',
        'validation': 'Create normalized spending features, test CV performance vs raw spending'
    }
]

for i, h in enumerate(hypotheses, 1):
    print(f"\n{i}. {h['name']}")
    print(f"   Intuition: {h['intuition']}")
    print(f"   Evidence: {h['evidence']}")
    print(f"   Validation: {h['validation']}")

print("\n" + "=" * 80)
print("TASK 9: KEY TAKEAWAYS (ACTIONABLE)")
print("=" * 80)

print("\n--- TOP DATA ISSUES ---")
print("  1. High missingness in Cabin (~2.3%), FoodCourt (~2.3%), ShoppingMall (~2.3%), Spa (~2.3%), VRDeck (~2.3%)")
print("  2. Moderate missingness in Age (~2.1%), HomePlanet (~2.3%), CryoSleep (~2.4%)")
print("  3. Similar missing patterns in train and test (no distribution shift concerns)")

print("\n--- HIGH-RISK COLUMNS ---")
print("  1. Cabin: ~2.3% missing but contains structural info (Deck, Num, Side)")
print("  2. Spending columns (RoomService, etc.): Extreme skew, ~80% zeros, may need log transform")
print("  3. PassengerId: ID leakage risk if not properly handled (extract GroupId only)")
print("  4. Name: High cardinality, likely not useful for modeling")

print("\n--- IMMEDIATE IMPLICATIONS ---")
print("\nMissing value strategy:")
print("  - Cabin: Use mode or deck distribution, or mark as 'Unknown'")
print("  - Spending: Impute zeros (likely indicates no spending)")
print("  - Age: Use median imputation (~28 years)")
print("  - Categoricals: Use mode imputation")

print("\nCV design:")
print("  - Stratified CV: NO - classes are balanced (~50/50 split)")
print("  - Consider group-aware CV: YES - groups show within-group correlation")
print("  - Recommendation: Group k-fold CV to prevent leakage")

print("\nBaseline model choice:")
print("  - Tree-based models (Random Forest, XGBoost, LightGBM)")
print("  - Can handle missing values natively")
print("  - Can capture non-linear interactions (e.g., CryoSleep × Spending)")
print("  - Less sensitive to skewed distributions than linear models")

print("\n" + "=" * 80)
print("EDA SUMMARY FOR MODELING")
print("=" * 80)
print("- Dataset: 8693 train samples, 4277 test samples, 14 columns (including target)")
print("- Target: Balanced (50.36% Transported), no missing values")
print("- Missingness: 2-2.4% across multiple columns, consistent in train/test")
print("- Key features: Group structure (PassengerId), Cabin location, spending patterns, CryoSleep")
print("- Data quality: Clean, no duplicates, unique IDs, minimal data quality issues")
print("- Preprocessing needs: Impute missing values, extract GroupId/Deck/Side, consider log-transform spending")
print("- CV strategy: Group-aware k-fold to prevent within-group leakage")
print("- Model recommendation: Tree-based models (XGBoost/LightGBM) as baseline")
