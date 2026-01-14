import pandas as pd
import numpy as np
from scipy import stats

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print("=" * 80)
print("TASK 1: MISSING RATE PER COLUMN")
print("=" * 80)

# Calculate missing rates
train_missing = train.isnull().mean() * 100
test_missing = test.isnull().mean() * 100

# Create comparison table
missing_comparison = pd.DataFrame({
    'Train_Missing_%': train_missing.round(2),
    'Test_Missing_%': test_missing.round(2),
    'Diff_%': (train_missing - test_missing).round(2)
})
missing_comparison = missing_comparison[missing_comparison['Train_Missing_%'] > 0]
print("\nMissing rates comparison (columns with any missing values):")
print(missing_comparison.sort_values('Train_Missing_%', ascending=False))

print("\n" + "=" * 80)
print("TASK 2: CO-MISSINGNESS GROUPS")
print("=" * 80)

# Calculate co-missingness matrix
missing_cols = train.columns[train.isnull().any()].tolist()
print(f"\nColumns with missing values: {missing_cols}")

# Pairwise co-missingness
print("\nPairwise co-missingness (percentage of rows where both are missing):")
for i, col1 in enumerate(missing_cols):
    co_missing = {}
    for col2 in missing_cols[i+1:]:
        both_missing = train[col1].isnull() & train[col2].isnull()
        co_missing[col2] = (both_missing.sum() / len(train) * 100).round(2)
    
    # Show only significant co-missingness (>1%)
    significant = {k: v for k, v in co_missing.items() if v > 1.0}
    if significant:
        print(f"\n{col1}:")
        for col2, pct in sorted(significant.items(), key=lambda x: -x[1]):
            print(f"  {col2}: {pct}%")

# Triple co-missingness (check groups of 3)
print("\n\nTriple co-missingness (percentage of rows where all 3 are missing):")
from itertools import combinations

for col1, col2, col3 in combinations(missing_cols, 3):
    all_missing = train[col1].isnull() & train[col2].isnull() & train[col3].isnull()
    pct = (all_missing.sum() / len(train) * 100).round(2)
    if pct > 1.0:
        print(f"{col1} + {col2} + {col3}: {pct}%")

# Check CryoSleep and spending columns
print("\n\nCo-missingness analysis for CryoSleep:")
cryo_sleep_missing = train['CryoSleep'].isnull()
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("\nPercentage of CryoSleep missing rows where spending columns are also missing:")
for col in spending_cols:
    both = cryo_sleep_missing & train[col].isnull()
    pct = (both.sum() / cryo_sleep_missing.sum() * 100).round(2)
    print(f"  {col}: {pct}%")

# Check if CryoSleep missing rows have zero spending
print("\nPercentage of CryoSleep missing rows with zero spending:")
for col in spending_cols:
    zero_spend = (train[col] == 0)
    both = cryo_sleep_missing & zero_spend
    pct = (both.sum() / cryo_sleep_missing.sum() * 100).round(2)
    print(f"  {col}: {pct}%")

print("\n" + "=" * 80)
print("TASK 3: MISSINGNESS CORRELATION WITH TARGET (Transported)")
print("=" * 80)

# Chi-square test for categorical variables
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

print("\nChi-square tests: Missingness vs Target")
for col in categorical_cols:
    if train[col].isnull().any():
        # Create contingency table
        contingency = pd.crosstab(train[col].isnull(), train['Transported'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n{col}:")
        print(f"  Chi2 statistic: {chi2:.4f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  => SIGNIFICANT: Missingness is correlated with target")
        else:
            print(f"  => Not significant (p >= 0.05)")

# Mann-Whitney U test for numerical variables
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("\n\nMann-Whitney U tests: Missingness vs Target")
for col in numeric_cols:
    if train[col].isnull().any():
        # Split target by missingness
        is_missing = train[col].isnull()
        transported_missing = train.loc[is_missing, 'Transported']
        transported_present = train.loc[~is_missing, 'Transported']
        
        stat, p_value = stats.mannwhitneyu(transported_missing, transported_present)
        print(f"\n{col}:")
        print(f"  U statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        # Also show proportions
        pct_missing = transported_missing.mean() * 100
        pct_present = transported_present.mean() * 100
        print(f"  Transported % (missing): {pct_missing:.2f}%")
        print(f"  Transported % (present): {pct_present:.2f}%")
        print(f"  Difference: {abs(pct_missing - pct_present):.2f}%")
        
        if p_value < 0.05:
            print(f"  => SIGNIFICANT: Missingness is correlated with target")
        else:
            print(f"  => Not significant (p >= 0.05)")

# Cabin analysis
print("\n\nCabin missingness vs Target:")
cabin_missing = train['Cabin'].isnull()
transported_cabin_missing = train.loc[cabin_missing, 'Transported'].mean() * 100
transported_cabin_present = train.loc[~cabin_missing, 'Transported'].mean() * 100
print(f"  Transported % (Cabin missing): {transported_cabin_missing:.2f}%")
print(f"  Transported % (Cabin present): {transported_cabin_present:.2f}%")
print(f"  Difference: {abs(transported_cabin_missing - transported_cabin_present):.2f}%")

# Name analysis
print("\n\nName missingness vs Target:")
name_missing = train['Name'].isnull()
transported_name_missing = train.loc[name_missing, 'Transported'].mean() * 100
transported_name_present = train.loc[~name_missing, 'Transported'].mean() * 100
print(f"  Transported % (Name missing): {transported_name_missing:.2f}%")
print(f"  Transported % (Name present): {transported_name_present:.2f}%")
print(f"  Difference: {abs(transported_name_missing - transported_name_present):.2f}%")

print("\n" + "=" * 80)
print("TASK 4: IMPUTATION STRATEGIES")
print("=" * 80)

strategies = {
    'HomePlanet': {
        'type': 'Categorical',
        'missing_rate': f"{train_missing['HomePlanet']:.2f}%",
        'hypothesis': 'Missingness may be random or related to entry procedures',
        'strategy': 'Mode imputation within groups (e.g., by Destination)',
        'verification': 'Compare feature distribution before/after imputation',
        'risk': 'May bias towards majority class'
    },
    
    'CryoSleep': {
        'type': 'Binary',
        'missing_rate': f"{train_missing['CryoSleep']:.2f}%",
        'hypothesis': 'CryoSleep passengers should have zero spending',
        'strategy': 'If all spending cols=0 → True; if any spending>0 → False; otherwise use mode',
        'verification': 'Check consistency: CryoSleep=True implies all spending=0',
        'risk': 'If the rule is wrong, systematic bias introduced'
    },
    
    'Cabin': {
        'type': 'String (Deck/Num/Side)',
        'missing_rate': f"{train_missing['Cabin']:.2f}%",
        'hypothesis': 'Cabin may be correlated with HomePlanet or VIP status',
        'strategy': 'Extract features from Cabin; impute based on HomePlanet/VIP patterns',
        'verification': 'Check Cabin distribution by HomePlanet and VIP',
        'risk': 'Cabin is complex; simple imputation may lose information'
    },
    
    'Destination': {
        'type': 'Categorical',
        'missing_rate': f"{train_missing['Destination']:.2f}%",
        'hypothesis': 'Missingness may be correlated with HomePlanet',
        'strategy': 'Mode imputation within HomePlanet groups',
        'verification': 'Cross-tabulate HomePlanet vs Destination',
        'risk': 'Over-simplifies destination preferences'
    },
    
    'Age': {
        'type': 'Numerical',
        'missing_rate': f"{train_missing['Age']:.2f}%",
        'hypothesis': 'Age may vary by HomePlanet and VIP status',
        'strategy': 'Median imputation within HomePlanet/VIP groups',
        'verification': 'Compare age distributions before/after imputation',
        'risk': 'May underestimate variance if imputed values are clustered'
    },
    
    'VIP': {
        'type': 'Binary',
        'missing_rate': f"{train_missing['VIP']:.2f}%",
        'hypothesis': 'VIP status correlates with spending and HomePlanet',
        'strategy': 'If total spending > threshold → True; else False (use 75th percentile)',
        'verification': 'Check VIP distribution vs total spending',
        'risk': 'Threshold choice is arbitrary; may misclassify'
    },
    
    'RoomService': {
        'type': 'Numerical (spending)',
        'missing_rate': f"{train_missing['RoomService']:.2f}%",
        'hypothesis': 'Missing spending likely means 0 (no service)',
        'strategy': 'Impute with 0',
        'verification': 'Check proportion of zeros in present values',
        'risk': 'If missing ≠ 0, introduces bias'
    },
    
    'FoodCourt': {
        'type': 'Numerical (spending)',
        'missing_rate': f"{train_missing['FoodCourt']:.2f}%",
        'hypothesis': 'Missing spending likely means 0',
        'strategy': 'Impute with 0',
        'verification': 'Check proportion of zeros in present values',
        'risk': 'If missing ≠ 0, introduces bias'
    },
    
    'ShoppingMall': {
        'type': 'Numerical (spending)',
        'missing_rate': f"{train_missing['ShoppingMall']:.2f}%",
        'hypothesis': 'Missing spending likely means 0',
        'strategy': 'Impute with 0',
        'verification': 'Check proportion of zeros in present values',
        'risk': 'If missing ≠ 0, introduces bias'
    },
    
    'Spa': {
        'type': 'Numerical (spending)',
        'missing_rate': f"{train_missing['Spa']:.2f}%",
        'hypothesis': 'Missing spending likely means 0',
        'strategy': 'Impute with 0',
        'verification': 'Check proportion of zeros in present values',
        'risk': 'If missing ≠ 0, introduces bias'
    },
    
    'VRDeck': {
        'type': 'Numerical (spending)',
        'missing_rate': f"{train_missing['VRDeck']:.2f}%",
        'hypothesis': 'Missing spending likely means 0',
        'strategy': 'Impute with 0',
        'verification': 'Check proportion of zeros in present values',
        'risk': 'If missing ≠ 0, introduces bias'
    },
    
    'Name': {
        'type': 'String',
        'missing_rate': f"{train_missing['Name']:.2f}%",
        'hypothesis': 'Name may help identify families but is not predictive',
        'strategy': 'Drop Name column (not useful for modeling)',
        'verification': 'Check if Name correlates with target',
        'risk': 'May lose information about groups/families'
    }
}

for col, info in strategies.items():
    print(f"\n{col}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("RISKS")
print("=" * 80)

risks = [
    "1. Target Leakage Risk: Using CryoSleep = all spending=0 rule implicitly uses spending patterns that may correlate with Transported",
    "2. Bias Risk: Imputing spending with 0 may systematically underestimate spending for certain passenger groups",
    "3. Distribution Distortion Risk: Mode/median imputation can create artificial clusters and reduce variance",
    "4. Cascade Risk: Errors in one imputation (e.g., CryoSleep) can propagate to other features",
    "5. Overfitting Risk: Complex imputation rules based on multiple features may overfit training data"
]

for risk in risks:
    print(risk)

print("\n" + "=" * 80)
print("VERIFICATION METHODS FOR STRATEGIES")
print("=" * 80)

verifications = {
    'CryoSleep rule': 'Validate: In training data, count how many CryoSleep=True have >0 spending. If >5%, rule needs revision.',
    'Spending=0 imputation': 'Validate: Check if zero spending correlates with other features (e.g., CryoSleep, VIP). Patterns suggest missing≠0.',
    'Cabin extraction': 'Validate: After extracting Deck/Num/Side, check if each sub-feature correlates with Target.',
    'VIP threshold': 'Validate: Plot VIP vs total spending histogram. Choose threshold at natural break point.',
    'Age group imputation': 'Validate: Compare age distributions by group; if similar, group-based imputation adds little value.',
    'Mode imputation': 'Validate: Check if feature distribution is balanced (no single mode >70%). If skewed, mode is biased.'
}

for method, verification in verifications.items():
    print(f"\n{method}:")
    print(f"  {verification}")
