import pandas as pd
import numpy as np
from scipy import stats

# Load data
train = pd.read_csv('data/raw/train.csv')

print("=" * 80)
print("CRITICAL VERIFICATION: MISSING SPENDING HYPOTHESIS")
print("=" * 80)

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("\nHypothesis: Missing spending = 0")
print("Test: Compare spending distribution for CryoSleep=False passengers")
print("       1. Rows with missing spending")
print("       2. Rows with zero spending (present)")

cryo_false = train[train['CryoSleep'] == False]

print(f"\nTotal CryoSleep=False passengers: {len(cryo_false)}")

for col in spending_cols:
    print(f"\n{'='*60}")
    print(f"{col}")
    print(f"{'='*60}")
    
    # Split into three groups
    missing = cryo_false[cryo_false[col].isnull()]
    zero = cryo_false[(cryo_false[col] == 0) & (cryo_false[col].notna())]
    non_zero = cryo_false[cryo_false[col] > 0]
    
    print(f"\nGroup sizes:")
    print(f"  Missing: {len(missing)} ({len(missing)/len(cryo_false)*100:.2f}%)")
    print(f"  Zero (present): {len(zero)} ({len(zero)/len(cryo_false)*100:.2f}%)")
    print(f"  Non-zero: {len(non_zero)} ({len(non_zero)/len(cryo_false)*100:.2f}%)")
    
    # Calculate total spending (excluding current column)
    other_cols = [c for c in spending_cols if c != col]
    missing['TotalOther'] = missing[other_cols].sum(axis=1, skipna=True)
    zero['TotalOther'] = zero[other_cols].sum(axis=1, skipna=True)
    non_zero['TotalOther'] = non_zero[other_cols].sum(axis=1, skipna=True)
    
    print(f"\nTotal spending (other columns):")
    print(f"  Missing: mean={missing['TotalOther'].mean():.2f}, median={missing['TotalOther'].median():.2f}")
    print(f"  Zero: mean={zero['TotalOther'].mean():.2f}, median={zero['TotalOther'].median():.2f}")
    print(f"  Non-zero: mean={non_zero['TotalOther'].mean():.2f}, median={non_zero['TotalOther'].median():.2f}")
    
    # Check if missing group is similar to zero group
    if len(missing) > 0 and len(zero) > 0:
        stat, p_value = stats.mannwhitneyu(missing['TotalOther'], zero['TotalOther'])
        print(f"\nMann-Whitney U test (Missing vs Zero):")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  => MISSING != ZERO (cannot impute with 0)")
        else:
            print(f"  => MISSING approx= ZERO (can impute with 0)")

print("\n" + "=" * 80)
print("ALTERNATIVE: GROUP-BASED IMPUTATION TEST")
print("=" * 80)

print("\nTest: If spending is missing, can we predict it from other features?")
print("Strategy: For CryoSleep=False passengers, use median spending by HomePlanet")

cryo_false = train[train['CryoSleep'] == False]

for col in spending_cols:
    print(f"\n{col}:")
    
    # Calculate median by HomePlanet
    present = cryo_false[cryo_false[col].notna()]
    medians_by_planet = present.groupby('HomePlanet')[col].median()
    print(f"\n  Median by HomePlanet:")
    for planet, median in medians_by_planet.items():
        print(f"    {planet}: {median:.2f}")
    
    # Check if missing rows have other spending
    missing = cryo_false[cryo_false[col].isnull()]
    other_cols = [c for c in spending_cols if c != col]
    missing['TotalOther'] = missing[other_cols].sum(axis=1, skipna=True)
    
    print(f"\n  Missing rows:")
    print(f"    Count: {len(missing)}")
    print(f"    Total other spending (mean): {missing['TotalOther'].mean():.2f}")
    print(f"    Total other spending (median): {missing['TotalOther'].median():.2f}")
    
    # Calculate error if using median by HomePlanet
    if len(missing) > 0 and 'HomePlanet' in missing.columns:
        print(f"\n  Imputation with HomePlanet median:")
        for planet in medians_by_planet.index:
            planet_missing = missing[missing['HomePlanet'] == planet]
            if len(planet_missing) > 0:
                # We can't know actual error, but we can see distribution
                print(f"    {planet}: {len(planet_missing)} rows, would impute {medians_by_planet[planet]:.2f}")

print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

print("\nBased on verification:")
print("\n1. If Missing approx= Zero (p >= 0.05) -> Impute with 0")
print("2. If Missing != Zero (p < 0.05) -> Use group-based imputation")
print("3. Consider using 'is_missing' flags for spending columns")
