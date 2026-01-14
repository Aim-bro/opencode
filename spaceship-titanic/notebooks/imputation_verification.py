import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('data/raw/train.csv')

print("=" * 80)
print("VERIFICATION 1: SPENDING DISTRIBUTIONS AND ZERO PATTERNS")
print("=" * 80)

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for col in spending_cols:
    print(f"\n{col}:")
    present = train[col].notna()
    print(f"  Total rows: {len(train)}")
    print(f"  Missing: {train[col].isnull().sum()} ({train[col].isnull().mean()*100:.2f}%)")
    print(f"  Present: {present.sum()} ({present.mean()*100:.2f}%)")
    
    # For present values, count zeros
    zero_count = (train[col] == 0).sum()
    print(f"  Zero values (in present): {zero_count} ({zero_count/present.sum()*100:.2f}% of present)")
    print(f"  Non-zero values: {(train[col] > 0).sum()} ({(train[col] > 0).sum()/present.sum()*100:.2f}%)")
    
    # Statistics for non-zero values
    non_zero = train[train[col] > 0][col]
    if len(non_zero) > 0:
        print(f"  Non-zero stats: min={non_zero.min():.0f}, median={non_zero.median():.0f}, max={non_zero.max():.0f}")

print("\n" + "=" * 80)
print("VERIFICATION 2: CRYOSLEEP VS SPENDING CONSISTENCY")
print("=" * 80)

print("\nChecking if CryoSleep=True passengers have zero spending:")
cryo_true = train[train['CryoSleep'] == True]
print(f"CryoSleep=True count: {len(cryo_true)}")

for col in spending_cols:
    non_zero_count = (cryo_true[col] > 0).sum()
    pct = non_zero_count / len(cryo_true) * 100
    print(f"  {col}: {non_zero_count} passengers with >0 spending ({pct:.2f}%)")

print("\nChecking if CryoSleep=False passengers have any spending:")
cryo_false = train[train['CryoSleep'] == False]
print(f"CryoSleep=False count: {len(cryo_false)}")

for col in spending_cols:
    zero_count = (cryo_false[col] == 0).sum()
    pct = zero_count / len(cryo_false) * 100
    print(f"  {col}: {zero_count} passengers with 0 spending ({pct:.2f}%)")

print("\n" + "=" * 80)
print("VERIFICATION 3: VIP VS TOTAL SPENDING")
print("=" * 80)

# Calculate total spending
train['TotalSpending'] = train[spending_cols].sum(axis=1)

print("\nTotal spending statistics by VIP status:")
vip_true = train[train['VIP'] == True]
vip_false = train[train['VIP'] == False]

print(f"\nVIP=True: {len(vip_true)} passengers")
print(f"  Total spending: mean={vip_true['TotalSpending'].mean():.0f}, median={vip_true['TotalSpending'].median():.0f}")
print(f"  Zero spending: {(vip_true['TotalSpending'] == 0).sum()} ({(vip_true['TotalSpending'] == 0).sum()/len(vip_true)*100:.2f}%)")

print(f"\nVIP=False: {len(vip_false)} passengers")
print(f"  Total spending: mean={vip_false['TotalSpending'].mean():.0f}, median={vip_false['TotalSpending'].median():.0f}")
print(f"  Zero spending: {(vip_false['TotalSpending'] == 0).sum()} ({(vip_false['TotalSpending'] == 0).sum()/len(vip_false)*100:.2f}%)")

# Find natural threshold
print("\nThreshold analysis for VIP imputation:")
all_spending = train['TotalSpending'].describe(percentiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9])
print(all_spending)

print("\nPercent of VIP=True by spending percentiles:")
percentiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
for p in percentiles:
    threshold = train['TotalSpending'].quantile(p)
    above_threshold = train[train['TotalSpending'] >= threshold]
    if len(above_threshold) > 0:
        vip_pct = (above_threshold['VIP'] == True).mean() * 100
        print(f"  Spending >= {threshold:.0f} ({p*100:.0f}th percentile): {vip_pct:.2f}% are VIP")

print("\n" + "=" * 80)
print("VERIFICATION 4: CABIN FEATURE EXTRACTION")
print("=" * 80)

# Extract Cabin features
train['Deck'] = train['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else None)
train['Num'] = train['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else None)
train['Side'] = train['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else None)

print("\nDeck distribution:")
print(train['Deck'].value_counts())

print("\nSide distribution:")
print(train['Side'].value_counts())

print("\nDeck vs HomePlanet cross-tab:")
deck_planet = pd.crosstab(train['Deck'], train['HomePlanet'], normalize='index') * 100
print(deck_planet.round(2))

print("\nDeck vs VIP cross-tab:")
deck_vip = pd.crosstab(train['Deck'], train['VIP'], normalize='index') * 100
print(deck_vip.round(2))

print("\n" + "=" * 80)
print("VERIFICATION 5: AGE DISTRIBUTIONS BY GROUP")
print("=" * 80)

print("\nAge statistics by HomePlanet:")
for planet in train['HomePlanet'].dropna().unique():
    subset = train[train['HomePlanet'] == planet]
    ages = subset['Age'].dropna()
    print(f"\n{planet}: {len(ages)} passengers")
    print(f"  Mean: {ages.mean():.2f}, Median: {ages.median():.2f}, Std: {ages.std():.2f}")

print("\nAge statistics by VIP:")
for vip_status in [True, False]:
    subset = train[train['VIP'] == vip_status]
    ages = subset['Age'].dropna()
    print(f"\nVIP={vip_status}: {len(ages)} passengers")
    print(f"  Mean: {ages.mean():.2f}, Median: {ages.median():.2f}, Std: {ages.std():.2f}")

print("\n" + "=" * 80)
print("VERIFICATION 6: MODE IMPUTATION CHECK")
print("=" * 80)

categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

print("\nMode distribution check:")
for col in categorical_cols:
    value_counts = train[col].value_counts(normalize=True) * 100
    mode_pct = value_counts.iloc[0]
    print(f"\n{col}:")
    print(f"  Mode: {value_counts.index[0]} ({mode_pct:.2f}%)")
    print(f"  Other categories: {', '.join([f'{idx} ({pct:.2f}%)' for idx, pct in value_counts.iloc[1:].items()])}")

print("\n" + "=" * 80)
print("VERIFICATION 7: HOMEPLANET VS DESTINATION CROSS-TAB")
print("=" * 80)

print("\nHomePlanet vs Destination cross-tab:")
planet_dest = pd.crosstab(train['HomePlanet'], train['Destination'], normalize='columns') * 100
print(planet_dest.round(2))

print("\nDestination distribution by HomePlanet (rows):")
planet_dest_row = pd.crosstab(train['HomePlanet'], train['Destination'], normalize='index') * 100
print(planet_dest_row.round(2))
