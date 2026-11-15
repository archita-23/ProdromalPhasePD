"""
CORRECTED PPMI Preprocessing with Actual Column Names
For your specific PPMI dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CORRECTED PPMI PREPROCESSING")
print("Using actual column names from your dataset")
print("="*60)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nSTEP 1: Loading PPMI Data")
print("-"*60)

df_part1 = pd.read_csv('MDS_UPDRS_Part_I.csv', low_memory=False)
df_part3 = pd.read_csv('MDS_UPDRS_Part_III.csv', low_memory=False)

print(f"✓ Part I loaded: {df_part1.shape}")
print(f"✓ Part III loaded: {df_part3.shape}")

# ============================================================================
# STEP 2: MERGE DATA
# ============================================================================

print("\nSTEP 2: Merging Part I & Part III")
print("-"*60)

df_merged = df_part1.merge(
    df_part3,
    on=['PATNO', 'EVENT_ID'],
    how='inner',
    suffixes=('_part1', '_part3')
)

print(f"✓ Merged successfully: {df_merged.shape[0]} records")
print(f"  Participants: {df_merged['PATNO'].nunique()}")
print(f"  Avg visits per participant: {len(df_merged) / df_merged['PATNO'].nunique():.1f}")

# ============================================================================
# STEP 3: EXTRACT FEATURES (CORRECTED COLUMN NAMES)
# ============================================================================

print("\nSTEP 3: Extracting Features")
print("-"*60)

df_features = df_merged[['PATNO', 'EVENT_ID']].copy()

# Feature 1: Sleep Duration/Quality
# NP1DDS = Daytime Sleepiness (closest to sleep quality)
# Scale: 0-4 where 0=normal, 4=severe
df_features['sleep_duration'] = df_merged['NP1DDS'].fillna(df_merged['NP1DDS'].median())
print("✓ sleep_duration: from NP1DDS (Daytime Sleepiness)")

# Feature 2: Sleep Quality (also daytime sleepiness)
# Using NP1DDS as proxy (best available in this dataset)
df_features['sleep_quality'] = df_merged['NP1DDS'].fillna(df_merged['NP1DDS'].median())
print("✓ sleep_quality: from NP1DDS (Daytime Sleepiness)")

# Feature 3: Mood (Average Depression + Anxiety)
# NP1DPRS = Depression, NP1ANXS = Anxiety
mood_cols = [df_merged['NP1DPRS'].fillna(df_merged['NP1DPRS'].median()),
             df_merged['NP1ANXS'].fillna(df_merged['NP1ANXS'].median())]
df_features['mood'] = pd.concat(mood_cols, axis=1).mean(axis=1)
print("✓ mood: Average of NP1DPRS (Depression) + NP1ANXS (Anxiety)")

# Feature 4: Fatigue/Apathy
# NP1APAT = Apathy (closest to fatigue/low energy)
df_features['fatigue'] = df_merged['NP1APAT'].fillna(df_merged['NP1APAT'].median())
print("✓ fatigue: from NP1APAT (Apathy)")

# Feature 5: Tremor (Maximum of all tremor types)
# Multiple tremor columns available: postural, kinetic, resting
tremor_cols = [
    'NP3PTRMR', 'NP3PTRML',      # Postural tremor
    'NP3KTRMR', 'NP3KTRML',      # Kinetic tremor
    'NP3RTARU', 'NP3RTALU',      # Resting tremor arms
    'NP3RTARL', 'NP3RTALL'       # Resting tremor legs
]

# Fill missing values with median, then take max
tremor_data = []
for col in tremor_cols:
    if col in df_merged.columns:
        tremor_data.append(df_merged[col].fillna(df_merged[col].median()))

if tremor_data:
    df_features['tremor'] = pd.concat(tremor_data, axis=1).max(axis=1)
    print(f"✓ tremor: Maximum of {len(tremor_cols)} tremor measurements")
else:
    df_features['tremor'] = 0
    print("⚠ tremor: No tremor columns found, using 0")

print(f"\nFeatures shape: {df_features.shape}")

# ============================================================================
# STEP 4: ORGANIZE LONGITUDINALLY
# ============================================================================

print("\nSTEP 4: Organizing Longitudinal Data")
print("-"*60)

df_features = df_features.sort_values(['PATNO', 'EVENT_ID']).reset_index(drop=True)
df_features['visit_num'] = df_features.groupby('PATNO').cumcount()

print(f"Participants: {df_features['PATNO'].nunique()}")
print(f"Total visits: {len(df_features)}")
print(f"Avg visits/participant: {len(df_features) / df_features['PATNO'].nunique():.1f}")

# ============================================================================
# STEP 5: NORMALIZE FEATURES
# ============================================================================

print("\nSTEP 5: Normalizing Features")
print("-"*60)

feature_cols = ['sleep_duration', 'sleep_quality', 'mood', 'fatigue', 'tremor']

print("Before normalization:")
print(df_features[feature_cols].describe().round(3))

scaler = StandardScaler()
df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

print("\nAfter normalization:")
print(df_features[feature_cols].describe().round(3))
print("✓ Normalized to Mean≈0, SD≈1")

# ============================================================================
# STEP 6: DATA QUALITY CHECKS
# ============================================================================

print("\nSTEP 6: Data Quality Validation")
print("-"*60)

# Check duplicates
dups = df_features.duplicated(subset=['PATNO', 'visit_num']).sum()
print(f"Duplicate visits: {dups}")

# Check variance
for col in feature_cols:
    var = df_features[col].var()
    status = "✓" if var > 0.1 else "⚠"
    print(f"  {status} {col}: variance = {var:.3f}")

# Check missing
missing = df_features[feature_cols].isna().sum()
if missing.sum() == 0:
    print("✓ No missing values")
else:
    print(f"⚠ Missing values: {missing[missing > 0].to_dict()}")

# ============================================================================
# STEP 7: TRAIN ISOLATION FOREST
# ============================================================================

print("\nSTEP 7: Training Isolation Forest")
print("-"*60)

X = df_features[feature_cols].values

iso_forest = IsolationForest(
    contamination=0.1,      # 10% anomaly rate
    random_state=42,
    n_estimators=100
)

iso_forest.fit(X)
predictions = iso_forest.predict(X)
scores = iso_forest.score_samples(X)

df_features['anomaly_score'] = scores
df_features['is_anomaly'] = predictions

print("✓ Model trained")
print(f"  Normal visits: {(predictions == 1).sum()}")
print(f"  Anomalous visits: {(predictions == -1).sum()}")

print("\nAnomaly Score Statistics:")
print(f"  Min: {scores.min():.4f}")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Max: {scores.max():.4f}")

# ============================================================================
# STEP 8: CATEGORIZE RISK
# ============================================================================

print("\nSTEP 8: Categorizing Risk Levels")
print("-"*60)

def categorize_risk(score):
    if score < -0.5:
        return "High Risk"
    elif score < -0.2:
        return "Moderate Risk"
    else:
        return "Normal"

df_features['risk_category'] = df_features['anomaly_score'].apply(categorize_risk)

risk_counts = df_features['risk_category'].value_counts()
print("Risk Distribution:")
for risk, count in risk_counts.items():
    pct = 100 * count / len(df_features)
    print(f"  {risk}: {count} ({pct:.1f}%)")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\nSTEP 9: Saving Results")
print("-"*60)

# Save processed data
output_cols = ['PATNO', 'EVENT_ID', 'visit_num'] + feature_cols + ['anomaly_score', 'is_anomaly', 'risk_category']
df_output = df_features[output_cols]
df_output.to_csv('ppmi_processed.csv', index=False)
print(f"✓ Saved ppmi_processed.csv ({len(df_output)} rows)")

# Save models
import pickle

with open('isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
print("✓ Saved isolation_forest_model.pkl")

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved feature_scaler.pkl")

# ============================================================================
# STEP 10: VISUALIZATION
# ============================================================================

print("\nSTEP 10: Creating Visualizations")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distributions (After Normalization)', fontsize=14)

for idx, col in enumerate(feature_cols):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df_features[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df_features[col].mean(), color='red', linestyle='--', label='Mean')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.legend()

axes[1, 2].remove()
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved feature_distributions.png")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*60)
print("✓ PREPROCESSING COMPLETE!")
print("="*60)

print(f"""
Summary:
  • Processed {df_output['PATNO'].nunique()} participants
  • {len(df_output)} total visits
  • Features: {', '.join(feature_cols)}
  • Anomaly detection: Isolation Forest
  
Output Files:
  ✓ ppmi_processed.csv (main data)
  ✓ isolation_forest_model.pkl (trained model)
  ✓ feature_scaler.pkl (normalizer)
  ✓ feature_distributions.png (visualization)

Next Steps:
  1. Verify data with: python verify_fixed.py
  2. Run Streamlit app: streamlit run app_simple.py
  3. Start monitoring patients!
""")