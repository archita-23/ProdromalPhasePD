"""
PPMI Data Preprocessing Pipeline for Prodromal Parkinson's Detection
Features: Sleep, Tremor, Mood, Sleep Quality, Fatigue
Model: Isolation Forest with Longitudinal Monitoring
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PPMIPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_mapping = {
            'sleep_duration': ['MDSUPDRS1_1'],  # Sleep problems
            'sleep_quality': ['MDSUPDRS1_2'],   # Daytime sleepiness
            'mood': ['MDSUPDRS1_5', 'MDSUPDRS1_6'],  # Mood/Depression/Anxiety
            'fatigue': ['MDSUPDRS1_7', 'MDSUPDRS1_8'],  # Fatigue/Apathy
            'tremor': ['MDSUPDRS3_20', 'MDSUPDRS3_21']  # Resting & postural tremor
        }
        self.feature_names = list(self.feature_mapping.keys())
    
    def load_ppmi_data(self, mds_updrs_part1_path, mds_updrs_part3_path):
        """
        Load PPMI CSV files
        Expected format: CSV with columns like PATNO (participant ID), EVENT_ID, MDSUPDRS1_1, etc.
        """
        try:
            df_part1 = pd.read_csv(mds_updrs_part1_path)
            df_part3 = pd.read_csv(mds_updrs_part3_path)
            print(f"✓ Loaded Part I: {df_part1.shape}")
            print(f"✓ Loaded Part III: {df_part3.shape}")
            return df_part1, df_part3
        except Exception as e:
            print(f"✗ Error loading files: {e}")
            return None, None
    
    def merge_data(self, df_part1, df_part3):
        """Merge Part I and Part III on PATNO and visit date"""
        # Merge on participant ID and event
        df_merged = df_part1.merge(
            df_part3,
            on=['PATNO', 'EVENT_ID'],
            how='inner',
            suffixes=('_part1', '_part3')
        )
        print(f"✓ Merged data: {df_merged.shape[0]} records")
        return df_merged
    
    def extract_features(self, df_merged):
        """
        Extract and aggregate features from MDS-UPDRS questionnaires
        MDS-UPDRS uses 0-4 scale for most items
        """
        features_df = df_merged[['PATNO', 'EVENT_ID']].copy()
        
        # Sleep Duration (MDSUPDRS1_1: Sleep problems, 0=None to 4=Severe)
        features_df['sleep_duration'] = df_merged['MDSUPDRS1_1'].fillna(0)
        
        # Sleep Quality (MDSUPDRS1_2: Daytime sleepiness)
        features_df['sleep_quality'] = df_merged['MDSUPDRS1_2'].fillna(0)
        
        # Mood (Average of depression + anxiety if both present)
        mood_cols = []
        if 'MDSUPDRS1_5' in df_merged.columns:
            mood_cols.append('MDSUPDRS1_5')  # Mood/depression
        if 'MDSUPDRS1_6' in df_merged.columns:
            mood_cols.append('MDSUPDRS1_6')  # Anxiety
        
        if mood_cols:
            features_df['mood'] = df_merged[mood_cols].fillna(0).mean(axis=1)
        else:
            features_df['mood'] = 0
        
        # Fatigue (Average of fatigue + apathy)
        fatigue_cols = []
        if 'MDSUPDRS1_7' in df_merged.columns:
            fatigue_cols.append('MDSUPDRS1_7')  # Fatigue
        if 'MDSUPDRS1_8' in df_merged.columns:
            fatigue_cols.append('MDSUPDRS1_8')  # Apathy
        
        if fatigue_cols:
            features_df['fatigue'] = df_merged[fatigue_cols].fillna(0).mean(axis=1)
        else:
            features_df['fatigue'] = 0
        
        # Tremor (Maximum of resting + postural tremor - indicates presence)
        tremor_cols = []
        if 'MDSUPDRS3_20' in df_merged.columns:
            tremor_cols.append('MDSUPDRS3_20')  # Resting tremor
        if 'MDSUPDRS3_21' in df_merged.columns:
            tremor_cols.append('MDSUPDRS3_21')  # Postural/kinetic tremor
        
        if tremor_cols:
            features_df['tremor'] = df_merged[tremor_cols].fillna(0).max(axis=1)
        else:
            features_df['tremor'] = 0
        
        print(f"✓ Extracted features: {features_df.shape}")
        print(f"  Features: {self.feature_names}")
        return features_df
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in feature columns
        Options: 'mean', 'median', 'knn'
        """
        feature_cols = self.feature_names
        
        # Count missing before
        missing_before = df[feature_cols].isna().sum()
        print(f"\n📊 Missing values before imputation:")
        print(missing_before[missing_before > 0])
        
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        
        print(f"✓ Imputation complete ({strategy})")
        return df
    
    def normalize_features(self, df, method='standard', scale_range=(0, 1)):
        """
        Normalize features for Isolation Forest
        'standard': Mean=0, SD=1 (recommended for IF)
        'robust': Resistant to outliers
        'minmax': Scale to [0, 1]
        """
        feature_cols = self.feature_names
        
        if method == 'standard':
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scaler = scaler
            print(f"✓ Applied StandardScaler")
        
        elif method == 'robust':
            scaler = RobustScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scaler = scaler
            print(f"✓ Applied RobustScaler")
        
        elif method == 'minmax':
            df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / \
                              (df[feature_cols].max() - df[feature_cols].min())
            print(f"✓ Applied Min-Max scaling to [{scale_range[0]}, {scale_range[1]}]")
        
        # Show summary statistics
        print("\nNormalized Feature Statistics:")
        print(df[feature_cols].describe().round(3))
        return df
    
    def organize_longitudinal(self, df):
        """
        Organize data by participant and visit chronologically
        Creates time-series sequences for each participant
        """
        # Convert EVENT_ID to numeric if needed (e.g., 'BL' = 0, 'V01' = 1, etc.)
        # Adjust mapping based on your PPMI data
        
        df = df.sort_values(['PATNO', 'EVENT_ID']).reset_index(drop=True)
        
        # Create visit sequence number per participant
        df['visit_number'] = df.groupby('PATNO').cumcount()
        
        print(f"\n📈 Longitudinal Organization:")
        print(f"Total participants: {df['PATNO'].nunique()}")
        print(f"Total visits: {len(df)}")
        print(f"Avg visits per participant: {len(df) / df['PATNO'].nunique():.1f}")
        
        return df
    
    def detect_outliers_iqr(self, df, multiplier=1.5):
        """
        Detect outliers using IQR method before Isolation Forest
        Helps identify potentially erroneous entries
        """
        feature_cols = self.feature_names
        outliers = pd.DataFrame()
        
        for col in feature_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            col_outliers = df[(df[col] < lower) | (df[col] > upper)]
            if len(col_outliers) > 0:
                outliers = pd.concat([outliers, col_outliers])
        
        print(f"\n⚠️  IQR Outliers detected: {len(outliers.drop_duplicates())} records")
        if len(outliers) > 0:
            print(outliers[['PATNO', 'visit_number'] + feature_cols].head(10))
        
        return outliers.drop_duplicates()
    
    def validate_data_quality(self, df):
        """
        Comprehensive data quality checks
        """
        print("\n🔍 Data Quality Validation:")
        print(f"Shape: {df.shape}")
        print(f"Duplicates: {df.duplicated(subset=['PATNO', 'visit_number']).sum()}")
        print(f"Zero variance features: {(df[self.feature_names].std() == 0).sum()}")
        
        # Check for extreme values (should be 0-4 after normalization)
        for col in self.feature_names:
            if (df[col] < -5).any() or (df[col] > 5).any():
                print(f"  ⚠️  {col}: Contains extreme values")
        
        print("✓ Validation complete")
        return True
    
    def save_processed_data(self, df, output_path='ppmi_processed.csv'):
        """Save preprocessed data"""
        df.to_csv(output_path, index=False)
        print(f"\n✓ Data saved to {output_path}")
        return df
    
    def visualize_distributions(self, df):
        """Visualize feature distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        feature_cols = self.feature_names
        
        for idx, col in enumerate(feature_cols):
            row = idx // 3
            col_pos = idx % 3
            ax = axes[row, col_pos]
            
            ax.hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Frequency')
        
        axes[1, 2].remove()  # Remove empty subplot
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to feature_distributions.png")
        plt.show()
    
    def create_training_dataset(self, df):
        """
        Prepare final dataset for Isolation Forest
        Create windows of consecutive visits if needed
        """
        X = df[self.feature_names].values
        metadata = df[['PATNO', 'visit_number']].values
        
        print(f"\n📦 Training Dataset Prepared:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {self.feature_names}")
        
        return X, metadata, df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = PPMIPreprocessor()
    
    # Load data
    df_part1, df_part3 = preprocessor.load_ppmi_data(
        'MDS_UPDRS_Part_I.csv',
        'MDS_UPDRS_Part_III.csv'
    )
    
    if df_part1 is not None and df_part3 is not None:
        # Merge datasets
        df_merged = preprocessor.merge_data(df_part1, df_part3)
        
        # Extract features
        df_features = preprocessor.extract_features(df_merged)
        
        # Handle missing values
        df_features = preprocessor.handle_missing_values(df_features, strategy='mean')
        
        # Organize longitudinally
        df_features = preprocessor.organize_longitudinal(df_features)
        
        # Normalize
        df_features = preprocessor.normalize_features(df_features, method='standard')
        
        # Quality checks
        outliers = preprocessor.detect_outliers_iqr(df_features)
        preprocessor.validate_data_quality(df_features)
        
        # Visualization
        preprocessor.visualize_distributions(df_features)
        
        # Save and prepare for training
        df_features = preprocessor.save_processed_data(df_features)
        X, metadata, df_final = preprocessor.create_training_dataset(df_features)
        
        print("\n" + "="*50)
        print("✓ PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Ready for Isolation Forest training!")
        print(f"Input shape: {X.shape}")
