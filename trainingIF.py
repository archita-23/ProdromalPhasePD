"""
Prodromal Phase Detection using Isolation Forest
================================================

This module implements a complete machine learning pipeline for detecting
anomalies (prodromal phases) using an Isolation Forest model.

Pipeline:
1. Load and explore dataset
2. Preprocess data (handle missing values, ensure correct types)
3. Feature engineering and normalization
4. Train Isolation Forest model
5. Generate predictions and save results
6. Persist model and scaler for future use
7. Provide inference function for new data

Author: ML Pipeline Generator
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os
from pathlib import Path


class ProdromalDetectionPipeline:
    """
    Complete pipeline for training and using an Isolation Forest model
    for prodromal phase detection.
    """
    
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        """
        Initialize the pipeline with model parameters.
        
        Parameters:
        -----------
        contamination : float, default=0.1
            The proportion of anomalies in the dataset (0 to 0.5).
            Represents expected contamination level for prodromal indicators.
        
        n_estimators : int, default=100
            Number of base estimators in the Isolation Forest.
            Higher values improve model stability.
        
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Initialize placeholders for model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.dataset = None
        
    def load_dataset(self, filepath):
        """
        Load the CSV dataset and perform initial validation.
        
        Parameters:
        -----------
        filepath : str
            Path to the input CSV file.
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset.
        
        Raises:
        -------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        self.dataset = df.copy()
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        return df
    
    def preprocess_data(self, df, feature_columns):
        """
        Preprocess the dataset:
        - Handle missing values
        - Ensure correct data types
        - Select relevant features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset.
        
        feature_columns : list
            List of column names to use as features.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset with selected features.
        """
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Step 1: Select relevant features
        print(f"\nSelecting features: {feature_columns}")
        df_processed = df[feature_columns].copy()
        
        # Step 2: Handle missing values
        print("\nChecking for missing values...")
        missing_count = df_processed.isnull().sum()
        
        if missing_count.sum() > 0:
            print(f"Found missing values:\n{missing_count[missing_count > 0]}")
            print("Handling missing values with forward fill, then backward fill...")
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
            
            # If still missing (e.g., entire column is NaN), use mean imputation
            df_processed = df_processed.fillna(df_processed.mean())
            print(f"Missing values after imputation:\n{df_processed.isnull().sum()}")
        else:
            print("No missing values found.")
        
        # Step 3: Ensure correct data types (convert to numeric)
        print("\nEnsuring numeric data types...")
        for col in feature_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle any conversion errors with mean imputation
        if df_processed.isnull().sum().sum() > 0:
            print("Coercing non-numeric values to mean...")
            df_processed = df_processed.fillna(df_processed.mean())
        
        print(f"\nFinal data types:\n{df_processed.dtypes}")
        print(f"Processed dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def normalize_features(self, df_processed):
        """
        Normalize features using StandardScaler (zero mean, unit variance).
        
        Parameters:
        -----------
        df_processed : pd.DataFrame
            Preprocessed dataset with selected features.
        
        Returns:
        --------
        np.ndarray
            Normalized feature matrix.
        """
        print("\n" + "="*60)
        print("NORMALIZING FEATURES")
        print("="*60)
        
        print("\nApplying StandardScaler to normalize features...")
        self.scaler = StandardScaler()
        features_normalized = self.scaler.fit_transform(df_processed)
        
        # Display normalization statistics
        print(f"\nNormalization Statistics:")
        for i, col in enumerate(df_processed.columns):
            print(f"  {col}:")
            print(f"    Original Mean: {df_processed[col].mean():.4f}, "
                  f"Std: {df_processed[col].std():.4f}")
            print(f"    Normalized Mean: {features_normalized[:, i].mean():.6f}, "
                  f"Std: {features_normalized[:, i].std():.6f}")
        
        self.feature_names = df_processed.columns.tolist()
        
        return features_normalized
    
    def train_model(self, features_normalized):
        """
        Train the Isolation Forest model.
        
        Parameters:
        -----------
        features_normalized : np.ndarray
            Normalized feature matrix.
        
        Returns:
        --------
        IsolationForest
            Trained model.
        """
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST MODEL")
        print("="*60)
        
        print(f"\nModel Configuration:")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  contamination: {self.contamination}")
        print(f"  random_state: {self.random_state}")
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1  # Use all available CPU cores
        )
        
        print("\nFitting model to data...")
        self.model.fit(features_normalized)
        
        print("Model training complete!")
        
        return self.model
    
    def predict_anomalies(self, features_normalized):
        """
        Predict anomalies using the trained model.
        
        Parameters:
        -----------
        features_normalized : np.ndarray
            Normalized feature matrix.
        
        Returns:
        --------
        np.ndarray
            Array of predictions (-1 for anomaly, 1 for normal).
            Converted to (1 for anomaly, 0 for normal) in output.
        """
        print("\n" + "="*60)
        print("PREDICTING ANOMALIES")
        print("="*60)
        
        print("\nGenerating predictions on the dataset...")
        predictions = self.model.predict(features_normalized)
        
        # Convert to binary format: 1 = anomaly, 0 = normal
        # (Isolation Forest uses -1 for anomalies, 1 for normal)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies
    
    def generate_output(self, anomalies, output_filepath):
        """
        Add anomaly column to original dataset and save as CSV.
        
        Parameters:
        -----------
        anomalies : np.ndarray
            Array of anomaly predictions (0 or 1).
        
        output_filepath : str
            Path to save the output CSV file.
        
        Returns:
        --------
        pd.DataFrame
            Dataset with anomaly predictions.
        """
        print("\n" + "="*60)
        print("GENERATING OUTPUT")
        print("="*60)
        
        # Add anomaly column to original dataset
        result_df = self.dataset.copy()
        result_df['anomaly'] = anomalies
        
        # Print summary statistics
        anomaly_count = anomalies.sum()
        normal_count = len(anomalies) - anomaly_count
        
        print(f"\nAnomalies detected: {anomaly_count} ({100*anomaly_count/len(anomalies):.2f}%)")
        print(f"Normal samples: {normal_count} ({100*normal_count/len(anomalies):.2f}%)")
        
        print(f"\nFirst 10 rows with predictions:\n{result_df.head(10)}")
        
        # Save to CSV
        print(f"\nSaving results to {output_filepath}...")
        result_df.to_csv(output_filepath, index=False)
        print(f"Results saved successfully!")
        
        return result_df
    
    def save_model_and_scaler(self, model_filepath, scaler_filepath):
        """
        Save the trained model and scaler using joblib for later use.
        
        Parameters:
        -----------
        model_filepath : str
            Path to save the model.
        
        scaler_filepath : str
            Path to save the scaler.
        """
        print("\n" + "="*60)
        print("SAVING MODEL AND SCALER")
        print("="*60)
        
        print(f"\nSaving model to {model_filepath}...")
        joblib.dump(self.model, model_filepath)
        print(f"Model saved!")
        
        print(f"Saving scaler to {scaler_filepath}...")
        joblib.dump(self.scaler, scaler_filepath)
        print(f"Scaler saved!")
        
        print(f"\nModel and scaler saved and ready for inference!")
    
    def predict_new_data(self, new_data, model_filepath, scaler_filepath):
        """
        Load saved model and scaler, then predict on new data.
        
        This is a standalone inference function that can be used after training
        to make predictions on new, unseen data.
        
        Parameters:
        -----------
        new_data : pd.DataFrame or np.ndarray
            New data for prediction. Can be:
            - DataFrame with same feature columns as training data
            - 2D numpy array with shape (n_samples, n_features)
        
        model_filepath : str
            Path to the saved model file.
        
        scaler_filepath : str
            Path to the saved scaler file.
        
        Returns:
        --------
        np.ndarray
            Predictions (0 for normal, 1 for anomaly).
        """
        print("\n" + "="*60)
        print("INFERENCE ON NEW DATA")
        print("="*60)
        
        # Load model and scaler
        print(f"\nLoading model from {model_filepath}...")
        loaded_model = joblib.load(model_filepath)
        
        print(f"Loading scaler from {scaler_filepath}...")
        loaded_scaler = joblib.load(scaler_filepath)
        
        # Convert DataFrame to numpy array if needed
        if isinstance(new_data, pd.DataFrame):
            print(f"\nInput data shape: {new_data.shape}")
            print(f"Features: {list(new_data.columns)}")
            new_data_array = new_data.values
        else:
            print(f"\nInput data shape: {new_data.shape}")
            new_data_array = new_data
        
        # Normalize the new data
        print("Normalizing new data using saved scaler...")
        new_data_normalized = loaded_scaler.transform(new_data_array)
        
        # Make predictions
        print("Making predictions...")
        predictions_raw = loaded_model.predict(new_data_normalized)
        predictions = (predictions_raw == -1).astype(int)
        
        anomaly_count = predictions.sum()
        normal_count = len(predictions) - anomaly_count
        
        print(f"\nResults:")
        print(f"  Anomalies detected: {anomaly_count} ({100*anomaly_count/len(predictions):.2f}%)")
        print(f"  Normal samples: {normal_count} ({100*normal_count/len(predictions):.2f}%)")
        
        return predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function: complete pipeline from training to persistence.
    """
    print("\n" + "="*60)
    print("PRODROMAL PHASE DETECTION - ISOLATION FOREST PIPELINE")
    print("="*60)
    
    # Configuration
    INPUT_CSV = "prodromal_data.csv"  # Change this to your CSV file path
    OUTPUT_CSV = "prodromal_data_with_predictions.csv"
    MODEL_FILE = "isolation_forest_model.pkl"
    SCALER_FILE = "feature_scaler.pkl"
    
    FEATURE_COLUMNS = ["sleep_hours", "tremor", "fatigue_level", "mood_score"]
    
    # Model hyperparameters
    CONTAMINATION = 0.10  # Expected proportion of anomalies (10%)
    N_ESTIMATORS = 100     # Number of trees in the forest
    RANDOM_STATE = 42      # For reproducibility
    
    try:
        # Step 1: Initialize pipeline
        pipeline = ProdromalDetectionPipeline(
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )
        
        # Step 2: Load dataset
        df = pipeline.load_dataset(INPUT_CSV)
        
        # Step 3: Preprocess data
        df_processed = pipeline.preprocess_data(df, FEATURE_COLUMNS)
        
        # Step 4: Normalize features
        features_normalized = pipeline.normalize_features(df_processed)
        
        # Step 5: Train model
        pipeline.train_model(features_normalized)
        
        # Step 6: Generate predictions
        anomalies = pipeline.predict_anomalies(features_normalized)
        
        # Step 7: Save results
        result_df = pipeline.generate_output(anomalies, OUTPUT_CSV)
        
        # Step 8: Save model and scaler
        pipeline.save_model_and_scaler(MODEL_FILE, SCALER_FILE)
        
        # ====== BONUS: Inference on new data ======
        print("\n" + "="*60)
        print("BONUS: DEMONSTRATING INFERENCE ON SAMPLE DATA")
        print("="*60)
        
        # Create sample new data for demonstration
        sample_new_data = pd.DataFrame({
            'sleep_hours': [7.5, 4.2, 8.1], 
            'tremor': [0.5, 2.1, 0.3],                     
            'fatigue_level': [2, 5, 1],
            'mood_score': [3, 1, 4]
        })
        
        print("\nSample new data for inference:")
        print(sample_new_data)
        
        # Use the inference function
        new_predictions = pipeline.predict_new_data(
            sample_new_data,
            MODEL_FILE,
            SCALER_FILE
        )
        
        print(f"\nPredictions for new data: {new_predictions}")
        print("(0 = Normal, 1 = Anomaly)")
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nOutput files created:")
        print(f"  - {OUTPUT_CSV} (predictions)")
        print(f"  - {MODEL_FILE} (trained model)")
        print(f"  - {SCALER_FILE} (feature scaler)")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nPlease ensure your CSV file '{INPUT_CSV}' exists in the working directory.")
        print(f"Expected columns: {', '.join(FEATURE_COLUMNS)}")
    
    except Exception as e:
        print(f"\n Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# STANDALONE INFERENCE FUNCTION (Can be imported and used separately)
# ============================================================================

def inference_on_new_data(new_data_df, model_file, scaler_file):
    """
    Standalone function for running inference on new data using saved model.
    
    This function can be imported and used in other scripts without
    running the full training pipeline.
    
    Parameters:
    -----------
    new_data_df : pd.DataFrame
        DataFrame with same features as training data.
        Expected columns: tremor, mood, sleep_quality, sleep_hours, fatigue
    
    model_file : str
        Path to saved model file.
    
    scaler_file : str
        Path to saved scaler file.
    
    Returns:
    --------
    pd.DataFrame
        Input data with added 'anomaly' column (0 = normal, 1 = anomaly).
    
    Example:
    --------
    >>> new_data = pd.DataFrame({
    ...     'tremor': [0.5, 2.1],
    ...     'mood': [3, 1],
    ...     'sleep_quality': [4, 2],
    ...     'sleep_hours': [7.5, 4.2],
    ...     'fatigue': [2, 5]
    ... })
    >>> result = inference_on_new_data(new_data, 'isolation_forest_model.pkl', 'feature_scaler.pkl')
    >>> print(result)
    """
    print("Loading saved model and scaler...")
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    print("Normalizing new data...")
    data_normalized = scaler.transform(new_data_df.values)
    
    print("Generating predictions...")
    predictions_raw = model.predict(data_normalized)
    predictions = (predictions_raw == -1).astype(int)
    
    result_df = new_data_df.copy()
    result_df['anomaly'] = predictions
    
    return result_df


if __name__ == "__main__":
    main()