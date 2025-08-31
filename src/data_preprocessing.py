"""
Data Preprocessing Module for Soil Health Monitoring

This module contains functions for cleaning, preprocessing, and transforming
the soil health dataset for machine learning applications.

Author: Soil Health Monitoring Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SoilDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.target_columns = [
            'soil_health_index', 'crop_yield_potential', 
            'fertilizer_n_recommendation', 'fertilizer_p_recommendation',
            'fertilizer_k_recommendation', 'irrigation_requirement',
            'soil_degradation_risk'
        ]
        
    def load_data(self, filepath='data/soil_health_dataset.csv'):
        """Load the soil health dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData types:")
        print(df.dtypes.value_counts())
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found!")
        
        # Statistical summary
        print("\nNumerical features summary:")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numerical_cols].describe())
        
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        print(f"\nCategorical features: {len(categorical_cols)}")
        for col in categorical_cols:
            if col != 'date':  # Skip date column
                print(f"\n{col} value counts:")
                print(df[col].value_counts())
        
        return {
            'numerical_cols': numerical_cols.tolist(),
            'categorical_cols': categorical_cols.tolist(),
            'missing_values': missing_values.to_dict()
        }
    
    def detect_outliers(self, df, method='iqr'):
        """Detect outliers in numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                outliers = df[z_scores > 3]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outliers_info
    
    def handle_outliers(self, df, method='cap', outliers_info=None):
        """Handle outliers using specified method"""
        df_processed = df.copy()
        
        if outliers_info is None:
            outliers_info = self.detect_outliers(df)
        
        for col, info in outliers_info.items():
            if info['count'] > 0 and info['percentage'] < 10:  # Only handle if < 10% outliers
                if method == 'cap':
                    # Cap outliers to 1st and 99th percentiles
                    lower_cap = df[col].quantile(0.01)
                    upper_cap = df[col].quantile(0.99)
                    df_processed[col] = np.clip(df_processed[col], lower_cap, upper_cap)
                
                elif method == 'remove':
                    # Remove outlier rows (use carefully)
                    outlier_indices = info['indices']
                    df_processed = df_processed.drop(outlier_indices)
        
        return df_processed
    
    def encode_categorical_features(self, df, encoding_method='onehot'):
        """Encode categorical features"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'date']
        
        for col in categorical_cols:
            if encoding_method == 'onehot':
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_features = encoder.fit_transform(df[[col]])
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0][1:]]
                
                # Add encoded features to dataframe
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
                
                self.encoders[col] = encoder
                
            elif encoding_method == 'label':
                # Label encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
        
        return df_encoded
    
    def create_feature_interactions(self, df):
        """Create interaction features"""
        df_interactions = df.copy()
        
        # pH and nutrient interactions
        if all(col in df.columns for col in ['soil_ph', 'nitrogen_content']):
            df_interactions['ph_nitrogen_interaction'] = df['soil_ph'] * df['nitrogen_content']
        
        if all(col in df.columns for col in ['soil_ph', 'phosphorus_content']):
            df_interactions['ph_phosphorus_interaction'] = df['soil_ph'] * df['phosphorus_content']
        
        # Temperature and moisture interaction
        if all(col in df.columns for col in ['soil_temperature', 'soil_moisture']):
            df_interactions['temp_moisture_interaction'] = df['soil_temperature'] * df['soil_moisture']
        
        # NPK ratio features
        if all(col in df.columns for col in ['nitrogen_content', 'phosphorus_content', 'potassium_content']):
            total_npk = df['nitrogen_content'] + df['phosphorus_content'] + df['potassium_content']
            df_interactions['n_ratio'] = df['nitrogen_content'] / (total_npk + 1e-8)
            df_interactions['p_ratio'] = df['phosphorus_content'] / (total_npk + 1e-8)
            df_interactions['k_ratio'] = df['potassium_content'] / (total_npk + 1e-8)
        
        # Sand to clay ratio
        if all(col in df.columns for col in ['sand_percentage', 'clay_percentage']):
            df_interactions['sand_clay_ratio'] = df['sand_percentage'] / (df['clay_percentage'] + 1e-8)
        
        return df_interactions
    
    def create_temporal_features(self, df):
        """Create temporal features from date column"""
        if 'date' not in df.columns:
            return df
        
        df_temporal = df.copy()
        df_temporal['date'] = pd.to_datetime(df_temporal['date'])
        
        # Extract temporal features
        df_temporal['year'] = df_temporal['date'].dt.year
        df_temporal['day_of_year'] = df_temporal['date'].dt.dayofyear
        df_temporal['week_of_year'] = df_temporal['date'].dt.isocalendar().week
        
        # Cyclical encoding for seasonal patterns
        df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
        
        # Drop original date column
        df_temporal = df_temporal.drop('date', axis=1)
        
        return df_temporal
    
    def scale_features(self, X_train, X_test, method='standard'):
        """Scale numerical features"""
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit scaler on training data only
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        self.scalers['feature_scaler'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def prepare_target_variables(self, df):
        """Prepare target variables for different ML tasks"""
        targets = {}
        
        # Regression targets
        if 'soil_health_index' in df.columns:
            targets['soil_health_regression'] = df['soil_health_index']
            
            # Classification target (soil health categories)
            health_categories = pd.cut(
                df['soil_health_index'], 
                bins=[0, 40, 70, 100], 
                labels=['Poor', 'Fair', 'Good']
            )
            targets['soil_health_classification'] = health_categories
        
        if 'crop_yield_potential' in df.columns:
            targets['yield_regression'] = df['crop_yield_potential']
        
        # Multi-output regression (fertilizer recommendations)
        fertilizer_cols = [col for col in df.columns if 'fertilizer_' in col and 'recommendation' in col]
        if fertilizer_cols:
            targets['fertilizer_recommendations'] = df[fertilizer_cols]
        
        # Classification target (degradation risk)
        if 'soil_degradation_risk' in df.columns:
            targets['degradation_risk_classification'] = df['soil_degradation_risk']
        
        return targets
    
    def create_train_test_splits(self, df, test_size=0.2, random_state=42):
        """Create train-test splits for different tasks"""
        # Prepare features
        feature_cols = [col for col in df.columns if col not in self.target_columns]
        X = df[feature_cols]
        
        # Prepare targets
        targets = self.prepare_target_variables(df)
        
        splits = {}
        
        for target_name, y in targets.items():
            # Handle missing values in target if any
            valid_indices = y.dropna().index if hasattr(y, 'dropna') else y.index
            X_valid = X.loc[valid_indices]
            y_valid = y.loc[valid_indices] if hasattr(y, 'loc') else y
            
            # Stratify for classification tasks
            stratify = y_valid if 'classification' in target_name else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify
            )
            
            splits[target_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        return splits
    
    def preprocess_pipeline(self, df, include_interactions=True, include_temporal=True, 
                          outlier_method='cap', encoding_method='onehot', scaling_method='standard'):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # 1. Handle outliers
        print("Handling outliers...")
        outliers_info = self.detect_outliers(df)
        df_processed = self.handle_outliers(df, method=outlier_method, outliers_info=outliers_info)
        
        # 2. Create temporal features
        if include_temporal:
            print("Creating temporal features...")
            df_processed = self.create_temporal_features(df_processed)
        
        # 3. Encode categorical features
        print("Encoding categorical features...")
        df_processed = self.encode_categorical_features(df_processed, encoding_method=encoding_method)
        
        # 4. Create feature interactions
        if include_interactions:
            print("Creating feature interactions...")
            df_processed = self.create_feature_interactions(df_processed)
        
        # 5. Create train-test splits
        print("Creating train-test splits...")
        splits = self.create_train_test_splits(df_processed)
        
        # 6. Scale features for each split
        print("Scaling features...")
        for task_name, split_data in splits.items():
            X_train_scaled, X_test_scaled = self.scale_features(
                split_data['X_train'], 
                split_data['X_test'], 
                method=scaling_method
            )
            splits[task_name]['X_train_scaled'] = X_train_scaled
            splits[task_name]['X_test_scaled'] = X_test_scaled
        
        print("Preprocessing pipeline completed!")
        
        return df_processed, splits
    
    def save_preprocessed_data(self, splits, filepath_prefix='data/preprocessed_'):
        """Save preprocessed data for each task"""
        for task_name, split_data in splits.items():
            # Save training data
            train_data = pd.concat([
                split_data['X_train_scaled'], 
                split_data['y_train']
            ], axis=1)
            train_data.to_csv(f"{filepath_prefix}{task_name}_train.csv", index=False)
            
            # Save test data
            test_data = pd.concat([
                split_data['X_test_scaled'], 
                split_data['y_test']
            ], axis=1)
            test_data.to_csv(f"{filepath_prefix}{task_name}_test.csv", index=False)
        
        print(f"Preprocessed data saved with prefix: {filepath_prefix}")

def main():
    """Main function to demonstrate preprocessing"""
    # Initialize preprocessor
    preprocessor = SoilDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    if df is None:
        print("Please run data_generation.py first to generate the dataset!")
        return
    
    # Explore data
    data_info = preprocessor.explore_data(df)
    
    # Run preprocessing pipeline
    df_processed, splits = preprocessor.preprocess_pipeline(df)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(splits)
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original dataset shape: {df.shape}")
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Number of ML tasks prepared: {len(splits)}")
    print("Tasks prepared:")
    for task_name, split_data in splits.items():
        print(f"  - {task_name}: {split_data['X_train'].shape[0]} train, {split_data['X_test'].shape[0]} test samples")

if __name__ == "__main__":
    main()
