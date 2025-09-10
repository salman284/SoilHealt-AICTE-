"""
Artificial Neural Network (ANN) Model for Soil Health Prediction

This module implements neural network models for soil health classification and regression
using scikit-learn's MLPClassifier and MLPRegressor.

Author: Soil Health Monitoring Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class SoilHealthANN:
    def __init__(self, task_type='classification'):
        """
        Initialize ANN model for soil health prediction
        
        Args:
            task_type (str): 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_scores = []
        self.input_shape = None
        
    def create_model(self, input_shape, num_classes=None, hidden_layers=(128, 64, 32), 
                    max_iter=500, learning_rate_init=0.001):
        """
        Create neural network architecture
        
        Args:
            input_shape (int): Number of input features
            num_classes (int): Number of classes for classification
            hidden_layers (tuple): Hidden layer sizes
            max_iter (int): Maximum number of iterations
            learning_rate_init (float): Initial learning rate
        """
        self.input_shape = input_shape
        
        if self.task_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.0001,  # L2 regularization
                activation='relu',
                solver='adam'
            )
        else:  # regression
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.0001,  # L2 regularization
                activation='relu',
                solver='adam'
            )
    
    def prepare_data(self, X, y):
        """
        Prepare data for training
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            
        Returns:
            tuple: Prepared X and y
        """
        # Handle categorical features
        X_processed = X.copy()
        
        # Encode categorical columns
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Prepare target variable
        if self.task_type == 'classification':
            if y.dtype == 'object':
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y
        else:
            y_processed = y
            
        return X_scaled, y_processed
    
    def train(self, X, y, test_size=0.2, validation_split=0.1):
        """
        Train the neural network model
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (float): Test set size
            validation_split (float): Validation set size
            
        Returns:
            dict: Training results
        """
        print(f"Training ANN model for {self.task_type}...")
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed if self.task_type == 'classification' else None
        )
        
        # Create model if not exists
        if self.model is None:
            num_classes = len(np.unique(y_processed)) if self.task_type == 'classification' else None
            self.create_model(X_processed.shape[1], num_classes)
        
        # Train model
        print("Training in progress...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")
            
            results = {
                'test_accuracy': accuracy,
                'test_predictions': y_pred,
                'test_true': y_test,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test R²: {r2:.4f}")
            
            results = {
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_r2': r2,
                'test_predictions': y_pred,
                'test_true': y_test
            }
        
        return results
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (pd.DataFrame): Feature data
            
        Returns:
            np.array: Predictions
        """
        # Handle categorical features
        X_processed = X.copy()
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities for classification
        
        Args:
            X (pd.DataFrame): Feature data
            
        Returns:
            np.array: Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        # Handle categorical features
        X_processed = X.copy()
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y, predictions)
            report = classification_report(y, predictions, output_dict=True)
            cm = confusion_matrix(y, predictions)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
        else:
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
    
    def plot_training_history(self):
        """
        Plot training history (simplified for sklearn)
        """
        if hasattr(self.model, 'loss_curve_'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.loss_curve_, label='Training Loss')
            plt.title('Training Loss Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Training history not available")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Plot confusion matrix for classification
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            class_names (list): Class names
        """
        if self.task_type != 'classification':
            print("Confusion matrix is only available for classification tasks")
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'task_type': self.task_type,
            'input_shape': self.input_shape
        }
        joblib.dump(model_data, f"{filepath}_model.pkl")
        print(f"Model saved to {filepath}_model.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the model file
        """
        model_data = joblib.load(f"{filepath}_model.pkl")
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.task_type = model_data['task_type']
        self.input_shape = model_data['input_shape']
        print(f"Model loaded from {filepath}_model.pkl")
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            param_grid (dict): Parameter grid for grid search
            cv (int): Cross-validation folds
            
        Returns:
            dict: Best parameters and score
        """
        if param_grid is None:
            if self.task_type == 'classification':
                param_grid = {
                    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            else:
                param_grid = {
                    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                }
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Create base model
        if self.task_type == 'classification':
            base_model = MLPClassifier(random_state=42, max_iter=300)
        else:
            base_model = MLPRegressor(random_state=42, max_iter=300)
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='accuracy' if self.task_type == 'classification' else 'r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_processed, y_processed)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

def main():
    """
    Example usage of the SoilHealthANN class
    """
    # Load data
    try:
        df = pd.read_csv('data/soil_health_dataset.csv')
    except FileNotFoundError:
        print("Please generate the dataset first by running data generation.")
        return
    
    # Prepare features and targets
    feature_columns = [
        'latitude', 'longitude', 'elevation', 'month',
        'soil_moisture', 'soil_temperature', 'soil_ph', 'bulk_density',
        'sand_percentage', 'silt_percentage', 'clay_percentage',
        'nitrogen_content', 'phosphorus_content', 'potassium_content',
        'organic_matter', 'electrical_conductivity',
        'air_temperature', 'relative_humidity', 'rainfall', 'solar_radiation',
        'days_since_fertilization', 'fertilizer_type', 'irrigation_frequency',
        'crop_rotation'
    ]
    
    X = df[feature_columns]
    
    # Classification example - soil degradation risk
    print("=== Classification Task: Soil Degradation Risk ===")
    y_class = df['soil_degradation_risk']
    
    ann_classifier = SoilHealthANN(task_type='classification')
    class_results = ann_classifier.train(X, y_class)
    
    print(f"Classification Accuracy: {class_results['test_accuracy']:.4f}")
    
    # Regression example - soil health index
    print("\n=== Regression Task: Soil Health Index ===")
    y_reg = df['soil_health_index']
    
    ann_regressor = SoilHealthANN(task_type='regression')
    reg_results = ann_regressor.train(X, y_reg)
    
    print(f"Regression R²: {reg_results['test_r2']:.4f}")
    print(f"Regression RMSE: {reg_results['test_rmse']:.4f}")

if __name__ == "__main__":
    main()
