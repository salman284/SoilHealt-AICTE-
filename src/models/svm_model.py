"""
Support Vector Machine (SVM) Model for Soil Health Classification and Regression

This module implements SVM models for various soil health prediction tasks
including classification and regression problems.

Author: Soil Health Monitoring Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class SoilHealthSVM:
    def __init__(self, task_type='classification'):
        """
        Initialize SVM model for soil health prediction
        
        Args:
            task_type (str): 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
    def create_model(self, kernel='rbf', **kwargs):
        """Create SVM model based on task type"""
        if self.task_type == 'classification':
            self.model = SVC(
                kernel=kernel,
                random_state=42,
                probability=True,  # Enable probability estimates
                **kwargs
            )
        elif self.task_type == 'regression':
            self.model = SVR(
                kernel=kernel,
                **kwargs
            )
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        print(f"Starting hyperparameter tuning for {self.task_type} SVM...")
        
        # Define parameter grids for different kernels and tasks
        if self.task_type == 'classification':
            param_grid = [
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                },
                {
                    'kernel': ['poly'],
                    'C': [0.1, 1, 10],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto']
                },
                {
                    'kernel': ['linear'],
                    'C': [0.1, 1, 10, 100]
                }
            ]
            scoring = 'accuracy'
            
        else:  # regression
            param_grid = [
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                },
                {
                    'kernel': ['poly'],
                    'C': [0.1, 1, 10],
                    'degree': [2, 3, 4],
                    'epsilon': [0.1, 0.2]
                },
                {
                    'kernel': ['linear'],
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            ]
            scoring = 'neg_mean_squared_error'
        
        # Create base model
        if self.task_type == 'classification':
            base_model = SVC(random_state=42, probability=True)
        else:
            base_model = SVR()
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.best_score_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {self.cv_scores:.4f}")
        
        return self.model
    
    def train(self, X_train, y_train, tune_hyperparameters=True):
        """Train the SVM model"""
        if tune_hyperparameters:
            self.hyperparameter_tuning(X_train, y_train)
        else:
            if self.model is None:
                self.create_model()
            print(f"Training {self.task_type} SVM model...")
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X_test)
        
        # Get prediction probabilities for classification
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test)
            return predictions, probabilities
        
        return predictions
    
    def evaluate_classification(self, X_test, y_test, class_labels=None):
        """Evaluate classification model performance"""
        if self.task_type != 'classification':
            raise ValueError("This method is only for classification tasks")
        
        predictions, probabilities = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print("Classification Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix - SVM Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }
    
    def evaluate_regression(self, X_test, y_test):
        """Evaluate regression model performance"""
        if self.task_type != 'regression':
            raise ValueError("This method is only for regression tasks")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("Regression Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Prediction vs Actual plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual - SVM Regression')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - SVM Regression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }
    
    def feature_importance_analysis(self, X_train, feature_names=None):
        """Analyze feature importance for linear SVM"""
        if self.model is None:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        if self.model.kernel != 'linear':
            print("Feature importance analysis is only available for linear SVM models")
            return None
        
        if self.task_type == 'classification':
            # For classification, use the coefficients
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_[0])
            else:
                print("No coefficients available for this model")
                return None
        else:
            # For regression, use the coefficients
            if hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
            else:
                print("No coefficients available for this model")
                return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)  # Show top 20 features
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Feature Importance - Linear SVM')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def cross_validation_analysis(self, X, y, cv_folds=5):
        """Perform cross-validation analysis"""
        if self.model is None:
            raise ValueError("Model must be created before cross-validation")
        
        if self.task_type == 'classification':
            scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        else:
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring=metric)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'task_type': self.task_type,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.task_type = model_data['task_type']
        self.best_params = model_data.get('best_params')
        self.cv_scores = model_data.get('cv_scores')
        print(f"Model loaded from {filepath}")

def run_soil_health_classification_example():
    """Example function for soil health classification"""
    print("="*60)
    print("SOIL HEALTH CLASSIFICATION WITH SVM")
    print("="*60)
    
    # Load preprocessed data (assuming it exists)
    try:
        train_data = pd.read_csv('data/preprocessed_soil_health_classification_train.csv')
        test_data = pd.read_csv('data/preprocessed_soil_health_classification_test.csv')
        
        # Separate features and target
        feature_cols = [col for col in train_data.columns if col not in ['Poor', 'Fair', 'Good']]
        
        # Assuming the target is encoded as the class name
        target_col = [col for col in train_data.columns if col in ['Poor', 'Fair', 'Good']][0]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Initialize and train SVM classifier
        svm_classifier = SoilHealthSVM(task_type='classification')
        svm_classifier.train(X_train, y_train, tune_hyperparameters=True)
        
        # Evaluate model
        class_labels = ['Poor', 'Fair', 'Good']
        results = svm_classifier.evaluate_classification(X_test, y_test, class_labels)
        
        # Feature importance analysis (if linear kernel)
        if svm_classifier.model.kernel == 'linear':
            feature_importance = svm_classifier.feature_importance_analysis(X_train, feature_cols)
        
        # Cross-validation analysis
        cv_results = svm_classifier.cross_validation_analysis(X_train, y_train)
        
        # Save model
        svm_classifier.save_model('models/svm_soil_health_classifier.pkl')
        
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data preprocessing first.")

def run_soil_health_regression_example():
    """Example function for soil health regression"""
    print("="*60)
    print("SOIL HEALTH REGRESSION WITH SVM")
    print("="*60)
    
    # Load preprocessed data (assuming it exists)
    try:
        train_data = pd.read_csv('data/preprocessed_soil_health_regression_train.csv')
        test_data = pd.read_csv('data/preprocessed_soil_health_regression_test.csv')
        
        # Separate features and target
        target_col = 'soil_health_index'
        feature_cols = [col for col in train_data.columns if col != target_col]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Initialize and train SVM regressor
        svm_regressor = SoilHealthSVM(task_type='regression')
        svm_regressor.train(X_train, y_train, tune_hyperparameters=True)
        
        # Evaluate model
        results = svm_regressor.evaluate_regression(X_test, y_test)
        
        # Feature importance analysis (if linear kernel)
        if svm_regressor.model.kernel == 'linear':
            feature_importance = svm_regressor.feature_importance_analysis(X_train, feature_cols)
        
        # Cross-validation analysis
        cv_results = svm_regressor.cross_validation_analysis(X_train, y_train)
        
        # Save model
        svm_regressor.save_model('models/svm_soil_health_regressor.pkl')
        
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data preprocessing first.")

def main():
    """Main function to demonstrate SVM models"""
    print("SVM Models for Soil Health Monitoring")
    print("Please ensure you have run data generation and preprocessing first.")
    
    # Run examples
    run_soil_health_classification_example()
    run_soil_health_regression_example()

if __name__ == "__main__":
    main()
