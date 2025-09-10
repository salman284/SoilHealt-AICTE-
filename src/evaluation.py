"""
Model Evaluation and Performance Analysis

This module provides comprehensive evaluation metrics and visualization tools
for soil health prediction models.

Author: Soil Health Monitoring Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model, model_name="Model"):
        """
        Initialize model evaluator
        
        Args:
            model: Trained model object
            model_name (str): Name of the model for display
        """
        self.model = model
        self.model_name = model_name
        self.task_type = self._detect_task_type()
        
    def _detect_task_type(self):
        """Auto-detect if this is classification or regression"""
        # Try to detect based on model type or methods
        if hasattr(self.model, 'predict_proba') or hasattr(self.model, 'classes_'):
            return 'classification'
        else:
            return 'regression'
    
    def evaluate_classification(self, X_test, y_test, average='weighted', 
                              plot_results=True):
        """
        Comprehensive classification evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            average (str): Averaging strategy for metrics
            plot_results (bool): Whether to plot results
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if self.task_type != 'classification':
            print("Warning: Model appears to be for regression, not classification")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC AUC (for binary or multiclass)
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # Multiclass
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average=average)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred
        }
        
        # Print results
        print(f"=== {self.model_name} Classification Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        if plot_results:
            self.plot_classification_results(X_test, y_test, results)
        
        return results
    
    def evaluate_regression(self, X_test, y_test, plot_results=True):
        """
        Comprehensive regression evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            plot_results (bool): Whether to plot results
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if self.task_type != 'regression':
            print("Warning: Model appears to be for classification, not regression")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (handle division by zero)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # Residuals
        residuals = y_test - y_pred
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'residuals': residuals,
            'predictions': y_pred
        }
        
        # Print results
        print(f"=== {self.model_name} Regression Results ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        if plot_results:
            self.plot_regression_results(X_test, y_test, results)
        
        return results
    
    def plot_classification_results(self, X_test, y_test, results):
        """Plot classification evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve (for binary classification)
        if results['roc_auc'] is not None and len(np.unique(y_test)) == 2:
            try:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.2f})')
                axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curve')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            except:
                axes[0, 1].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Class Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1, 0].bar(unique, counts, alpha=0.7)
        axes[1, 0].set_title('True Class Distribution')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        
        # Prediction Distribution
        unique_pred, counts_pred = np.unique(results['predictions'], return_counts=True)
        axes[1, 1].bar(unique_pred, counts_pred, alpha=0.7, color='orange')
        axes[1, 1].set_title('Predicted Class Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def plot_regression_results(self, X_test, y_test, results):
        """Plot regression evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_test, results['predictions'], alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {results["r2_score"]:.3f})')
        axes[0, 0].grid(True)
        
        # Residuals Plot
        axes[0, 1].scatter(results['predictions'], results['residuals'], alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True)
        
        # Residuals Histogram
        axes[1, 0].hist(results['residuals'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True)
        
        # Q-Q Plot for residuals normality
        from scipy import stats
        stats.probplot(results['residuals'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, cv=5, scoring=None):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            cv (int): Number of folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scoring_metric': scoring
        }
        
        print(f"=== Cross-Validation Results ({cv}-fold) ===")
        print(f"Scoring: {scoring}")
        print(f"Mean Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual Scores: {cv_scores}")
        
        return results
    
    def plot_learning_curve(self, X, y, cv=5, n_jobs=-1):
        """Plot learning curves to analyze bias/variance"""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {self.model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return train_sizes, train_scores, val_scores
    
    def plot_validation_curve(self, X, y, param_name, param_range, cv=5):
        """Plot validation curve for hyperparameter tuning"""
        train_scores, val_scores = validation_curve(
            self.model, X, y, param_name=param_name, param_range=param_range, cv=cv
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curve - {self.model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return train_scores, val_scores

class ModelComparison:
    def __init__(self):
        """Initialize model comparison utility"""
        self.models = {}
        self.results = {}
        
    def add_model(self, name, model, task_type='auto'):
        """Add a model to comparison"""
        self.models[name] = {
            'model': model,
            'evaluator': ModelEvaluator(model, name)
        }
        
        if task_type != 'auto':
            self.models[name]['evaluator'].task_type = task_type
    
    def compare_models(self, X_test, y_test, plot_results=True):
        """
        Compare multiple models on the same test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            plot_results (bool): Whether to plot comparison
            
        Returns:
            dict: Comparison results
        """
        comparison_results = {}
        
        for name, model_info in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print('='*50)
            
            evaluator = model_info['evaluator']
            
            if evaluator.task_type == 'classification':
                results = evaluator.evaluate_classification(X_test, y_test, plot_results=False)
                comparison_results[name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'roc_auc': results['roc_auc']
                }
            else:
                results = evaluator.evaluate_regression(X_test, y_test, plot_results=False)
                comparison_results[name] = {
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'r2_score': results['r2_score'],
                    'mape': results['mape']
                }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        print(f"\n{'='*50}")
        print("MODEL COMPARISON SUMMARY")
        print('='*50)
        print(comparison_df)
        
        if plot_results:
            self.plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison results"""
        n_metrics = len(comparison_df.columns)
        n_models = len(comparison_df)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(comparison_df.columns):
            values = comparison_df[metric].dropna()
            
            bars = axes[i].bar(values.index, values.values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values.values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def feature_importance_analysis(model, feature_names, X_test=None, y_test=None, 
                              method='auto', top_n=20):
    """
    Analyze feature importance using various methods
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_test, y_test: Test data for permutation importance
        method: Method to use ('auto', 'builtin', 'permutation', 'shap')
        top_n: Number of top features to display
    """
    importance_dict = {}
    
    # Built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importance_dict['builtin'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance_dict['builtin'] = np.abs(model.coef_).flatten()
    
    # Permutation importance
    if X_test is not None and y_test is not None and method in ['auto', 'permutation']:
        try:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importance_dict['permutation'] = perm_importance.importances_mean
        except Exception as e:
            print(f"Permutation importance failed: {e}")
    
    # SHAP values (if available)
    if method in ['auto', 'shap'] and X_test is not None:
        try:
            import shap
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test[:100])  # Use subset for speed
            importance_dict['shap'] = np.abs(shap_values.values).mean(0)
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    # Plot results
    n_methods = len(importance_dict)
    if n_methods == 0:
        print("No feature importance methods available")
        return
    
    fig, axes = plt.subplots(1, n_methods, figsize=(8*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, (method_name, importances) in enumerate(importance_dict.items()):
        # Get top features
        if len(importances.shape) > 1:
            importances = importances.flatten()
        
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        y_pos = np.arange(len(top_features))
        axes[i].barh(y_pos, top_importances)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(top_features)
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'Feature Importance ({method_name.title()})')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return importance_dict

def main():
    """Example usage of evaluation tools"""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    print("Model Evaluation Example")
    
    # Generate sample data
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    
    # Train models
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_c, y_train_c)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_r, y_train_r)
    
    # Evaluate classification
    print("\n=== Classification Evaluation ===")
    class_evaluator = ModelEvaluator(rf_classifier, "Random Forest Classifier")
    class_results = class_evaluator.evaluate_classification(X_test_c, y_test_c)
    
    # Evaluate regression
    print("\n=== Regression Evaluation ===")
    reg_evaluator = ModelEvaluator(rf_regressor, "Random Forest Regressor")
    reg_results = reg_evaluator.evaluate_regression(X_test_r, y_test_r)
    
    # Feature importance
    feature_names = [f'Feature_{i}' for i in range(20)]
    print("\n=== Feature Importance Analysis ===")
    importance_results = feature_importance_analysis(rf_classifier, feature_names, X_test_c, y_test_c)

if __name__ == "__main__":
    main()
