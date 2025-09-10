"""
Data Visualization Tools for Soil Health Analysis

This module provides comprehensive visualization tools for soil health data
analysis, model results, and insights generation.

Author: Soil Health Monitoring Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SoilHealthVisualizer:
    def __init__(self, data=None):
        """
        Initialize the visualizer
        
        Args:
            data (DataFrame): Soil health dataset
        """
        self.data = data
        
    def load_data(self, data):
        """Load new dataset"""
        self.data = data
        
    def plot_dataset_overview(self, save_path=None):
        """Create comprehensive dataset overview"""
        if self.data is None:
            print("No data loaded")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Dataset info
        info_text = f"""Dataset Overview:
        
        Shape: {self.data.shape}
        Features: {self.data.shape[1]}
        Samples: {self.data.shape[0]}
        
        Data Types:
        Numerical: {len(self.data.select_dtypes(include=[np.number]).columns)}
        Categorical: {len(self.data.select_dtypes(include=['object']).columns)}
        
        Missing Values: {self.data.isnull().sum().sum()}
        """
        
        axes[0, 0].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        axes[0, 0].set_title('Dataset Information')
        axes[0, 0].axis('off')
        
        # Missing values heatmap
        if self.data.isnull().sum().sum() > 0:
            sns.heatmap(self.data.isnull(), cbar=True, ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values Pattern')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            axes[0, 1].set_title('Missing Values Pattern')
            axes[0, 1].axis('off')
        
        # Data types distribution
        dtype_counts = self.data.dtypes.value_counts()
        axes[0, 2].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Data Types Distribution')
        
        # Numerical features distribution
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.data[numerical_cols].hist(bins=20, ax=axes[1, 0], alpha=0.7)
            axes[1, 0].set_title('Numerical Features Distribution')
        
        # Categorical features
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]  # Use first categorical column
            self.data[cat_col].value_counts().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title(f'{cat_col} Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap (subset)
        if len(numerical_cols) > 0:
            corr_cols = numerical_cols[:10]  # Limit to first 10 for readability
            corr_matrix = self.data[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
            axes[1, 2].set_title('Feature Correlations (Top 10)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_soil_properties_distribution(self, properties=None, save_path=None):
        """Plot distribution of soil properties"""
        if self.data is None:
            print("No data loaded")
            return
        
        if properties is None:
            # Auto-detect soil property columns
            soil_columns = [col for col in self.data.columns if any(keyword in col.lower() 
                          for keyword in ['ph', 'moisture', 'temperature', 'nitrogen', 
                                        'phosphorus', 'potassium', 'organic'])]
            properties = soil_columns[:8]  # Limit to 8 for visualization
        
        if not properties:
            print("No soil properties found in data")
            return
        
        n_props = len(properties)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, prop in enumerate(properties):
            if prop in self.data.columns:
                # Histogram with KDE
                sns.histplot(data=self.data, x=prop, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {prop}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(properties), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_analysis(self, season_col='season', target_col=None, save_path=None):
        """Analyze seasonal patterns in soil data"""
        if self.data is None or season_col not in self.data.columns:
            print(f"No data loaded or '{season_col}' column not found")
            return
        
        # Detect numerical columns for analysis
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numerical_cols:
            # Focus on target column
            analysis_cols = [target_col]
        else:
            # Use top 4 numerical columns
            analysis_cols = numerical_cols[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(analysis_cols):
            if i < 4 and col in self.data.columns:
                sns.boxplot(data=self.data, x=season_col, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by Season')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(analysis_cols), 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regional_analysis(self, region_col='region', target_col=None, save_path=None):
        """Analyze regional patterns in soil data"""
        if self.data is None or region_col not in self.data.columns:
            print(f"No data loaded or '{region_col}' column not found")
            return
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numerical_cols:
            analysis_cols = [target_col]
        else:
            analysis_cols = numerical_cols[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(analysis_cols):
            if i < 4 and col in self.data.columns:
                sns.violinplot(data=self.data, x=region_col, y=col, ax=axes[i])
                axes[i].set_title(f'{col} by Region')
                axes[i].tick_params(axis='x', rotation=45)
        
        for i in range(len(analysis_cols), 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self, method='pearson', save_path=None):
        """Create comprehensive correlation analysis"""
        if self.data is None:
            print("No data loaded")
            return
        
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            print("Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr(method=method)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Full correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], fmt='.2f')
        axes[0].set_title(f'{method.title()} Correlation Matrix')
        
        # High correlation pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for high correlation
                    corr_pairs.append({
                        'Feature1': corr_matrix.columns[i],
                        'Feature2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            y_pos = np.arange(len(corr_df))
            bars = axes[1].barh(y_pos, corr_df['Correlation'])
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"{row['Feature1']} - {row['Feature2']}" 
                                   for _, row in corr_df.iterrows()])
            axes[1].set_xlabel('Correlation Coefficient')
            axes[1].set_title('High Correlation Pairs (|r| > 0.5)')
            axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Color bars
            for bar, corr_val in zip(bars, corr_df['Correlation']):
                bar.set_color('red' if corr_val > 0 else 'blue')
        else:
            axes[1].text(0.5, 0.5, 'No high correlations found\n(|r| > 0.5)', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('High Correlation Pairs')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_soil_health_analysis(self, health_col='health_category', score_col=None, save_path=None):
        """Analyze soil health patterns"""
        if self.data is None:
            print("No data loaded")
            return
        
        # Try to find health-related columns
        if health_col not in self.data.columns:
            health_candidates = [col for col in self.data.columns 
                               if 'health' in col.lower() or 'category' in col.lower()]
            if health_candidates:
                health_col = health_candidates[0]
            else:
                print("No health category column found")
                return
        
        if score_col is None:
            score_candidates = [col for col in self.data.columns 
                              if 'score' in col.lower() and self.data[col].dtype in [np.float64, np.int64]]
            score_col = score_candidates[0] if score_candidates else None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Health category distribution
        health_counts = self.data[health_col].value_counts()
        axes[0, 0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Soil Health Category Distribution')
        
        # Health score distribution (if available)
        if score_col and score_col in self.data.columns:
            sns.histplot(data=self.data, x=score_col, hue=health_col, kde=True, ax=axes[0, 1])
            axes[0, 1].set_title(f'{score_col} Distribution by Health Category')
        else:
            axes[0, 1].text(0.5, 0.5, 'No health score\ncolumn found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Health by categorical features
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != health_col]
        
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            health_cat_crosstab = pd.crosstab(self.data[cat_col], self.data[health_col])
            health_cat_crosstab.plot(kind='bar', stacked=True, ax=axes[1, 0])
            axes[1, 0].set_title(f'Health Category by {cat_col}')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Health trends over time (if date column exists)
        date_cols = [col for col in self.data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols and score_col:
            date_col = date_cols[0]
            try:
                # Convert to datetime if not already
                if self.data[date_col].dtype == 'object':
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                
                # Group by month and calculate mean score
                self.data['month_year'] = self.data[date_col].dt.to_period('M')
                monthly_health = self.data.groupby('month_year')[score_col].mean()
                
                monthly_health.plot(kind='line', ax=axes[1, 1])
                axes[1, 1].set_title(f'Average {score_col} Over Time')
                axes[1, 1].tick_params(axis='x', rotation=45)
            except:
                axes[1, 1].text(0.5, 0.5, 'Cannot process\ndate column', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No date column\nor score found', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, n_components=2, save_path=None):
        """Perform and visualize PCA analysis"""
        if self.data is None:
            print("No data loaded")
            return
        
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            print("Not enough numerical features for PCA")
            return
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data.fillna(numerical_data.mean()))
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance ratio
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0, 0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Scree plot
        axes[0, 1].plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
                       pca.explained_variance_ratio_[:20], 'ro-')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Explained Variance Ratio')
        axes[0, 1].set_title('Scree Plot (First 20 Components)')
        axes[0, 1].grid(True)
        
        # 2D PCA scatter plot
        if n_components >= 2:
            scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                       alpha=0.6, c=range(len(pca_result)), cmap='viridis')
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1, 0].set_title('PCA Scatter Plot (PC1 vs PC2)')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Feature contributions to first two PCs
        if len(numerical_data.columns) <= 20:  # Only show if not too many features
            feature_names = numerical_data.columns
            pc1_contrib = pca.components_[0]
            pc2_contrib = pca.components_[1] if len(pca.components_) > 1 else np.zeros_like(pc1_contrib)
            
            y_pos = np.arange(len(feature_names))
            width = 0.35
            
            axes[1, 1].barh(y_pos - width/2, pc1_contrib, width, label='PC1', alpha=0.7)
            axes[1, 1].barh(y_pos + width/2, pc2_contrib, width, label='PC2', alpha=0.7)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(feature_names)
            axes[1, 1].set_xlabel('Component Loading')
            axes[1, 1].set_title('Feature Contributions to PC1 & PC2')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Too many features\nto display loadings', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca, pca_result
    
    def create_interactive_dashboard(self, health_col='health_category'):
        """Create interactive dashboard using Plotly"""
        if self.data is None:
            print("No data loaded")
            return
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distributions', 'Correlation Heatmap', 
                          'Scatter Plot Matrix', 'Health Category Analysis'),
            specs=[[{'type': 'histogram'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Feature distributions
        if len(numerical_cols) > 0:
            for col in numerical_cols[:3]:  # Show first 3 numerical columns
                fig.add_trace(
                    go.Histogram(x=self.data[col], name=col, opacity=0.7),
                    row=1, col=1
                )
        
        # Correlation heatmap
        if len(numerical_cols) > 1:
            corr_matrix = self.data[numerical_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          colorscale='RdBu'),
                row=1, col=2
            )
        
        # Scatter plot
        if len(numerical_cols) >= 2:
            color_col = health_col if health_col in self.data.columns else None
            fig.add_trace(
                go.Scatter(x=self.data[numerical_cols[0]], 
                          y=self.data[numerical_cols[1]],
                          mode='markers',
                          marker=dict(color=self.data[color_col] if color_col else 'blue',
                                    colorscale='viridis' if color_col else None),
                          name='Data Points'),
                row=2, col=1
            )
        
        # Health category analysis
        if health_col in self.data.columns:
            health_counts = self.data[health_col].value_counts()
            fig.add_trace(
                go.Bar(x=health_counts.index, y=health_counts.values, name='Health Categories'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Soil Health Interactive Dashboard")
        fig.show()
        
        return fig
    
    def export_visualizations(self, output_dir='visualizations'):
        """Export all visualizations to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.data is None:
            print("No data loaded")
            return
        
        print(f"Exporting visualizations to {output_dir}/")
        
        # Dataset overview
        self.plot_dataset_overview(save_path=f'{output_dir}/dataset_overview.png')
        
        # Soil properties
        self.plot_soil_properties_distribution(save_path=f'{output_dir}/soil_properties.png')
        
        # Seasonal analysis
        if 'season' in self.data.columns:
            self.plot_seasonal_analysis(save_path=f'{output_dir}/seasonal_analysis.png')
        
        # Regional analysis
        region_cols = [col for col in self.data.columns if 'region' in col.lower()]
        if region_cols:
            self.plot_regional_analysis(region_col=region_cols[0], 
                                      save_path=f'{output_dir}/regional_analysis.png')
        
        # Correlation analysis
        self.plot_correlation_analysis(save_path=f'{output_dir}/correlation_analysis.png')
        
        # Soil health analysis
        health_cols = [col for col in self.data.columns if 'health' in col.lower()]
        if health_cols:
            self.plot_soil_health_analysis(health_col=health_cols[0],
                                         save_path=f'{output_dir}/soil_health_analysis.png')
        
        # PCA analysis
        self.plot_pca_analysis(save_path=f'{output_dir}/pca_analysis.png')
        
        print("All visualizations exported successfully!")

def main():
    """Example usage of SoilHealthVisualizer"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'soil_ph': np.random.normal(6.5, 1.0, n_samples),
        'soil_moisture': np.random.normal(30, 10, n_samples),
        'soil_temperature': np.random.normal(20, 5, n_samples),
        'nitrogen_content': np.random.lognormal(3, 0.5, n_samples),
        'phosphorus_content': np.random.lognormal(2.5, 0.7, n_samples),
        'potassium_content': np.random.lognormal(4, 0.4, n_samples),
        'organic_matter': np.random.gamma(2, 1.5, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
        'soil_health_score': np.random.uniform(0, 1, n_samples)
    }
    
    # Create health categories based on score
    sample_data['health_category'] = pd.cut(
        sample_data['soil_health_score'], 
        bins=[0, 0.3, 0.6, 0.8, 1.0], 
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    
    df = pd.DataFrame(sample_data)
    
    # Initialize visualizer
    visualizer = SoilHealthVisualizer(df)
    
    # Create visualizations
    print("Creating soil health visualizations...")
    
    visualizer.plot_dataset_overview()
    visualizer.plot_soil_properties_distribution()
    visualizer.plot_seasonal_analysis()
    visualizer.plot_correlation_analysis()
    visualizer.plot_soil_health_analysis()
    visualizer.plot_pca_analysis()
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard()

if __name__ == "__main__":
    main()
