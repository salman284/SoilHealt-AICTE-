"""
Clustering Models for Soil Health Analysis

This module implements various clustering algorithms to identify patterns
in soil health data and group similar soil conditions.

Author: Soil Health Monitoring Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SoilHealthClustering:
    def __init__(self, algorithm='kmeans'):
        """
        Initialize clustering model for soil health analysis
        
        Args:
            algorithm (str): Clustering algorithm to use
                           ('kmeans', 'hierarchical', 'dbscan', 'gmm')
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.labels_ = None
        self.n_clusters = None
        self.scaled_data = None
        self.pca_data = None
        
    def create_model(self, n_clusters=4, **kwargs):
        """
        Create clustering model based on algorithm type
        
        Args:
            n_clusters (int): Number of clusters (not used for DBSCAN)
            **kwargs: Additional parameters for the clustering algorithm
        """
        self.n_clusters = n_clusters
        
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                **kwargs
            )
        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                **kwargs
            )
        elif self.algorithm == 'dbscan':
            self.model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5),
                **{k: v for k, v in kwargs.items() if k not in ['eps', 'min_samples']}
            )
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return self.model
    
    def fit(self, X, feature_names=None):
        """
        Fit the clustering model to the data
        
        Args:
            X (array-like): Input features
            feature_names (list): Names of features for interpretation
        """
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(X)
        
        # Fit PCA for visualization
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Fit the clustering model
        if self.algorithm == 'gmm':
            self.model.fit(self.scaled_data)
            self.labels_ = self.model.predict(self.scaled_data)
        else:
            self.labels_ = self.model.fit_predict(self.scaled_data)
        
        # Update number of clusters for DBSCAN
        if self.algorithm == 'dbscan':
            self.n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        return self.labels_
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.algorithm == 'gmm':
            return self.model.predict(X_scaled)
        elif self.algorithm in ['kmeans']:
            return self.model.predict(X_scaled)
        else:
            # For DBSCAN and hierarchical, we need to fit on new data
            # This is a limitation of these algorithms
            print("Warning: Prediction not directly supported for this algorithm")
            return None
    
    def evaluate_clustering(self, X):
        """
        Evaluate clustering performance using various metrics
        
        Args:
            X (array-like): Original data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        # Remove noise points for DBSCAN
        mask = self.labels_ != -1
        X_eval = self.scaled_data[mask]
        labels_eval = self.labels_[mask]
        
        if len(set(labels_eval)) < 2:
            print("Warning: Less than 2 clusters found, cannot compute metrics")
            return {}
        
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X_eval, labels_eval)
        except:
            metrics['silhouette_score'] = None
            
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_eval, labels_eval)
        except:
            metrics['calinski_harabasz_score'] = None
            
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_eval, labels_eval)
        except:
            metrics['davies_bouldin_score'] = None
        
        # Additional metrics
        metrics['n_clusters'] = self.n_clusters
        metrics['n_noise'] = np.sum(self.labels_ == -1) if self.algorithm == 'dbscan' else 0
        
        return metrics
    
    def find_optimal_clusters(self, X, max_clusters=10, method='elbow'):
        """
        Find optimal number of clusters using elbow method or silhouette analysis
        
        Args:
            X (array-like): Input data
            max_clusters (int): Maximum number of clusters to test
            method (str): Method to use ('elbow' or 'silhouette')
        """
        if self.algorithm not in ['kmeans', 'gmm']:
            print("Optimal cluster finding only supported for KMeans and GMM")
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        range_clusters = range(2, max_clusters + 1)
        scores = []
        inertias = []
        
        for n_clusters in range_clusters:
            if self.algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                inertias.append(model.inertia_)
            else:  # GMM
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                inertias.append(-model.score(X_scaled))  # Negative log-likelihood
            
            if method == 'silhouette':
                score = silhouette_score(X_scaled, labels)
                scores.append(score)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow plot
        axes[0].plot(range_clusters, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia' if self.algorithm == 'kmeans' else 'Negative Log-Likelihood')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        # Silhouette plot
        if method == 'silhouette':
            axes[1].plot(range_clusters, scores, 'ro-')
            axes[1].set_xlabel('Number of Clusters')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].set_title('Silhouette Analysis')
            axes[1].grid(True)
            
            optimal_clusters = range_clusters[np.argmax(scores)]
            print(f"Optimal number of clusters (silhouette): {optimal_clusters}")
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return inertias, scores if method == 'silhouette' else None
    
    def plot_clusters_2d(self, feature_names=None):
        """Plot clusters in 2D using PCA"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        plt.figure(figsize=(10, 8))
        
        # Plot each cluster
        unique_labels = set(self.labels_)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points (for DBSCAN)
                mask = self.labels_ == label
                plt.scatter(self.pca_data[mask, 0], self.pca_data[mask, 1], 
                           c='black', marker='x', s=50, label='Noise')
            else:
                mask = self.labels_ == label
                plt.scatter(self.pca_data[mask, 0], self.pca_data[mask, 1], 
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.xlabel(f'First Principal Component')
        plt.ylabel(f'Second Principal Component')
        plt.title(f'{self.algorithm.upper()} Clustering Results (PCA Projection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_clusters_3d(self, features_3d=None):
        """Plot clusters in 3D using plotly"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        if features_3d is not None and len(features_3d) == 3:
            # Use specific 3 features
            data_3d = self.scaled_data[:, features_3d]
            labels = ['Feature 1', 'Feature 2', 'Feature 3']
        else:
            # Use PCA with 3 components
            pca_3d = PCA(n_components=3)
            data_3d = pca_3d.fit_transform(self.scaled_data)
            labels = ['PC1', 'PC2', 'PC3']
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        unique_labels = set(self.labels_)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            name = 'Noise' if label == -1 else f'Cluster {label}'
            color = 'black' if label == -1 else colors[i % len(colors)]
            
            fig.add_trace(go.Scatter3d(
                x=data_3d[mask, 0],
                y=data_3d[mask, 1],
                z=data_3d[mask, 2],
                mode='markers',
                marker=dict(size=5, color=color),
                name=name
            ))
        
        fig.update_layout(
            title=f'{self.algorithm.upper()} Clustering Results (3D)',
            scene=dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2]
            )
        )
        
        fig.show()
    
    def analyze_clusters(self, X, feature_names=None):
        """
        Analyze cluster characteristics
        
        Args:
            X (array-like): Original data
            feature_names (list): Names of features
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        df = pd.DataFrame(X, columns=feature_names or [f'Feature_{i}' for i in range(X.shape[1])])
        df['Cluster'] = self.labels_
        
        # Calculate cluster statistics
        cluster_stats = df.groupby('Cluster').agg(['mean', 'std', 'count'])
        
        print("Cluster Statistics:")
        print("=" * 50)
        print(cluster_stats)
        
        # Plot cluster characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Cluster')
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):
                df.boxplot(column=col, by='Cluster', ax=axes[i])
                axes[i].set_title(f'{col} by Cluster')
                axes[i].set_xlabel('Cluster')
            
            # Hide unused subplots
            for i in range(len(numeric_cols), 4):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return cluster_stats
    
    def get_cluster_centers(self):
        """Get cluster centers (for applicable algorithms)"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        if self.algorithm == 'kmeans':
            # Transform centers back to original scale
            centers_scaled = self.model.cluster_centers_
            centers = self.scaler.inverse_transform(centers_scaled)
            return centers
        elif self.algorithm == 'gmm':
            centers_scaled = self.model.means_
            centers = self.scaler.inverse_transform(centers_scaled)
            return centers
        else:
            print(f"Cluster centers not available for {self.algorithm}")
            return None

def compare_clustering_algorithms(X, feature_names=None, n_clusters=4):
    """
    Compare different clustering algorithms on the same dataset
    
    Args:
        X (array-like): Input data
        feature_names (list): Names of features
        n_clusters (int): Number of clusters to use
    """
    algorithms = ['kmeans', 'hierarchical', 'gmm']
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, algorithm in enumerate(algorithms):
        print(f"\n=== {algorithm.upper()} ===")
        
        # Fit clustering model
        clusterer = SoilHealthClustering(algorithm=algorithm)
        clusterer.create_model(n_clusters=n_clusters)
        labels = clusterer.fit(X, feature_names)
        
        # Evaluate
        metrics = clusterer.evaluate_clustering(X)
        results[algorithm] = metrics
        
        print(f"Number of clusters: {metrics.get('n_clusters', 'N/A')}")
        print(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
        print(f"Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.4f}")
        print(f"Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
        
        # Plot
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                mask = labels == label
                axes[i].scatter(clusterer.pca_data[mask, 0], clusterer.pca_data[mask, 1], 
                               c='black', marker='x', s=50, alpha=0.7)
            else:
                mask = labels == label
                axes[i].scatter(clusterer.pca_data[mask, 0], clusterer.pca_data[mask, 1], 
                               c=[color], s=50, alpha=0.7)
        
        axes[i].set_title(f'{algorithm.upper()}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    """Example usage of SoilHealthClustering"""
    print("SoilHealthClustering example usage")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 8
    
    # Create synthetic soil health data with clusters
    centers = np.array([[2, 2, 2, 2, 2, 2, 2, 2],
                       [-2, -2, -2, -2, -2, -2, -2, -2],
                       [2, -2, 2, -2, 2, -2, 2, -2],
                       [-2, 2, -2, 2, -2, 2, -2, 2]])
    
    X = []
    for center in centers:
        cluster_data = np.random.multivariate_normal(
            center, np.eye(n_features) * 0.5, n_samples // 4
        )
        X.append(cluster_data)
    
    X = np.vstack(X)
    feature_names = [f'Soil_Feature_{i+1}' for i in range(n_features)]
    
    # Compare algorithms
    results = compare_clustering_algorithms(X, feature_names, n_clusters=4)
    
    # Detailed analysis with KMeans
    print("\n=== Detailed KMeans Analysis ===")
    kmeans_clusterer = SoilHealthClustering(algorithm='kmeans')
    kmeans_clusterer.find_optimal_clusters(X, max_clusters=8, method='silhouette')
    
    # Fit final model
    kmeans_clusterer.create_model(n_clusters=4)
    labels = kmeans_clusterer.fit(X, feature_names)
    
    # Analyze clusters
    cluster_stats = kmeans_clusterer.analyze_clusters(X, feature_names)

if __name__ == "__main__":
    main()
