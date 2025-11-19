import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

# COMPLETELY SUPPRESS ALL WARNINGS INCLUDING NUMPY
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress numpy DLL warnings specifically
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

class PhishingEDA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.data_path)
        self.df['status'] = self.df['status'].map({'phishing': 1, 'legitimate': 0})
        print("✅ Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
    
    def basic_info(self):
        """Display basic dataset information"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nMissing Values:\n{self.df.isnull().sum().sum()}")
        print(f"Duplicate Rows: {self.df.duplicated().sum()}")
        
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        
        print("\nTarget Variable Distribution:")
        print(self.df['status'].value_counts())
        print(f"Phishing Ratio: {self.df['status'].mean():.2%}")
    
    def visualize_distributions(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*50)
        print("DATA VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Target variable
        status_counts = self.df['status'].value_counts()
        axes[0,0].pie(status_counts.values, labels=['Legitimate', 'Phishing'], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('URL Status Distribution')
        
        # URL length distribution
        sns.histplot(data=self.df, x='length_url', hue='status', ax=axes[0,1], bins=30)
        axes[0,1].set_title('URL Length Distribution by Status')
        
        # Page rank comparison
        sns.boxplot(data=self.df, x='status', y='page_rank', ax=axes[0,2])
        axes[0,2].set_title('Page Rank vs URL Status')
        axes[0,2].set_xticklabels(['Legitimate', 'Phishing'])
        
        # Number of dots
        sns.boxplot(data=self.df, x='status', y='nb_dots', ax=axes[1,0])
        axes[1,0].set_title('Number of Dots vs URL Status')
        axes[1,0].set_xticklabels(['Legitimate', 'Phishing'])
        
        # Number of slashes
        sns.boxplot(data=self.df, x='status', y='nb_slash', ax=axes[1,1])
        axes[1,1].set_title('Number of Slashes vs URL Status')
        axes[1,1].set_xticklabels(['Legitimate', 'Phishing'])
        
        # Domain age
        sns.boxplot(data=self.df, x='status', y='domain_age', ax=axes[1,2])
        axes[1,2].set_title('Domain Age vs URL Status')
        axes[1,2].set_xticklabels(['Legitimate', 'Phishing'])
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Correlation with target
        target_corr = numeric_df.corr()['status'].sort_values(ascending=False)
        print("\nTop 10 Features Correlated with Target:")
        print(target_corr.head(10))
        
        print("\nBottom 10 Features Correlated with Target:")
        print(target_corr.tail(10))
        
        # Plot correlation heatmap for top 15 features
        top_features = target_corr.abs().sort_values(ascending=False).head(15).index
        corr_matrix = numeric_df[top_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5)
        plt.title('Top 15 Features Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def pca_analysis(self):
        """Perform PCA analysis for feature importance"""
        print("\n" + "="*50)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*50)
        
        # Prepare data for PCA
        X = self.df.select_dtypes(include=[np.number]).drop('status', axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Plot explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scree plot
        ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
        ax1.set_xlabel('Principal Components')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot')
        ax1.grid(True)
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
        ax2.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance from PCA
        num_components = (cumulative_variance <= 0.95).sum() + 1
        pca_loadings = abs(pca.components_[:num_components])
        feature_importance = np.sum(pca_loadings, axis=0)
        
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nNumber of components explaining 95% variance: {num_components}")
        print("\nTop 15 Most Important Features from PCA:")
        print(feature_importance_df.head(15))
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
        plt.title('Top 15 Features by PCA Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def run_complete_analysis(self):
        """Run complete EDA pipeline"""
        print("🚀 STARTING COMPREHENSIVE EDA")
        print("="*60)
        
        self.basic_info()
        self.visualize_distributions()
        self.correlation_analysis()
        feature_importance_df = self.pca_analysis()
        
        print("\n" + "="*60)
        print("✅ EDA COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return feature_importance_df

# Execute if run directly
if __name__ == "__main__":
    eda = PhishingEDA("dataset_phishing_updated.csv")
    feature_importance_df = eda.run_complete_analysis()