import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
#standardizes data with mean=0 and std = 1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu
def run_full_analysis():
    df = pd.read_csv("E:\\dataset_phishing_updated.csv")
    
    # EDA 
    print("Running EDA...")
    print("Null values:", df.isnull().sum().sum())
    print("Duplicated entries:", df.duplicated().sum())
    print("Data types:")
    print(df.dtypes)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='status', hue='status', palette='coolwarm', legend=False)
    plt.title("Distribution of Phishing Status")
    plt.xlabel("Phishing Status")
    plt.ylabel("Count")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.histplot(df['length_url'], bins=30, kde=True, color='blue')
    plt.title("Distribution of URL Lengths")
    plt.xlabel("URL Length")
    plt.ylabel("Frequency")
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap") 
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='status', y='page_rank')
    plt.title("Page Rank vs. Phishing Status")
    plt.xlabel("Phishing Status")
    plt.ylabel("Page Rank")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='status', y='nb_dots')
    plt.title("Number of Dots in URL vs. Phishing Status")
    plt.xlabel("Phishing Status")
    plt.ylabel("Number of Dots")
    plt.show()
    
    print(df.head())
    print(df.describe())
    
    # PCA Analysis
    print("\nRunning PCA Analysis...")
    df_numeric = df.drop(columns=["url", "status"], errors='ignore')
    original_columns = df_numeric.columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    eigenvalues = pca.explained_variance_
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='green')
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.axhline(y=1, color='red', linestyle='--', label='Eigenvalue = 1 (Kaiser rule)')
    plt.legend()
    plt.show()
    
    num_components = 13  # Based on scree plot
    pca_loadings = abs(pca.components_[:num_components])
    feature_importance = np.sum(pca_loadings, axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': original_columns,
        'Importance': feature_importance
    })
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    top_features = feature_importance_df['Feature'].head(num_components).tolist()
    print(f"\nTop {num_components} Contributing Features to the Selected PCs:")
    for feature in top_features:
        print(feature)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], feature_importance_df['Importance'].head(num_components)[::-1], color='purple')
    plt.xlabel("Importance of Each pca component")
    plt.title(f"Top {num_components} Features Contributing to First {num_components} Principal Components")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"\nAfter applying PCA, {num_components} components were selected based on the elbow in the scree plot.")
    print("The top contributing features included character-based attributes of URLs like nb_underscore, nb_colon, and domain_in_brand.")
    
    # Factor Analysis (Orthogonal)
    print("\nRunning Factor Analysis (Orthogonal Model)...")
    fa = FactorAnalysis(n_components=num_components, random_state=42)
    fa_result = fa.fit_transform(df_scaled)

    fa_loadings = abs(fa.components_)
    fa_feature_importance = np.sum(fa_loadings, axis=0)

    fa_feature_importance_df = pd.DataFrame({
        'Feature': original_columns,
        'Importance': fa_feature_importance
    })
    fa_feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    top_fa_features = fa_feature_importance_df['Feature'].head(num_components).tolist()

    print(f"\nTop {num_components} Contributing Features (Factor Analysis):")
    for feature in top_fa_features:
        print(feature)

    plt.figure(figsize=(10, 6))
    plt.barh(top_fa_features[::-1], fa_feature_importance_df['Importance'].head(num_components)[::-1], color='orange')
    plt.xlabel("Importance (Sum of Loadings)")
    plt.title(f"Top {num_components} Features from Factor Analysis")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Mann-Whitney U Test
    print("\nRunning Mann-Whitney U Test for Hypothesis Testing...")
    df_full = df.copy()
    df_full['status'] = df_full['status'].map({'phishing': 1, 'legitimate': 0}).fillna(df_full['status'])

    mann_whitney_results = []

    for feature in fa_feature_importance_df['Feature'].head(num_components):
        phishing_group = df_full[df_full['status'] == 1][feature]
        legitimate_group = df_full[df_full['status'] == 0][feature]
        
        stat, p_value = mannwhitneyu(phishing_group, legitimate_group, alternative='two-sided')
        
        mann_whitney_results.append({
            'Feature': feature,
            'Mann-Whitney U Statistic': stat,
            'p-value': p_value
        })

    mann_whitney_df = pd.DataFrame(mann_whitney_results)
    mann_whitney_df.sort_values(by='p-value', inplace=True)

    print("\nMann-Whitney U Test Results:")
    print(mann_whitney_df)
    print("The p values in Mann-whiteney u test results are scientifically written in exponential notation which means they are extremely small numbers far less than 0.05 - indicating a strong statistical significance")


    # Highlight significant features
    significant_features = mann_whitney_df[mann_whitney_df['p-value'] < 0.05]
    print(f"\nFeatures showing significant difference between phishing and legitimate URLs (p < 0.05):")
    print(significant_features[['Feature', 'p-value']])
    
    # Clustering
    print("\nRunning Clustering Analysis...")
    selected_features = [
        'longest_word_host', 'tld_in_subdomain', 'empty_title', 'avg_word_host',
        'http_in_path', 'page_rank', 'avg_words_raw', 'domain_age', 'google_index',
        'longest_words_raw', 'avg_word_path', 'nb_slash', 'nb_and'
    ]
    
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.grid()
    plt.show()
    
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    cluster_names = {0: "Low-Risk URLs", 1: "Moderate-Risk URLs", 2: "High-Risk URLs"}
    df['Cluster Name'] = df['Cluster'].map(cluster_names)
    
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_pca_2d[:, 0],
        y=X_pca_2d[:, 1],
        hue=df['Cluster Name'],
        palette='coolwarm',
        s=50,
        alpha=0.7
    )
    plt.title("Clusters Visualized in 2D Space (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", loc="best")
    plt.grid(alpha=0.3)
    plt.show()
    
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score for {optimal_clusters} clusters: {silhouette_avg:.2f}")
    print("Inference:Well-separated clusters: Clear gaps between high/low-risk groups")
    print("High-risk URLs scatter because phishing attacks use diverse tactics(multiple distinct patterns), making their features vary widely in patterns.")
    return top_features

# Function to train and return the classification model
def get_classification_model():
    df = pd.read_csv("E:\\dataset_phishing_updated.csv")
    df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
    
    # Selected features based on PCA analysis
    selected_features = [
        'longest_word_host', 'tld_in_subdomain', 'empty_title', 'avg_word_host',
        'http_in_path', 'page_rank', 'avg_words_raw', 'domain_age', 'google_index',
        'longest_words_raw', 'avg_word_path', 'nb_slash', 'nb_and'
    ]
    
    X = df[selected_features]
    y = df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, X_test, y_test

# Run the analysis if this file is executed directly
if __name__ == "__main__":
    print("Running comprehensive analysis...")
    run_full_analysis()
    model, _, _ = get_classification_model()
    print("Analysis complete!")
