import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
df = pd.read_csv("E:\\dataset_phishing_updated.csv")
'''
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated()) 
print(df.dtypes)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df['year'] = df['Timestamp'].dt.year
df['month'] = df['Timestamp'].dt.month
df['day'] = df['Timestamp'].dt.day
df['hour'] = df['Timestamp'].dt.hour
#Distribution of target variable(status)
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='status', hue='status', palette='coolwarm', legend=False)
plt.title("Distribution of Phishing Status")
plt.xlabel("Phishing Status")
plt.ylabel("Count")
plt.show() 
print(df['status'].value_counts())
#indicates the url length is between 0-200(shorter url length)
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
#lower page rank indicates phishing and higher page rank indicates legitimate
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='status', y='page_rank')
plt.title("Page Rank vs. Phishing Status")
plt.xlabel("Phishing Status")
plt.ylabel("Page Rank")
plt.show()
#phishing url contains more dots(outliers will be high) whereas legitimate url contains fewer dots(outliers will be less)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='status', y='nb_dots')
plt.title("Number of Dots in URL vs. Phishing Status")
plt.xlabel("Phishing Status")
plt.ylabel("Number of Dots")
plt.show()
print(f"Maximum number of scams: {df['Number_of_Scams'].max()}")
print(f"Maximum web traffic: {df['web_traffic'].max()}")
print(f"Longest url length: {df['length_url'].max()}")
print(f"Highest ratio which may indicate phishing is : {df['ratio_digits_url'].max()}")
print(df["Timestamp"].head())
print(df["Timestamp"].dtype)
print(df["Timestamp"].isna().sum())
print(df.columns)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Dropping non-numeric columns
df_numeric = df.drop(columns=["url", "status"], errors='ignore')
original_columns = df_numeric.columns
# Standardizing the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
# Applying PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled)
explained_variance = pca.explained_variance_ratio_.cumsum()
num_components_95 = (explained_variance < 0.95).sum() + 1
pca_loadings = abs(pca.components_[:num_components_95])
feature_importance = np.sum(pca_loadings, axis=0)
feature_importance_df = pd.DataFrame({'Feature': original_columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
selected_features = feature_importance_df.head(num_components_95)['Feature'].tolist()
reduced_features = feature_importance_df.tail(len(original_columns) - num_components_95)['Feature'].tolist()
print(f"Total original features: {len(original_columns)}")
print(f"Number of Principal Components selected for 95% variance: {num_components_95}")
print("Selected Features (Important in PCA):", selected_features)
print("Reduced Features (Less Important in PCA):", reduced_features)
X = df_numeric 
X_selected = df_numeric[selected_features]
y = df["status"] if "status" in df else np.random.randint(0, 2, size=len(df))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy_all_features = accuracy_score(y_test, model.predict(X_test))
model.fit(X_train_pca, y_train)
accuracy_pca_features = accuracy_score(y_test, model.predict(X_test_pca))
plt.figure(figsize=(8, 5))
plt.bar(["All Features", "PCA Features"], [accuracy_all_features, accuracy_pca_features], color=["blue", "green"])
plt.ylim(0.9, 1.0)  # Adjust y-axis for better visibility
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison: All Features vs PCA Selected Features")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for i, acc in enumerate([accuracy_all_features, accuracy_pca_features]):
    plt.text(i, acc + 0.001, f"{acc:.4f}", ha='center', fontsize=12, fontweight='bold')

plt.show()
