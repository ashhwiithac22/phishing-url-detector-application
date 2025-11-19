import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings
import joblib
import time
import os

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

class AdvancedPhishingClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.model_performance = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset using ONLY 15 PCA features"""
        print("📊 Loading and preparing data...")
        self.df = pd.read_csv(self.data_path)
        self.df['status'] = self.df['status'].map({'phishing': 1, 'legitimate': 0})
        
        # Use ONLY the 15 features from PCA analysis
        selected_features = [
            'domain_in_title', 'nb_dslash', 'domain_with_copyright', 
            'shortening_service', 'prefix_suffix', 'web_traffic',
            'ratio_extRedirection', 'nb_extCSS', 'nb_at', 'nb_underscore',
            'nb_colon', 'nb_redirection', 'nb_hyperlinks', 'nb_space', 'dns_record'
        ]
        
        # Use available features only
        available_features = [f for f in selected_features if f in self.df.columns]
        print(f"Available PCA features: {len(available_features)}")
        
        if len(available_features) < 10:
            st.error("❌ Not enough PCA features found in dataset!")
            return
        
        self.X = self.df[available_features]
        self.y = self.df['status']
        
        print(f"✅ Selected {len(available_features)} PCA features for modeling")
        print(f"✅ Dataset shape: {self.X.shape}")
        print(f"✅ Features: {available_features}")
        
    def preprocess_data(self):
        """Split and scale the data"""
        print("⚙️ Preprocessing data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {self.X_train_scaled.shape}")
        print(f"✅ Test set: {self.X_test_scaled.shape}")
        
    def initialize_models(self):
        """Initialize all machine learning models"""
        print("🤖 Initializing machine learning models...")
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Bagging Classifier': BaggingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        return models
    
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auc_score = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else 0
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        overfitting_gap = train_accuracy - test_accuracy
        
        cm = confusion_matrix(y_test, y_test_pred)
        fn_cost = cm[1, 0] * 5
        fp_cost = cm[0, 1] * 1
        total_cost = fn_cost + fp_cost
        
        performance = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'cv_mean': cv_mean,
            'overfitting_gap': overfitting_gap,
            'false_negatives': cm[1, 0],
            'false_positives': cm[0, 1],
            'total_cost': total_cost
        }
        
        return performance
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("🎯 Training and evaluating models...")
        models = self.initialize_models()
        
        for name, model in models.items():
            print(f"🔧 Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(self.X_train_scaled, self.y_train)
                performance = self.evaluate_model(
                    model, name, self.X_train_scaled, self.X_test_scaled, 
                    self.y_train, self.y_test
                )
                
                self.model_performance[name] = performance
                training_time = time.time() - start_time
                print(f"✅ {name}: {training_time:.1f}s | Test Acc: {performance['test_accuracy']:.3f}")
                      
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
    
    def select_best_model(self):
        """Automatically select the best model"""
        print("\n🏆 Selecting best model...")
        
        if not self.model_performance:
            print("❌ No models were successfully trained!")
            return None
        
        scores_df = pd.DataFrame({
            'Model': list(self.model_performance.keys()),
            'Test_Accuracy': [p['test_accuracy'] for p in self.model_performance.values()],
            'F1_Score': [p['f1_score'] for p in self.model_performance.values()],
            'CV_Mean': [p['cv_mean'] for p in self.model_performance.values()],
            'Overfitting_Gap': [p['overfitting_gap'] for p in self.model_performance.values()],
            'False_Negatives': [p['false_negatives'] for p in self.model_performance.values()],
        })
        
        scores_df['Composite_Score'] = (
            scores_df['Test_Accuracy'] * 0.3 +
            scores_df['F1_Score'] * 0.3 +
            scores_df['CV_Mean'] * 0.2 +
            (1 - scores_df['Overfitting_Gap']) * 0.1 +
            (1 - scores_df['False_Negatives'] / len(self.y_test)) * 0.1
        )
        
        best_model_row = scores_df.loc[scores_df['Composite_Score'].idxmax()]
        self.best_model_name = best_model_row['Model']
        self.best_model = self.model_performance[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*60)
        display_df = scores_df.round(4).sort_values('Composite_Score', ascending=False)
        print(display_df.to_string(index=False))
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"🎯 Test Accuracy: {best_model_row['Test_Accuracy']:.4f}")
        print(f"⚖️ F1 Score: {best_model_row['F1_Score']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def save_best_model(self):
        """Save the best model and scaler"""
        print("\n💾 Saving best model and artifacts...")
        
        joblib.dump(self.best_model, "best_phishing_model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        joblib.dump(list(self.X.columns), "feature_names.pkl")
        
        print(f"✅ Best model saved: {self.best_model_name}")
        print(f"✅ Features used: {len(self.X.columns)}")
        
    def run_complete_analysis(self):
        """Run the complete model training pipeline"""
        print("🚀 STARTING MODEL TRAINING WITH 15 PCA FEATURES")
        print("="*60)
        
        self.load_and_prepare_data()
        self.preprocess_data()
        self.train_and_evaluate_models()
        best_model, best_model_name = self.select_best_model()
        self.save_best_model()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.best_model, self.best_model_name

if __name__ == "__main__":
    analyzer = AdvancedPhishingClassifier("dataset_phishing_updated.csv")
    best_model, best_model_name = analyzer.run_complete_analysis()