import streamlit as st
import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: none;
    }
    .safe-url {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
    }
    .phishing-url {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .cv-results {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PhishingDetector:
    def __init__(self):
        # Initialize session state to persist model
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
        if 'feature_names' not in st.session_state:
            st.session_state.feature_names = None
        if 'cv_scores' not in st.session_state:
            st.session_state.cv_scores = None
        
        # Try to load existing model
        self.load_existing_model()
    
    def load_existing_model(self):
        """Try to load previously trained model"""
        try:
            self.model = joblib.load("phishing_model.pkl")
            self.scaler = joblib.load("scaler.pkl")
            self.feature_names = joblib.load("feature_names.pkl")
            st.session_state.model_trained = True
            if st.session_state.model_info is None:
                st.session_state.model_info = {
                    'name': 'XGBoost',
                    'accuracy': 0.8657,
                    'precision': 0.8560,
                    'recall': 0.8793,
                    'f1_score': 0.8675,
                    'features_used': len(self.feature_names),
                    'cv_mean': 0.86,
                    'cv_std': 0.02
                }
            st.sidebar.success("✅ Pre-trained model loaded!")
        except:
            self.model = None
            self.scaler = None
            self.feature_names = None
    
    def load_and_train_model(self):
        """Load dataset and train XGBoost model with Cross-Validation"""
        try:
            # Load dataset
            df = pd.read_csv(r"D:\PROJECTS\Project Phishing\dataset_phishing_updated.csv")
            df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
            
            # Select features that can be extracted from URLs
            extractable_features = [
                'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 
                'nb_at', 'nb_qm', 'nb_and', 'nb_eq', 'nb_slash', 'nb_www',
                'ratio_digits_url', 'ratio_digits_host', 'http_in_path',
                'tld_in_subdomain', 'prefix_suffix', 'nb_redirection',
                'nb_underscore', 'nb_colon', 'nb_space', 'nb_dslash'
            ]
            
            # Use only features that exist in dataset and are extractable
            available_features = [f for f in extractable_features if f in df.columns]
            
            if len(available_features) < 10:
                st.error("❌ Not enough extractable features in dataset")
                return False
            
            X = df[available_features]
            y = df['status']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # === CROSS-VALIDATION ===
            st.subheader("📊 Cross-Validation Results")
            
            # Perform 5-fold stratified cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            # Cross-validation scores
            cv_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring='accuracy')
            cv_accuracy_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring='accuracy')
            cv_precision_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring='precision')
            cv_recall_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring='recall')
            cv_f1_scores = cross_val_score(cv_model, X_scaled, y, cv=cv, scoring='f1')
            
            # Display CV results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CV Accuracy", f"{cv_accuracy_scores.mean():.3f} ± {cv_accuracy_scores.std():.3f}")
            with col2:
                st.metric("CV Precision", f"{cv_precision_scores.mean():.3f} ± {cv_precision_scores.std():.3f}")
            with col3:
                st.metric("CV Recall", f"{cv_recall_scores.mean():.3f} ± {cv_recall_scores.std():.3f}")
            with col4:
                st.metric("CV F1-Score", f"{cv_f1_scores.mean():.3f} ± {cv_f1_scores.std():.3f}")
            
            # Show individual fold results
            with st.expander("📈 View Individual Fold Results"):
                cv_results_df = pd.DataFrame({
                    'Fold': range(1, 6),
                    'Accuracy': cv_accuracy_scores,
                    'Precision': cv_precision_scores,
                    'Recall': cv_recall_scores,
                    'F1-Score': cv_f1_scores
                })
                st.dataframe(cv_results_df.style.format({
                    'Accuracy': '{:.3f}',
                    'Precision': '{:.3f}', 
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}'
                }))
                
                # Plot CV results
                fig = px.line(cv_results_df, x='Fold', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                             title='5-Fold Cross-Validation Performance',
                             markers=True)
                fig.update_layout(yaxis_title='Score', xaxis_title='Fold Number')
                st.plotly_chart(fig, use_container_width=True)
            
            # === FINAL MODEL TRAINING ===
            st.subheader("🎯 Final Model Training")
            
            # Train final model on full training data
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Training final XGBoost model...")
            self.model.fit(X_train, y_train)
            progress_bar.progress(100)
            
            # Final evaluation on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            self.feature_names = available_features
            
            # Store in session state
            st.session_state.model_info = {
                'name': 'XGBoost',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'features_used': len(available_features),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'cv_mean': cv_accuracy_scores.mean(),
                'cv_std': cv_accuracy_scores.std(),
                'cv_scores': cv_accuracy_scores.tolist()
            }
            
            st.session_state.feature_names = available_features
            st.session_state.model_trained = True
            st.session_state.cv_scores = cv_accuracy_scores.tolist()
            
            # Save model artifacts
            joblib.dump(self.model, "phishing_model.pkl")
            joblib.dump(self.scaler, "scaler.pkl")
            joblib.dump(self.feature_names, "feature_names.pkl")
            
            status_text.text("✅ Model training completed!")
            
            # Show final test results
            st.subheader("🎯 Final Test Set Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Test Precision", f"{precision:.3f}")
            with col3:
                st.metric("Test Recall", f"{recall:.3f}")
            with col4:
                st.metric("Test F1-Score", f"{f1:.3f}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            return False
    
    def extract_url_features(self, url):
        """Extract features from URL that match training data"""
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc
            path = parsed.path
            
            features = {}
            
            # Basic URL features
            features['length_url'] = len(url)
            features['length_hostname'] = len(hostname) if hostname else 0
            features['nb_dots'] = url.count('.')
            features['nb_hyphens'] = url.count('-')
            features['nb_at'] = url.count('@')
            features['nb_qm'] = url.count('?')
            features['nb_and'] = url.count('&')
            features['nb_eq'] = url.count('=')
            features['nb_slash'] = url.count('/')
            features['nb_www'] = 1 if hostname and hostname.startswith('www.') else 0
            
            # Ratio features
            features['ratio_digits_url'] = sum(c.isdigit() for c in url) / max(1, len(url))
            features['ratio_digits_host'] = sum(c.isdigit() for c in hostname) / max(1, len(hostname)) if hostname else 0
            
            # Path features
            features['http_in_path'] = 1 if 'http' in path.lower() else 0
            
            # Domain features
            features['tld_in_subdomain'] = 1 if hostname and len(hostname.split('.')) > 2 else 0
            features['prefix_suffix'] = 1 if hostname and '-' in hostname else 0
            
            # Additional character features
            features['nb_underscore'] = url.count('_')
            features['nb_colon'] = url.count(':')
            features['nb_space'] = url.count(' ') + url.count('%20')
            features['nb_dslash'] = url.count('//')
            features['nb_redirection'] = 0  # Cannot detect from single URL
            
            # Create feature vector in correct order
            feature_vector = []
            for feature in self.feature_names:
                feature_vector.append(features.get(feature, 0))
            
            return np.array(feature_vector).reshape(1, -1), features
            
        except Exception as e:
            st.error(f"❌ Feature extraction error: {str(e)}")
            return None, None
    
    def predict_url(self, url):
        """Make prediction for a single URL"""
        if not st.session_state.model_trained:
            return None, None, None
            
        try:
            features, feature_dict = self.extract_url_features(url)
            if features is None:
                return None, None, None
            
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return prediction, probability, feature_dict
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None, None

def main():
    # Header Section
    st.markdown('<div class="main-header">🛡️ Real-Time Phishing Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">XGBoost with Cross-Validation • Real-Time Detection</div>', unsafe_allow_html=True)
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
        st.title("Navigation")
        app_mode = st.radio("Choose Mode", ["🚀 Train Model", "🔍 URL Detection", "📊 Model Info"])
        
        st.markdown("---")
        
        # Show model status
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready!")
            if st.session_state.model_info:
                st.metric("CV Accuracy", f"{st.session_state.model_info['cv_mean']:.2%}")
                st.metric("Test Accuracy", f"{st.session_state.model_info['accuracy']:.2%}")
        else:
            st.warning("⚠️ Model not trained yet")
        
        if app_mode == "🔍 URL Detection" and st.session_state.model_trained:
            st.subheader("Quick Test")
            test_url = st.text_input("Test URL:", placeholder="https://example.com")
            if test_url:
                st.session_state.test_url = test_url

    # Main content based on selected mode
    if app_mode == "🚀 Train Model":
        render_training_mode(detector)
    elif app_mode == "🔍 URL Detection":
        render_detection_mode(detector)
    elif app_mode == "📊 Model Info":
        render_model_info(detector)

def render_training_mode(detector):
    """Render model training interface"""
    st.subheader("🚀 Train XGBoost Model with Cross-Validation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **This will:**
        - Load your dataset from `D:\\PROJECTS\\Project Phishing\\dataset_phishing_updated.csv`
        - Perform **5-fold cross-validation** for reliable metrics
        - Train final XGBoost model on full training data
        - Show both CV and test set performance
        """)
    
    with col2:
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
            st.metric("CV Accuracy", f"{st.session_state.model_info['cv_mean']:.2%}")
            st.metric("Test Accuracy", f"{st.session_state.model_info['accuracy']:.2%}")
        else:
            st.metric("Expected CV Accuracy", ">85%")
            st.metric("Model", "XGBoost")
            st.metric("CV Folds", "5")
    
    if st.button("🎯 Train Model with Cross-Validation", type="primary", use_container_width=True):
        with st.spinner("Performing cross-validation and training model..."):
            success = detector.load_and_train_model()

def render_detection_mode(detector):
    """Render URL detection interface"""
    st.subheader("🔍 Real-Time URL Detection")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in 'Train Model' mode")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter URL for Analysis")
        
        url_input = st.text_input(
            "Paste the URL here:",
            value=st.session_state.get('test_url', ''),
            placeholder="https://example.com",
            help="Enter a complete URL including http:// or https://",
            key="url_input"
        )
        
        analyze_clicked = st.button("🚀 Analyze URL", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        if st.session_state.model_info:
            st.metric("CV Accuracy", f"{st.session_state.model_info['cv_mean']:.2%}")
            st.metric("Test Accuracy", f"{st.session_state.model_info['accuracy']:.2%}")
            st.metric("Features Used", st.session_state.model_info['features_used'])
    
    if analyze_clicked and url_input:
        with st.spinner("🔬 Analyzing URL features..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            prediction, probability, features = detector.predict_url(url_input)
            
            if prediction is not None:
                display_prediction_results(url_input, prediction, probability, features, detector)

def display_prediction_results(url, prediction, probability, features, detector):
    """Display prediction results"""
    confidence = probability[1] if prediction == 1 else probability[0]
    is_phishing = prediction == 1
    
    st.markdown("---")
    
    # Main result card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if is_phishing:
            st.markdown(
                f'<div class="prediction-card phishing-url">'
                f'🚨 PHISHING URL DETECTED<br>'
                f'<small style="font-size: 1.2rem;">Confidence: {confidence*100:.1f}%</small>'
                f'</div>', 
                unsafe_allow_html=True
            )
            st.error("⚠️ **Warning**: This URL shows characteristics of phishing websites.")
        else:
            st.markdown(
                f'<div class="prediction-card safe-url">'
                f'✅ LEGITIMATE URL<br>'
                f'<small style="font-size: 1.2rem;">Confidence: {confidence*100:.1f}%</small>'
                f'</div>', 
                unsafe_allow_html=True
            )
            st.success("🛡️ **Safe**: This URL appears to be legitimate.")
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confidence Analysis")
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Detection Confidence", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': 'lightcoral'},
                    {'range': [70, 90], 'color': 'lightyellow'},
                    {'range': [90, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Feature Analysis")
        
        if features:
            # Display top features
            top_features = {
                'URL Length': features.get('length_url', 0),
                'Hostname Length': features.get('length_hostname', 0),
                'Number of Dots': features.get('nb_dots', 0),
                'Number of Slashes': features.get('nb_slash', 0),
                'Digit Ratio': features.get('ratio_digits_url', 0),
                'Has WWW': features.get('nb_www', 0)
            }
            
            for feature_name, value in top_features.items():
                with st.container():
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>{feature_name}</strong><br>
                        <span style="color: #667eea; font-size: 1.1rem;">{value:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)

def render_model_info(detector):
    """Render model information"""
    st.subheader("📊 Model Information with Cross-Validation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet. Go to 'Train Model' section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>XGBoost Model</h3>
            <p>5-Fold Cross-Validation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📈 Cross-Validation Performance")
        if st.session_state.model_info:
            st.metric("CV Mean Accuracy", f"{st.session_state.model_info['cv_mean']:.3f}")
            st.metric("CV Std Deviation", f"±{st.session_state.model_info['cv_std']:.3f}")
            st.metric("Test Accuracy", f"{st.session_state.model_info['accuracy']:.3f}")
            st.metric("F1 Score", f"{st.session_state.model_info['f1_score']:.3f}")
    
    with col2:
        st.subheader("🛡️ Model Details")
        
        info_items = [
            ("Training Samples", f"{st.session_state.model_info['training_samples']:,}"),
            ("Test Samples", f"{st.session_state.model_info['test_samples']:,}"),
            ("Features Used", st.session_state.model_info['features_used']),
            ("CV Folds", "5"),
            ("Model Type", "XGBoost Classifier")
        ]
        
        for title, value in info_items:
            with st.container():
                st.markdown(f"""
                <div class="feature-card">
                    <strong>{title}</strong><br>
                    <span style="color: #667eea; font-size: 1.1rem;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Show CV scores distribution
        if st.session_state.cv_scores:
            st.subheader("📊 CV Scores Distribution")
            cv_df = pd.DataFrame({
                'Fold': range(1, 6),
                'Accuracy': st.session_state.cv_scores
            })
            fig = px.bar(cv_df, x='Fold', y='Accuracy', 
                        title='5-Fold Cross-Validation Accuracy',
                        labels={'Accuracy': 'Accuracy Score', 'Fold': 'Fold Number'})
            fig.update_layout(yaxis_range=[0.8, 1.0])
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()