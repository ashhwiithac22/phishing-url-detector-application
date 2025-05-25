# Phishing Website Detection System

A Python-based web application to detect phishing websites using machine learning techniques. This tool analyzes URLs based on several key features and predicts whether a given website is legitimate or a phishing attempt.

---

## Features

### 1. URL-Based Feature Extraction
- Extracts over 30 URL-related features automatically
- Includes URL length, presence of '@', use of IP address, redirections, SSL certificate info, etc.

### 2. Phishing Detection Model
- Trained machine learning model using algorithms such as Random Forest and Logistic Regression
- Fast and accurate predictions
- Supports batch prediction and real-time URL checks

### 3. Streamlit Web Interface
- Clean and interactive web UI
- Input any URL to check for phishing activity
- Displays results and a breakdown of analyzed features

### 4. Modular and Extendable Design
- Feature extraction, model prediction, and UI are modularized
- Easy to integrate with other detection systems or APIs

---

## Project Structure
project phishing/
├──analysis.py
├──main.py 

## Setup and Installation

### 1. Clone the Repository
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector

Install Dependencies
pip install -r requirements.txt

Run the Application
streamlit run app.py

Usage
URL Classification
Launch the web interface

Enter any website URL into the input field

Click the "Predict" button to receive a result:

Legitimate

Phishing

Feature Display
The application may also show key features such as:
Presence of ‘https’
Number of subdomains
Domain age
Length and structure of the URL