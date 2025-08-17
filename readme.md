## Phishing URL Detection Web Application 🔐

- A Python-based web application that detects whether a URL is legitimate ✅ or phishing ❌ using machine learning.
-  Built with Streamlit for an intuitive UI and trained with a feature-rich dataset of URLs.

## Features
1. URL Classification
- Predicts whether a given URL is legitimate or phishing
- Uses a trained machine learning model (Logiatic regression and Linear Regression
- Highlights risky patterns commonly used in phishing links

2. Feature Extraction
- Extracts over 30 lexical, domain-based, and content-based features from the URL

Key features include:

- URL length
- Use of '@' symbols, redirections, subdomains
- HTTPS usage

3. Interactive Web Interface
- Built with Streamlit for fast and user-friendly interaction
- Simple textbox to enter any URL
- Real-time prediction with explanation of prediction confidence

4. Model Training Script
- Train your own model using the provided phishing_dataset.csv
- Evaluate accuracy, precision, recall, and F1-score
- Save the trained model as a .pkl file for deployment

Project Structure

    phishing_detector_app/

       ├── app.py                  # Streamlit main application

       ├── main.py

       ├── requirements.txt        # All required Python libraries

      └── README.md               # Project documentation

Setup and Installation
1.Clone the repository


    git clone https://github.com/yourusername/phishing-url-detector.git

    cd phishing-url-detector

2.Install dependencies


    pip install -r requirements.txt

3.Run the application


    streamlit run app.py

## Usage

1.Open the app in your browser using the link Streamlit provides

2.Paste any URL into the input box

3.Click “Check URL” to get prediction and risk assessment

Results will show:

URL Status: ✅ Legitimate or ❌ Phishing


Requirements

     Python 3.8 or higher

     Libraries:

     scikit-learn

     pandas

     numpy

     streamlit

     joblib

