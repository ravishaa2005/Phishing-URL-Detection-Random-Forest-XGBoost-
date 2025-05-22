from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import tldextract
import os

app = Flask(__name__)

# Precompile regular expressions and setup caching
IP_PATTERN = re.compile(r'\d+\.\d+\.\d+\.\d+')
tldextract.extract.cache_file = '.tld_set'

# Define keywords and domains
SUSPICIOUS_KEYWORDS = {
    'login', 'signin', 'account', 'verify', 'secure', 'update', 'confirm',
    'password', 'bank', 'billing', 'security', 'pay', 'wallet'
}

LEGITIMATE_DOMAINS = {
    'google', 'facebook', 'amazon', 'microsoft', 'apple', 'netflix',
    'paypal', 'instagram', 'twitter', 'linkedin', 'github', 'youtube'
}

# Global variables for model and label encoder
MODEL = None
LABEL_ENCODER = None

def extract_features(url):
    try:
        # Domain analysis
        ext = tldextract.extract(url)
        domain = ext.domain.lower()
        subdomain = ext.subdomain.lower()
        suffix = ext.suffix.lower()
        
        # Extract all 17 specified features
        features = {
            'url_length': len(url),
            'digit_count': sum(c.isdigit() for c in url),
            'special_char_count': len(re.findall(r'[^\w]', url)),
            'has_https': int(url.startswith("https")),
            'has_ip': int(bool(IP_PATTERN.search(url))),
            'count_dash': url.count('-'),
            'count_at': url.count('@'),
            'count_dot': url.count('.'),
            'suspicious_words': sum(1 for word in SUSPICIOUS_KEYWORDS if word in url.lower()),
            'domain_length': len(domain),
            'subdomain_length': len(subdomain),
            'suffix_length': len(suffix),
            'is_legit_domain': int(domain in LEGITIMATE_DOMAINS),
            'is_suspicious_tld': int(suffix in ['xyz', 'ru', 'top', 'tk', 'gq', 'ml']),
            'keyword_in_domain': int(any(word in domain for word in SUSPICIOUS_KEYWORDS)),
            'keyword_in_subdomain': int(any(word in subdomain for word in SUSPICIOUS_KEYWORDS)),
            'hyphenated_domain': int('-' in domain or '-' in subdomain)
        }
        
        return pd.DataFrame([features])
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return None

# Load model at startup
def initialize_model():
    global MODEL, LABEL_ENCODER
    try:
        print("Loading model... This may take a moment...")
        model_path = os.path.join(os.path.dirname(__file__), 'url_security_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model_data = joblib.load(model_path)
        MODEL = model_data['model']
        LABEL_ENCODER = model_data['label_encoder']
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('index.html', error="Please enter a URL")
            
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://' + url
        
        # Check if model is loaded
        if MODEL is None or LABEL_ENCODER is None:
            return render_template('index.html', error="Model not initialized. Please restart the server.")
        
        # Extract features with error handling
        features = extract_features(url)
        if features is None:
            return render_template('index.html', error="Error processing URL. Please check the URL format and try again.")
        
        try:
            # Get prediction
            prediction_idx = MODEL.predict(features)[0]
            probabilities = MODEL.predict_proba(features)[0]
            confidence = max(probabilities) * 100
            
            # Convert prediction index to label
            prediction = LABEL_ENCODER.inverse_transform([prediction_idx])[0]
            
            # Format feature values for display
            feature_dict = {}
            for column, value in features.iloc[0].items():
                if isinstance(value, (bool, np.bool_)):
                    feature_dict[column] = bool(value)
                elif isinstance(value, (int, np.integer)):
                    feature_dict[column] = int(value)
                elif isinstance(value, (float, np.floating)):
                    feature_dict[column] = round(float(value), 2)
                else:
                    feature_dict[column] = str(value)
            
            # Get recommendations based on prediction
            recommendations = []
            if prediction == "benign":
                recommendations = [
                    "The URL appears to be safe to visit",
                    "Always verify the website's SSL certificate",
                    "Keep your browser and antivirus software up to date"
                ]
            elif prediction == "phishing":
                recommendations = [
                    "Do not enter any personal information on this site",
                    "Do not click on any links within the website",
                    "Report this URL to your security team or relevant authorities"
                ]
            elif prediction == "malware":
                recommendations = [
                    "Do not download any files from this website",
                    "Close the website immediately",
                    "Run a security scan on your system if you've visited this site"
                ]
            elif prediction == "defacement":
                recommendations = [
                    "The website may have been compromised",
                    "Do not trust the content displayed on this site",
                    "Contact the legitimate website owner if you know them"
                ]
            
            # Determine result class for styling
            result_class = "safe" if prediction == "benign" else "danger"
            
            result = {
                "url": url,
                "prediction": prediction,
                "confidence": f"{confidence:.1f}",
                "recommendations": recommendations,
                "class": result_class,
                "features": feature_dict
            }
            
            return render_template('index.html', result=result)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('index.html', error="Error during prediction. Please try again.")
    
    return render_template('index.html')

if __name__ == '__main__':
    if initialize_model():
        app.run(debug=True, host='0.0.0.0', port=5000)  # Make the server externally visible
    else:
        print("Failed to initialize model. Please check the model file and try again.")
