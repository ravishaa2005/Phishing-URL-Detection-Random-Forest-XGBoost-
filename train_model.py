import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import tldextract
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Precompile regular expressions
IP_PATTERN = re.compile(r'\d+\.\d+\.\d+\.\d+')

# Cache for tldextract to avoid repeated DNS lookups
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
        
        return features
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return None

def process_url_batch(urls):
    return [extract_features(url) for url in urls]

def add_legitimate_examples(df):
    # Add known legitimate websites
    legitimate_urls = [
        'https://www.google.com',
        'https://www.microsoft.com',
        'https://www.apple.com',
        'https://www.amazon.com',
        'https://www.facebook.com',
        'https://www.youtube.com',
        'https://www.twitter.com',
        'https://www.linkedin.com',
        'https://www.github.com',
        'https://www.netflix.com',
        'https://www.paypal.com',
        'https://www.instagram.com'
    ]
    
    legitimate_df = pd.DataFrame({
        'url': legitimate_urls,
        'type': ['benign'] * len(legitimate_urls)
    })
    
    # Add these examples multiple times to increase their weight
    legitimate_df = pd.concat([legitimate_df] * 5, ignore_index=True)
    
    return pd.concat([df, legitimate_df], ignore_index=True)

def main():
    print("Loading dataset...")
    try:
        # Load the dataset
        df = pd.read_csv('dataset.csv')
        print(f"Loaded {len(df)} samples from dataset.csv")
        
        # Add legitimate examples
        df = add_legitimate_examples(df)
        print(f"Added legitimate examples. Total samples: {len(df)}")
        
        # Check if the required columns exist
        if 'url' not in df.columns or 'type' not in df.columns:
            raise ValueError("Dataset must contain 'url' and 'type' columns")
        
        print("Extracting features using parallel processing...")
        # Calculate optimal batch size and number of workers
        num_samples = len(df)
        batch_size = max(1, num_samples // (os.cpu_count() or 1))
        
        # Split URLs into batches
        url_batches = [df['url'][i:i + batch_size] for i in range(0, len(df), batch_size)]
        
        # Process batches in parallel
        features_list = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_url_batch, batch) for batch in url_batches]
            
            # Use tqdm for progress bar
            for future in tqdm(futures, total=len(futures), desc="Processing URLs"):
                batch_features = future.result()
                features_list.extend([f for f in batch_features if f is not None])
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        
        # Remove rows with missing features
        valid_indices = X.index
        df_filtered = df.iloc[valid_indices]
        
        print("Encoding labels...")
        le = LabelEncoder()
        y = le.fit_transform(df_filtered['type'])
        
        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train the model
        print("Training model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        # Save the model and label encoder
        print("Saving model...")
        model_data = {
            'model': rf_model,
            'label_encoder': le
        }
        joblib.dump(model_data, 'url_security_model.joblib', compress=3)
        print("Model saved successfully!")
        
    except FileNotFoundError:
        print("Error: dataset.csv file not found in the current directory")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 