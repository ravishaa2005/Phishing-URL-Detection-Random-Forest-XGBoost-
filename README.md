## PhishScan – Phishing URL Detection Web App

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC4E24?style=for-the-badge&logo=xgboost&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge&logo=tree&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

### Overview
**PhishScan** is a web-based tool that analyzes URLs using trained **Random Forest** and **XGBoost** models to classify them as **Benign**, **Malware**, **Defacement**, or **Phishing**.  

It extracts **17 key features** from the URL and provides an **instant threat prediction** along with **security recommendations** based on the result.


### Core Features
- Real-Time URL Classification using trained ML models  
- Feature Extraction from input URLs (e.g., HTTPS status, domain age, suspicious symbols)  
- Dual Model Integration: XGBoost & Random Forest for enhanced accuracy  
- Security Recommendations based on threat type (phishing, malware, etc.)  
- Flask Web Interface for smooth user interaction  
- Visual Feedback: Displays predicted category and extracted features  
- Responsive UI: Mobile-friendly and accessible

### Dataset
Source: [Malicious URLs Dataset on Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)  
- Total URLs: ~650,000+  
- Categories: Benign, Malware, Phishing, Defacement  
- Features Extracted: 17 (e.g., URL length, presence of IP, '@' symbol, domain age)

### Setup & Configuration
1. Clone the Repository
```bash
git clone https://github.com/ravishaa2005/Phishing-URL-Detection-Random-Forest-XGBoost-.git
cd Phishing-URL-Detection-Random-Forest-XGBoost-
```

2. Create Virtual Environment & Install Dependencies
```bash
pip install -r requirements.txt
```

3. Model Training
```bash
python train_model.py
```

4. Run the Flask App
```bash
python app.py
```

5. Access the App on browser

### Model Accuracy

| Model         | Accuracy |
|---------------|----------|
| Random Forest | 95.3%    |
| XGBoost       | 96.7%    |


### App Screenshots

| Home Page                                | URL Input & Results                       | Feature Table                            |
|------------------------------------------|-------------------------------------------|------------------------------------------|
| ![Home](assets/images/home.png)          | ![Input](assets/images/input_result.png)  | ![Features](assets/images/features.png)  |
