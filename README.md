# Email & SMS Spam Classifier 📧📱

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/ML-Naive%20Bayes%20%7C%20RandomForest%20%7C%20SVM-orange.svg)](https://scikit-learn.org/stable/)
[![Dataset](https://img.shields.io/badge/Dataset-SMS%20Spam%20Collection-red.svg)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
[![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](https://streamlit.io/)
[![Testing](https://img.shields.io/badge/Testing-pytest-green.svg)](https://pytest.org/)

A **robust SMS and Email spam classifier** using multiple machine learning models including Naive Bayes, Random Forest, and SVM. Provides a **user-friendly Streamlit web interface** for real-time message classification with high accuracy and efficiency.

---

## 🏆 Project Achievements

### 🎯 Core System Performance
- **High Accuracy**: Achieves up to 98% accuracy using ensemble methods
- **Real-time Prediction**: Sub-second classification with caching
- **Flexible Dataset Support**: Handles multiple text datasets
- **Production Ready**: Modular code with clean architecture and error handling

### 🌐 Web Application
- **Interactive UI**: Streamlit-based interface for email and SMS classification
- **Instant Results**: Paste or type messages and see results immediately
- **Responsive Design**: Works seamlessly on desktop and mobile

### 🧠 Advanced ML Pipeline
- **Multi-Model Approach**: Naive Bayes, Random Forest, SVM, and ensemble methods
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature Engineering**: TF-IDF, CountVectorizer, n-grams
- **Model Comparison**: Evaluate accuracy, precision, recall, and F1-score
- **Hyperparameter Tuning**: GridSearch and cross-validation for optimal performance

### 🧪 Testing & Quality Assurance
- **Automated Testing**: pytest framework for all modules
- **Input Validation**: Robust handling of malformed text
- **Performance Monitoring**: Track classification speed and accuracy

---

## 🎯 Project Goals
1. Detect spam in **emails and SMS messages** accurately  
2. Provide **fast real-time predictions**  
3. Implement multiple **machine learning models** for comparison  
4. Deploy a **production-ready Streamlit web application**  

---

## 📊 Dataset
- **SMS Spam Collection Dataset (Kaggle)**: 5,574 SMS messages labeled as spam or ham  
- **Columns**: `label` (spam/ham), `message` (text content)

### Required Files
- `spam.csv` or similar CSV containing labeled messages

---

## 🛠️ Project Structure

Email-Sms-Spam-Classifier/
├── 📁 data/ # Dataset storage
│ └── spam.csv # SMS/Email dataset
├── 📁 src/ # Source code
│ ├── preprocessing.py # Text cleaning & tokenization
│ ├── feature_extraction.py # TF-IDF, CountVectorizer, n-grams
│ ├── ml_models.py # Naive Bayes, SVM, Random Forest models
│ └── classifier.py # Unified training and prediction pipeline
├── 📁 tests/ # Unit tests
│ ├── test_preprocessing.py
│ ├── test_ml_models.py
│ └── pytest.ini
├── 📁 notebooks/ # Jupyter notebooks for experimentation
│ └── spam_classification_analysis.ipynb
├── 🌐 app.py # Streamlit web app
├── 📋 requirements.txt # Dependencies
├── 🧪 pytest.ini # Test configuration
├── 📚 README.md # This documentation
├── 📖 SETUP_GUIDE.md # Installation instructions
├── 🐙 GITHUB_SETUP.md # GitHub deployment guide
└── 🚫 .gitignore # Ignore files


---

## 🚀 Quick Start Guide

### 📥 Installation
1. **Clone the repository**
```bash
git clone https://github.com/tarunkarimi/Email-Sms-Spam-Classifier.git
cd Email-Sms-Spam-Classifier
