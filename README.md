# 💳 AI Credit Card Fraud Detection System

An interactive machine learning web application that predicts credit card transaction fraud in real time using a trained Random Forest classifier and Streamlit.

---

## 📌 Project Overview
Credit card fraud detection poses a major challenge due to extreme class imbalance (fraudulent transactions make up < 0.2% of all activity). This project builds an end-to-end Machine Learning pipeline:
* Preprocesses transaction data and scales features using `RobustScaler`.
* Trains a high-precision Random Forest model to minimize false positives.
* Serves real-time inference through a user-friendly Streamlit web application.

---

## 📁 Repository Structure
assets/ - Visualizations & UI previews
app.py - Streamlit web application script
credit_card_fraud_model.pkl - Saved Random Forest classifier model
robust_scaler.pkl - Saved RobustScaler preprocessing object
requirements.txt - Python dependencies list
README.md - Project documentation

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Streamlit Application
streamlit run app.py

After executing, open your browser and navigate to http://localhost:8501.

---

## 📊 Model Performance
* Algorithm: Random Forest Classifier
* Precision: ~0.96
* PR-AUC: ~0.865
* Key Feature Handling: Robust scaling on skewed distributions (Amount & Time), baseline zero-padding on remaining PCA parameters.
