# 💳 PaySim Fraud Detection System
### Handling Highly Imbalanced Data with Machine Learning

---

## 🚀 Project Overview

This project builds a robust machine learning pipeline to detect fraudulent mobile money transactions using the **PaySim dataset**, which simulates real-world financial transaction behavior.

Fraud detection is a critical financial security problem where:

- Fraud cases are extremely rare (highly imbalanced dataset)
- Missing a fraud (False Negative) leads to direct financial loss
- High false positives harm customer experience
- Accuracy alone is misleading

This project focuses on **cost-sensitive learning**, **class imbalance handling**, and **business-driven evaluation metrics**.

---

## 🎯 Business Problem

Digital payment platforms process millions of transactions daily. Even a small fraud detection failure rate can lead to significant losses.

The goal is to:

- Detect fraudulent transactions accurately
- Minimize False Negatives (avoid financial loss)
- Maintain reasonable False Positives (avoid customer friction)
- Use evaluation metrics suitable for imbalanced datasets

---

## 📊 Dataset Information

**Dataset:** PaySim Mobile Money Fraud Detection Dataset (Kaggle)

PaySim is a synthetic dataset generated using real-world mobile money transaction patterns.

- ~6.3 million transactions
- 11 features
- Transaction types include:
  - CASH_IN
  - CASH_OUT
  - TRANSFER
  - PAYMENT
  - DEBIT

Target variable:

- `0` → Legitimate
- `1` → Fraud

### Key Features

- `type` → Transaction type  
- `amount` → Transaction amount  
- `oldbalanceOrg` → Sender balance before transaction  
- `newbalanceOrig` → Sender balance after transaction  
- `oldbalanceDest` → Receiver balance before transaction  
- `newbalanceDest` → Receiver balance after transaction  
- `isFraud` → Fraud label (Target)  
- `isFlaggedFraud` → System flagged fraud indicator  

---

## ⚠️ Class Imbalance Challenge

Fraud transactions represent a very small percentage of the dataset, making it a **highly imbalanced classification problem**.

This imbalance causes models to become biased toward predicting legitimate transactions.

---

## ⚠️ Why Accuracy is Misleading

If a model predicts all transactions as legitimate:

- Accuracy will still be extremely high
- Fraud detection rate = 0%

Therefore, this project evaluates models using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (preferred for imbalanced data)
- Confusion Matrix

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing

- Checked for missing values
- Encoded categorical feature `type`
- Standardized `amount` feature using StandardScaler
- Stratified train-test split
- Feature engineering (balance difference errors)
- Ensured no data leakage

---

### 2️⃣ Handling Class Imbalance

Multiple techniques were implemented and compared:

- SMOTE (Synthetic Minority Oversampling Technique)
- Random UnderSampling
- Class Weight adjustment
- Threshold tuning (custom decision threshold)
- Precision-Recall curve optimization

---

### 3️⃣ Models Implemented

- Logistic Regression (with class_weight)
- Random Forest
- Gradient Boosting
- XGBoost
- Isolation Forest (Anomaly Detection approach)

---

## 📈 Model Evaluation Strategy

Primary optimization focus:

> Maximize Recall (Fraud class) while maintaining acceptable Precision.

Key Metrics:

- Confusion Matrix
- Recall (Fraud Detection Rate)
- Precision
- F1-Score
- ROC-AUC Score
- PR-AUC Score

---

## 🏆 Best Model Performance

Best performing model:

**XGBoost with SMOTE + Threshold Tuning**

- ROC-AUC: ~0.98+
- High Recall for Fraud class
- Balanced Precision-Recall tradeoff
- Reduced False Negatives significantly

---

## 📊 Business Impact Perspective

In real-world systems:

- False Negative (FN) → Direct financial loss
- False Positive (FP) → Customer dissatisfaction & operational cost

This project prioritizes reducing False Negatives while controlling False Positives to maintain business balance.

---

## 📂 Project Structure

```text
paysim-fraud-detection/
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned & transformed data
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── xgboost_model.pkl
├── reports/
│   └── performance_metrics.png
├── requirements.txt
└── README.md

## 📂 Project Structure


## 🛠 Tech Stack

- Python

- Pandas

- NumPy

- Scikit-learn

- XGBoost

- Imbalanced-learn

- Matplotlib

- Seaborn

## 🚀 Future Improvements

- Deploy model using Flask / FastAPI

- Real-time fraud scoring API

- Model monitoring & drift detection

- Explainability using SHAP

- Cost-sensitive learning optimization

- Ensemble stacking

- Automated ML pipeline integration

## 🌍 Real-World Applications

- Fraud detection systems are used in:

- Banking & financial services

- Online payment gateways

- E-commerce platforms

- Insurance claim verification

- Loan approval systems

This project simulates a production-grade fraud detection workflow.

## 👩‍💻 Author

Manasi Gopale
Machine Learning Enthusiast

GitHub: https://github.com/gopalemansii

LinkedIn:https://www.linkedin.com/in/mansi-gopale-0926732ba/

