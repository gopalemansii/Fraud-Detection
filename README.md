# 💳 Credit Card Fraud Detection using Machine Learning
### Handling Highly Imbalanced Data (Production-Oriented Approach)

---

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques on a **highly imbalanced dataset**.

Fraud detection is a real-world classification problem where:

- Fraud cases are extremely rare
- Missing a fraud (False Negative) is very costly
- Accuracy alone is misleading

The dataset contains **284,807 transactions**, out of which only **492 are fraudulent (0.17%)**, making it a classic **imbalanced classification problem**.

---

## 🎯 Problem Statement

Build a machine learning model that:

- Identifies fraudulent transactions accurately
- Minimizes False Negatives (missed frauds)
- Handles extreme class imbalance effectively
- Uses proper evaluation metrics beyond accuracy

---

## 📊 Dataset Information

**Dataset:** Credit Card Fraud Detection Dataset (Kaggle)

- 284,807 transactions
- 30 features:
  - V1–V28 (PCA transformed features)
  - Time
  - Amount
  - Class (Target variable)

Target Variable:
- `0` → Legitimate transaction
- `1` → Fraud transaction

### Class Distribution

| Class        | Count   | Percentage |
|-------------|---------|------------|
| Legitimate  | 284,315 | 99.83%     |
| Fraud       | 492     | 0.17%      |

This dataset is **highly imbalanced**, which makes model training challenging.

---

## ⚠️ Why Accuracy is Not Enough

If a model predicts all transactions as legitimate:

- Accuracy = 99.83%
- Fraud detected = 0%

Therefore, this project focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (important for imbalanced datasets)
- Confusion Matrix

---

## 🧠 Approach

### 1️⃣ Data Preprocessing

- Checked for missing values
- Standardized the `Amount` feature using `StandardScaler`
- Used Stratified Train-Test split
- Handled imbalance using:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Class Weighting
  - Random UnderSampling (for comparison)

---

### 2️⃣ Models Implemented

- Logistic Regression (with class_weight)
- Random Forest
- XGBoost
- Gradient Boosting
- Isolation Forest (Anomaly Detection approach)

---

### 3️⃣ Handling Class Imbalance

Techniques applied:

- SMOTE Oversampling
- Class Weights adjustment
- Threshold tuning (instead of default 0.5)
- Precision-Recall curve optimization

---

## 📈 Model Evaluation Metrics

Since the dataset is imbalanced, evaluation focuses on:

- **Recall (Fraud class)** → Important to catch frauds
- **Precision** → Avoid too many false alarms
- **F1-Score**
- **ROC-AUC Score**
- **PR-AUC Score**
- **Confusion Matrix**

Primary goal:
> Maximize Recall while maintaining reasonable Precision.

---

## 🏆 Best Model Performance

- Model: XGBoost with SMOTE
- ROC-AUC: ~0.98+
- High Recall for Fraud class
- Balanced Precision & Recall

*(Exact results may vary based on hyperparameter tuning.)*

---

## 📂 Project Structure
fraud-detection/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ └── fraud_detection_analysis.ipynb
│
├── models/
│ └── saved_model.pkl
│
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│
├── requirements.txt
└── README.md


🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

XGBoost

Imbalanced-learn

Matplotlib / Seaborn

💡 Key Learnings

Understanding class imbalance

Why accuracy fails in fraud detection

Implementing SMOTE properly

Importance of Precision-Recall curve

Threshold tuning for business optimization

Trade-off between False Positives & False Negatives

🚀 Future Improvements

Deploy model using Flask / FastAPI

Real-time fraud detection API

Model monitoring & drift detection

Feature importance using SHAP

Cost-sensitive learning

Ensemble stacking

🌍 Real-World Applications

Fraud detection is widely used in:

Banking systems

Online payments

E-commerce platforms

Insurance claims

Loan approval systems

This project demonstrates practical handling of real-world imbalanced classification problems.

👩‍💻 Author

Manasi Gopale
Machine Learning Enthusiast
GitHub: https://github.com/gopalemansii

LinkedIn:[ https://github.com/gopalemansii](https://www.linkedin.com/in/mansi-gopale-0926732ba/)
