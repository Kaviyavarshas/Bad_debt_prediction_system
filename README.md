# Bad_debt_prediction_system

## Project Overview

The **Bad Debt Prediction System** is an end-to-end Machine Learning application developed to predict whether a customer is likely to become a bad debt before a sale is approved.

The system analyzes historical customer and transaction information to estimate customer credit risk and assists organizations in making informed business decisions. By identifying high-risk customers in advance, businesses can reduce financial losses, improve cash flow, and strengthen risk management.

The application is deployed using **Streamlit**, providing an easy-to-use interface for internal users to evaluate customer risk before approving sales.

---

# Business Problem

Organizations frequently sell products or services on credit to customers who later default on payments, resulting in bad debts.

These bad debts can lead to:

- Financial losses
- Poor cash flow
- Increased credit risk
- Reduced profitability
- Inefficient credit management

Currently, there is no predictive mechanism to evaluate customer payment behavior before approving a sale.

This project addresses that challenge by building a Machine Learning model capable of predicting whether a customer is likely to become a bad debt using historical customer and transaction data.

---

# Project Objectives

- Analyze historical customer and transaction data.
- Identify key features contributing to bad debt.
- Perform data cleaning and preprocessing.
- Handle missing values and duplicate records.
- Address class imbalance.
- Train multiple Machine Learning models.
- Compare model performance.
- Select the best-performing model.
- Predict customer credit risk.
- Recommend whether to approve or reject a sale.
- Deploy the model using Streamlit.
- Support business decisions through automated risk prediction.

---

# Solution Overview

The project follows a complete Machine Learning pipeline:

1. Load historical customer data.
2. Clean and preprocess the dataset.
3. Perform feature engineering.
4. Train multiple classification models.
5. Evaluate model performance.
6. Select the best-performing model.
7. Save the trained model.
8. Deploy the prediction system using Streamlit.
9. Predict customer risk in real time.

---

# Machine Learning Models

The project evaluates multiple Machine Learning algorithms, including:

- HistGradientBoosting Classifier
- CatBoost
- XGBoost
- LightGBM

The best-performing model is selected based on evaluation metrics.

---

# Features

- Customer Risk Prediction
- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Feature Scaling
- Class Imbalance Handling
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Streamlit Web Application
- Business Decision Support
- Risk Classification

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- HistGradientBoosting
- CatBoost
- XGBoost
- LightGBM
- Streamlit
- MLflow
- Matplotlib
- Seaborn
- Git
- GitHub
- VS Code

---

# Dataset

The model was trained using historical customer and transaction data containing information relevant to customer payment behavior and credit risk.

**Note:**

The original production dataset (`merged_production_data.parquet`) is **not included** in this repository because it contains confidential business information.

To run this project, replace the dataset path in `app.py` with your own dataset having the same schema.

---

# Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Duplicate removal
- Feature engineering
- Data encoding
- Feature scaling
- Class imbalance handling

---

# Exploratory Data Analysis

EDA was performed to:

- Understand customer behavior
- Analyze feature distributions
- Identify missing values
- Detect correlations
- Understand class imbalance

---

# Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Feature Importance

The best-performing model was selected for deployment.

---

# Prediction Workflow

```
Customer Information
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
Risk Prediction
        │
        ▼
Business Decision
        │
        ├── Allow Purchase
        └── Reject
```

---

# Web Application

The project includes a Streamlit application that allows users to:

- Enter customer information
- Predict customer risk
- View instant results
- Support credit approval decisions

---

# Sample Output

### Example 1

Customer Risk: Low

Decision: **Allow Purchase**

---

### Example 2

Customer Risk: High

Decision: **Reject**

---

# Business Benefits

This solution helps organizations:

- Reduce bad debts
- Improve cash flow
- Improve credit risk assessment
- Support data-driven decisions
- Increase operational efficiency
- Improve profitability

---

# Future Enhancements

- Cloud Deployment
- Real-time Prediction API
- Explainable AI Dashboard
- Automated Model Retraining
- Customer Risk Dashboard
- ERP/CRM Integration
- MLflow Monitoring

---

# Project Structure

```
Bad_debt_prediction_system/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   └── merged_production_data.parquet (Not Included - Confidential)
│
├── models/
│   ├── best_catboost.cbm
│   └── model_features.pkl
│
├── model_building.ipynb
│
├── catboost_info/      (Ignored)
├── mlruns/             (Ignored)
└── mlflow.db           (Ignored)
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Kaviyavarshas/Bad_debt_prediction_system.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Important Note

The production dataset used during development is confidential and is therefore excluded from this repository.

To execute the application successfully, provide your own dataset with the same structure or modify the data loading path in `app.py`.

---

# Author

**Kaviyavarsha S**

Machine Learning Engineer | Data Science Enthusiast

---

# License

This project is licensed under the MIT License.
