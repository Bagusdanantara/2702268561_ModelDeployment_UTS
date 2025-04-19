# Install libraries if not installed
import os
import subprocess
import sys

# List of required libraries
required_libraries = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'streamlit', 'joblib']

# Check and install missing libraries
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ModelTrainer Class
class ModelTrainer:
    def __init__(self, df, categorical_cols, numerical_cols, target_col, model):
        self.df = df
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        self.X = None
        self.Y = None
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.model = model

    def preprocess_data(self):
        # Preprocessing
        self.df['person_real_exp'] = self.df['person_age'] - self.df['person_emp_exp']
        self.df['person_real_exp'] = self.df.apply(
            lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age']
            else (row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan),
            axis=1
        )
        self.df['person_real_exp_status'] = self.df['person_real_exp'].apply(
            lambda x: 'valid' if pd.notna(x) else 'invalid'
        )
        self.df['person_income'] = self.df['person_income'].fillna(self.df['person_income'].median())
        self.df['cleaned_real_gender'] = self.df['person_gender'].replace({'fe male': 'female', 'Male': 'male'})
        
        self.X = self.df[self.categorical_cols + self.numerical_cols]
        self.Y = self.df[self.target_col]
        
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        self.xtrain = pd.get_dummies(self.xtrain, columns=self.categorical_cols, drop_first=True)
        self.xtest = pd.get_dummies(self.xtest, columns=self.categorical_cols, drop_first=True)

        self.xtest = self.xtest.reindex(columns=self.xtrain.columns, fill_value=0)

        scaler = StandardScaler()
        self.xtrain[self.numerical_cols] = scaler.fit_transform(self.xtrain[self.numerical_cols])
        self.xtest[self.numerical_cols] = scaler.transform(self.xtest[self.numerical_cols])

    def train_xgboost(self):
        self.xgb_model = XGBClassifier(n_estimators=100, random_state=42)
        self.xgb_model.fit(self.xtrain, self.ytrain)

    def evaluate_model(self):
        y_pred = self.xgb_model.predict(self.xtest)
        print("XGBoost Accuracy:", accuracy_score(self.ytest, y_pred))

    def predict(self, input_data):
        input_data = pd.DataFrame(input_data)
        input_data = pd.get_dummies(input_data, columns=self.categorical_cols, drop_first=True)
        input_data = input_data.reindex(columns=self.xtrain.columns, fill_value=0)

        scaler = StandardScaler()
        input_data[self.numerical_cols] = scaler.fit_transform(input_data[self.numerical_cols])

        prediction = self.xgb_model.predict(input_data)
        return prediction

# Streamlit UI
def main():
    # Load model (Assuming XGB model is saved as 'XGB_model.pkl')
    model = joblib.load('XGB_model.pkl')

    # Test Data for Two Cases (Test Cases)
    test_case_1 = {
        'person_age': [35],
        'person_income': [50000],
        'person_emp_exp': [10],
        'loan_amnt': [15000],
        'loan_int_rate': [7.5],
        'loan_percent_income': [30],
        'cb_person_cred_hist_length': [12],
        'credit_score': [700],
        'person_gender': ['male'],
        'person_education': ['Bachelor'],
        'loan_intent': ['PERSONAL'],
        'person_home_ownership': ['OWN'],
        'previous_loan_defaults_on_file': ['No']
    }

    test_case_2 = {
        'person_age': [25],
        'person_income': [35000],
        'person_emp_exp': [3],
        'loan_amnt': [10000],
        'loan_int_rate': [5.5],
        'loan_percent_income': [25],
        'cb_person_cred_hist_length': [8],
        'credit_score': [690],
        'person_gender': ['female'],
        'person_education': ['Master'],
        'loan_intent': ['EDUCATION'],
        'person_home_ownership': ['RENT'],
        'previous_loan_defaults_on_file': ['Yes']
    }

    # Simulating the use of ModelTrainer
    trainer = ModelTrainer(df=None, 
                           categorical_cols=['person_gender', 'person_education', 'loan_intent', 'person_home_ownership', 'previous_loan_defaults_on_file', 'cleaned_real_gender'],
                           numerical_cols=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'person_real_exp'],
                           target_col='loan_status',
                           model=model)

    st.title("Loan Status Prediction")

    # Test Case 1 Prediction
    if st.button('Test Case 1: Predict Loan Status'):
        prediction_1 = trainer.predict(test_case_1)
        if prediction_1 == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

    # Test Case 2 Prediction
    if st.button('Test Case 2: Predict Loan Status'):
        prediction_2 = trainer.predict(test_case_2)
        if prediction_2 == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

if __name__ == '__main__':
    main()
