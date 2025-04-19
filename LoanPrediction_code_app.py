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
    def __init__(self, df, categorical_cols, numerical_cols, target_col, model=None):
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
        self.model = model if model else XGBClassifier(n_estimators=100, random_state=42)

    def preprocess_data(self):
        # Preprocessing, deteksi outlier dan handling missing data
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
        
        # Pastikan kolom sudah ada dan digunakan
        self.X = self.df[self.categorical_cols + self.numerical_cols]  # Gabungkan kolom kategorikal dan numerik
        self.Y = self.df[self.target_col]  # Kolom target
        
        # Membagi data menjadi train dan test set
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        # One-Hot Encoding untuk kolom kategorikal
        self.xtrain = pd.get_dummies(self.xtrain, columns=self.categorical_cols, drop_first=True)
        self.xtest = pd.get_dummies(self.xtest, columns=self.categorical_cols, drop_first=True)

        # Menyelaraskan kolom antara xtrain dan xtest jika perlu
        self.xtest = self.xtest.reindex(columns=self.xtrain.columns, fill_value=0)

        # Scaling kolom numerik
        scaler = StandardScaler()
        self.xtrain[self.numerical_cols] = scaler.fit_transform(self.xtrain[self.numerical_cols])
        self.xtest[self.numerical_cols] = scaler.transform(self.xtest[self.numerical_cols])

    def train_model(self):
        # Jika model tidak diinisialisasi, menggunakan XGBoost
        if self.model is None:
            self.model = XGBClassifier(n_estimators=100, random_state=42)
        
        # Latih model XGBoost
        self.model.fit(self.xtrain, self.ytrain)

    def evaluate_model(self):
        # Evaluasi model
        y_pred = self.model.predict(self.xtest)
        print("Model Accuracy:", accuracy_score(self.ytest, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.ytest, y_pred))
        print("Classification Report:\n", classification_report(self.ytest, y_pred))

    def predict(self, input_data):
        # Melakukan prediksi dengan model yang sudah dilatih
        input_data = pd.DataFrame(input_data)
        
        # Pastikan kolom kategorikal di-dummies-kan
        input_data = pd.get_dummies(input_data, columns=self.categorical_cols, drop_first=True)
        
        # Menyesuaikan kolom dengan xtrain
        input_data = input_data.reindex(columns=self.xtrain.columns, fill_value=0)
        
        # Normalisasi untuk kolom numerik
        scaler = StandardScaler()
        input_data[self.numerical_cols] = scaler.fit_transform(input_data[self.numerical_cols])

        # Prediksi status pinjaman
        prediction = self.model.predict(input_data)
        return prediction


# Streamlit UI
def main():
    # Membaca data secara manual
    data = {
        'person_age': [30, 45, 36, 50],
        'person_income': [50000, 60000, 55000, 70000],
        'person_emp_exp': [5, 20, 12, 25],
        'loan_amnt': [15000, 20000, 18000, 22000],
        'loan_int_rate': [7.5, 8.0, 7.2, 6.8],
        'loan_percent_income': [30, 35, 30, 28],
        'cb_person_cred_hist_length': [12, 15, 10, 20],
        'credit_score': [700, 720, 690, 750],
        'person_gender': ['male', 'female', 'male', 'female'],
        'person_education': ['Bachelor', 'Master', 'High School', 'Doctorate'],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'PERSONAL', 'HOMEIMPROVEMENT'],
        'person_home_ownership': ['OWN', 'RENT', 'MORTGAGE', 'OWN'],
        'previous_loan_defaults_on_file': ['No', 'Yes', 'No', 'Yes'],
        'loan_status': [1, 0, 1, 0]  # 1 = Approved, 0 = Rejected
    }

    # Membuat DataFrame dari dictionary
    df = pd.DataFrame(data)

    # Mengimpor model yang disimpan
    model = joblib.load('/Users/bagusdanantaras/Downloads/XGB_model.pkl')  # Ganti dengan path yang sesuai ke file model Anda

    # Inisialisasi ModelTrainer dan preprocessing
    trainer = ModelTrainer(df, 
                           categorical_cols=['person_gender', 'person_education', 'loan_intent', 'person_home_ownership', 'previous_loan_defaults_on_file', 'cleaned_real_gender'],
                           numerical_cols=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'person_real_exp'],
                           target_col='loan_status',
                           model=model)
    trainer.preprocess_data()

    st.title("Loan Status Prediction")

    # Formulir input dari pengguna
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
    person_emp_exp = st.number_input("Experience (years)", min_value=0, max_value=50, value=5)
    loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=15000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.0, max_value=20.0, value=7.5)
    loan_percent_income = st.number_input("Loan Percentage from Income (%)", min_value=0.0, max_value=100.0, value=30.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=30, value=12)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    person_gender = st.selectbox("Gender", ['male', 'female'])
    person_education = st.selectbox("Education Level", ['Bachelor', 'Master', 'High School', 'Doctorate'])
    loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    person_home_ownership = st.selectbox("Home Ownership Status", ['OWN', 'RENT', 'MORTGAGE'])
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", ['Yes', 'No'])

    # Button untuk memulai prediksi
    if st.button('Predict Loan Status'):
        # Menyiapkan data input pengguna dalam bentuk DataFrame
        input_data = {
            'person_age': [person_age],
            'person_income': [person_income],
            'person_emp_exp': [person_emp_exp],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
            'credit_score': [credit_score],
            'person_gender': [person_gender],
            'person_education': [person_education],
            'loan_intent': [loan_intent],
            'person_home_ownership': [person_home_ownership],
            'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
        }

        # Melakukan prediksi
        prediction = trainer.predict(input_data)

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

if __name__ == '__main__':
    main()
