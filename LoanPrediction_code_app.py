import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaller

# Memuat model yang disimpan
with open('XGB_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fungsi untuk preprocessing input dari pengguna
def preprocess_input(data):
    # Menghitung 'person_real_exp' berdasarkan usia dan pengalaman kerja
    data['person_real_exp'] = data['person_age'] - data['person_emp_exp']

    # Validasi nilai person_real_exp (menggunakan threshold 16 dan 85)
    data['person_real_exp'] = data.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age']
        else (row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan),
        axis=1
    )

    # Menandai data yang tidak valid (NaN)
    data['person_real_exp_status'] = data['person_real_exp'].apply(
        lambda x: 'valid' if pd.notna(x) else 'invalid'
    )

    # Menggantikan missing value dengan median untuk kolom person_income
    data['person_income'] = data['person_income'].fillna(data['person_income'].median())

    # Mengubah 'fe male' dan 'Male' menjadi 'female' dan 'male' pada gender
    data['cleaned_real_gender'] = data['person_gender'].replace({'fe male': 'female', 'Male': 'male'})

    return data

# Fungsi untuk melakukan prediksi
def predict(data):
    # Melakukan scaling pada kolom numerik
    numerical_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 
                      'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'person_real_exp']
    
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # One-Hot Encoding untuk kolom kategorikal
    categorical_cols = ['person_gender', 'person_education', 'loan_intent', 'person_home_ownership', 
                        'previous_loan_defaults_on_file', 'cleaned_real_gender']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Prediksi dengan model yang sudah dimuat
    prediction = model.predict(data)
    return prediction

# Streamlit UI
def main():
    st.title("Loan Status Prediction")

    st.write("Masukkan data untuk memprediksi status pinjaman (Approved atau Rejected):")

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

        input_df = pd.DataFrame(input_data)

        # Preprocessing data
        input_df = preprocess_input(input_df)

        # Melakukan prediksi
        prediction = predict(input_df)

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

if __name__ == '__main__':
    main()
