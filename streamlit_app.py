import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan tools preprocessing
with open("/Users/bagusdanantaras/Downloads/xgb_model-2.pkl", "rb") as f:
    model = pickle.load(f)

with open("/Users/bagusdanantaras/Downloads/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("/Users/bagusdanantaras/Downloads/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Fungsi prediksi
def predict(input_data):
    df = pd.DataFrame([input_data])

    # Feature engineering
    df['person_real_exp'] = df['person_age'] - df['person_emp_exp']
    df['person_real_exp'] = df.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age']
        else (row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan),
        axis=1
    )
    df['person_real_exp'] = df['person_real_exp'].fillna(df['person_real_exp'].median())

    numerical_cols = ['person_income', 'person_age', 'person_emp_exp', 'person_real_exp']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]

    # Encoding kategori
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            # Pastikan semua nilai dikenal LabelEncoder
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            # Tambahkan 'unknown' ke classes_ jika belum ada
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            df[col] = le.transform(df[col])
        else:
            raise ValueError(f"Tidak ada encoder untuk kolom: {col}")

    # Scaling
    df[numerical_cols] = scaler.transform(df[numerical_cols].astype(float))

    # Gabungkan fitur akhir
    final_cols = categorical_cols + numerical_cols
    X = df[final_cols]

    # Prediksi
    pred = model.predict(X.values)[0]
    label = label_encoders['loan_status'].inverse_transform([pred])[0]

    return label

# UI Streamlit
st.title("📊 XGBoost Loan Prediction App")
st.write("Masukkan data peminjam di bawah ini:")

# Input form
with st.form("prediction_form"):
    person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
    person_emp_exp = st.number_input("Lama Bekerja (tahun)", min_value=0, max_value=80, value=5)
    person_income = st.number_input("Pendapatan per Tahun", value=50000)
    person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ['rent', 'own', 'mortgage', 'other'])
    person_gender = st.selectbox("Jenis Kelamin", ['male', 'female'])
    loan_intent = st.selectbox("Tujuan Pinjaman", ['personal', 'education', 'medical', 'venture', 'home', 'debt_consolidation'])
    loan_grade = st.selectbox("Grade Pinjaman", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input("Jumlah Pinjaman", value=10000)
    loan_int_rate = st.number_input("Suku Bunga (%)", value=13.5)
    loan_percent_income = st.number_input("Persentase Pinjaman terhadap Pendapatan", value=0.25)
    cb_person_default_on_file = st.selectbox("Pernah Default?", ['n', 'y'])
    cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", value=5)

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = {
            'person_age': person_age,
            'person_emp_exp': person_emp_exp,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_gender': person_gender,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length
        }

        result = predict(input_data)
        st.success(f"🎯 Hasil Prediksi: **{result.upper()}**")

# Tambahkan test case (Demo)
st.sidebar.header("🔍 Test Case")
if st.sidebar.button("🧪 Jalankan Test Case 1"):
    test_input_1 = {
        'person_age': 35,
        'person_emp_exp': 10,
        'person_income': 60000,
        'person_home_ownership': 'own',
        'person_gender': 'male',
        'loan_intent': 'education',
        'loan_grade': 'C',
        'loan_amnt': 12000,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.2,
        'cb_person_default_on_file': 'n',
        'cb_person_cred_hist_length': 7
    }
    prediction = predict(test_input_1)
    st.sidebar.write("✅ Prediksi Test Case 1:", prediction)

if st.sidebar.button("🧪 Jalankan Test Case 2"):
    test_input_2 = {
        'person_age': 45,
        'person_emp_exp': 25,
        'person_income': 90000,
        'person_home_ownership': 'mortgage',
        'person_gender': 'female',
        'loan_intent': 'home',
        'loan_grade': 'B',
        'loan_amnt': 30000,
        'loan_int_rate': 9.5,
        'loan_percent_income': 0.33,
        'cb_person_default_on_file': 'y',
        'cb_person_cred_hist_length': 15
    }
    prediction = predict(test_input_2)
    st.sidebar.write("✅ Prediksi Test Case 2:", prediction)
