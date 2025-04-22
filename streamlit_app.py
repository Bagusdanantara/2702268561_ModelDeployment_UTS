import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model
with open('/Users/bagusdanantaras/Downloads/xgb_model_real.pkl', 'rb') as f:
    model = pickle.load(f)
# Load saved scaler
with open('/Users/bagusdanantaras/Downloads/scaler_real.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load saved label encoders
with open('/Users/bagusdanantaras/Downloads/label_encoders_real.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define feature columns
categorical_columns = [
    'person_gender',
    'person_education',
    'loan_intent',
    'person_home_ownership',
    'previous_loan_defaults_on_file'
]
numerical_columns = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'person_real_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

def predict(input_data: dict):
    # Create DataFrame
    df = pd.DataFrame([input_data])

    # Feature engineering: person_real_exp
    df['person_real_exp'] = df['person_age'] - df['person_emp_exp']
    df['person_real_exp'] = df.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age'] else (
            row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan
        ),
        axis=1
    )
    # Impute missing with training mean
    mean_val = scaler.mean_[numerical_columns.index('person_real_exp')]
    df['person_real_exp'] = df['person_real_exp'].fillna(mean_val)

    # Encode categorical features
    for col in categorical_columns:
        le = label_encoders.get(col)
        if le:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            df[col] = le.transform(df[col])
        else:
            st.error(f"Encoder for column {col} not found!")
            return None

    # Scale numerical features
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Reorder columns to match training
    X = df[categorical_columns + numerical_columns]

        # Predict using correctly ordered features
    st.write("Debug: X.columns", X.columns.tolist())
    st.write("Debug: Model.feature_names_in_", model.feature_names_in_.tolist())
    missing_in_X = set(model.feature_names_in_) - set(X.columns)
    if missing_in_X:
        st.write("Missing in X (features the model expects but not in input):", missing_in_X)
    extra_in_X = set(X.columns) - set(model.feature_names_in_)
    if extra_in_X:
        st.write("Extra in X (input features not expected by model):", extra_in_X)

    pred = model.predict(X)[0]

    # Decode prediction
    target_le = label_encoders.get('loan_status')
    if target_le:
        return target_le.inverse_transform([pred])[0]
    return pred

# Streamlit UI setup
st.set_page_config(page_title='Loan Approval Predictor', layout='wide')
st.title('📊 Loan Approval Prediction')

# Input form
with st.form(key='input_form'):
    st.subheader('Masukkan Detail Peminjam')
    person_age = st.number_input('Usia', min_value=18, max_value=100, value=30)
    person_emp_exp = st.number_input('Lama Bekerja (tahun)', min_value=0, max_value=50, value=5)
    person_income = st.number_input('Pendapatan Tahunan', value=50000)
    loan_amnt = st.number_input('Jumlah Pinjaman', value=10000)
    loan_int_rate = st.number_input('Suku Bunga (%)', value=13.5)
    loan_percent_income = st.number_input('Pinjaman sebagai % Pendapatan', value=0.25)
    cb_person_cred_hist_length = st.number_input('Lama Riwayat Kredit (tahun)', value=5)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    person_gender = st.selectbox('Jenis Kelamin', ['male', 'female'])
    person_education = st.selectbox('Pendidikan', ['High School', 'Bachelor', 'Master', 'PhD', 'Other'])
    loan_intent = st.selectbox('Tujuan Pinjaman', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOME', 'DEBT_CONSOLIDATION'])
    person_home_ownership = st.selectbox('Status Kepemilikan Rumah', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    previous_loan_defaults_on_file = st.selectbox('Pernah Default Sebelumnya?', ['Yes', 'No'])

    submit = st.form_submit_button('Prediksi')

if submit:
    input_data = {
        'person_age': person_age,
        'person_emp_exp': person_emp_exp,
        'person_income': person_income,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'person_gender': person_gender,
        'person_education': person_education,
        'loan_intent': loan_intent,
        'person_home_ownership': person_home_ownership,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    result = predict(input_data)
    if result is not None:
        st.success(f'Hasil Prediksi: **{result}**')

# Sidebar Test Cases
st.sidebar.header('Test Case')
if st.sidebar.button('Test Case 1'):
    tc1 = {
        'person_age': 35,
        'person_emp_exp': 10,
        'person_income': 60000,
        'loan_amnt': 15000,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 7,
        'credit_score': 700,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'loan_intent': 'EDUCATION',
        'person_home_ownership': 'OWN',
        'previous_loan_defaults_on_file': 'No'
    }
    st.sidebar.write('Prediksi TC1:', predict(tc1))

if st.sidebar.button('Test Case 2'):
    tc2 = {
        'person_age': 45,
        'person_emp_exp': 20,
        'person_income': 90000,
        'loan_amnt': 30000,
        'loan_int_rate': 9.5,
        'loan_percent_income': 0.33,
        'cb_person_cred_hist_length': 15,
        'credit_score': 750,
        'person_gender': 'female',
        'person_education': 'Master',
        'loan_intent': 'HOME',
        'person_home_ownership': 'MORTGAGE',
        'previous_loan_defaults_on_file': 'Yes'
    }
    st.sidebar.write('Prediksi TC2:', predict(tc2))

