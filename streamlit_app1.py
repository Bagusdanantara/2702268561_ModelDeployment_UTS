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

# Define feature lists (must match training!)
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

def predict(input_data: dict) -> str:
    #  Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Feature engineering: create person_real_exp
    df['person_real_exp'] = df['person_age'] - df['person_emp_exp']
    df['person_real_exp'] = df.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age'] else (
            row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan
        ),
        axis=1
    )
    # Impute missing feature with training mean
    mean_val = scaler.mean_[numerical_columns.index('person_real_exp')]
    df['person_real_exp'] = df['person_real_exp'].fillna(mean_val)

    # Encode categorical features with saved encoders
    for col in categorical_columns:
        le = label_encoders.get(col)
        if not le:
            raise ValueError(f"Encoder for '{col}' not found!")
        # 🍂 Replace unseen values with 'unknown'
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')
        df[col] = le.transform(df[col])

    # Scale numeric features
df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Prepare array for model input
    cat_vals = df[categorical_columns].values
    num_vals = df[numerical_columns].values
    X_input = np.hstack([cat_vals, num_vals])

    # Predict and decode
    pred = model.predict(X_input)[0]
    target_le = label_encoders.get('loan_status')
    if target_le:
        return target_le.inverse_transform([pred])[0]
    return str(pred)

# Streamlit App Configuration
st.set_page_config(
    page_title='🌟 Loan Approval Predictor',
    layout='centered'
)

# App Header with Emoji
st.title('🌟 Loan Approval Prediction 🌟')
st.markdown('**Isi form berikut untuk mendapatkan prediksi persetujuan pinjaman!** ✍️')

# Main Input Form
with st.form('input_form'):
    st.subheader('🖊️ Masukkan Detail Peminjam')
    inputs = {}
    # Numeric inputs
    inputs['person_age'] = st.number_input('Usia (tahun)', min_value=18, max_value=100, value=30)
    inputs['person_emp_exp'] = st.number_input('Lama Bekerja (tahun)', min_value=0, max_value=50, value=5)
    inputs['person_income'] = st.number_input('Pendapatan Tahunan', value=50000)
    inputs['loan_amnt'] = st.number_input('Jumlah Pinjaman', value=10000)
    inputs['loan_int_rate'] = st.number_input('Suku Bunga (%)', value=13.5)
    inputs['loan_percent_income'] = st.number_input('Persentase Pinjaman terhadap Pendapatan', value=0.25)
    inputs['cb_person_cred_hist_length'] = st.number_input('Lama Riwayat Kredit (tahun)', value=5)
    inputs['credit_score'] = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    # Categorical inputs
    inputs['person_gender'] = st.selectbox('Jenis Kelamin', label_encoders['person_gender'].classes_.tolist())
    inputs['person_education'] = st.selectbox('Pendidikan', label_encoders['person_education'].classes_.tolist())
    inputs['loan_intent'] = st.selectbox('Tujuan Pinjaman', label_encoders['loan_intent'].classes_.tolist())
    inputs['person_home_ownership'] = st.selectbox('Kepemilikan Rumah', label_encoders['person_home_ownership'].classes_.tolist())
    inputs['previous_loan_defaults_on_file'] = st.selectbox('Pernah Default Sebelumnya?', label_encoders['previous_loan_defaults_on_file'].classes_.tolist())

    submit = st.form_submit_button('🚀 Prediksi')

# Show result after submission
if submit:
    result = predict(inputs)
    st.success(f'✅ Hasil Prediksi: **{result}**')

# Sidebar Test Cases
st.sidebar.title('🧪 Test Cases')

# Inisialisasi hasil test case
tc1_result = None
tc2_result = None

# Tekan untuk menjalankan Test Case 1
tc1_data = {
    'person_age': 35, 'person_emp_exp': 10, 'person_income': 60000,
    'loan_amnt': 15000, 'loan_int_rate': 11.5, 'loan_percent_income': 0.2,
    'cb_person_cred_hist_length': 7, 'credit_score': 700,
    'person_gender': 'male', 'person_education': 'Bachelor', 'loan_intent': 'EDUCATION',
    'person_home_ownership': 'OWN', 'previous_loan_defaults_on_file': 'No'
}
if st.sidebar.button('Test Case 1'):
    tc1_result = predict(tc1_data)

# Tekan untuk menjalankan Test Case 2
tc2_data = {
    'person_age': 45, 'person_emp_exp': 20, 'person_income': 90000,
    'loan_amnt': 30000, 'loan_int_rate': 9.5, 'loan_percent_income': 0.33,
    'cb_person_cred_hist_length': 15, 'credit_score': 750,
    'person_gender': 'female', 'person_education': 'Master', 'loan_intent': 'HOME',
    'person_home_ownership': 'MORTGAGE', 'previous_loan_defaults_on_file': 'Yes'
}
if st.sidebar.button('Test Case 2'):
    tc2_result = predict(tc2_data)

# Tampilkan hasil Test Case di area utama jika ada
if tc1_result is not None:
    st.info(f'🧪 Test Case 1 Prediksi: {tc1_result}')
if tc2_result is not None:
    st.info(f'🧪 Test Case 2 Prediksi: {tc2_result}')

