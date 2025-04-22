import streamlit as st  # 🚀 Streamlit framework
# 🚀 Set page config must be first Streamlit command
st.set_page_config(
    page_title='🌟 Loan Approval Predictor',
    layout='centered'
)
import pandas as pd       # 📊 Data manipulation
import numpy as np        # 🔢 Numerical operations
import pickle             # 🗄️ Model serialization

# 🎁 Load the trained model and preprocessing objects
st.sidebar.header('🔄 Upload Model Files')
model_file = st.sidebar.file_uploader('Upload model (xgb_model.pkl)', type=['pkl'])
scaler_file = st.sidebar.file_uploader('Upload scaler (scaler.pkl)', type=['pkl'])
encoders_file = st.sidebar.file_uploader('Upload label encoders (label_encoders.pkl)', type=['pkl'])

if not model_file or not scaler_file or not encoders_file:
    st.sidebar.warning('⚠️ Silakan upload ketiga file pkl di atas untuk menjalankan aplikasi.')
    st.stop()

# Load from uploaded files
model = pickle.load(model_file)
scaler = pickle.load(scaler_file)
label_encoders = pickle.load(encoders_file)

# ✨ Define feature lists (must match training!)
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
    # 📝 Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # 🛠️ Feature engineering: create person_real_exp
    df['person_real_exp'] = df['person_age'] - df['person_emp_exp']
    df['person_real_exp'] = df.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age'] else (
            row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan
        ),
        axis=1
    )
    # 🎯 Impute missing feature with training mean
    mean_val = scaler.mean_[numerical_columns.index('person_real_exp')]
    df['person_real_exp'] = df['person_real_exp'].fillna(mean_val)

    # 🔄 Encode categorical features with saved encoders
    for col in categorical_columns:
        le = label_encoders.get(col)
        if not le:
            raise ValueError(f"Encoder for '{col}' not found!")
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')
        df[col] = le.transform(df[col])

    # 🔢 Scale numeric features using numpy array
    num_array = scaler.transform(df[numerical_columns].values)
    for idx, col in enumerate(numerical_columns):
        df[col] = num_array[:, idx]

    # 👉 Prepare array for model input
    cat_vals = df[categorical_columns].values
    num_vals = df[numerical_columns].values
    X_input = np.hstack([cat_vals, num_vals])

    # 🎯 Predict and decode
    pred = model.predict(X_input)[0]
    target_le = label_encoders.get('loan_status')
    if target_le:
        return target_le.inverse_transform([pred])[0]
    return str(pred)

# 🔄 Initialize default input values in session state
default_vals = {
    'person_age': 30,
    'person_emp_exp': 5,
    'person_income': 50000,
    'loan_amnt': 10000,
    'loan_int_rate': 13.5,
    'loan_percent_income': 0.25,
    'cb_person_cred_hist_length': 5,
    'credit_score': 650,
    'person_gender': label_encoders['person_gender'].classes_[0],
    'person_education': label_encoders['person_education'].classes_[0],
    'loan_intent': label_encoders['loan_intent'].classes_[0],
    'person_home_ownership': label_encoders['person_home_ownership'].classes_[0],
    'previous_loan_defaults_on_file': label_encoders['previous_loan_defaults_on_file'].classes_[0]
}
for key, val in default_vals.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 🧪 Sidebar: Test Cases
st.sidebar.header('🧪 Test Cases')

# Define Test Case inputs directly for quick validation
if st.sidebar.button('Test Case 1'):
    tc1 = {
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'loan_intent': 'EDUCATION',
        'person_home_ownership': 'OWN',
        'previous_loan_defaults_on_file': 'No',
        'person_age': 35,
        'person_emp_exp': 10,
        'person_income': 60000,
        'loan_amnt': 15000,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 7,
        'credit_score': 700
    }
    pred1 = predict(tc1)
    st.sidebar.success(f'🧪 Test Case 1 Prediksi: {pred1}')

if st.sidebar.button('Test Case 2'):
    tc2 = {
        'person_gender': 'female',
        'person_education': 'Master',
        'loan_intent': 'HOME',
        'person_home_ownership': 'MORTGAGE',
        'previous_loan_defaults_on_file': 'Yes',
        'person_age': 45,
        'person_emp_exp': 20,
        'person_income': 90000,
        'loan_amnt': 30000,
        'loan_int_rate': 9.5,
        'loan_percent_income': 0.33,
        'cb_person_cred_hist_length': 15,
        'credit_score': 750
    }
    pred2 = predict(tc2)
    st.sidebar.success(f'🧪 Test Case 2 Prediksi: {pred2}')

# 🚀 Main Predict Button
if st.button('🚀 Prediksi'):
    result = predict(inputs)
    st.success(f'✅ Hasil Prediksi: **{result}**')
