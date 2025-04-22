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

# 🧪 Sidebar: Test Cases - set session state values when clicked
tc1 = {
    'person_age': 35, 'person_emp_exp': 10, 'person_income': 60000,
    'loan_amnt': 15000, 'loan_int_rate': 11.5, 'loan_percent_income': 0.2,
    'cb_person_cred_hist_length': 7, 'credit_score': 700,
    'person_gender': 'male', 'person_education': 'Bachelor', 'loan_intent': 'EDUCATION',
    'person_home_ownership': 'OWN', 'previous_loan_defaults_on_file': 'No'
}
tc2 = {
    'person_age': 45, 'person_emp_exp': 20, 'person_income': 90000,
    'loan_amnt': 30000, 'loan_int_rate': 9.5, 'loan_percent_income': 0.33,
    'cb_person_cred_hist_length': 15, 'credit_score': 750,
    'person_gender': 'female', 'person_education': 'Master', 'loan_intent': 'HOME',
    'person_home_ownership': 'MORTGAGE', 'previous_loan_defaults_on_file': 'Yes'
}
st.sidebar.title('🧪 Test Cases')
if st.sidebar.button('Load Test Case 1'):
    for k, v in tc1.items(): st.session_state[k] = v
if st.sidebar.button('Load Test Case 2'):
    for k, v in tc2.items(): st.session_state[k] = v

# 🎨 App Header with Emoji
st.title('🌟 Loan Approval Prediction 🌟')
st.markdown('**Isi form berikut untuk mendapatkan prediksi persetujuan pinjaman!** ✍️')

# 🖥️ Main Input Form
with st.form('input_form'):
    st.subheader('🖊️ Masukkan Detail Peminjam')
    inputs = {}
    # Numeric inputs with session state defaults
    inputs['person_age'] = st.number_input('Usia (tahun)', min_value=18, max_value=100,
                                          value=st.session_state['person_age'], key='person_age')
    inputs['person_emp_exp'] = st.number_input('Lama Bekerja (tahun)', min_value=0, max_value=50,
                                               value=st.session_state['person_emp_exp'], key='person_emp_exp')
    inputs['person_income'] = st.number_input('Pendapatan Tahunan',
                                              value=st.session_state['person_income'], key='person_income')
    inputs['loan_amnt'] = st.number_input('Jumlah Pinjaman',
                                          value=st.session_state['loan_amnt'], key='loan_amnt')
    inputs['loan_int_rate'] = st.number_input('Suku Bunga (%)',
                                             value=st.session_state['loan_int_rate'], key='loan_int_rate')
    inputs['loan_percent_income'] = st.number_input('Persentase Pinjaman terhadap Pendapatan',
                                                   value=st.session_state['loan_percent_income'], key='loan_percent_income')
    inputs['cb_person_cred_hist_length'] = st.number_input('Lama Riwayat Kredit (tahun)',
                                                          value=st.session_state['cb_person_cred_hist_length'],
                                                          key='cb_person_cred_hist_length')
    inputs['credit_score'] = st.number_input('Credit Score', min_value=300, max_value=900,
                                              value=st.session_state['credit_score'], key='credit_score')
    # Categorical inputs with session state
    inputs['person_gender'] = st.selectbox('Jenis Kelamin', label_encoders['person_gender'].classes_,
                                           index=list(label_encoders['person_gender'].classes_).index(
                                               st.session_state['person_gender']), key='person_gender')
    inputs['person_education'] = st.selectbox('Pendidikan', label_encoders['person_education'].classes_,
                                              index=list(label_encoders['person_education'].classes_).index(
                                                  st.session_state['person_education']), key='person_education')
    inputs['loan_intent'] = st.selectbox('Tujuan Pinjaman', label_encoders['loan_intent'].classes_,
                                         index=list(label_encoders['loan_intent'].classes_).index(
                                             st.session_state['loan_intent']), key='loan_intent')
    inputs['person_home_ownership'] = st.selectbox('Kepemilikan Rumah', label_encoders['person_home_ownership'].classes_,
                                                   index=list(label_encoders['person_home_ownership'].classes_).index(
                                                       st.session_state['person_home_ownership']),
                                                   key='person_home_ownership')
    inputs['previous_loan_defaults_on_file'] = st.selectbox('Pernah Default Sebelumnya?',
                                                            label_encoders['previous_loan_defaults_on_file'].classes_,
                                                            index=list(label_encoders['previous_loan_defaults_on_file'].classes_)
                                                            .index(st.session_state['previous_loan_defaults_on_file']),
                                                            key='previous_loan_defaults_on_file')

    submit = st.form_submit_button('🚀 Prediksi')

# 🎉 Show result after submission
if submit:
    result = predict(inputs)
    st.success(f'✅ Hasil Prediksi: **{result}**')
