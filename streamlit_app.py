import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Class untuk load model dan inference
class ModelInference:
    def __init__(self, model_path):
        # Memuat model yang telah disimpan
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def preprocess(self, input_data):
       #memisahkan data categorical dan numerical
        numeric_data = input_data.select_dtypes(include=['float64', 'int64'])
        categorical_data = input_data.select_dtypes(include=['object'])
        
        # Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_data)

        #one-hot encoding pada categorical data
        categorical_data_encoded = pd.get_dummies(categorical_data, drop_first=True)
        #merge data numeric yg sudah discale dan data categorical yg sudah di encode
        final_data = pd.concat([pd.DataFrame(numeric_data_scaled, columns=numeric_data.columns), categorical_data_encoded], axis=1)
        return final_data
    
    def predict(self, input_data):
        # Mengambil input, lakukan preprocessing, kemudian prediksi
        processed_data = self.preprocess(input_data)
        prediction = self.model.predict(processed_data)
        return prediction

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title('Loan Status Prediction App')

    # Input Form
    person_age = st.number_input("Enter Age:", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Enter Income:", min_value=1000, max_value=1000000, value=50000)
    loan_amnt = st.number_input("Enter Loan Amount:", min_value=1000, max_value=500000, value=10000)
    loan_int_rate = st.number_input("Enter Loan Interest Rate (%):", min_value=1.0, max_value=30.0, value=12.0)
    credit_score = st.number_input("Enter Credit Score:", min_value=300, max_value=850, value=650)

    # Simulasi input kategorikal (misalnya bisa menggunakan dropdown untuk memilih gender, status rumah, dll)
    person_gender = st.selectbox("Select Gender:", ['male', 'female'], index=1)

    # Membuat DataFrame untuk input
    new_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'credit_score': [credit_score],
        'person_gender': [person_gender]
    })
    
    # Load Model dan buat prediksi
    inference_model = ModelInference("/Users/bagusdanantaras/Downloads/best_model_xgboost.pkl")
    if st.button("Predict"):
        prediction = inference_model.predict(new_data)
        st.write(f"Predicted Loan Status: {'Approved' if prediction[0] == 1 else 'Denied'}")

if __name__ == "__main__":
    main()
