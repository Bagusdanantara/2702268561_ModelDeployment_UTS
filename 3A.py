import pandas as pd
import numpy as np
import pickle

# Load model dan tools preprocessing
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Fungsi untuk inference (bisa dipanggil dari API nanti)
def predict_from_input(input_data):
    """
    input_data: dictionary, misalnya:
    {
        'person_age': 35,
        'person_emp_exp': 10,
        'person_income': 50000,
        'person_home_ownership': 'rent',
        'person_gender': 'male',
        ...
    }
    """
    # Ubah ke dataframe
    df = pd.DataFrame([input_data])

    # Feature engineering yang sama
    df['person_real_exp'] = df['person_age'] - df['person_emp_exp']
    df['person_real_exp'] = df.apply(
        lambda row: row['person_emp_exp'] if row['person_emp_exp'] <= row['person_age']
        else (row['person_real_exp'] if 16 <= row['person_real_exp'] <= 85 else np.nan),
        axis=1
    )
    df['person_real_exp'] = df['person_real_exp'].fillna(df['person_real_exp'].median())

    # Ubah urutan kolom sesuai dengan training
    numerical_cols = ['person_income', 'person_age', 'person_emp_exp', 'person_real_exp']
    categorical_cols = [col for col in df.columns if col not in numerical_cols and col != 'person_real_exp_status']

    # Encode kategorikal pakai encoder yang udah dilatih
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le:
            df[col] = le.transform(df[col].astype(str))
        else:
            raise ValueError(f"Tidak ditemukan encoder untuk kolom: {col}")

    # Scale numerikal pakai scaler
    df[numerical_cols] = scaler.transform(df[numerical_cols].astype(float))

    # Gabungkan kolom final untuk prediksi
    final_cols = categorical_cols + numerical_cols
    X = df[final_cols]

    # Prediksi
    pred = model.predict(X.values)[0]

    # Decode label target jika tersedia
    target_enc = label_encoders.get('loan_status')  # ganti sesuai target aslinya
    if target_enc:
        pred_label = target_enc.inverse_transform([pred])[0]
    else:
        pred_label = pred

    return pred_label

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh input (ganti dengan input aktual)
    sample_input = {
        'person_age': 40,
        'person_emp_exp': 15,
        'person_income': 75000,
        'person_home_ownership': 'rent',
        'person_gender': 'male',
        'loan_intent': 'personal',
        'loan_grade': 'B',
        'loan_amnt': 20000,
        'loan_int_rate': 12.5,
        'loan_percent_income': 0.3,
        'cb_person_default_on_file': 'n',
        'cb_person_cred_hist_length': 5
    }

    prediction = predict_from_input(sample_input)
    print("Prediksi Model:", prediction)
