import pickle
import pandas as pd
import numpy as np

# Load model dan preprocessing tools
with open("xgboost_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data['model']
    label_encoders = data['label_encoders']
    scaler = data['scaler']

# input col
categorical_column = ['person_gender', 'person_education', 'loan_intent', 'person_home_ownership',
                      'previous_loan_defaults_on_file']
numerical_column = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

def predict(input_data: dict):
    df = pd.DataFrame([input_data])

    # Encoding kolom kategorikal
    for col in categorical_column:
        le = label_encoders.get(col)
        if le:
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            df[col] = le.transform(df[col])
        else:
            raise ValueError(f"Encoder untuk kolom {col} tidak ditemukan!")

    # Scaling kolom numerik
    df[numerical_column] = scaler.transform(df[numerical_column])

    # Prediksi
    prediction = model.predict(df)

    # Decode hasil prediksi ke label asli
    target_encoder = label_encoders.get('loan_status')
    if target_encoder:
        prediction_label = target_encoder.inverse_transform(prediction)[0]
    else:
        prediction_label = prediction[0]

    return prediction_label

# use case 
if __name__ == "__main__":
    input_sample = {
        "person_gender": "female",
        "person_education": "High School",
        "loan_intent": "PERSONAL",
        "person_home_ownership": "RENT",
        "previous_loan_defaults_on_file": "No",
        "person_age": 28,
        "person_income": 48000,
        "person_emp_exp": 5,
        "loan_amnt": 12000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.25,
        "cb_person_cred_hist_length": 3,
        "credit_score": 690
    }

    result = predict(input_sample)
    print("Hasil prediksi:", result)
